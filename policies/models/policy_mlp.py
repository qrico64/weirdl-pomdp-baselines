""" Recommended Architecture
Separate RNN architecture is inspired by a popular RL repo
https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/POMDP/common/value_networks.py#L110
which has another branch to encode current state (and action)

Hidden state update functions get_hidden_state() is inspired by varibad encoder 
https://github.com/lmzintgraf/varibad/blob/master/models/encoder.py
"""

import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from utils import helpers as utl
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu
from policies.models.recurrent_critic import Critic_RNN
from policies.models.recurrent_actor import Actor_RNN
from policies.models.transformer_encoder_critic import Critic_TransformerEncoder
from policies.models.transformer_encoder_actor import Actor_TransformerEncoder
from utils import logger
from policies.rl import RL_ALGORITHM_PROPERTIES


class ModelFreeOffPolicy_MLP(nn.Module):
    """Recommended Architecture
    Recurrent Actor and Recurrent Critic with separate RNNs
    """

    ARCH = "markov"

    def __init__(
        self,
        obs_dim,
        action_dim,
        encoder,
        algo_name,
        dqn_layers,
        policy_layers,
        lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        # pixel obs
        image_encoder_fn=lambda: None,
        max_len=None,
        feature_extractor_type = 'separate',
        combined_embedding_size: int = None,
        nominal_embedding_size: int = 0,
        num_trajectories: int = 2,
        use_residuals: bool = False,
        **kwargs
    ):
        super().__init__()

        self.algo_name = algo_name
        self.use_value_fn = RL_ALGORITHM_PROPERTIES['use_value_fn'][algo_name]
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.algo = RL_ALGORITHMS[algo_name](**kwargs[algo_name], action_dim=action_dim)
        
        self.use_residuals = use_residuals

        # Critics
        self.qf1, self.qf2 = self.algo.build_critic(
            obs_dim=obs_dim,
            hidden_sizes=dqn_layers,
            action_dim=action_dim,
        )
        self.qf1_optimizer = Adam(self.qf1.parameters(), lr=lr)
        self.qf2_optimizer = Adam(self.qf2.parameters(), lr=lr)
        self.qf1_target = deepcopy(self.qf1)
        self.qf2_target = deepcopy(self.qf2)

        self.value = None
        self.value_optimizer = None
        if self.use_value_fn:
            self.value = self.algo.build_value(
                hidden_sizes=dqn_layers,
                input_size=obs_dim,
            )
            self.value_optimizer = Adam(self.value.parameters(), lr=lr)
            # no need for value target since it's iql

        # Actor
        self.actor = self.algo.build_actor(
            input_size=obs_dim,
            action_dim=action_dim,
            hidden_sizes=policy_layers,
        )
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        # target networks
        self.actor_target = deepcopy(self.actor)

    @torch.no_grad()
    def act(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
        nominals=None,
        base_actions=None,
    ):
        assert base_actions is None
        return self.algo.select_action(
            actor=self.actor,
            observ=obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
        )

    def forward(self, actions, rewards, observs, dones, masks, next_observs, nominals = None, base_actions = None):
        if self.use_value_fn:
            value_loss = self.algo.value_loss(
                markov_actor=True,
                markov_critic=True,
                transformer=False,
                actor=self.actor,
                actor_target=self.actor_target,
                critic=(self.qf1, self.qf2),
                critic_target=(self.qf1_target, self.qf2_target),
                observs=observs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                gamma=self.gamma,
                next_observs=next_observs,
                nominals=nominals,
                base_actions=base_actions,
                masks=masks,
                value_fn=self.value,
            )
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        ### 1. Critic loss
        qf1_loss, qf2_loss = self.algo.critic_loss(
            markov_actor=True,
            markov_critic=True,
            transformer=False,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=(self.qf1, self.qf2),
            critic_target=(self.qf1_target, self.qf2_target),
            observs=observs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
            next_observs=next_observs,
            nominals=nominals,
            base_actions=base_actions,
            masks=masks,
            value_fn=self.value,
        )

        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        ### 2. Actor loss
        policy_loss, log_probs = self.algo.actor_loss(
            markov_actor=True,
            markov_critic=True,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=(self.qf1, self.qf2),
            critic_target=(self.qf1_target, self.qf2_target),
            observs=observs,
            actions=actions,
            rewards=rewards,
            nominals=nominals,
            base_actions=base_actions,
            masks=masks,
            value_fn=self.value,
        )

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        outputs = {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "policy_loss": policy_loss.item(),
        }

        ### 3. soft update
        self.soft_target_update()

        ### 4. update others like alpha
        if log_probs is not None and self.algo_name != "iql":
            if masks is not None:
                num_valid = torch.clamp(masks.sum(), min=1.0)
                # extract valid log_probs
                with torch.no_grad():
                    current_log_probs = (log_probs[:-1] * masks).sum() / num_valid
                    current_log_probs = current_log_probs.item()
            else:
                with torch.no_grad():
                    current_log_probs = log_probs[:-1].mean()
                    current_log_probs = current_log_probs.item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.actor, self.actor_target, self.tau)

    def report_grad_norm(self):
        # may add qf1, policy, etc.
        data = {
            "q1_grad_norm": utl.get_grad_norm(self.qf1),
            "q2_grad_norm": utl.get_grad_norm(self.qf2),
            "pi_grad_norm": utl.get_grad_norm(self.actor),
        }
        if self.use_value_fn:
            data = data | {
                "value_grad_norm": utl.get_grad_norm(self.value),
            }
        return data

    def update(self, batch):
        return self.forward(
            batch["act"],
            batch["rew"],
            batch["obs"],
            batch["term"],
            None,
            batch["obs2"],
            nominals=None,
            base_actions=batch.get("base_actions"),
        )
