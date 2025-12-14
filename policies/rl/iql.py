import torch
from .base import RLAlgorithmBase
from policies.models.actor import TanhGaussianPolicy, DeterministicPolicy
from torchkit.networks import FlattenMlp


class IQL(RLAlgorithmBase):
    """
    Implicit Q-Learning (IQL)
    Paper: https://arxiv.org/abs/2110.06169

    IQL is an offline RL algorithm that uses:
    - Two Q-functions (like SAC/TD3)
    - A value function V(s) trained via expectile regression
    - A policy trained via advantage-weighted behavioral cloning
    """
    name = "iql"
    continuous_action = True
    use_target_actor = False  # IQL doesn't update the actor via target networks

    def __init__(
        self,
        tau=0.7,
        temperature=3.0,
        deterministic_policy=False,
        action_dim=None,
    ):
        """
        Args:
            expectile: Expectile parameter for value function (tau in paper).
                      Default 0.7 means V learns upper expectile of Q values.
            temperature: Temperature (beta) for advantage weighting in policy loss.
                        Higher = more aggressive advantage weighting.
            deterministic_policy: If True, use deterministic policy.
            action_dim: Dimension of action space.
        """
        self.tau = tau
        self.temperature = temperature
        self.deterministic_policy = deterministic_policy

    def update_others(self, **kwargs):
        """IQL doesn't have additional components to update."""
        return {}

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, deterministic_policy=False, **kwargs):
        """
        Build actor network for IQL.

        Args:
            input_size: Dimension of input (observation).
            action_dim: Dimension of action.
            hidden_sizes: List of hidden layer sizes.
            deterministic_policy: If True, use DeterministicPolicy, else TanhGaussianPolicy.
        """
        if deterministic_policy:
            return DeterministicPolicy(
                obs_dim=input_size,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                **kwargs,
            )
        else:
            return TanhGaussianPolicy(
                obs_dim=input_size,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                **kwargs,
            )

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        """
        Build critic networks for IQL: two Q-functions.
        Note: IQL also uses a value function V(s), which should be built separately
        by the calling code (similar to how critics are built).

        Args:
            hidden_sizes: List of hidden layer sizes.
            input_size: Combined obs + action dimension.
            obs_dim: Observation dimension.
            action_dim: Action dimension.
        """
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim

        qf1 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        qf2 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        return qf1, qf2

    @staticmethod
    def build_value(hidden_sizes, input_size=None, obs_dim=None):
        """
        Build value function V(s) for IQL.
        This is a separate network from Q-functions.

        Args:
            hidden_sizes: List of hidden layer sizes.
            input_size: Observation dimension.
            obs_dim: Observation dimension (alternative to input_size).
        """
        if obs_dim is not None:
            input_size = obs_dim

        vf = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        return vf

    def select_action(self, actor, observ, deterministic: bool, return_log_prob: bool = False):
        """
        Select action during evaluation/inference.

        Args:
            actor: The policy network.
            observ: Observation tensor.
            deterministic: Whether to use deterministic action selection.
            return_log_prob: Whether to return log probability.
        """
        if self.deterministic_policy:
            # DeterministicPolicy only returns action
            action = actor(observ)
            return action, None, None, None
        else:
            # TanhGaussianPolicy returns (action, mean, log_std, log_prob)
            return actor(observ, False, deterministic, return_log_prob)

    @staticmethod
    def forward_actor(actor, observ):
        """
        Forward pass through actor for computing actions and log probs.

        Args:
            actor: The policy network.
            observ: Observation tensor.

        Returns:
            new_actions: Sampled actions.
            log_probs: Log probabilities of the actions (None for deterministic).
        """
        # Check if it's a deterministic policy
        if isinstance(actor, DeterministicPolicy):
            new_actions = actor(observ)
            log_probs = None
        else:
            new_actions, _, _, log_probs = actor(observ, return_log_prob=True)
        return new_actions, log_probs

    def forward_actor_log_prob(self, actor, observ, actions):
        assert isinstance(actor, TanhGaussianPolicy)
        log_prob = actor.log_prob(observ, actions)
        return log_prob

    def expectile_loss(self, diff, expectile):
        """
        Asymmetric L2 loss for expectile regression.

        Args:
            diff: Difference (Q - V)
            expectile: Expectile parameter (tau)

        Returns:
            Expectile loss
        """
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)

    def value_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        transformer: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions,
        rewards,
        dones,
        gamma,
        next_observs=None,
        nominals=None,
        base_actions=None,
        masks=None,
        value_fn=None,  # IQL requires value function
    ):
        """
        Compute IQL value function loss via expectile regression.

        V(s) is trained to match the expectile of Q(s, a) over actions in the dataset.

        Args:
            markov_critic: Whether critic is Markovian.
            value_fn: Value function V(s).
            critic: Tuple of (qf1, qf2).
            critic_target: Target critic (not used in IQL value loss).
            observs: Observations.
            actions: Actions from dataset.
            rewards: Rewards (for non-Markovian critics).
            nominals: Nominal actions (for residual policies).

        Returns:
            value_loss: Expectile regression loss.
        """
        if markov_critic:
            # V(s)
            v_pred = value_fn(observs)  # (B, 1) or (T, B, 1)

            # Q(s, a) - use target Q for stability
            with torch.no_grad():
                q1 = critic_target[0](observs, actions)
                q2 = critic_target[1](observs, actions)
                q_target = torch.min(q1, q2)  # (B, 1) or (T, B, 1)
        else:
            # Non-Markovian (recurrent) case
            # V(h(t))
            v_pred = value_fn(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                nominals=nominals,
            )  # (T+1, B, 1)

            # Q(h(t), a(t))
            with torch.no_grad():
                q1, q2 = critic_target(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=actions[1:],
                    nominals=nominals,
                )  # (T, B, 1)
                q_target = torch.min(q1, q2)

            # Align dimensions: V has T+1 timesteps, Q has T timesteps
            v_pred = v_pred[:-1]  # (T, B, 1)
            # this is because the last v_pred is the value of the last next_obs, which we're not dealing with.

        # Expectile loss: asymmetric L2
        diff = q_target - v_pred
        value_loss = self.expectile_loss(diff, self.tau)
        
        # Apply masking and normalization
        if masks is not None:
            num_valid = torch.clamp(masks.sum(), min=1.0)
            value_loss = (value_loss * masks).sum() / num_valid
        else:
            # For Markovian case, just take the mean
            value_loss = value_loss.mean()

        return value_loss

    def critic_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        transformer: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions,
        rewards,
        dones,
        gamma,
        next_observs=None,
        nominals=None,
        base_actions=None,
        masks=None,
        value_fn=None,  # IQL requires value function
    ):
        """
        Compute IQL critic loss.

        Unlike TD3/SAC, IQL uses V(s') instead of Q(s', pi(s')) for the target.
        This avoids out-of-distribution actions in offline RL.

        Q_target(s, a) = r + gamma * V(s')

        Args:
            markov_actor: Whether actor is Markovian (not used in critic loss).
            markov_critic: Whether critic is Markovian.
            transformer: Whether using transformer architecture.
            actor: Policy network (not used in IQL critic loss).
            actor_target: Target policy (not used in IQL critic loss).
            critic: Tuple of (qf1, qf2).
            critic_target: Target critics (not used in IQL, we use value function).
            observs: Observations.
            actions: Actions from dataset.
            rewards: Rewards.
            dones: Done flags.
            gamma: Discount factor.
            next_observs: Next observations (for Markovian critics).
            nominals: Nominal actions (for residual policies).
            base_actions: Base actions (for residual policies).
            value_fn: Value function V(s) - required for IQL.
            masks: Mask tensor for valid timesteps.

        Returns:
            qf1_loss, qf2_loss
        """
        assert value_fn is not None
        # Q_target = r + gamma * (1 - done) * V(s')
        with torch.no_grad():
            if markov_critic:
                # V(s')
                next_v = value_fn(next_observs)  # (B, 1)
            else:
                # Non-Markovian: V(h(t+1))
                next_v = value_fn(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    nominals=nominals,
                )  # (T+1, B, 1)

            # Q target
            q_target = rewards + (1.0 - dones) * gamma * next_v

            if not markov_critic:
                q_target = q_target[1:]  # (T, B, 1)

        # Q predictions
        if markov_critic:
            q1_pred = critic[0](observs, actions)
            q2_pred = critic[1](observs, actions)
        else:
            q1_pred, q2_pred = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=actions[1:],
                nominals=nominals,
            )  # (T, B, 1)

        # Compute losses with masking
        if masks is not None:
            # Apply masking and compute loss
            num_valid = torch.clamp(masks.sum(), min=1.0)
            q1_pred_masked = q1_pred * masks
            q2_pred_masked = q2_pred * masks
            q_target_masked = q_target * masks
            qf1_loss = ((q1_pred_masked - q_target_masked) ** 2).sum() / num_valid
            qf2_loss = ((q2_pred_masked - q_target_masked) ** 2).sum() / num_valid
        else:
            # Markovian case: use mean
            qf1_loss = ((q1_pred - q_target) ** 2).mean()
            qf2_loss = ((q2_pred - q_target) ** 2).mean()

        return qf1_loss, qf2_loss

    def actor_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions=None,
        rewards=None,
        nominals=None,
        base_actions=None,
        masks=None,
        value_fn=None,  # IQL requires value function
    ):
        """
        Compute IQL policy loss via advantage-weighted regression.

        The policy is trained to imitate actions with positive advantage:
        L_policy = -exp(A(s,a) / beta) * log pi(a|s)

        where A(s,a) = Q(s,a) - V(s) and beta is the temperature.

        Args:
            markov_actor: Whether actor is Markovian.
            markov_critic: Whether critic is Markovian.
            actor: Policy network.
            actor_target: Target policy (not used in IQL).
            critic: Tuple of (qf1, qf2).
            critic_target: Target critics.
            observs: Observations.
            actions: Expert/dataset actions.
            rewards: Rewards (for non-Markovian policies).
            nominals: Nominal actions (for residual policies).
            base_actions: Base actions (for residual policies).
            value_fn: Value function V(s) - required for IQL.
            masks: Mask tensor for valid timesteps.

        Returns:
            policy_loss: Advantage-weighted BC loss.
            log_probs: Log probabilities.
        """
        assert value_fn is not None
        # Compute advantage: A(s, a) = Q(s, a) - V(s)
        with torch.no_grad():
            if markov_critic:
                # Q(s, a)
                q1 = critic_target[0](observs, actions)
                q2 = critic_target[1](observs, actions)
                q = torch.min(q1, q2)  # (B, 1)

                # V(s)
                v = value_fn(observs)  # (B, 1)
            else:
                # Non-Markovian
                # Q(h(t), a(t))
                q1, q2 = critic_target(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=actions[1:],
                    nominals=nominals,
                )  # (T, B, 1)
                q = torch.min(q1, q2)

                # V(h(t))
                v = value_fn(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    nominals=nominals,
                )  # (T+1, B, 1)
                v = v[:-1]  # (T, B, 1)

            # Advantage
            adv = q - v  # (B, 1) or (T, B, 1)

            # Advantage weight: exp(A / beta)
            # Clamp for numerical stability
            adv_weight = torch.exp(adv / self.temperature).clamp(max=100.0)

        # Compute log probability of dataset actions under current policy
        if markov_actor:
            if self.deterministic_policy:
                # Deterministic policy: weighted MSE
                pred_actions = actor(observs)

                # MSE per dimension, then sum
                se = (pred_actions - actions) ** 2
                se = se.sum(dim=-1, keepdim=True)  # (B, 1)

                # Weighted MSE
                policy_loss = adv_weight * se
                log_probs = None
            else:
                # Stochastic policy: weighted negative log likelihood
                _, mean, log_std, _ = actor(observs, return_log_prob=False)
                std = torch.exp(log_std)

                # Compute log prob of dataset actions
                from torchkit.distributions import TanhNormal
                tanh_normal = TanhNormal(mean, std)
                log_probs = tanh_normal.log_prob_from_action(actions)
                log_probs = log_probs.sum(dim=-1, keepdim=True)  # (B, 1)

                # Advantage-weighted negative log likelihood
                policy_loss = -adv_weight * log_probs
        else:
            # Non-Markovian (recurrent) policy
            if self.deterministic_policy:
                pred_actions = actor(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    nominals=nominals,
                    base_actions=base_actions,
                )  # (T+1, B, A)

                if base_actions is not None:
                    pred_actions = pred_actions + base_actions

                # Compare with dataset actions
                expert_actions = actions[1:]  # (T, B, A)
                pred_actions_for_loss = pred_actions[:-1]  # (T, B, A)

                se = (pred_actions_for_loss - expert_actions) ** 2
                se = se.sum(dim=-1, keepdim=True)  # (T, B, 1)

                # Weighted MSE
                policy_loss = adv_weight * se
                log_probs = None
            else:
                log_probs = actor.forward_log_prob(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    expert_actions=actions[1:],
                    nominals=nominals,
                    base_actions=base_actions,
                )
                log_probs = log_probs.sum(dim=-1, keepdim=True)  # (T, B, 1)

                # Advantage-weighted negative log likelihood
                policy_loss = -adv_weight * log_probs

        # Apply masking and normalization
        if masks is not None:
            num_valid = torch.clamp(masks.sum(), min=1.0)
            policy_loss = (policy_loss * masks).sum() / num_valid
        else:
            # For Markovian case, just take the mean
            policy_loss = policy_loss.mean()

        return policy_loss, log_probs

    #### Below are used in shared RNN setting
    def forward_actor_in_target(self, actor, actor_target, next_observ):
        """IQL doesn't use target networks for actor."""
        return self.forward_actor(actor, next_observ)

    def entropy_bonus(self, log_probs):
        """IQL doesn't use entropy regularization."""
        return 0.0
