import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl
from torchkit.constant import *
import torchkit.pytorch_utils as ptu
from utils import logger
from policies.models.transformer_related import positional_embeddings
import time
import numpy as np


def collectivize_inputs(prev_actions, rewards, obs):
    # assert isinstance(rewards, list)
    # assert isinstance(prev_actions, list)
    # assert isinstance(obs, list)
    # assert len(prev_actions) == len(rewards) == len(obs)
    N = len(rewards)
    Ts = [traj.shape[0] for traj in rewards]
    T = max(Ts)
    rewards_tensor = ptu.zeros(T, N, rewards[0].shape[-1], dtype=rewards[0].dtype)
    obs_tensor = ptu.zeros(T, N, obs[0].shape[-1], dtype=obs[0].dtype)
    prev_actions_tensor = ptu.zeros(T, N, prev_actions[0].shape[-1], dtype=prev_actions[0].dtype)
    for i in range(N):
        rewards_tensor[:rewards[i].shape[0],i,:] = rewards[i]
        obs_tensor[:obs[i].shape[0],i,:] = obs[i]
        prev_actions_tensor[:prev_actions[i].shape[0],i,:] = prev_actions[i]
    
    return prev_actions_tensor, rewards_tensor, obs_tensor, ptu.tensor(Ts)


class Actor_TransformerEncoder(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        encoder,
        algo,
        action_embedding_size,
        observ_embedding_size,
        reward_embedding_size,
        rnn_hidden_size,
        policy_layers,
        rnn_num_layers,
        max_len, # Maximum sequence length
        image_encoder=None,
        **kwargs
    ):
        assert max_len is not None
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.algo = algo
        self.observ_embedding_size = observ_embedding_size
        self.action_embedding_size = action_embedding_size
        self.reward_embedding_size = reward_embedding_size
        self.policy_layers = policy_layers
        self.max_len = max_len

        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)

        self.image_encoder = image_encoder
        if self.image_encoder is None:
            self.observ_embedder = utl.FeatureExtractor(
                obs_dim, observ_embedding_size, F.relu
            )
        else:  # for pixel observation, use external encoder
            assert observ_embedding_size == 0
            observ_embedding_size = self.image_encoder.embed_size  # reset it

        self.action_embedder = utl.FeatureExtractor(action_dim, action_embedding_size, F.relu)
        self.reward_embedder = utl.FeatureExtractor(1, reward_embedding_size, F.relu)

        ## 2. build RNN model
        assert self.observ_embedding_size == self.action_embedding_size == self.reward_embedding_size
        self.rnn_hidden_size = rnn_hidden_size

        self.num_layers = rnn_num_layers

        self.hidden_size = self.observ_embedding_size
        logger.log(f"\n****** Creating actor transformer ******")
        logger.log(f"d_model = {self.hidden_size}")
        logger.log(f"nhead = {4}")
        logger.log(f"dim_feedforward = {self.hidden_size * 4}")
        logger.log(f"dropout = {0.1}")
        logger.log(f"activation = {'relu'}")
        logger.log(f"num_layers = {self.num_layers}")
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=4,
            dim_feedforward=self.hidden_size * 4,
            dropout=0.1,
            activation="relu",
        )
        norm = nn.LayerNorm(self.hidden_size)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=self.num_layers, norm=norm)
        logger.log(f"****** Created actor transformer ******\n")

        ## 4. build policy
        self.policy = self.algo.build_actor(
            input_size=self.hidden_size,
            action_dim=self.action_dim,
            hidden_sizes=policy_layers,
        )

        self.MAX_3T = max_len * 3 + 4
        self.positional_embedding = positional_embeddings.SinusoidalPositionalEncoding(d_model=self.hidden_size, max_len=self.MAX_3T)
        self.mask = torch.triu(ptu.ones(self.MAX_3T, self.MAX_3T), diagonal=1).float()
        self.mask = self.mask.masked_fill(self.mask == 1, float('-inf'))
        self.register_buffer("my_mask", self.mask)

    def _get_obs_embedding(self, observs: torch.Tensor) -> torch.Tensor:
        if self.image_encoder is None:  # vector obs
            return self.observ_embedder(observs)
        else:  # pixel obs
            return self.image_encoder(observs)

    def get_hidden_states(self, obs, prev_actions, rewards) -> torch.Tensor:
        T, N, _ = rewards.shape
        # assert rewards.dim() == 3 and rewards.shape[2] == 1, f"{rewards.shape}"
        # assert obs.shape == (T, N, self.obs_dim), f"{obs.shape} != {(T, N, self.obs_dim)}"
        # assert prev_actions.shape == (T, N, self.action_dim), f"{prev_actions.shape} != {(T, N, self.action_dim)}"
        obs_encs = self._get_obs_embedding(obs.reshape(T * N, self.obs_dim)).reshape(T, N, self.hidden_size)
        action_encs = self.action_embedder(prev_actions.reshape(T * N, self.action_dim)).reshape(T, N, self.hidden_size)
        reward_encs = self.reward_embedder(rewards.reshape(T * N, 1)).reshape(T, N, self.hidden_size)
        context = torch.stack([action_encs, reward_encs, obs_encs], dim=1).flatten(0, 1)
        pos = self.positional_embedding(context.transpose(0, 1)).transpose(0, 1)
        context += pos

        return context  # (T * 3, N, self.hidden_size)

    def forward(self, prev_actions, rewards, observs, return_log_prob:bool=False):
        """
        For prev_actions a, rewards r, observs o: (T+1, B, dim)
                a[t] -> r[t], o[t]

        return current actions a' (T+1, B, dim) based on previous history

        """
        obs = observs
        
        T, N, _ = rewards.shape
        context = self.get_hidden_states(obs, prev_actions, rewards)
        # assert context.shape == (T * 3, N, self.hidden_size)

        mask = self.mask[:T * 3, :T * 3]
        decoded = self.transformer(context, mask=mask)
        # assert isinstance(decoded, torch.Tensor) and decoded.shape == (T * 3, N, self.hidden_size), f"{decoded.shape} != {(T * 3, N, self.hidden_size)}"
        obs_embeds = decoded[torch.arange(2, T * 3, 3), :, :]
        # assert obs_embeds.shape == (T, N, self.hidden_size)
        actions, log_probs = self.algo.forward_actor(actor=self.policy, observ=obs_embeds)
        # assert actions.shape == (T, N, prev_actions.shape[2])
        # assert log_probs.shape == (T, N, 1)
        return actions, log_probs
        # final_embed = decoded[T * 3 - 1, :, :]
        # assert final_embed.shape == (N, self.hidden_size), f"{final_embed.shape} != {(N, self.hidden_size)}"

        # return self.algo.forward_actor(actor=self.policy, observ=final_embed)

    @torch.no_grad()
    def get_initial_info(self):
        # here we assume batch_size = 1

        ## here we set the ndim = 2 for action and reward for compatibility
        prev_action = ptu.zeros((1, self.action_dim)).float()
        reward = ptu.zeros((1, 1)).float()

        return prev_action, reward

    @torch.no_grad()
    def act(
        self,
        prev_actions,
        rewards,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        prev_actions, rewards, obs, Ts = collectivize_inputs(prev_actions, rewards, obs)
        
        T, N, _ = rewards.shape
        context = self.get_hidden_states(obs, prev_actions, rewards)
        # assert context.shape == (T * 3, N, self.hidden_size)

        mask = self.mask[:T * 3, :T * 3]
        decoded = self.transformer(context, mask=mask)
        # assert isinstance(decoded, torch.Tensor) and decoded.shape == (T * 3, N, self.hidden_size), f"{decoded.shape} != {(T * 3, N, self.hidden_size)}"
        final_embed = decoded[Ts * 3 - 1, torch.arange(N), :]
        # assert final_embed.shape == (N, self.hidden_size), f"{final_embed.shape} != {(N, self.hidden_size)}"

        # 4. Actor head, generate action tuple
        action_tuple = self.algo.select_action(
            actor=self.policy,
            observ=final_embed,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
        )

        return action_tuple
