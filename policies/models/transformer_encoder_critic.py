import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl
from torchkit.constant import *
from utils import logger
from policies.models.transformer_related import positional_embeddings
from torchkit import pytorch_utils as ptu
from typing import Literal


class Critic_TransformerEncoder(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        encoder,
        algo,
        action_embedding_size,
        observ_embedding_size,
        reward_embedding_size,
        dqn_layers,
        rnn_num_layers,
        max_len, # Maximum sequence length
        image_encoder=None,
        feature_extractor_type: Literal['separate', 'combined1', 'combined2'] = 'separate',
        combined_embedding_size: int = None,
        nominal_embedding_size: int = 0,
        num_trajectories: int = 2,
        **kwargs
    ):
        assert max_len is not None
        assert nominal_embedding_size is not None
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.algo = algo
        self.observ_embedding_size = observ_embedding_size
        self.action_embedding_size = action_embedding_size
        self.reward_embedding_size = reward_embedding_size
        self.max_len = max_len
        self.nominal_embedding_size = nominal_embedding_size
        self.num_trajectories = num_trajectories

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

        logger.log(f"\n****** Creating actor transformer ******")
        logger.log(f"num_trajectories = {self.num_trajectories}")
        self.feature_extractor_type = feature_extractor_type

        if self.nominal_embedding_size > 0:
            self.nominal_embedder = nn.Embedding(self.num_trajectories, self.nominal_embedding_size)

        if self.feature_extractor_type == 'separate':
            self.action_embedder = utl.FeatureExtractor(action_dim, action_embedding_size, F.relu)
            self.reward_embedder = utl.FeatureExtractor(1, reward_embedding_size, F.relu)
            assert self.observ_embedding_size == self.action_embedding_size == self.reward_embedding_size
            self.hidden_size = self.observ_embedding_size + self.nominal_embedding_size
            self.max_context_len = max_len * 3 + 4
            logger.log(f"feature_extractor_type = {self.feature_extractor_type} ({observ_embedding_size} + {nominal_embedding_size}, {action_embedding_size} + {nominal_embedding_size}, {reward_embedding_size} + {nominal_embedding_size})")
        elif self.feature_extractor_type == 'combined1':
            self.action_embedder = utl.FeatureExtractor(action_dim, action_embedding_size, F.relu)
            self.reward_embedder = utl.FeatureExtractor(1, reward_embedding_size, F.relu)
            assert isinstance(combined_embedding_size, int)
            self.combined_embedding_size = combined_embedding_size
            self.combined_embedder = utl.FeatureExtractor(
                self.observ_embedding_size + self.action_embedding_size + self.reward_embedding_size + self.nominal_embedding_size,
                self.combined_embedding_size,
                F.relu,
            )
            self.hidden_size = self.combined_embedding_size
            self.max_context_len = max_len + 1
            logger.log(f"feature_extractor_type = {self.feature_extractor_type} ({observ_embedding_size}, {action_embedding_size}, {reward_embedding_size}, {nominal_embedding_size}) -> ({self.combined_embedding_size})")
        elif self.feature_extractor_type == "combined2":
            self.action_embedder = utl.FeatureExtractor(action_dim, action_embedding_size, F.relu)
            self.reward_embedder = utl.FeatureExtractor(1, reward_embedding_size, F.relu)
            self.hidden_size = self.observ_embedding_size + self.action_embedding_size + self.reward_embedding_size + self.nominal_embedding_size
            self.max_context_len = max_len + 1
            logger.log(f"feature_extractor_type = {self.feature_extractor_type} ({observ_embedding_size} + {action_embedding_size} + {reward_embedding_size} + {nominal_embedding_size} = {self.hidden_size})")
        else:
            raise NotImplementedError(f"{self.feature_extractor_type}")

        self.num_layers = rnn_num_layers

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

        ## 4. build q networks
        self.qf1, self.qf2 = self.algo.build_critic(
            input_size=self.hidden_size + self.action_embedding_size,
            hidden_sizes=dqn_layers,
            action_dim=action_dim,
        )

        self.positional_embedding = positional_embeddings.SinusoidalPositionalEncoding(d_model=self.hidden_size, max_len=self.max_context_len)
        self.mask = torch.triu(ptu.ones(self.max_context_len, self.max_context_len), diagonal=1).float()
        self.mask = self.mask.masked_fill(self.mask == 1, float('-inf'))
        self.register_buffer("my_mask", self.mask)

    def _get_obs_embedding(self, observs: torch.Tensor) -> torch.Tensor:
        if self.image_encoder is None:  # vector obs
            return self.observ_embedder(observs)
        else:  # pixel obs
            return self.image_encoder(observs)

    def get_hidden_states(self, obs, prev_actions, rewards, nominals) -> torch.Tensor:
        T, N, _ = rewards.shape
        # assert rewards.dim() == 3 and rewards.shape[2] == 1, f"{rewards.shape}"
        # assert obs.shape == (T, N, self.obs_dim), f"{obs.shape} != {(T, N, self.obs_dim)}"
        # assert prev_actions.shape == (T, N, self.action_dim), f"{prev_actions.shape} != {(T, N, self.action_dim)}"
        obs_encs = self._get_obs_embedding(obs)
        action_encs = self.action_embedder(prev_actions)
        reward_encs = self.reward_embedder(rewards)

        if self.nominal_embedding_size > 0:
            assert nominals is not None
            assert nominals.dtype == torch.int64, f"{nominals.dtype}"
            if nominals.dim() == 3:
                nominals = nominals.squeeze(-1)
            nominal_encs = self.nominal_embedder(nominals)
            assert nominal_encs.shape == (T, N, self.nominal_embedding_size), f"{nominal_encs.shape} != {(T, N, self.nominal_embedding_size)}"
        
        if self.feature_extractor_type == 'separate':
            if self.nominal_embedding_size > 0:
                obs_encs = torch.cat([obs_encs, nominal_encs], dim=-1)
                reward_encs = torch.cat([reward_encs, nominal_encs], dim=-1)
                action_encs = torch.cat([action_encs, nominal_encs], dim=-1)
            context = torch.stack([action_encs, reward_encs, obs_encs], dim=1).flatten(0, 1)
        elif self.feature_extractor_type == 'combined1':
            if self.nominal_embedding_size > 0:
                concat_encs = torch.cat([action_encs, reward_encs, obs_encs, nominal_encs], dim=-1)
            else:
                concat_encs = torch.cat([action_encs, reward_encs, obs_encs], dim=-1)
            context = self.combined_embedder(concat_encs)
        elif self.feature_extractor_type == 'combined2':
            if self.nominal_embedding_size > 0:
                context = torch.cat([action_encs, reward_encs, obs_encs, nominal_encs], dim=-1)
            else:
                context = torch.cat([action_encs, reward_encs, obs_encs], dim=-1)
        else:
            raise NotImplementedError()
        pos = self.positional_embedding(context.transpose(0, 1)).transpose(0, 1)
        context += pos

        return context  # (T * 3, N, self.hidden_size)

    def forward(self, prev_actions, rewards, observs, current_actions, nominals = None):
        """
        For prev_actions a, rewards r, observs o: (T+1, B, dim)
                a[t] -> r[t], o[t]
        current_actions (or action probs for discrete actions) a': (T or T+1, B, dim)
                o[t] -> a'[t]
        NOTE: there is one timestep misalignment in prev_actions and current_actions
        """
        obs = observs

        context = self.get_hidden_states(obs, prev_actions, rewards, nominals)
        T, N, _ = context.shape
        # assert context.shape == (T * 3, N, self.hidden_size)

        # assert current_actions is not None
        # assert current_actions.shape == (T, N, prev_actions.shape[2]) or current_actions.shape == (T - 1, N, prev_actions.shape[2]), f"{current_actions.shape} != {(T, N, prev_actions.shape[2])} or {(T - 1, N, prev_actions.shape[2])}"
        curaT = current_actions.shape[0]
        current_actions_encs = self.action_embedder(current_actions)

        mask = self.mask[:T, :T]
        decoded = self.transformer(context, mask=mask)
        # assert isinstance(decoded, torch.Tensor) and decoded.shape == (T * 3, N, self.hidden_size), f"{decoded.shape} != {(T * 3, N, self.hidden_size)}"
        obs_embed_indx = torch.arange(2, T, 3) if self.feature_extractor_type == 'separate' else torch.arange(T)
        obs_embeds = decoded[obs_embed_indx, :, :]
        # assert obs_embeds.shape == (T, N, self.hidden_size), f"{obs_embeds.shape} != {(T, N, self.hidden_size)}"
        final_embeds = torch.cat([obs_embeds[:curaT], current_actions_encs], dim=-1)
        # assert final_embeds.shape == (curaT, N, self.hidden_size * 2)

        q1 = self.qf1(final_embeds)
        q2 = self.qf2(final_embeds)

        return q1, q2  # (T or T+1, B, 1 or A)
