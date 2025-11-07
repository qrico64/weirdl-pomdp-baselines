import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl
from torchkit.constant import *
from utils import logger
from policies.models.transformer_related import positional_embeddings


class Critic_Transformer(nn.Module):
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
        dqn_layers,
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
        logger.log(f"Creating critic transformer with {self.hidden_size} embedding size.")
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=4,
            dim_feedforward=self.hidden_size * 4,
            dropout=0.1,
            activation="relu",
        )
        norm = nn.LayerNorm(self.hidden_size)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers, norm=norm)

        ## 4. build q networks
        assert len(dqn_layers) == 0
        self.qf1, self.qf2 = self.algo.build_critic(
            input_size=self.hidden_size,
            hidden_sizes=dqn_layers,
            action_dim=action_dim,
        )

        self.positional_embedding = positional_embeddings.SinusoidalPositionalEncoding(d_model=self.hidden_size, max_len=max_len*3+4)

    def _get_obs_embedding(self, observs: torch.Tensor) -> torch.Tensor:
        if self.image_encoder is None:  # vector obs
            return self.observ_embedder(observs)
        else:  # pixel obs
            return self.image_encoder(observs)

    def get_hidden_states(self, obs, prev_actions, rewards) -> torch.Tensor:
        T, N, _ = rewards.shape
        assert rewards.dim() == 3 and rewards.shape[2] == 1, f"{rewards.shape}"
        assert obs.shape == (T, N, self.obs_dim), f"{obs.shape} != {(T, N, self.obs_dim)}"
        assert prev_actions.shape == (T + 1, N, self.action_dim), f"{prev_actions.shape} != {(T + 1, N, self.action_dim)}"
        obs_encs = self._get_obs_embedding(obs.reshape(T * N, self.obs_dim)).reshape(T, N, self.hidden_size)
        action_encs = self.action_embedder(prev_actions.reshape((T + 1) * N, self.action_dim)).reshape((T + 1), N, self.hidden_size)
        reward_encs = self.reward_embedder(rewards.reshape(T * N, 1)).reshape(T, N, self.hidden_size)
        context = torch.zeros(T * 3 + 1, N, self.hidden_size, dtype=action_encs.dtype, device=action_encs.device)
        context[torch.arange(T * 3 + 1) % 3 == 0, :, :] = action_encs
        context[torch.arange(T * 3 + 1) % 3 == 1, :, :] = reward_encs
        context[torch.arange(T * 3 + 1) % 3 == 2, :, :] = obs_encs
        context += self.positional_embedding(context.transpose(0, 1)).transpose(0, 1)

        return context  # (T * 3 + 1, N, self.hidden_size)

    def forward(self, prev_actions, rewards, observs, current_actions):
        """
        For prev_actions a, rewards r, observs o: (T+1, B, dim)
                a[t] -> r[t], o[t]
        current_actions (or action probs for discrete actions) a': (T or T+1, B, dim)
                o[t] -> a'[t]
        NOTE: there is one timestep misalignment in prev_actions and current_actions
        """
        obs = observs

        T, N, _ = rewards.shape
        assert current_actions is not None
        assert current_actions.shape == (T, N, prev_actions.shape[2])
        context = self.get_hidden_states(obs, prev_actions, rewards)
        assert context.shape == (T * 3 + 1, N, self.hidden_size)

        decoded = self.transformer(context, memory=torch.zeros_like(context))
        assert isinstance(decoded, torch.Tensor) and decoded.shape == (T * 3 + 1, N, self.hidden_size), f"{decoded.shape} != {(T * 3 + 1, N, self.hidden_size)}"
        final_embed = decoded[T * 3, :, :]
        assert final_embed.shape == (N, self.hidden_size), f"{final_embed.shape} != {(N, self.hidden_size)}"

        # 4. q value
        q1 = self.qf1(final_embed)
        q2 = self.qf2(final_embed)

        return q1, q2  # (T or T+1, B, 1 or A)
