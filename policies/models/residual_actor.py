"""
Residual Actor for Meta-Learning with Transformers

This module implements a residual policy that adds learned corrections to a base policy.
The base policy is assumed to be a transformer that takes a list of transitions and outputs actions.
"""

import torch
import torch.nn as nn
from utils import logger
from typing import Optional


class ResidualActor_Transformer(nn.Module):
    """
    Residual actor that wraps a base transformer policy and learns to add corrections.

    The residual policy takes the same inputs as the base policy (transitions history)
    and outputs action corrections that are added to the base policy's actions.

    Args:
        base_policy: The base transformer policy module (e.g., Actor_TransformerEncoder)
        residual_scale: Scaling factor for residual actions (default: 0.1)
        freeze_base: Whether to freeze the base policy parameters (default: True)
        residual_hidden_sizes: Hidden layer sizes for the residual network
        residual_type: Type of residual connection ('additive' or 'gated')
    """

    def __init__(
        self,
        base_policy: nn.Module,
        residual_scale: float = 0.1,
        freeze_base: bool = True,
        residual_hidden_sizes: Optional[list] = None,
        residual_type: str = 'additive',
    ):
        super().__init__()

        self.base_policy = base_policy
        self.residual_scale = residual_scale
        self.residual_type = residual_type

        # Freeze base policy if requested
        if freeze_base:
            for param in self.base_policy.parameters():
                param.requires_grad = False
            logger.log("Residual Actor: Base policy parameters frozen")

        # Extract dimensions from base policy
        self.obs_dim = base_policy.obs_dim
        self.action_dim = base_policy.action_dim
        self.hidden_size = base_policy.hidden_size
        self.algo = base_policy.algo

        # Build residual network
        if residual_hidden_sizes is None:
            residual_hidden_sizes = [self.hidden_size, self.hidden_size // 2]

        if self.residual_type == 'additive':
            # Simple additive residual: learns a correction to add to base actions
            self.residual_network = self._build_residual_mlp(
                input_size=self.hidden_size,
                output_size=self.action_dim,
                hidden_sizes=residual_hidden_sizes,
            )
            logger.log(f"Residual Actor: Additive residual network with scale {residual_scale}")

        elif self.residual_type == 'gated':
            # Gated residual: learns both correction and gating weights
            self.residual_network = self._build_residual_mlp(
                input_size=self.hidden_size,
                output_size=self.action_dim,
                hidden_sizes=residual_hidden_sizes,
            )
            self.gate_network = self._build_residual_mlp(
                input_size=self.hidden_size,
                output_size=self.action_dim,
                hidden_sizes=residual_hidden_sizes,
                output_activation='sigmoid',
            )
            logger.log(f"Residual Actor: Gated residual network with scale {residual_scale}")
        else:
            raise ValueError(f"Unknown residual_type: {residual_type}")

        logger.log(f"Residual Actor: Created with {residual_type} residual connection")

    def _build_residual_mlp(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        output_activation: Optional[str] = None,
    ) -> nn.Module:
        """Build a simple MLP for residual learning"""
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def get_hidden_states(self, obs, prev_actions, rewards):
        """Get hidden states from base policy's transformer encoder"""
        return self.base_policy.get_hidden_states(obs, prev_actions, rewards)

    @torch.no_grad()
    def get_initial_info(self):
        """Get initial action and reward (delegates to base policy)"""
        return self.base_policy.get_initial_info()

    def forward(self, prev_actions, rewards, observs, return_log_prob: bool = False):
        """
        Forward pass through residual actor.

        Args:
            prev_actions: (T+1, B, action_dim) - previous actions
            rewards: (T+1, B, 1) - previous rewards
            observs: (T+1, B, obs_dim) - observations
            return_log_prob: whether to return log probabilities

        Returns:
            actions: (T+1, B, action_dim) - residual-corrected actions
            log_probs: (T+1, B, 1) - log probabilities (if return_log_prob=True)
        """
        # Get hidden states from transformer encoder
        context = self.get_hidden_states(observs, prev_actions, rewards)
        T, _, _ = context.shape

        # Get base policy actions (detached to prevent gradients flowing to base)
        with torch.no_grad():
            base_actions, _ = self.base_policy.forward(
                prev_actions, rewards, observs, return_log_prob=False
            )

        # Process through transformer to get embeddings
        mask = self.base_policy.mask[:T, :T]
        decoded = self.base_policy.transformer(context, mask=mask)

        # Extract observation embeddings
        if self.base_policy.feature_extractor_type == 'separate':
            obs_embed_indx = torch.arange(2, T, 3)
        else:
            obs_embed_indx = torch.arange(T)
        obs_embeds = decoded[obs_embed_indx, :, :]

        # Compute residual corrections
        if self.residual_type == 'additive':
            # Simple additive residual
            residual_actions = self.residual_network(obs_embeds)
            residual_actions = self.residual_scale * torch.tanh(residual_actions)
            final_actions = base_actions + residual_actions

        elif self.residual_type == 'gated':
            # Gated residual: gate controls how much residual to add
            residual_actions = self.residual_network(obs_embeds)
            residual_actions = torch.tanh(residual_actions)
            gate_weights = self.gate_network(obs_embeds)
            final_actions = base_actions + self.residual_scale * gate_weights * residual_actions

        # Clip to action bounds (assuming [-1, 1] for continuous actions)
        final_actions = torch.clamp(final_actions, -1.0, 1.0)

        if return_log_prob:
            # Compute log probabilities using Gaussian assumption
            # This is a simplified approach - for better results, could use a learned std
            action_std = 0.1  # Fixed standard deviation
            log_probs = -0.5 * ((final_actions - base_actions) / action_std) ** 2
            log_probs = log_probs.sum(dim=-1, keepdim=True)
            return final_actions, log_probs

        return final_actions, None

    @torch.no_grad()
    def act(
        self,
        prev_actions,
        rewards,
        obs,
        lengths,
        deterministic=False,
        return_log_prob=False,
    ):
        """
        Select action at inference time.

        Args:
            prev_actions: (T, B, action_dim) - history of previous actions
            rewards: (T, B, 1) - history of rewards
            obs: (T, B, obs_dim) - history of observations
            lengths: (B,) - actual lengths of each sequence
            deterministic: whether to act deterministically
            return_log_prob: whether to return log probabilities

        Returns:
            Tuple of (action, action_mean, action_log_std, log_prob)
        """
        # Get hidden states from transformer encoder
        context = self.get_hidden_states(obs, prev_actions, rewards)
        T, N, _ = context.shape

        # Get base policy actions
        with torch.no_grad():
            base_action_tuple = self.base_policy.act(
                prev_actions, rewards, obs, lengths,
                deterministic=deterministic,
                return_log_prob=False,
            )
            base_actions = base_action_tuple[0]

        # Process through transformer
        mask = self.base_policy.mask[:T, :T]
        decoded = self.base_policy.transformer(context, mask=mask)

        # Extract final embedding for each sequence
        if self.base_policy.feature_extractor_type == 'separate':
            obs_embed_idx = lengths * 3 - 1
        else:
            obs_embed_idx = lengths - 1
        final_embed = decoded[obs_embed_idx, torch.arange(N), :]

        # Compute residual correction
        if self.residual_type == 'additive':
            residual_actions = self.residual_network(final_embed)
            residual_actions = self.residual_scale * torch.tanh(residual_actions)
            final_actions = base_actions + residual_actions

        elif self.residual_type == 'gated':
            residual_actions = self.residual_network(final_embed)
            residual_actions = torch.tanh(residual_actions)
            gate_weights = self.gate_network(final_embed)
            final_actions = base_actions + self.residual_scale * gate_weights * residual_actions

        # Clip to action bounds
        final_actions = torch.clamp(final_actions, -1.0, 1.0)

        # Return in the same format as base policy
        # (action, action_mean, action_log_std, log_prob)
        if return_log_prob:
            # Simple log prob computation
            action_std = 0.1
            log_prob = -0.5 * ((final_actions - base_actions) / action_std) ** 2
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            return (final_actions, final_actions, None, log_prob)
        else:
            return (final_actions, final_actions, None, None)

    def parameters(self, recurse: bool = True):
        """
        Override parameters() to only return residual network parameters.
        This ensures optimizer only updates the residual, not the base policy.
        """
        if self.residual_type == 'additive':
            return self.residual_network.parameters(recurse=recurse)
        elif self.residual_type == 'gated':
            return list(self.residual_network.parameters(recurse=recurse)) + \
                   list(self.gate_network.parameters(recurse=recurse))

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Override to only return residual network named parameters"""
        if self.residual_type == 'additive':
            for name, param in self.residual_network.named_parameters(prefix=prefix + 'residual_network.', recurse=recurse):
                yield name, param
        elif self.residual_type == 'gated':
            for name, param in self.residual_network.named_parameters(prefix=prefix + 'residual_network.', recurse=recurse):
                yield name, param
            for name, param in self.gate_network.named_parameters(prefix=prefix + 'gate_network.', recurse=recurse):
                yield name, param

    def train(self, mode: bool = True):
        """Override train mode to keep base policy in eval mode if frozen"""
        super().train(mode)
        # Always keep base policy in eval mode if it's frozen
        if mode:
            self.base_policy.eval()
        return self

    def eval(self):
        """Set to evaluation mode"""
        return self.train(False)
