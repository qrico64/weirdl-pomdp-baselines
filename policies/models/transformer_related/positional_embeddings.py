# Python 3.8.5 / PyTorch 1.7.1 compatible
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from utils import logger


class SinusoidalPositionalEncoding(nn.Module):
    """
    Original 'Attention Is All You Need' sinusoidal encoding.

    Usage:
        pe = SinusoidalPositionalEncoding(d_model=512, max_len=4096)
        x = tok_emb + pe(tok_emb)   # tok_emb: (B, L, D)
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        assert d_model % 2 == 0, f"{d_model}"
        self.d_model = d_model
        self.max_len = max_len

        logger.log(f"\n****** Creating SinusoidalPositionalEncoding ******")
        logger.log(f"d_model = {d_model}")
        logger.log(f"max_len = {max_len}")
        logger.log(f"****** Created SinusoidalPositionalEncoding ******\n")

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (D/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even channels
        pe[:, 1::2] = torch.cos(position * div_term)  # odd  channels
        pe = pe.unsqueeze(0)  # (1, L, D)

        # In 1.7.1, avoid persistent= keyword for portability.
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, L, D). Returns pe[:, :L, :] (broadcastable to x) for addition.
        """
        assert x.shape[2] == self.d_model and x.shape[1] <= self.max_len, f"{x.shape} != {(1, self.max_len, self.d_model)}"
        L = x.size(1)
        return self.pe[:, :L, :]



class DTPositionalEncoding(nn.Module):
    """
    Positional scheme for interleaved [RTG_t, s_t, a_t] tokens repeated over steps.
    Returns encodings to ADD to token embeddings:

        out = time_step_emb (per env step, broadcast to 3 slots)
            + local_pos_emb (0..L-1 in the window)
            + optional type_emb (0=RTG, 1=state, 2=action)

    Args:
        d_model:      model dimension
        max_time:     maximum absolute environment timestep (exclusive)
        max_len:      maximum token length in a single window (must cover 3*T)
        use_type_emb: add per-slot embeddings for {RTG, state, action}

    Forward:
        - timesteps: (B, T) absolute steps, OR (B,) start steps (then pass T)
        - T:         number of env steps in the window (required if timesteps is (B,))
        - L:         total token length; must equal 3*T

    Returns:
        (B, L, d_model)
    """
    def __init__(self,
                 d_model: int,
                 max_time: int,
                 max_len: int,
                 use_type_emb: bool = True):
        super(DTPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_time = max_time
        self.max_len = max_len
        self.use_type_emb = use_type_emb

        # Learned absolute time-step embedding (per env step t)
        self.time_emb = nn.Embedding(max_time, d_model)

        # Learned local position embedding within the current window (0..L-1)
        self.pos_emb = nn.Embedding(max_len, d_model)

        if use_type_emb:
            # 0=RTG, 1=state, 2=action
            self.type_emb = nn.Embedding(3, d_model)
        else:
            self.type_emb = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.time_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight,  mean=0.0, std=0.02)
        if self.type_emb is not None:
            nn.init.normal_(self.type_emb.weight, mean=0.0, std=0.02)

    @staticmethod
    def _slots_pattern(batch: int, T: int, device) -> Tensor:
        """
        Returns slot ids of shape (B, 3T): [0,1,2, 0,1,2, ..., 0,1,2]
        where 0=RTG, 1=state, 2=action.
        """
        pattern = torch.tensor([0, 1, 2], device=device, dtype=torch.long).repeat(T)  # (3T,)
        return pattern.unsqueeze(0).expand(batch, -1)  # (B, 3T)

    @staticmethod
    def _local_positions(L: int, device) -> Tensor:
        """
        Local positions 0..L-1, shape (1, L) int64 (for Embedding).
        """
        return torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)

    def forward(self,
                timesteps: Tensor,
                T: Optional[int] = None,
                L: Optional[int] = None) -> Tensor:
        """
        Produce positional encodings to ADD to token embeddings.

        timesteps:
            - (B, T) absolute env steps for each triplet position
            - OR (B,) start steps, then you must pass T
        T:
            Required if timesteps is (B,)
        L:
            Required total token length; must equal 3*T

        Returns:
            (B, L, d_model)
        """
        if timesteps.dim() == 1:
            # timesteps: (B,)
            assert T is not None, "When timesteps is (B,), you must pass T."
            start = timesteps.unsqueeze(1)  # (B, 1)
            incr = torch.arange(T, device=timesteps.device, dtype=start.dtype)  # (T,)
            timesteps = start + incr  # (B, T)
        else:
            # timesteps: (B, T)
            assert timesteps.dim() == 2, "timesteps must be (B,) or (B, T)"
            B_, T_in = timesteps.shape
            if T is None:
                T = T_in
            else:
                assert T == T_in, "Provided T does not match timesteps.shape[1]"

        assert L is not None and L == 3 * T, "L must equal 3*T for [RTG, s, a] format"
        B = timesteps.size(0)
        device = timesteps.device

        # 1) absolute time-step embedding per env step -> (B, T, D)
        # Ensure Long dtype for embedding indices
        step_idx = timesteps.long()
        step_e = self.time_emb(step_idx)  # (B, T, D)

        # Repeat each step embedding across its three tokens -> (B, 3T, D)
        step_e = step_e.repeat_interleave(3, dim=1)  # (B, L, D)

        # 2) local position embedding (0..L-1) -> (B, L, D)
        pos_idx = self._local_positions(L, device)  # (1, L)
        pos_e = self.pos_emb(pos_idx)               # (1, L, D)
        pos_e = pos_e.expand(B, -1, -1)            # (B, L, D)

        # 3) type embedding (optional) -> (B, L, D)
        if self.type_emb is not None:
            slot_idx = self._slots_pattern(B, T, device)  # (B, L) longs
            type_e = self.type_emb(slot_idx)
        else:
            type_e = 0.0

        return step_e + pos_e + type_e