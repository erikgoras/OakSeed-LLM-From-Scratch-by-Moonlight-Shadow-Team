from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim if head_dim is not None else embed_dim
        self.scale = self.head_dim ** -0.5
        target_device = torch.device(device) if device is not None else get_default_device()

        self.q_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.head_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.to(target_device)

    @property
    def device(self) -> torch.device:
        return self.q_proj.weight.device

    def compute_attention_scores(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        x_dev = x.to(device=self.device, dtype=torch.float32)
        q = self.q_proj(x_dev)
        k = self.k_proj(x_dev)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        return self.attn_dropout(probs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dev = x.to(device=self.device, dtype=torch.float32)
        v = self.v_proj(x_dev)
        attn_weights = self.compute_attention_scores(x_dev)
        context = attn_weights @ v
        return self.resid_dropout(self.out_proj(context))


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        target_device = torch.device(device) if device is not None else get_default_device()

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.to(target_device)

    @property
    def device(self) -> torch.device:
        return self.qkv_proj.weight.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_dev = x.to(device=self.device, dtype=torch.float32)
        q, k, v = self.qkv_proj(x_dev).chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        probs = self.attn_dropout(probs)

        context = (probs @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.resid_dropout(self.out_proj(context))
