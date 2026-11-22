from typing import Optional
import torch
from torch import nn


class _BaseAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.causal = causal

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values.reshape(N, value_len, self.heads, self.head_dim))
        keys = self.keys(keys.reshape(N, key_len, self.heads, self.head_dim))
        queries = self.queries(query.reshape(N, query_len, self.heads, self.head_dim))

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if self.causal:
            causal_mask = torch.tril(
                torch.ones((query_len, key_len), device=energy.device, dtype=torch.bool),
                diagonal=key_len - query_len,
            ).unsqueeze(0).unsqueeze(0)
            energy = energy.masked_fill(~causal_mask, float("-inf"))

        if mask is not None:
            while mask.ndim < 4:
                mask = mask.unsqueeze(0)
            energy = energy.masked_fill(mask.to(energy.device) == 0, float("-inf"))

        scale = self.head_dim ** 0.5
        attention = torch.softmax(energy / scale, dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        out = self.dropout(out)
        return out


class CausalSelfAttention(_BaseAttention):
    def __init__(self, embed_size: int, heads: int, dropout: float = 0.1):
        super().__init__(embed_size=embed_size, heads=heads, dropout=dropout, causal=True)


class MultiHeadAttention(_BaseAttention):
    def __init__(self, embed_size: int, heads: int, dropout: float = 0.1):
        super().__init__(embed_size=embed_size, heads=heads, dropout=dropout, causal=False)
