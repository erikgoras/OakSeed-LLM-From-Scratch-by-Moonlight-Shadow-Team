from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a lower-triangular causal mask of shape (1, 1, seq_len, seq_len)."""
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


class FeedForward(nn.Module):
    """Position-wise MLP with 4x expansion, GELU activation, and dropout."""

    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = 4 * embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LayerNorm Transformer block with residual connections."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_size=embed_dim,
            heads=num_heads,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim=embed_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
        dropout: float = 0.1,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.to(resolve_device(device))

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}; block_size is {self.block_size}"
            )

        model_device = next(self.parameters()).device
        idx = idx.to(model_device)
        positions = torch.arange(seq_len, device=model_device)
        causal_mask = build_causal_mask(seq_len=seq_len, device=model_device)

        x = self.token_embedding_table(idx) + self.position_embedding_table(positions)
        for block in self.blocks:
            x = block(x, mask=causal_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        targets = targets.to(model_device)
        batch_size, time_steps, channels = logits.shape
        loss = F.cross_entropy(
            logits.reshape(batch_size * time_steps, channels),
            targets.reshape(batch_size * time_steps),
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        model_device = next(self.parameters()).device
        out = idx.to(model_device)
        for _ in range(max_new_tokens):
            idx_cond = out[:, -self.block_size :]
            logits, _ = self(idx_cond)
            scaled_logits = logits[:, -1, :] / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat((out, next_token), dim=1)
        return out
