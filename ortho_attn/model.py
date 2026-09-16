"""Decoder-only Transformer with concatenated multi-head attention.

All experiment variants share this module tree so parameter count and
initialization stay matched. The orthogonality penalty is a loss, not a
different mixer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 32
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class Head(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hs = config.head_size
        self.key = nn.Linear(config.n_embd, hs, bias=False)
        self.query = nn.Linear(config.n_embd, hs, bias=False)
        self.value = nn.Linear(config.n_embd, hs, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)
        self.scale = hs**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, t, _c = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = (q @ k.transpose(-2, -1)) * self.scale
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        # Concat mixer: 4 * head_size == n_embd. Do not switch this to a sum.
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head_outputs = [h(x) for h in self.heads]
        stacked = torch.stack(head_outputs, dim=0)  # (n_head, B, T, head_size)
        out = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, stacked


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa_out, head_outputs = self.sa(self.ln1(x))
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, head_outputs


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        _b, t = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(
            torch.arange(t, device=idx.device)
        )
        block_head_outputs: list[torch.Tensor] = []
        for block in self.blocks:
            x, head_outputs = block(x)
            block_head_outputs.append(head_outputs)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, block_head_outputs
