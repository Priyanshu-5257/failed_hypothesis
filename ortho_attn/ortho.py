"""Gram-matrix orthogonality penalties.

Both penalties operate on already-normalized vectors and apply a soft-margin
ReLU on off-diagonal cosine similarity. Defaults have no margin; callers pass
the experiment value explicitly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import GPT


def _offdiag_penalty(gram: torch.Tensor, margin: float) -> torch.Tensor:
    n = gram.size(-1)
    mask = ~torch.eye(n, device=gram.device, dtype=torch.bool)
    off = gram[..., mask]
    return F.relu(off.abs() - margin).mean()


def activation_orthogonality_loss(
    block_head_outputs: list[torch.Tensor], margin: float
) -> torch.Tensor:
    """Penalize per-token cosine similarity across heads, before concat.

    Each entry is (n_head, B, T, head_size).
    """
    if not block_head_outputs:
        raise ValueError("block_head_outputs is empty")
    losses = []
    for layer_heads in block_head_outputs:
        h = layer_heads.permute(1, 2, 0, 3)  # (B, T, n_head, head_size)
        h_norm = F.normalize(h, p=2, dim=-1)
        gram = h_norm @ h_norm.transpose(-1, -2)
        losses.append(_offdiag_penalty(gram, margin))
    return torch.stack(losses).mean()


def value_weight_orthogonality_loss(model: GPT, margin: float) -> torch.Tensor:
    """Penalize cosine similarity of flattened per-head Value matrices."""
    losses = []
    for block in model.blocks:
        weights = [head.value.weight.reshape(-1) for head in block.sa.heads]
        w = torch.stack(weights)
        w_norm = F.normalize(w, p=2, dim=1)
        gram = w_norm @ w_norm.T
        losses.append(_offdiag_penalty(gram, margin))
    return torch.stack(losses).mean()


def mean_abs_head_cosine(block_head_outputs: list[torch.Tensor]) -> torch.Tensor:
    """Population mean |cos| over all tokens, head pairs, and layers."""
    return activation_orthogonality_loss(block_head_outputs, margin=0.0)
