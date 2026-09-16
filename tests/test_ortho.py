import torch

from ortho_attn.model import GPT, GPTConfig
from ortho_attn.ortho import (
    activation_orthogonality_loss,
    mean_abs_head_cosine,
    value_weight_orthogonality_loss,
)


def _heads(n_head: int, similar: bool) -> list[torch.Tensor]:
    # (n_head, B, T, head_size)
    b, t, d = 2, 3, 8
    if similar:
        base = torch.randn(1, b, t, d)
        stacked = base.repeat(n_head, 1, 1, 1)
    else:
        stacked = torch.zeros(n_head, b, t, d)
        for i in range(n_head):
            stacked[i, ..., i] = 1.0
    return [stacked]


def test_identical_heads_have_high_penalty():
    loss = activation_orthogonality_loss(_heads(4, similar=True), margin=0.0)
    assert loss.item() > 0.9


def test_axis_aligned_heads_have_near_zero_penalty():
    loss = activation_orthogonality_loss(_heads(4, similar=False), margin=0.0)
    assert loss.item() < 1e-6


def test_margin_zeroes_small_overlap():
    h = torch.eye(4).view(4, 1, 1, 4).repeat(1, 2, 2, 1)
    h = h + 0.01 * torch.randn_like(h)
    tight = activation_orthogonality_loss([h], margin=0.0)
    loose = activation_orthogonality_loss([h], margin=0.5)
    assert tight.item() > 0
    assert loose.item() == 0.0


def test_mean_abs_cosine_matches_zero_margin_penalty():
    heads = _heads(4, similar=True)
    a = mean_abs_head_cosine(heads)
    b = activation_orthogonality_loss(heads, margin=0.0)
    assert torch.allclose(a, b)


def test_value_weight_penalty_zero_on_orthogonal_wv():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=10, n_embd=8, n_head=2, n_layer=1, block_size=4))
    with torch.no_grad():
        for block in model.blocks:
            for i, head in enumerate(block.sa.heads):
                head.value.weight.zero_()
                head.value.weight[i, i] = 1.0
    loss = value_weight_orthogonality_loss(model, margin=0.0)
    assert loss.item() < 1e-6
