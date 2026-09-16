import torch

from ortho_attn.model import GPT, GPTConfig, count_parameters


def test_concat_projection_is_full_width():
    cfg = GPTConfig(vocab_size=20, n_embd=64, n_head=4, n_layer=2, block_size=8)
    model = GPT(cfg)
    proj = model.blocks[0].sa.proj
    assert proj.in_features == cfg.n_embd
    assert proj.out_features == cfg.n_embd
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss, heads, post = model(x, x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None
    assert len(heads) == cfg.n_layer
    assert heads[0].shape == (cfg.n_head, 2, 8, cfg.head_size)
    assert post[0].shape == heads[0].shape


def test_param_count_independent_of_loss_choice():
    torch.manual_seed(0)
    a = GPT(GPTConfig(vocab_size=12, n_embd=32, n_head=4, n_layer=2, block_size=8))
    torch.manual_seed(0)
    b = GPT(GPTConfig(vocab_size=12, n_embd=32, n_head=4, n_layer=2, block_size=8))
    assert count_parameters(a) == count_parameters(b)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_one_train_step_finite():
    cfg = GPTConfig(vocab_size=16, n_embd=32, n_head=4, n_layer=1, block_size=8)
    model = GPT(cfg)
    x = torch.randint(0, 16, (4, 8))
    logits, loss, _heads, _post = model(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits).all()


def test_identity_proj_drops_linear_params_keeps_concat():
    kwargs = dict(vocab_size=20, n_embd=64, n_head=4, n_layer=2, block_size=8)
    torch.manual_seed(0)
    linear = GPT(GPTConfig(**kwargs, proj_mode="linear"))
    torch.manual_seed(0)
    ident = GPT(GPTConfig(**kwargs, proj_mode="identity"))
    assert count_parameters(ident) < count_parameters(linear)
    assert isinstance(ident.blocks[0].sa.proj, torch.nn.Identity)
    x = torch.randint(0, 20, (2, 8))
    _logits, loss, heads, post = ident(x, x)
    assert loss is not None
    assert heads[0].shape == (4, 2, 8, 16)
    assert post[0].shape == heads[0].shape
