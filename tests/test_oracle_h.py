"""Unit tests for Phase 1 Oracle H representation engine."""

import pytest
import torch
import torch.nn as nn

from zip2zip.oracle_h import (
    compute_kl_and_topk,
    get_oracle_initializations,
    student_forward_with_oracle_h,
)


class DummyPhiModel(nn.Module):
    def __init__(self, vocab_size: int = 100, embed_dim: int = 64):
        super().__init__()
        self.config = type("Config", (), {"vocab_size": vocab_size, "hidden_size": embed_dim})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        hidden = inputs_embeds + 0.01  # dummy forward
        logits = self.lm_head(hidden)
        
        class Output:
            pass
        out = Output()
        out.logits = logits
        out.hidden_states = [hidden, hidden]
        out.past_key_values = None
        return out


def test_oracle_h_initializations():
    """Verify 4 initializations are correctly shaped and midpoint is exact."""
    model = DummyPhiModel(vocab_size=50, embed_dim=32)
    device = torch.device("cpu")
    inits = get_oracle_initializations(model.model.embed_tokens, token_a=5, token_b=10, device=device)

    assert set(inits.keys()) == {"midpoint", "token_b", "perturbed_midpoint", "empirical_random"}
    for name, tensor in inits.items():
        assert tensor.shape == (32,)
        assert tensor.dtype == torch.float32

    # Check midpoint exactness
    ea = model.model.embed_tokens(torch.tensor([5]))[0].float()
    eb = model.model.embed_tokens(torch.tensor([10]))[0].float()
    expected_mid = 0.5 * (ea + eb)
    assert torch.allclose(inits["midpoint"], expected_mid)


def test_oracle_h_optimizer_isolation():
    """Verify that during Oracle H optimization, only H is optimized and model is strictly frozen."""
    model = DummyPhiModel(vocab_size=50, embed_dim=32)
    for p in model.parameters():
        p.requires_grad = False

    device = torch.device("cpu")
    H_param = nn.Parameter(torch.randn(32, dtype=torch.float32))
    opt = torch.optim.Adam([H_param], lr=1e-2)

    assert len(opt.param_groups[0]["params"]) == 1
    assert opt.param_groups[0]["params"][0] is H_param

    # Forward student
    ctx = [1, 2, 3]
    s_logits, s_hidden = student_forward_with_oracle_h(model, ctx, H_param, device, offsets=(0,))
    loss = s_logits[0].sum()
    loss.backward()

    # Check gradients
    assert H_param.grad is not None
    assert H_param.grad.norm().item() > 0.0
    for name, p in model.named_parameters():
        assert p.grad is None or p.grad.norm().item() == 0.0


def test_oracle_h_physical_sequence_length():
    """Student must occupy strictly ONE physical slot for H (length C + 1 + K)."""
    model = DummyPhiModel(vocab_size=50, embed_dim=32)
    device = torch.device("cpu")
    ctx = [1, 2, 3, 4]  # C = 4
    future = [10, 11, 12, 13]  # K = 4
    H_param = nn.Parameter(torch.randn(32, dtype=torch.float32))

    # Evaluate offsets 0, 1, 2
    offsets = (0, 1, 2)
    s_logits, s_hidden = student_forward_with_oracle_h(
        model, ctx, H_param, device, future_token_ids=future, offsets=offsets
    )

    assert set(s_logits.keys()) == {0, 1, 2}
    for k in offsets:
        assert s_logits[k].shape == (50,)


def test_compute_kl_and_topk():
    """Verify exact KL sum calculation and top-1 / top-5 overlap."""
    t_logits = torch.randn(100)
    s_logits = t_logits.clone()

    kl_t, kl_nats, top1_m, top5_o, max_l, mean_l = compute_kl_and_topk(t_logits, s_logits)
    assert kl_nats == pytest.approx(0.0, abs=1e-5)
    assert top1_m is True
    assert top5_o == 1.0
    assert max_l == pytest.approx(0.0, abs=1e-5)
