"""Unit tests for context-conditioned H-encoder and frozen-Phi representation harness."""

import pytest
import torch
import torch.nn as nn

from zip2zip.frozen_phi_h import (
    ContextConditionedHEncoder,
    assert_model_strictly_frozen,
    compute_model_parameter_hash,
    continuation_kl_loss,
)


def test_h_encoder_initialization_zero_gate():
    """Verify that at initialization (gate=0), H exactly equals 0.5 * (embed_a + embed_b)."""
    embed_dim = 128
    hidden_dim = 64
    encoder = ContextConditionedHEncoder(embed_dim=embed_dim, hidden_dim=hidden_dim)

    h_ctx = torch.randn(2, embed_dim)
    embed_a = torch.randn(2, embed_dim)
    embed_b = torch.randn(2, embed_dim)

    with torch.no_grad():
        H = encoder(h_ctx, embed_a, embed_b)
        expected_base = 0.5 * (embed_a + embed_b)
        delta = (H - expected_base).abs().max().item()

    assert delta == 0.0, f"At step 0, H must exactly equal mean(A, B), got delta={delta}"


def test_h_encoder_parameter_accounting():
    """Verify parameter accounting for standard 3072/2048 dimensions."""
    encoder = ContextConditionedHEncoder(embed_dim=3072, hidden_dim=2048)
    params = encoder.parameter_count
    # In dim = 3 * 3072 = 9216
    # LayerNorm: 9216 * 2 = 18432
    # fc1: 9216 * 2048 + 2048 = 18,876,416
    # fc2: 2048 * 3072 + 3072 = 6,294,528
    # gate: 1
    # Total = 18432 + 18876416 + 6294528 + 1 = 25,189,377
    assert 25_000_000 < params < 26_000_000
    assert params / 3.82e9 < 0.007  # Less than 0.7% of Phi-3.5


def test_continuation_kl_loss_exact_identity():
    """KL between identical logit vectors must be 0.0 nats."""
    vocab_size = 100
    teacher_logits = torch.randn(4, vocab_size)
    student_logits = teacher_logits.clone()
    offsets = [0, 1, 2, 4]

    loss, per_offset = continuation_kl_loss(teacher_logits, student_logits, offsets)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
    for off in offsets:
        assert per_offset[off] == pytest.approx(0.0, abs=1e-5)


def test_model_freezing_and_hash_invariance():
    """Verify strictly frozen model check and SHA256 parameter hashing."""
    linear = nn.Linear(16, 16)
    hash_before = compute_model_parameter_hash(linear)

    # Initially requires_grad is True -> assert should fail
    with pytest.raises(AssertionError):
        assert_model_strictly_frozen(linear)

    # Freeze parameters
    for p in linear.parameters():
        p.requires_grad = False
    assert_model_strictly_frozen(linear)

    hash_after = compute_model_parameter_hash(linear)
    assert hash_before == hash_after
