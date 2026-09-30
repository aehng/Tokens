"""Unit tests for Arm B block cache control utilities."""

import pytest
import torch

from zip2zip.block_cache import (
    clone_cache,
    compute_kl_divergence,
    get_cache_seq_len,
    get_layer_kv,
)


def test_clone_cache_tuple():
    """Verify deep cloning of tuple-based past_key_values."""
    k0 = torch.randn(1, 4, 10, 32)
    v0 = torch.randn(1, 4, 10, 32)
    cache = ((k0, v0),)

    cloned = clone_cache(cache)
    assert get_cache_seq_len(cloned) == 10

    # Modify original -> cloned must not change
    k0.add_(10.0)
    k_cloned, v_cloned = get_layer_kv(cloned, 0)
    assert (k_cloned - v0).shape == k0.shape
    assert not torch.allclose(k_cloned, k0)


def test_compute_kl_divergence_identity():
    """KL between identical distributions must be zero."""
    logits = torch.randn(1, 1000)
    kl = compute_kl_divergence(logits, logits.clone())
    assert kl == pytest.approx(0.0, abs=1e-5)
