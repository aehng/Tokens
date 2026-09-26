"""CPU checks for vLLM warmup admission and request-local H activation."""

import torch

from tokens_vllm.warmup import (
    activate_codebook_slot,
    admission_kind,
    clear_predictive_slot,
    is_vllm_internal_warmup_request,
    mask_inactive_h_logits,
    prepare_warmup_slot,
)


class _Slots:
    def __init__(self) -> None:
        self.h_input = torch.ones(2, 32, 3)
        self.h_output = torch.ones(2, 32, 3)
        self.h_spans = torch.full((2, 32), 3, dtype=torch.int64)
        self.semantic_offset = torch.tensor([9, 9])
        self.physical_accounted = torch.tensor([9, 9])
        self.h_active = torch.tensor([True, True])
        self.pending_semantic_advance = torch.tensor([4, 4])
        self.pending_physical_advance = torch.tensor([2, 2])


def test_warmup_ids_match_only_the_pinned_pattern():
    assert is_vllm_internal_warmup_request("_warmup_0_")
    assert is_vllm_internal_warmup_request("_warmup_12_")
    assert is_vllm_internal_warmup_request("_warmup_27_")
    assert not is_vllm_internal_warmup_request("_warmup_")
    assert not is_vllm_internal_warmup_request("_warmup_user")
    assert not is_vllm_internal_warmup_request("_warmup_user_")
    assert not is_vllm_internal_warmup_request("user-1")
    assert not is_vllm_internal_warmup_request("foo_warmup_0_")


def test_missing_codebook_still_fails_for_a_normal_request():
    try:
        admission_kind("user-1", {})
    except RuntimeError as exc:
        assert "predictive_codebook" in str(exc)
    else:
        raise AssertionError("a normal request without a codebook must fail")
    try:
        admission_kind("_warmup_user_", None)
    except RuntimeError as exc:
        assert "predictive_codebook" in str(exc)
    else:
        raise AssertionError("a lookalike warmup id must fail")


def test_warmup_request_is_inactive_with_unit_spans():
    assert admission_kind("_warmup_0_", None) == "warmup"
    state = _Slots()
    prepare_warmup_slot(state, 0, num_computed_tokens=4)
    assert state.h_active[0].item() is False
    assert torch.equal(state.h_spans[0], torch.ones(32, dtype=torch.int64))
    assert int(state.semantic_offset[0]) == 4
    assert int(state.physical_accounted[0]) == 4
    assert torch.count_nonzero(state.h_input[0]) == 0
    assert torch.count_nonzero(state.h_output[0]) == 0
    assert state.h_active[1].item() is True


def test_codebook_request_activates_only_its_slot():
    assert admission_kind("req-a", {"predictive_codebook": {"k": 32}}) == "codebook"
    state = _Slots()
    prepare_warmup_slot(state, 0, 0)
    activate_codebook_slot(state, 1)
    assert state.h_active[0].item() is False
    assert state.h_active[1].item() is True


def test_removal_clears_h_active():
    state = _Slots()
    activate_codebook_slot(state, 0)
    clear_predictive_slot(state, 0)
    assert state.h_active[0].item() is False
    assert torch.count_nonzero(state.h_spans[0]) == 0
    assert torch.count_nonzero(state.h_input[0]) == 0
    assert torch.count_nonzero(state.h_output[0]) == 0
    assert int(state.semantic_offset[0]) == 0
    assert int(state.physical_accounted[0]) == 0
    assert int(state.pending_semantic_advance[0]) == 0
    assert int(state.pending_physical_advance[0]) == 0
    assert state.h_active[1].item() is True


def test_inactive_rows_mask_h_logits_and_active_rows_keep_them():
    h_logits = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]])
    active = torch.tensor([True, False])
    masked = mask_inactive_h_logits(h_logits, active)
    assert torch.equal(masked[0], h_logits[0])
    assert torch.isneginf(masked[1]).all()
    assert torch.equal(h_logits, torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]]))
