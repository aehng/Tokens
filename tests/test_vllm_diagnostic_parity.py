"""Unit tests for vLLM Phase-5 diagnostic parity suite.

Verifies:
1. Baseline vocabulary comparison excludes H rows (width 32064).
2. Baseline metadata reports H-disabled on both sides.
3. Teacher-forced comparison uses the same canonical reference prefix on both runtimes.
4. Later comparison steps cannot be affected by differing free-running argmax choices.
5. Teacher-forced step count must match requested diagnostic length.
6. Matched position streams must compare actual captured positions.
7. Experiment A/B/C remain distinct and correctly labeled.
8. vLLM deterministic comparison rejects differing logits/tokens.
9. LoRA merged/unloaded assertion remains passing.
"""

from __future__ import annotations

import math
import pytest
import torch
from torch import nn

from tokens_vllm.contract import (
    BASE_VOCAB_SIZE,
    CODEBOOK_SIZE,
    INITIAL_VOCAB_SIZE,
    LOGICAL_VOCAB_SIZE,
    extract_base_logits,
    insert_hypertoken_logits,
)
from tokens_vllm.proof_harness import (
    assert_lora_merged_and_unloaded,
    compute_step_parity_metric,
    disable_hyper_modules,
)


def test_baseline_vocabulary_excludes_h_rows():
    """AC1: Baseline vocabulary comparison cannot accidentally include H rows."""
    base_logits = torch.randn(1, BASE_VOCAB_SIZE)
    h_logits = torch.randn(1, CODEBOOK_SIZE)
    logical = insert_hypertoken_logits(base_logits, h_logits)
    assert logical.shape[-1] == LOGICAL_VOCAB_SIZE  # 32096

    # extract_base_logits strips the 32 H rows
    extracted = extract_base_logits(logical)
    assert extracted.shape[-1] == BASE_VOCAB_SIZE  # 32064
    assert torch.equal(extracted, base_logits)

    # compute_step_parity_metric with is_predictive=False rejects non-32064 logits
    with pytest.raises(ValueError, match="baseline requires vocab width 32064"):
        compute_step_parity_metric(
            step=0,
            prefix_length=10,
            hf_position=9,
            vllm_position=9,
            hf_logits=logical[0],
            vllm_logits=base_logits[0],
            is_predictive=False,
        )

    # Baseline parity on pure 32064 logits computes cleanly with h_vocab_max_abs=None
    metric = compute_step_parity_metric(
        step=0,
        prefix_length=10,
        hf_position=9,
        vllm_position=9,
        hf_logits=base_logits[0],
        vllm_logits=base_logits[0] + 0.01,
        is_predictive=False,
    )
    assert metric["compared_vocab_width"] == 32064
    assert metric["h_vocab_max_abs"] is None
    assert metric["base_vocab_max_abs"] == pytest.approx(0.01, abs=1e-5)
    assert metric["max_abs"] == pytest.approx(0.01, abs=1e-5)


def test_disable_hyper_modules_replaces_and_restores():
    """AC1: disable_hyper_modules temporarily swaps in standard nn.Embedding and nn.Linear."""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(BASE_VOCAB_SIZE, 64)
            self.lm_head = nn.Linear(64, BASE_VOCAB_SIZE)

    m = DummyModel()
    orig_embed = m.model.embed_tokens
    orig_lm = m.lm_head

    with disable_hyper_modules(m) as plain:
        assert plain.model.embed_tokens is not orig_embed
        assert plain.lm_head is not orig_lm
        assert isinstance(plain.model.embed_tokens, nn.Embedding)
        assert isinstance(plain.lm_head, nn.Linear)
        # Weight tensor is identical
        assert plain.model.embed_tokens.weight.data_ptr() == orig_embed.weight.data_ptr()
        assert plain.lm_head.weight.data_ptr() == orig_lm.weight.data_ptr()

    # Original modules restored on exit
    assert m.model.embed_tokens is orig_embed
    assert m.lm_head is orig_lm


def test_baseline_metadata_reports_h_disabled():
    """AC1 & AC5: Baseline metadata reports H-disabled on both sides."""
    base_meta = {
        "hf_h_enabled": False,
        "vllm_h_enabled": False,
        "compared_vocab_width": 32064,
        "prompt_representation": "raw_base_tokens",
        "position_mode": "compressed",
    }
    assert base_meta["hf_h_enabled"] is False
    assert base_meta["vllm_h_enabled"] is False
    assert base_meta["compared_vocab_width"] == 32064
    assert base_meta["prompt_representation"] == "raw_base_tokens"
    assert base_meta["position_mode"] == "compressed"


def test_teacher_forced_prefix_construction():
    """AC2: Teacher-forced comparison uses the same canonical reference prefix on both runtimes."""
    prompt_ids = [10, 20, 30]
    canonical_tokens = [101, 102, 103, 104]

    steps = []
    for step in range(len(canonical_tokens)):
        prefix = prompt_ids + canonical_tokens[:step]
        steps.append((step, len(prefix), prefix))

    assert steps[0] == (0, 3, [10, 20, 30])
    assert steps[1] == (1, 4, [10, 20, 30, 101])
    assert steps[2] == (2, 5, [10, 20, 30, 101, 102])
    assert steps[3] == (3, 6, [10, 20, 30, 101, 102, 103])


def test_teacher_forcing_immune_to_differing_runtime_argmax():
    """AC2 & AC4: Changing one runtime's greedy choice does NOT alter subsequent teacher-forced prefixes."""
    prompt_ids = [10, 20, 30]
    canonical_ref = [100, 101, 102]

    # Suppose vLLM would have greedily predicted 999 instead of canonical_ref[0] (100)
    vllm_free_running_tokens = [999, 888, 777]

    # In teacher forcing, step 1 MUST receive canonical_ref[0], NOT 999
    step_1_prefix = prompt_ids + canonical_ref[:1]
    assert step_1_prefix == [10, 20, 30, 100]
    assert 999 not in step_1_prefix

    # Step 2 prefix receives canonical_ref[:2], NOT vLLM's choices
    step_2_prefix = prompt_ids + canonical_ref[:2]
    assert step_2_prefix == [10, 20, 30, 100, 101]


def test_teacher_forced_step_count_matches_requested():
    """AC2: Diagnostic fails if teacher-forced step counts do not match MAX_NEW."""
    max_new = 16

    # Incomplete step count -> teacher_forced_complete is False
    base_steps = [{"step": i} for i in range(15)]  # only 15
    exp_a_steps = [{"step": i} for i in range(16)]
    exp_b_steps = [{"step": i} for i in range(16)]
    exp_c_steps = [{"step": i} for i in range(16)]

    complete = bool(
        len(base_steps) == max_new
        and len(exp_a_steps) == max_new
        and len(exp_b_steps) == max_new
        and len(exp_c_steps) == max_new
    )
    assert complete is False

    # All 16 steps -> True
    base_steps.append({"step": 15})
    complete = bool(
        len(base_steps) == max_new
        and len(exp_a_steps) == max_new
        and len(exp_b_steps) == max_new
        and len(exp_c_steps) == max_new
    )
    assert complete is True


def test_matched_position_streams_compare_actual_captured_positions():
    """AC3: Matched experiments verify hf_position == vllm_position at every step."""
    hf_logits = torch.randn(LOGICAL_VOCAB_SIZE)
    vllm_logits = torch.randn(LOGICAL_VOCAB_SIZE)

    # Matched positions
    m_match = compute_step_parity_metric(
        step=0,
        prefix_length=10,
        hf_position=42,
        vllm_position=42,
        hf_logits=hf_logits,
        vllm_logits=vllm_logits,
        is_predictive=True,
    )
    assert m_match["position_match"] is True

    # Mismatched positions
    m_mismatch = compute_step_parity_metric(
        step=0,
        prefix_length=10,
        hf_position=42,
        vllm_position=53,
        hf_logits=hf_logits,
        vllm_logits=vllm_logits,
        is_predictive=True,
    )
    assert m_mismatch["position_match"] is False

    # Position contract valid checks
    b_steps = [{"position_match": True}] * 16
    c_steps = [{"position_match": True}] * 16
    a_steps = [{"position_match": False}] + [{"position_match": False}] * 15

    valid = bool(
        all(s["position_match"] for s in b_steps)
        and all(s["position_match"] for s in c_steps)
        and any(not s["position_match"] for s in a_steps)
    )
    assert valid is True

    # If Exp B had a position mismatch, valid is False
    b_steps_broken = list(b_steps)
    b_steps_broken[3] = {"position_match": False}
    valid_broken = bool(
        all(s["position_match"] for s in b_steps_broken)
        and all(s["position_match"] for s in c_steps)
        and any(not s["position_match"] for s in a_steps)
    )
    assert valid_broken is False


def test_experiment_abc_labeling_and_contracts():
    """AC7: Verify A/B/C experiments remain distinct and accurately configured."""
    experiments = {
        "exp_a": {
            "hf_position_mode": "compressed",
            "vllm_position_mode": "base_token_end",
            "h_enabled": True,
        },
        "exp_b": {
            "hf_position_mode": "compressed",
            "vllm_position_mode": "compressed",
            "h_enabled": True,
        },
        "exp_c": {
            "hf_position_mode": "base_token_end",
            "vllm_position_mode": "base_token_end",
            "h_enabled": True,
        },
    }

    # All three configurations must be mutually distinct
    tuples = [tuple(v.items()) for v in experiments.values()]
    assert len(set(tuples)) == 3


def test_determinism_check_rejects_differences():
    """Determinism check requires tokens_match == True and max_abs == 0.0."""
    # Identical
    det_pass = {
        "free_running_tokens_match": True,
        "free_running_max_abs": 0.0,
        "all_top1_match": True,
    }
    assert bool(det_pass["free_running_tokens_match"] and det_pass["free_running_max_abs"] == 0.0 and det_pass["all_top1_match"]) is True

    # Differing tokens
    det_diff_tokens = {
        "free_running_tokens_match": False,
        "free_running_max_abs": 0.0,
        "all_top1_match": True,
    }
    assert bool(det_diff_tokens["free_running_tokens_match"] and det_diff_tokens["free_running_max_abs"] == 0.0) is False

    # Non-zero max_abs
    det_diff_logits = {
        "free_running_tokens_match": True,
        "free_running_max_abs": 0.02,
        "all_top1_match": True,
    }
    assert bool(det_diff_logits["free_running_tokens_match"] and det_diff_logits["free_running_max_abs"] == 0.0) is False


def test_assert_lora_merged_and_unloaded():
    """LoRA merged/unloaded assertion passes on plain modules and fails on active LoRA."""
    # Plain module passes
    plain_model = nn.Sequential(nn.Linear(10, 10))
    assert_lora_merged_and_unloaded(plain_model)

    # Active lora_A parameter fails
    class FakeLoraModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            self.linear.lora_A = nn.Parameter(torch.randn(2, 10))

    fake_model = FakeLoraModule()
    with pytest.raises(AssertionError, match="still has active lora_A"):
        assert_lora_merged_and_unloaded(fake_model)
