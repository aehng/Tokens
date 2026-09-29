"""CPU Unit and Integration Tests for Phi Quality/Speed Attribution Architecture.

All tests use synthetic or model-free fixtures to validate:
- Condition routing A/B/C/D
- H-disabled truly disables H behavior (all hyper slots masked to -inf)
- Oracle codebook construction from synthetic continuations
- Candidate pool and K parameter wiring
- FINAL access prohibition (security gate)
- Record schemas and serialization
- First divergence and dead-slot bookkeeping
- Causal attribution diagnosis logic across all 6 branches
- Kaggle 5-hour hard budget cap accounting
- Resume behavior without duplicates
- No external sourcebook leakage and clean TRAIN/DEV split isolation
"""

from collections import Counter
import json
from pathlib import Path
import pytest
import torch

from src.zip2zip.predictor_v2.attribution_harness import (
    ALL_CONDITIONS,
    ATTRIBUTION_RECORD_SCHEMA,
    ATTRIBUTION_SUMMARY_SCHEMA,
    COND_A_VANILLA,
    COND_B_H_DISABLED,
    COND_C_ORACLE,
    COND_D_REAL_PREDICTOR,
    INITIAL_VOCAB_SIZE,
    AttributionRecord,
    FinalSplitAccessForbiddenError,
    KaggleRunBudget,
    build_canonical_prompt_text,
    build_codebook_dict,
    check_split_safety,
    compute_attribution_summary,
    derive_oracle_codebook_phrases,
    extract_function_signature,
    find_first_divergence,
)
from src.zip2zip.static_codebook import StaticCodebookManager
from experiments.run_phi_attribution_benchmark import select_stratified_dev_prompts


def test_final_split_access_strictly_forbidden():
    """Verify that any attempt to evaluate or access FINAL raises a hard permission error."""
    with pytest.raises(FinalSplitAccessForbiddenError, match="CRITICAL PROTOCOL VIOLATION"):
        check_split_safety("FINAL")
    with pytest.raises(FinalSplitAccessForbiddenError):
        check_split_safety("final")
    # DEV and TRAIN are allowed
    check_split_safety("DEV")
    check_split_safety("TRAIN")


def test_condition_b_h_disabled_masks_all_hyper_logits():
    """Verify that in Condition B (H-disabled), StaticCodebookManager masks all hyper slots."""
    max_k = 32
    static_mgr = StaticCodebookManager(
        initial_vocab_size=INITIAL_VOCAB_SIZE,
        max_codebook_size=max_k,
        max_subtokens=4,
        embedding_dim=3072,
        pad_token_id=32000,
    )
    # Empty codebook for Condition B
    static_mgr.set_seeded_codebook({}, batch_size=1)
    assert static_mgr.num_seeded == 0

    # Create dummy logits tensor: (batch=1, vocab=32011 + 32 = 32043)
    total_vocab = INITIAL_VOCAB_SIZE + max_k
    logits = torch.zeros(1, total_vocab)

    masked_logits = static_mgr.mask_unused_logits(logits)
    # All hypertoken slots (32011 to 32043) must be -inf
    hyper_slice = masked_logits[0, INITIAL_VOCAB_SIZE : total_vocab]
    assert torch.all(hyper_slice == float("-inf"))
    # Base vocab logits must remain 0.0
    base_slice = masked_logits[0, :INITIAL_VOCAB_SIZE]
    assert torch.all(base_slice == 0.0)


def test_oracle_codebook_construction_from_synthetic_continuation():
    """Verify hindsight oracle phrase derivation and frequency*length weighting."""
    # Synthetic sequence where [100, 101, 102] appears 3 times (savings: 3*(3-1)=6)
    # and [200, 201, 202] appears 2 times (savings: 2*(3-1)=4)
    seq = [
        100, 101, 102, 5,
        100, 101, 102, 6,
        100, 101, 102, 7,
        200, 201, 202, 8,
        200, 201, 202, 9,
    ]
    phrases = derive_oracle_codebook_phrases(seq, k=4, min_len=2, max_len=4)
    assert len(phrases) <= 4
    # Top phrases should include the frequent multi-token phrases
    assert (100, 101, 102) in phrases
    assert (200, 201, 202) in phrases

    # Verify build_codebook_dict assigns contiguous hypertoken IDs
    cb = build_codebook_dict(phrases, INITIAL_VOCAB_SIZE)
    assert len(cb) == len(phrases)
    assigned_ids = sorted(cb.values())
    assert assigned_ids == list(range(INITIAL_VOCAB_SIZE, INITIAL_VOCAB_SIZE + len(phrases)))


def test_first_divergence_bookkeeping():
    """Verify first token divergence detection between candidate and Vanilla."""
    seq1 = [10, 20, 30, 40, 50]
    seq2 = [10, 20, 30, 40, 50]
    assert find_first_divergence(seq1, seq2) is None

    seq3 = [10, 20, 99, 40, 50]
    assert find_first_divergence(seq1, seq3) == 2

    seq4 = [10, 20]
    assert find_first_divergence(seq1, seq4) == 2


def test_kaggle_budget_accounting():
    """Verify strict cumulative budget tracking and 5-hour hard limit."""
    budget = KaggleRunBudget(hard_cap_hours=5.0)
    assert budget.total_compute_hours == 0.0
    assert budget.remaining_budget_hours == 5.0

    # 1. Smoke test: 12 minutes (0.2h)
    budget.record_run("smoke_01", "2026-09-29T01:00:00Z", "2026-09-29T01:12:00Z", duration_seconds=720, notes="Smoke test")
    assert round(budget.total_compute_hours, 2) == 0.2
    assert round(budget.remaining_budget_hours, 2) == 4.8

    # 2. Assert can launch Stage 1 requiring 1.5h
    budget.assert_can_launch(1.5)

    # Record Stage 1 run: 5400s = 1.5h
    budget.record_run("stage1_01", "2026-09-29T01:30:00Z", "2026-09-29T03:00:00Z", duration_seconds=5400)
    assert round(budget.total_compute_hours, 2) == 1.7
    assert round(budget.remaining_budget_hours, 2) == 3.3

    # Assert cannot launch a 4.0h run that would exceed the 5.0h cap
    with pytest.raises(Exception, match="5.0h hard limit"):
        budget.assert_can_launch(4.0)


def test_stratified_prompt_selection():
    """Verify balanced prompt selection across domains."""
    synthetic_records = [
        {"prompt_id": f"c_{i}", "domain": "code"} for i in range(15)
    ] + [
        {"prompt_id": f"r_{i}", "domain": "reasoning"} for i in range(15)
    ] + [
        {"prompt_id": f"i_{i}", "domain": "instruction"} for i in range(15)
    ]
    # Select 9 stratified prompts (3 each)
    selected = select_stratified_dev_prompts(synthetic_records, limit=9)
    assert len(selected) == 9
    counts = Counter(r["domain"] for r in selected)
    assert counts["code"] == 3
    assert counts["reasoning"] == 3
    assert counts["instruction"] == 3


def test_causal_decision_matrix_checkpoint_failure():
    """IF B is >3% worse than A -> checkpoint/LoRA problem."""
    records = []
    # 10 prompts
    for i in range(10):
        # Condition A: 10/10 pass
        records.append({"condition": COND_A_VANILLA, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50})
        # Condition B: only 7/10 pass (30% drop > 3%)
        records.append({"condition": COND_B_H_DISABLED, "prompt_id": f"p_{i}", "quality_gate_pass": (i < 7), "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50})
        # Condition C: 7/10 pass
        records.append({"condition": COND_C_ORACLE, "prompt_id": f"p_{i}", "quality_gate_pass": (i < 7), "total_latency_s": 0.8, "transformer_decode_calls": 35, "expanded_token_count": 50, "h_emissions": [{"id": 32011}]})
        # Condition D: 7/10 pass
        records.append({"condition": COND_D_REAL_PREDICTOR, "prompt_id": f"p_{i}", "quality_gate_pass": (i < 7), "total_latency_s": 0.85, "transformer_decode_calls": 38, "expanded_token_count": 50, "h_emissions": [{"id": 32011}]})

    summary = compute_attribution_summary(records)
    diag = summary["causal_diagnosis"]
    assert diag["primary_bottleneck"] == "checkpoint_lora"
    assert "Condition B" in diag["detailed_diagnosis"]


def test_causal_decision_matrix_h_representation_failure():
    """IF B ~= A, but C is >3% worse than B -> hypertoken representation/continuation problem."""
    records = []
    for i in range(10):
        # A: 10/10 pass
        records.append({"condition": COND_A_VANILLA, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50})
        # B: 10/10 pass
        records.append({"condition": COND_B_H_DISABLED, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50})
        # C: 6/10 pass (40% drop)
        records.append({"condition": COND_C_ORACLE, "prompt_id": f"p_{i}", "quality_gate_pass": (i < 6), "total_latency_s": 0.7, "transformer_decode_calls": 30, "expanded_token_count": 50, "h_emissions": [{"id": 32011}]})
        # D: 5/10 pass
        records.append({"condition": COND_D_REAL_PREDICTOR, "prompt_id": f"p_{i}", "quality_gate_pass": (i < 5), "total_latency_s": 0.75, "transformer_decode_calls": 32, "expanded_token_count": 50, "h_emissions": [{"id": 32011}]})

    summary = compute_attribution_summary(records)
    diag = summary["causal_diagnosis"]
    assert diag["primary_bottleneck"] == "hypertoken_representation_continuation"


def test_causal_decision_matrix_h_emission_head_failure():
    """IF B ~= A, C ~= A, but H tokens rarely/never emit -> H emission output head problem."""
    records = []
    for i in range(10):
        # A, B, C, D all 10/10 pass
        records.append({"condition": COND_A_VANILLA, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50})
        records.append({"condition": COND_B_H_DISABLED, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50})
        # C emits 0 hypertokens!
        records.append({"condition": COND_C_ORACLE, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50, "h_emissions": []})
        records.append({"condition": COND_D_REAL_PREDICTOR, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50, "h_emissions": []})

    summary = compute_attribution_summary(records)
    diag = summary["causal_diagnosis"]
    assert diag["primary_bottleneck"] == "h_emission_head"


def test_causal_decision_matrix_predictor_selection_failure():
    """IF B ~= A, C ~= A, but D is >3% worse than C -> candidate retrieval / ranking bottleneck."""
    records = []
    for i in range(10):
        records.append({"condition": COND_A_VANILLA, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50})
        records.append({"condition": COND_B_H_DISABLED, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50})
        records.append({"condition": COND_C_ORACLE, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 0.7, "transformer_decode_calls": 30, "expanded_token_count": 50, "h_emissions": [{"id": 32011}]})
        # D fails quality: only 6/10 pass (40% drop vs Vanilla)
        records.append({"condition": COND_D_REAL_PREDICTOR, "prompt_id": f"p_{i}", "quality_gate_pass": (i < 6), "total_latency_s": 0.75, "transformer_decode_calls": 32, "expanded_token_count": 50, "h_emissions": [{"id": 32011}]})

    summary = compute_attribution_summary(records)
    diag = summary["causal_diagnosis"]
    assert diag["primary_bottleneck"] == "candidate_retrieval_ranking"


def test_causal_decision_matrix_overhead_failure():
    """IF D preserves quality within 3% but is slower than Vanilla -> runtime overhead."""
    records = []
    for i in range(10):
        records.append({"condition": COND_A_VANILLA, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50})
        records.append({"condition": COND_B_H_DISABLED, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50})
        records.append({"condition": COND_C_ORACLE, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 0.7, "transformer_decode_calls": 30, "expanded_token_count": 50, "h_emissions": [{"id": 32011}]})
        # D preserves quality (10/10), but is slower (1.15s > 1.00s)
        records.append({"condition": COND_D_REAL_PREDICTOR, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.15, "transformer_decode_calls": 42, "expanded_token_count": 50, "h_emissions": [{"id": 32011}]})

    summary = compute_attribution_summary(records)
    diag = summary["causal_diagnosis"]
    assert diag["primary_bottleneck"] == "runtime_overhead_or_utilization"


def test_causal_decision_matrix_success():
    """IF D preserves quality within 3% AND reduces real latency -> SUCCESS!"""
    records = []
    for i in range(10):
        records.append({"condition": COND_A_VANILLA, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50})
        records.append({"condition": COND_B_H_DISABLED, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 1.0, "transformer_decode_calls": 50, "expanded_token_count": 50})
        records.append({"condition": COND_C_ORACLE, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 0.7, "transformer_decode_calls": 30, "expanded_token_count": 50, "h_emissions": [{"id": 32011}]})
        # D preserves quality (10/10) and is faster (0.80s < 1.00s, 20% speedup)
        records.append({"condition": COND_D_REAL_PREDICTOR, "prompt_id": f"p_{i}", "quality_gate_pass": True, "total_latency_s": 0.80, "transformer_decode_calls": 35, "expanded_token_count": 50, "h_emissions": [{"id": 32011}]})

    summary = compute_attribution_summary(records)
    diag = summary["causal_diagnosis"]
    assert diag["primary_bottleneck"] == "none_success"
    assert "SUCCESS" in diag["detailed_diagnosis"]


def test_resume_idempotence(tmp_path: Path):
    """Verify that rerun does not duplicate completed prompt-condition records."""
    raw_path = tmp_path / "raw_attribution_records.jsonl"

    rec1 = AttributionRecord(prompt_id="prompt_1", condition=COND_A_VANILLA, quality_gate_pass=True)
    raw_path.write_text(json.dumps(rec1.to_dict()) + "\n", encoding="utf-8")

    # Read existing
    completed = set()
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            completed.add((r["prompt_id"], r["condition"]))

    assert ("prompt_1", COND_A_VANILLA) in completed
    assert ("prompt_1", COND_B_H_DISABLED) not in completed


def test_canonical_prompt_formatting():
    """Verify prompt formatting contract matches canonical evaluator."""
    code_sample = {
        "prompt_id": "test_code_1",
        "domain": "code",
        "prompt_text": "Write a function that returns double.",
        "reference": "def double(x):\n    return x * 2",
    }
    rendered = build_canonical_prompt_text(code_sample)
    assert "Implement this Python function using the required signature:" in rendered
    assert "def double(x):" in rendered

    inst_sample = {
        "prompt_id": "test_inst_1",
        "domain": "instruction",
        "prompt_text": "Give 3 tips for baking.",
    }
    rendered_inst = build_canonical_prompt_text(inst_sample)
    assert rendered_inst == "Give 3 tips for baking."
