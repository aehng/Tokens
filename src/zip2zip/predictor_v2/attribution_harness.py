"""Unified Attribution Harness for Controlled Phi Quality/Speed Experiments.

Implements the controlled conditions:
  A. PURE VANILLA: Native Phi with no Tokens wrapper, checkpoint, or H.
  B0. TOKENS ARCHITECTURE/VANILLA WEIGHTS: Wrapper with adapters and H disabled.
  B1. TRAINED CHECKPOINT/H-DISABLED: Same wrapper plus the pinned Step-100 checkpoint.
  CF. FORCED ORACLE: Exact phrase matches are forced through H and expanded.
  C. ORACLE CODEBOOK LIVE: Hindsight codebook; the model chooses H or base tokens.
  D. REAL PREDICTOR: Phi-only TRAIN retrieval and the configured ranker.

Enforces:
  - Strict isolation: FINAL split access is strictly prohibited.
  - Reproducibility: Full machine-readable provenance, config hashes, and schemas.
  - No continuous polling / Kaggle compute cap (<= 5.0 hours).
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from src.zip2zip.predictor_v2.ablation_gates import (
    checkpoint_isolation_gate,
    forced_h_representation_gates,
    token_equivalence_gate,
)

ATTRIBUTION_RECORD_SCHEMA = "phi_attribution_record_v1"
ATTRIBUTION_SUMMARY_SCHEMA = "phi_attribution_summary_v1"

COND_A_VANILLA = "A_vanilla"
# B0 wraps the pinned Vanilla weights with Zip2Zip modules but leaves the
# checkpoint adapter disabled.  The historical Stage 1 B records are B1.
COND_B0_TOKENS_VANILLA_WEIGHTS = "B0_tokens_vanilla_weights"
COND_B_H_DISABLED = "B_h_disabled"
COND_B1_LORA_H_DISABLED = COND_B_H_DISABLED
COND_C_ORACLE = "C_oracle"
COND_CF_FORCED_ORACLE = "CF_forced_oracle"
COND_D_REAL_PREDICTOR = "D_real_predictor"

ALL_CONDITIONS = (
    COND_A_VANILLA,
    COND_B0_TOKENS_VANILLA_WEIGHTS,
    COND_B_H_DISABLED,
    COND_CF_FORCED_ORACLE,
    COND_C_ORACLE,
    COND_D_REAL_PREDICTOR,
)

CANONICAL_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
CANONICAL_MODEL_REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"
CANONICAL_ZIP2ZIP_ID = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
CANONICAL_ZIP2ZIP_REVISION = "11c461733a79d2a5de6b814585c3361ca2aacbe7"
INITIAL_VOCAB_SIZE = 32011
MAX_NEW_TOKENS = 1024
CANONICAL_EOS_TOKEN_IDS = (32007, 32001, 32000)
PAD_TOKEN_ID = 32000

# Pinned 12-prompt DEV subset: 4 Code / 4 Reasoning / 4 Instruction.
# IDs are the first four of each domain in the canonical DEV split_ids order.
STRATIFIED_DEV12_PROMPT_IDS: Tuple[str, ...] = (
    "mbpp_113",
    "mbpp_168",
    "mbpp_217",
    "mbpp_225",
    "gsm_2032",
    "gsm_2044",
    "gsm_2353",
    "gsm_2491",
    "alpaca_1",
    "alpaca_1024",
    "alpaca_1029",
    "alpaca_1132",
)


class AttributionError(Exception):
    """Raised when attribution harness constraints or split rules are violated."""


class FinalSplitAccessForbiddenError(AttributionError, PermissionError):
    """Raised if an attempt is made to evaluate or access FINAL split."""


@dataclass
class KaggleRunBudget:
    """Tracks cumulative Kaggle GPU compute against the hard 5-hour limit."""

    hard_cap_hours: float = 5.0
    runs: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_compute_hours(self) -> float:
        return sum(float(r.get("runtime_hours", 0.0)) for r in self.runs)

    @property
    def remaining_budget_hours(self) -> float:
        return max(0.0, self.hard_cap_hours - self.total_compute_hours)

    def record_run(
        self,
        run_id: str,
        start_time_iso: str,
        end_time_iso: str,
        duration_seconds: float,
        gpu_type: str = "T4",
        notes: str = "",
    ) -> Dict[str, Any]:
        hours = duration_seconds / 3600.0
        rec = {
            "run_id": run_id,
            "start_time": start_time_iso,
            "end_time": end_time_iso,
            "duration_seconds": round(duration_seconds, 2),
            "runtime_hours": round(hours, 4),
            "gpu_type": gpu_type,
            "notes": notes,
        }
        self.runs.append(rec)
        return rec

    def assert_can_launch(self, estimated_hours: float) -> None:
        if self.total_compute_hours + estimated_hours > self.hard_cap_hours:
            raise AttributionError(
                f"Cannot launch run requiring ~{estimated_hours:.2f}h: "
                f"Cumulative compute ({self.total_compute_hours:.2f}h) would exceed "
                f"the 5.0h hard limit (remaining: {self.remaining_budget_hours:.2f}h)."
            )


def select_stratified_dev_prompts(
    dev_records: Sequence[Any],
    limit: Optional[int] = None,
) -> List[Any]:
    """Select a balanced, stratified subset across Code, Reasoning, and Instruction.

    The 12-prompt diagnostic uses a pinned ID list so packaging, CPU tests, and
    GPU runs cannot silently disagree about which DEV rows were chosen.
    """
    records = list(dev_records)
    if limit == 12:
        by_id = {}
        for record in records:
            prompt_id = str(getattr(record, "prompt_id", None) or record.get("prompt_id"))
            by_id[prompt_id] = record
        missing = [prompt_id for prompt_id in STRATIFIED_DEV12_PROMPT_IDS if prompt_id not in by_id]
        if missing:
            raise AttributionError(
                "Pinned 12-prompt DEV subset is missing from the loaded split: "
                + ", ".join(missing)
            )
        return [by_id[prompt_id] for prompt_id in STRATIFIED_DEV12_PROMPT_IDS]

    by_domain: Dict[str, List[Any]] = {"code": [], "reasoning": [], "instruction": []}
    for record in records:
        domain = getattr(record, "domain", None) or record.get("domain")
        if domain in by_domain:
            by_domain[domain].append(record)

    if limit is None or limit >= len(records):
        ordered = []
        for domain in ("code", "reasoning", "instruction"):
            ordered.extend(by_domain[domain])
        return ordered

    per_domain = limit // 3
    remainder = limit % 3
    counts = {
        "code": per_domain + (1 if remainder > 0 else 0),
        "reasoning": per_domain + (1 if remainder > 1 else 0),
        "instruction": per_domain,
    }
    selected = []
    for domain in ("code", "reasoning", "instruction"):
        selected.extend(by_domain[domain][: counts[domain]])
    return selected


def extract_function_signature(
    reference: str,
    *,
    sample_id: str = "<unknown>",
    test_assert_statements: Optional[Sequence[str]] = None,
) -> str:
    """Return top-level Python function signature identified by tests."""
    try:
        module = ast.parse(reference)
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse reference for {sample_id}: {exc.msg}") from exc

    functions = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        raise ValueError(f"Expected at least 1 function, found 0 in {sample_id}")

    target = functions[0]
    if len(functions) > 1:
        test_modules = []
        if test_assert_statements:
            for test in test_assert_statements:
                if isinstance(test, str):
                    try:
                        test_modules.append(ast.parse(test))
                    except SyntaxError:
                        pass
        if not test_modules:
            test_modules = [
                ast.Module(
                    body=[n for n in module.body if isinstance(n, ast.Assert)],
                    type_ignores=[],
                )
            ]

        tested_names: set[str] = set()
        for t_mod in test_modules:
            for stmt in t_mod.body:
                expr = stmt.test if isinstance(stmt, ast.Assert) else stmt
                for n in ast.walk(expr):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                        tested_names.add(n.func.id)

        selected = [f for f in functions if f.name in tested_names]
        if len(selected) == 1:
            target = selected[0]

    sig_node = copy.copy(target)
    sig_node.decorator_list = []
    sig_node.body = [ast.Pass()]
    sig_node.type_comment = None
    ast.fix_missing_locations(sig_node)
    stub = ast.unparse(sig_node)
    sig = stub.removesuffix("\n    pass").rstrip()
    if not sig.endswith(":"):
        sig += ":"
    return sig


def build_canonical_prompt_text(sample: Mapping[str, Any]) -> str:
    """Format canonical prompt text identically across all 4 conditions."""
    if sample.get("rendered_prompt_text"):
        return str(sample["rendered_prompt_text"])
    raw_prompt = sample.get("prompt_text") or sample.get("prompt") or ""
    if raw_prompt.startswith("<|user|>"):
        return str(raw_prompt)
    domain = sample.get("domain")
    if domain == "code":
        ref = sample.get("reference") or sample.get("reference_response") or ""
        tests = sample.get("test_assert_statements") or sample.get("tests")
        try:
            sig = extract_function_signature(
                ref,
                sample_id=str(sample.get("prompt_id") or sample.get("id")),
                test_assert_statements=tests,
            )
            return f"{raw_prompt.rstrip()}\n\nImplement this Python function using the required signature:\n```python\n{sig}\n```"
        except Exception:
            return str(raw_prompt)
    return str(raw_prompt)


def derive_oracle_codebook_phrases(
    continuation_token_ids: Sequence[int],
    k: int = 32,
    min_len: int = 2,
    max_len: int = 4,
    disabled_ids: Optional[Set[int]] = None,
) -> List[Tuple[int, ...]]:
    """Derive hindsight oracle phrases directly from canonical continuation."""
    if not continuation_token_ids or k <= 0:
        return []
    disabled = disabled_ids or {0, 1, 2, *range(32000, 32011)}
    counts: Counter[Tuple[int, ...]] = Counter()
    n = len(continuation_token_ids)
    for length in range(min_len, min(max_len + 1, n + 1)):
        for i in range(n - length + 1):
            subseq = tuple(continuation_token_ids[i : i + length])
            if not any(t in disabled for t in subseq):
                counts[subseq] += 1

    # Utility: frequency * (length - 1)
    scored = sorted(
        counts.keys(),
        key=lambda phrase: (counts[phrase] * (len(phrase) - 1), counts[phrase], -len(phrase)),
        reverse=True,
    )
    return scored[:k]


def build_codebook_dict(phrases: Sequence[Sequence[int]], initial_vocab_size: int = INITIAL_VOCAB_SIZE) -> Dict[Tuple[int, ...], int]:
    """Map selected phrases to contiguous hypertoken IDs starting from initial_vocab_size."""
    return {tuple(phrase): initial_vocab_size + i for i, phrase in enumerate(phrases)}


def evaluate_output_quality(
    domain: str,
    output_text: str,
    sample: Mapping[str, Any],
    timeout_s: float = 5.0,
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate domain-specific correctness using canonical evaluator logic.

    Returns:
        (is_passed, scores_dict)
    """
    if domain == "code":
        from experiments.run_quality_benchmark import evaluate_mbpp_code
        gt = sample.get("reference") or sample.get("reference_response") or ""
        tests = sample.get("test_assert_statements") or sample.get("tests") or [
            line.strip() for line in gt.splitlines() if line.strip().startswith("assert")
        ]
        res = evaluate_mbpp_code(output_text, tests, timeout_s=timeout_s)
        return bool(res.get("problem_pass")), res

    if domain == "reasoning":
        from experiments.run_quality_benchmark import evaluate_gsm8k_reasoning
        gt = sample.get("reference") or sample.get("reference_response") or ""
        res = evaluate_gsm8k_reasoning(output_text, gt)
        return bool(res.get("exact_correct")), res

    if domain == "instruction":
        from experiments.run_quality_benchmark import evaluate_alpaca_instruction
        eos_hit = bool(sample.get("eos_reached", False))
        res = evaluate_alpaca_instruction(output_text, eos_reached=eos_hit)
        passed = bool(res.get("instruction_pass") or res.get("mechanical_instruction_pass"))
        return passed, res

    raise ValueError(f"Unknown domain: {domain}")


def find_first_divergence(seq1: Sequence[int], seq2: Sequence[int]) -> Optional[int]:
    """Return index of first token divergence between two sequences, or None if identical."""
    min_len = min(len(seq1), len(seq2))
    for i in range(min_len):
        if seq1[i] != seq2[i]:
            return i
    if len(seq1) != len(seq2):
        return min_len
    return None


@dataclass
class AttributionRecord:
    """Machine-readable record for one prompt evaluated under one condition."""

    schema: str = ATTRIBUTION_RECORD_SCHEMA
    prompt_id: str = ""
    domain: str = ""
    split: str = ""
    condition: str = ""
    created_at_utc: str = ""

    # Provenance
    model_id: str = CANONICAL_MODEL_ID
    model_revision: str = CANONICAL_MODEL_REVISION
    checkpoint_name: str = ""
    checkpoint_sha256: str = ""
    git_commit: str = ""
    runtime: Dict[str, Any] = field(default_factory=dict)

    # Output & Quality
    raw_output: str = ""
    expanded_output: str = ""
    generated_token_ids: List[int] = field(default_factory=list)
    expanded_token_ids: List[int] = field(default_factory=list)
    generated_token_count: int = 0
    expanded_token_count: int = 0
    termination_reason: str = ""
    termination_token_id: Optional[int] = None
    eos_reached: bool = False
    truncated: bool = False
    task_quality_scores: Dict[str, Any] = field(default_factory=dict)
    quality_gate_pass: bool = False
    severe_repetition_detected: bool = False

    # Hypertoken Diagnostics
    candidates_supplied: int = 0
    selected_h_slots: List[List[int]] = field(default_factory=list)
    phrase_length_per_slot: List[int] = field(default_factory=list)
    phrase_occurs_in_vanilla: List[bool] = field(default_factory=list)
    h_emissions: List[Dict[str, Any]] = field(default_factory=list)
    dead_h_slots: List[int] = field(default_factory=list)
    h_utilization_pct: float = 0.0
    first_h_emission_pos: Optional[int] = None
    first_divergence_from_vanilla_pos: Optional[int] = None
    continuation_tokens_after_last_h: Optional[int] = None
    forced_oracle_roundtrip_ok: Optional[bool] = None
    cf_semantic_positions_ok: Optional[bool] = None
    cf_continuation_stable: Optional[bool] = None

    # Performance
    ttft_s: Optional[float] = None
    decode_time_s: Optional[float] = None
    tpot_s: Optional[float] = None
    total_latency_s: float = 0.0
    time_to_eos_s: float = 0.0
    transformer_decode_calls: int = 0
    raw_model_steps: int = 0
    words_per_sec: float = 0.0
    setup_time_s: float = 0.0
    peak_vram_bytes: Optional[int] = None
    gpu_seconds_per_request: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_split_safety(split: str) -> None:
    """Enforce strict protection: FINAL split must NEVER be accessed."""
    if split.upper() == "FINAL":
        raise FinalSplitAccessForbiddenError(
            "CRITICAL PROTOCOL VIOLATION: Access to FINAL split is strictly forbidden in this experiment."
        )


def compute_attribution_summary(
    records: Sequence[Dict[str, Any]],
    external_gates: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute aggregate quality, performance, and causal attribution diagnosis."""
    by_condition: Dict[str, List[Dict[str, Any]]] = {c: [] for c in ALL_CONDITIONS}
    for r in records:
        c = r.get("condition")
        if c in by_condition:
            by_condition[c].append(r)

    summary: Dict[str, Any] = {
        "schema": ATTRIBUTION_SUMMARY_SCHEMA,
        "total_records": len(records),
        "conditions": {},
        "comparison_vs_vanilla": {},
        "paired_condition_comparisons": {},
        "causal_diagnosis": {},
    }
    gate_report = dict(external_gates or {})
    gate_report.setdefault("a_b0_token_equality", token_equivalence_gate(records))
    gate_report.setdefault("b0_b1_parameter_isolation", checkpoint_isolation_gate(records))
    gate_report.setdefault("forced_h_representation", forced_h_representation_gates(records))
    gate_report.setdefault("a_b0_logit_parity", {"status": "NOT_TESTED"})
    summary["ablation_gates"] = gate_report

    domains = ("code", "reasoning", "instruction")

    for cond, rows in by_condition.items():
        if not rows:
            continue
        total_prompts = len(rows)
        quality_passes = sum(1 for r in rows if r.get("quality_gate_pass"))
        agg_quality = quality_passes / total_prompts if total_prompts > 0 else 0.0

        domain_quality = {}
        for d in domains:
            d_rows = [r for r in rows if r.get("domain") == d]
            if d_rows:
                domain_quality[d] = {
                    "count": len(d_rows),
                    "passes": sum(1 for r in d_rows if r.get("quality_gate_pass")),
                    "rate": round(sum(1 for r in d_rows if r.get("quality_gate_pass")) / len(d_rows), 4),
                }

        total_decode_calls = sum(int(r.get("transformer_decode_calls", 0)) for r in rows)
        total_expanded = sum(int(r.get("expanded_token_count", 0)) for r in rows)
        compression_pct = round(100.0 * (1.0 - total_decode_calls / max(total_expanded, 1)), 2)

        mean_total_latency = sum(float(r.get("total_latency_s", 0.0)) for r in rows) / total_prompts
        mean_time_to_eos = sum(float(r.get("time_to_eos_s", 0.0)) for r in rows) / total_prompts
        mean_ttft = (
            sum(float(r.get("ttft_s", 0.0)) for r in rows if r.get("ttft_s") is not None)
            / max(sum(1 for r in rows if r.get("ttft_s") is not None), 1)
        )
        mean_tpot = (
            sum(float(r.get("tpot_s", 0.0)) for r in rows if r.get("tpot_s") is not None)
            / max(sum(1 for r in rows if r.get("tpot_s") is not None), 1)
        )

        h_emitted_total = sum(len(r.get("h_emissions", [])) for r in rows)
        mean_utilization = sum(float(r.get("h_utilization_pct", 0.0)) for r in rows) / total_prompts

        summary["conditions"][cond] = {
            "record_count": total_prompts,
            "aggregate_quality_rate": round(agg_quality, 4),
            "domain_quality": domain_quality,
            "total_decode_calls": total_decode_calls,
            "total_expanded_tokens": total_expanded,
            "realized_compression_pct": compression_pct,
            "mean_total_latency_s": round(mean_total_latency, 4),
            "mean_time_to_eos_s": round(mean_time_to_eos, 4),
            "mean_ttft_s": round(mean_ttft, 4),
            "mean_tpot_s": round(mean_tpot, 6),
            "total_h_emissions": h_emitted_total,
            "mean_h_utilization_pct": round(mean_utilization, 2),
            "eos_reached_count": sum(1 for r in rows if r.get("eos_reached")),
            "truncated_count": sum(1 for r in rows if r.get("truncated")),
            "severe_repetition_count": sum(1 for r in rows if r.get("severe_repetition_detected")),
        }

    # Descriptive comparisons vs Vanilla (A). A/B cannot isolate the
    # checkpoint from wrapper plumbing; B0 is required for that attribution.
    summary["paired_condition_comparisons"] = _paired_condition_comparisons(records)
    vanilla_stats = summary["conditions"].get(COND_A_VANILLA)
    if vanilla_stats:
        v_qual = vanilla_stats["aggregate_quality_rate"]
        v_lat = vanilla_stats["mean_total_latency_s"]
        v_steps = vanilla_stats["total_decode_calls"]

        for cond in (
            COND_B0_TOKENS_VANILLA_WEIGHTS,
            COND_B_H_DISABLED,
            COND_C_ORACLE,
            COND_CF_FORCED_ORACLE,
            COND_D_REAL_PREDICTOR,
        ):
            cond_stats = summary["conditions"].get(cond)
            if not cond_stats:
                continue
            c_qual = cond_stats["aggregate_quality_rate"]
            c_lat = cond_stats["mean_total_latency_s"]
            c_steps = cond_stats["total_decode_calls"]

            rel_qual_drop = ((v_qual - c_qual) / v_qual) if v_qual > 0 else 0.0
            meets_3pct_gate = rel_qual_drop <= 0.030001
            speedup_pct = round(100.0 * (v_lat - c_lat) / max(v_lat, 1e-6), 2)
            steps_saved = v_steps - c_steps

            summary["comparison_vs_vanilla"][cond] = {
                "absolute_quality_diff": round(c_qual - v_qual, 4),
                "relative_quality_drop_pct": round(rel_qual_drop * 100.0, 2),
                "meets_3pct_quality_gate": meets_3pct_gate,
                "latency_change_pct": round(-speedup_pct, 2),
                "speedup_pct": speedup_pct,
                "decode_steps_saved": steps_saved,
                "is_faster_than_vanilla": c_lat < v_lat,
            }

        # Root-cause classification is deliberately gated on controlled
        # comparisons.  Historical Stage 1 B bundled wrapper + checkpoint.
        comp_b0 = summary["comparison_vs_vanilla"].get(COND_B0_TOKENS_VANILLA_WEIGHTS)
        comp_b1 = summary["comparison_vs_vanilla"].get(COND_B_H_DISABLED)
        comp_c = summary["comparison_vs_vanilla"].get(COND_C_ORACLE)
        comp_d = summary["comparison_vs_vanilla"].get(COND_D_REAL_PREDICTOR)
        b0_stats = summary["conditions"].get(COND_B0_TOKENS_VANILLA_WEIGHTS)
        b1_stats = summary["conditions"].get(COND_B_H_DISABLED)
        stats_c = summary["conditions"].get(COND_C_ORACLE)
        stats_d = summary["conditions"].get(COND_D_REAL_PREDICTOR)
        cf_records = [r for r in records if r.get("condition") == COND_CF_FORCED_ORACLE]
        paired = summary["paired_condition_comparisons"]
        a_b0_controlled = _pair_field_all_equal(
            paired.get(f"{COND_A_VANILLA}_to_{COND_B0_TOKENS_VANILLA_WEIGHTS}", {}),
            "base_phi_hash_pairs_compared",
            "base_phi_hash_pairs_equal",
        ) and _pair_field_all_equal(
            paired.get(f"{COND_A_VANILLA}_to_{COND_B0_TOKENS_VANILLA_WEIGHTS}", {}),
            "generation_policy_pairs_compared",
            "generation_policy_pairs_equal",
        ) and _pair_field_all_equal(
            paired.get(f"{COND_A_VANILLA}_to_{COND_B0_TOKENS_VANILLA_WEIGHTS}", {}),
            "input_token_hash_pairs_compared",
            "input_token_hash_pairs_equal",
        ) and _pair_field_all_equal(
            paired.get(f"{COND_A_VANILLA}_to_{COND_B0_TOKENS_VANILLA_WEIGHTS}", {}),
            "dtype_pairs_compared",
            "dtype_pairs_equal",
        )
        b0_b1_controlled = _pair_field_all_equal(
            paired.get(f"{COND_B0_TOKENS_VANILLA_WEIGHTS}_to_{COND_B_H_DISABLED}", {}),
            "base_phi_hash_pairs_compared",
            "base_phi_hash_pairs_equal",
        ) and _pair_field_all_equal(
            paired.get(f"{COND_B0_TOKENS_VANILLA_WEIGHTS}_to_{COND_B_H_DISABLED}", {}),
            "generation_policy_pairs_compared",
            "generation_policy_pairs_equal",
        ) and _pair_field_all_equal(
            paired.get(f"{COND_B0_TOKENS_VANILLA_WEIGHTS}_to_{COND_B_H_DISABLED}", {}),
            "input_token_hash_pairs_compared",
            "input_token_hash_pairs_equal",
        ) and _pair_field_all_equal(
            paired.get(f"{COND_B0_TOKENS_VANILLA_WEIGHTS}_to_{COND_B_H_DISABLED}", {}),
            "dtype_pairs_compared",
            "dtype_pairs_equal",
        ) and all(
            bool(r.get("runtime", {}).get("lora_delta_verified"))
            for r in records
            if r.get("condition") == COND_B_H_DISABLED
        )
        b1_c_controlled = _pair_field_all_equal(
            paired.get(f"{COND_B_H_DISABLED}_to_{COND_C_ORACLE}", {}),
            "base_phi_hash_pairs_compared",
            "base_phi_hash_pairs_equal",
        ) and _pair_field_all_equal(
            paired.get(f"{COND_B_H_DISABLED}_to_{COND_C_ORACLE}", {}),
            "generation_policy_pairs_compared",
            "generation_policy_pairs_equal",
        ) and _pair_field_all_equal(
            paired.get(f"{COND_B_H_DISABLED}_to_{COND_C_ORACLE}", {}),
            "input_token_hash_pairs_compared",
            "input_token_hash_pairs_equal",
        ) and _pair_field_all_equal(
            paired.get(f"{COND_B_H_DISABLED}_to_{COND_C_ORACLE}", {}),
            "dtype_pairs_compared",
            "dtype_pairs_equal",
        )

        required_gates_pass = (
            gate_report.get("a_b0_token_equality", {}).get("status") == "PASS"
            and gate_report.get("a_b0_logit_parity", {}).get("status") == "PASS"
            and gate_report.get("b0_b1_parameter_isolation", {}).get("status") == "PASS"
        )
        forced_h_gate_pass = gate_report.get("forced_h_representation", {}).get("status") == "PASS"

        primary_bottleneck = "unresolved_architecture_vs_checkpoint"
        detailed_diagnosis = (
            "The available comparison does not yet separate Zip2Zip wrapper/plumbing from trained "
            "checkpoint weights. Run matched A/B0/B1 records before assigning this quality loss."
        )

        if not required_gates_pass:
            detailed_diagnosis = (
                "Causal attribution is blocked because one or more A/B0 token/logit parity or B0/B1 "
                "parameter-isolation gates did not pass. No downstream quality loss is assigned."
            )
        elif comp_b0 and comp_b0["relative_quality_drop_pct"] > 3.0 and a_b0_controlled:
            primary_bottleneck = "architecture_or_plumbing_candidate"
            detailed_diagnosis = (
                f"B0 differs from Vanilla by {comp_b0['relative_quality_drop_pct']}% quality. "
                "Base-Phi fingerprints and recorded generation-policy hashes match, so this comparison "
                "supports a wrapper/plumbing cause for the observed quality change."
            )
        elif comp_b0 and comp_b0["meets_3pct_quality_gate"] and comp_b1 and comp_b1["relative_quality_drop_pct"] > 3.0 and b0_b1_controlled:
            primary_bottleneck = "trained_checkpoint_candidate"
            detailed_diagnosis = (
                f"B0 is within the 3% quality gate versus Vanilla, while B1 is "
                f"{comp_b1['relative_quality_drop_pct']}% below Vanilla. B0/B1 base-Phi fingerprints and "
                "generation policies match, and B1 records verify the intended LoRA load."
            )
        elif forced_h_gate_pass and comp_b1 and comp_c and comp_c["relative_quality_drop_pct"] > comp_b1["relative_quality_drop_pct"] + 3.0 and b1_c_controlled:
            primary_bottleneck = "active_h_path_requires_forced_control"
            detailed_diagnosis = (
                "ORACLE CODEBOOK LIVE has a larger quality loss than H-disabled B1. This associates "
                "the change with active H use, but does not isolate representation/state from emission "
                "selection; CF FORCED ORACLE and exact expansion checks are still required."
            )
        elif forced_h_gate_pass and stats_c and stats_c["total_h_emissions"] == 0:
            primary_bottleneck = "h_emission_not_observed"
            detailed_diagnosis = (
                "ORACLE CODEBOOK LIVE emitted no H tokens in these records. Candidate opportunity was not "
                "accounted here, so this observation does not establish whether the cause is candidate coverage, "
                "the H output head, or another serving factor."
            )
        elif forced_h_gate_pass and comp_c and comp_d and comp_c["meets_3pct_quality_gate"] and comp_d["relative_quality_drop_pct"] > comp_c["relative_quality_drop_pct"] + 3.0:
            c_d_controlled = _pair_field_all_equal(
                paired.get(f"{COND_C_ORACLE}_to_{COND_D_REAL_PREDICTOR}", {}),
                "checkpoint_hash_pairs_compared",
                "checkpoint_hash_pairs_equal",
            ) and _pair_field_all_equal(
                paired.get(f"{COND_C_ORACLE}_to_{COND_D_REAL_PREDICTOR}", {}),
                "generation_policy_pairs_compared",
                "generation_policy_pairs_equal",
            ) and _pair_field_all_equal(
                paired.get(f"{COND_C_ORACLE}_to_{COND_D_REAL_PREDICTOR}", {}),
                "input_token_hash_pairs_compared",
                "input_token_hash_pairs_equal",
            ) and _pair_field_all_equal(
                paired.get(f"{COND_C_ORACLE}_to_{COND_D_REAL_PREDICTOR}", {}),
                "dtype_pairs_compared",
                "dtype_pairs_equal",
            )
            if c_d_controlled:
                primary_bottleneck = "predictor_codebook_candidate"
                detailed_diagnosis = (
                    "D is materially worse than ORACLE CODEBOOK LIVE while using a matched B1 checkpoint "
                    "and generation policy. This points toward candidate generation/ranking, subject to "
                    "opportunity and realization accounting for both supplied codebooks."
                )
        elif forced_h_gate_pass and comp_d and comp_d["meets_3pct_quality_gate"] and not comp_d["is_faster_than_vanilla"]:
            primary_bottleneck = "runtime_overhead_or_utilization"
            detailed_diagnosis = (
                f"Real Predictor (Condition D) preserves quality within 3% (drop: {comp_d['relative_quality_drop_pct']}%), "
                f"but real inference latency is higher than Vanilla (+{-comp_d['speedup_pct']}%). "
                f"Per-step overhead or low compression masks decode-call savings."
            )
        elif forced_h_gate_pass and comp_d and comp_d["meets_3pct_quality_gate"] and comp_d["is_faster_than_vanilla"]:
            primary_bottleneck = "none_success"
            detailed_diagnosis = (
                f"SUCCESS: Condition D achieved quality within 3% (drop: {comp_d['relative_quality_drop_pct']}%) "
                f"and reduced real inference time by {comp_d['speedup_pct']}%."
            )

        summary["causal_diagnosis"]["controlled_attribution_available"] = bool(
            required_gates_pass and comp_b0 and b0_stats and a_b0_controlled and b0_b1_controlled
        )
        summary["causal_diagnosis"]["forced_oracle_evidence_available"] = any(
            r.get("forced_oracle_roundtrip_ok") is not None for r in cf_records
        )
        summary["causal_diagnosis"]["forced_oracle_roundtrip_pass_count"] = sum(
            bool(r.get("forced_oracle_roundtrip_ok")) for r in cf_records
        )
        summary["causal_diagnosis"]["forced_oracle_roundtrip_record_count"] = len(cf_records)
        post_h = [
            r.get("runtime", {}).get("cf_post_h_top1_matches_target")
            for r in cf_records
            if r.get("runtime", {}).get("cf_post_h_top1_matches_target") is not None
        ]
        summary["causal_diagnosis"]["forced_oracle_post_h_top1_agreement_rate"] = (
            sum(bool(value) for value in post_h) / len(post_h) if post_h else None
        )

        summary["causal_diagnosis"].update({
            "primary_bottleneck": primary_bottleneck,
            "detailed_diagnosis": detailed_diagnosis,
            "required_architecture_and_checkpoint_gates_pass": required_gates_pass,
            "forced_h_representation_gates_pass": forced_h_gate_pass,
        })

    return summary


def _pair_field_all_equal(pair: Mapping[str, Any], compared_key: str, equal_key: str) -> bool:
    compared = int(pair.get(compared_key, 0))
    equal = int(pair.get(equal_key, 0))
    return compared > 0 and compared == equal


def _paired_condition_comparisons(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize per-prompt output equality without treating it as quality."""
    indexed: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for record in records:
        prompt_id = record.get("prompt_id")
        condition = record.get("condition")
        if prompt_id and condition in ALL_CONDITIONS:
            indexed.setdefault(str(prompt_id), {})[str(condition)] = record

    pairs = (
        (COND_A_VANILLA, COND_B0_TOKENS_VANILLA_WEIGHTS),
        (COND_B0_TOKENS_VANILLA_WEIGHTS, COND_B_H_DISABLED),
        (COND_B_H_DISABLED, COND_C_ORACLE),
        (COND_B_H_DISABLED, COND_CF_FORCED_ORACLE),
        (COND_C_ORACLE, COND_D_REAL_PREDICTOR),
    )
    result: Dict[str, Any] = {}
    for left, right in pairs:
        compared = []
        for prompt_id, row in indexed.items():
            if left not in row or right not in row:
                continue
            a = row[left]
            b = row[right]
            a_ids = a.get("expanded_token_ids", a.get("generated_token_ids"))
            b_ids = b.get("expanded_token_ids", b.get("generated_token_ids"))
            same_tokens = a_ids == b_ids if a_ids is not None and b_ids is not None else None
            a_hash = a.get("runtime", {}).get("base_phi_weight_sha256")
            b_hash = b.get("runtime", {}).get("base_phi_weight_sha256")
            a_policy = a.get("runtime", {}).get("generation_policy_sha256")
            b_policy = b.get("runtime", {}).get("generation_policy_sha256")
            a_checkpoint = a.get("checkpoint_sha256")
            b_checkpoint = b.get("checkpoint_sha256")
            a_input = a.get("runtime", {}).get("input_token_ids_sha256")
            b_input = b.get("runtime", {}).get("input_token_ids_sha256")
            a_dtype = a.get("runtime", {}).get("torch_dtype")
            b_dtype = b.get("runtime", {}).get("torch_dtype")
            compared.append((same_tokens, a_hash, b_hash, a_policy, b_policy, a_checkpoint, b_checkpoint, a_input, b_input, a_dtype, b_dtype))
        both_token_sequences = [row for row in compared if row[0] is not None]
        hashes = [(row[1], row[2]) for row in compared if row[1] is not None and row[2] is not None]
        policies = [(row[3], row[4]) for row in compared if row[3] is not None and row[4] is not None]
        checkpoints = [(row[5], row[6]) for row in compared if row[5] is not None and row[6] is not None]
        input_hashes = [(row[7], row[8]) for row in compared if row[7] is not None and row[8] is not None]
        dtypes = [(row[9], row[10]) for row in compared if row[9] is not None and row[10] is not None]
        result[f"{left}_to_{right}"] = {
            "paired_prompts": len(compared),
            "token_sequences_compared": len(both_token_sequences),
            "exact_token_sequence_matches": sum(1 for row in both_token_sequences if row[0]),
            "base_phi_hash_pairs_compared": len(hashes),
            "base_phi_hash_pairs_equal": sum(1 for a, b in hashes if a == b),
            "generation_policy_pairs_compared": len(policies),
            "generation_policy_pairs_equal": sum(1 for a, b in policies if a == b),
            "checkpoint_hash_pairs_compared": len(checkpoints),
            "checkpoint_hash_pairs_equal": sum(1 for a, b in checkpoints if a == b),
            "input_token_hash_pairs_compared": len(input_hashes),
            "input_token_hash_pairs_equal": sum(1 for a, b in input_hashes if a == b),
            "dtype_pairs_compared": len(dtypes),
            "dtype_pairs_equal": sum(1 for a, b in dtypes if a == b),
        }
    return result
