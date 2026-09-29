"""Unified Attribution Harness for Controlled Phi Quality/Speed Experiments.

Implements the four core experimental conditions:
  A. VANILLA: Normal canonical Phi-3.5-mini-instruct.
  B. PREDICTIVE CHECKPOINT (H-DISABLED): Trained checkpoint/LoRA loaded, but hypertokens disabled (K=0).
  C. ORACLE (HINDSIGHT): Predictive model + hindsight oracle codebook derived from canonical Vanilla continuation.
  D. REAL CURRENT PREDICTOR: Predictive model + Phi-only candidate retrieval (train_only index) + PooledMLP ranker.

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

ATTRIBUTION_RECORD_SCHEMA = "phi_attribution_record_v1"
ATTRIBUTION_SUMMARY_SCHEMA = "phi_attribution_summary_v1"

COND_A_VANILLA = "A_vanilla"
COND_B_H_DISABLED = "B_h_disabled"
COND_C_ORACLE = "C_oracle"
COND_D_REAL_PREDICTOR = "D_real_predictor"

ALL_CONDITIONS = (
    COND_A_VANILLA,
    COND_B_H_DISABLED,
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
    domain = sample.get("domain")
    if domain == "code":
        raw_prompt = sample.get("prompt_text") or sample.get("prompt") or ""
        ref = sample.get("reference") or sample.get("reference_response") or ""
        tests = sample.get("test_assert_statements") or sample.get("tests")
        sig = extract_function_signature(
            ref,
            sample_id=str(sample.get("prompt_id") or sample.get("id")),
            test_assert_statements=tests,
        )
        return f"{raw_prompt.rstrip()}\n\nImplement this Python function using the required signature:\n```python\n{sig}\n```"
    return str(sample.get("prompt_text") or sample.get("prompt") or "")


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
        from experiments.run_quality_benchmark import evaluate_restricted_mbpp
        tests = sample.get("test_assert_statements") or sample.get("tests") or []
        res = evaluate_restricted_mbpp(output_text, tests, timeout_s=timeout_s)
        return bool(res.get("problem_pass")), res

    if domain == "reasoning":
        from experiments.run_quality_benchmark import evaluate_gsm8k_reasoning
        gt = sample.get("reference") or sample.get("reference_response") or ""
        res = evaluate_gsm8k_reasoning(output_text, gt)
        return bool(res.get("exact_correct")), res

    if domain == "instruction":
        from experiments.run_quality_benchmark import evaluate_mechanical_instruction
        res = evaluate_mechanical_instruction(output_text)
        return bool(res.get("mechanical_instruction_pass")), res

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


def compute_attribution_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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
        "causal_diagnosis": {},
    }

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

    # Causal comparisons vs Vanilla (A)
    vanilla_stats = summary["conditions"].get(COND_A_VANILLA)
    if vanilla_stats:
        v_qual = vanilla_stats["aggregate_quality_rate"]
        v_lat = vanilla_stats["mean_total_latency_s"]
        v_steps = vanilla_stats["total_decode_calls"]

        for cond in (COND_B_H_DISABLED, COND_C_ORACLE, COND_D_REAL_PREDICTOR):
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

        # Root-cause causal classification
        comp_b = summary["comparison_vs_vanilla"].get(COND_B_H_DISABLED)
        comp_c = summary["comparison_vs_vanilla"].get(COND_C_ORACLE)
        comp_d = summary["comparison_vs_vanilla"].get(COND_D_REAL_PREDICTOR)
        stats_c = summary["conditions"].get(COND_C_ORACLE)
        stats_d = summary["conditions"].get(COND_D_REAL_PREDICTOR)

        primary_bottleneck = "unclear"
        detailed_diagnosis = ""

        if comp_b and comp_b["relative_quality_drop_pct"] > 3.0:
            primary_bottleneck = "checkpoint_lora"
            detailed_diagnosis = (
                f"Predictive checkpoint/LoRA without hypertokens (Condition B) loses "
                f"{comp_b['relative_quality_drop_pct']}% quality vs Vanilla. The fine-tuning "
                f"or base model state is degraded before any hypertoken mechanism operates."
            )
        elif comp_c and comp_c["relative_quality_drop_pct"] > 3.0:
            primary_bottleneck = "hypertoken_representation_continuation"
            detailed_diagnosis = (
                f"Oracle condition C loses {comp_c['relative_quality_drop_pct']}% quality despite "
                f"having perfect hindsight candidates. The H-encoder, HyperLinear decoder, semantic positions, "
                f"or post-H continuation state causes quality collapse."
            )
        elif stats_c and stats_c["total_h_emissions"] == 0:
            primary_bottleneck = "h_emission_head"
            detailed_diagnosis = (
                "Oracle condition C emitted 0 hypertokens. Candidate availability is fine, but the "
                "output head / HyperLinear fails to emit any hypertokens."
            )
        elif comp_d and comp_d["relative_quality_drop_pct"] > 3.0 and comp_c and comp_c["meets_3pct_quality_gate"]:
            primary_bottleneck = "candidate_retrieval_ranking"
            detailed_diagnosis = (
                f"Oracle condition C preserves quality (drop: {comp_c['relative_quality_drop_pct']}%), "
                f"but Real Predictor (Condition D) fails quality gate (drop: {comp_d['relative_quality_drop_pct']}%). "
                f"The candidate generation / ranking subsystem is the primary blocker."
            )
        elif comp_d and comp_d["meets_3pct_quality_gate"] and not comp_d["is_faster_than_vanilla"]:
            primary_bottleneck = "runtime_overhead_or_utilization"
            detailed_diagnosis = (
                f"Real Predictor (Condition D) preserves quality within 3% (drop: {comp_d['relative_quality_drop_pct']}%), "
                f"but real inference latency is higher than Vanilla (+{-comp_d['speedup_pct']}%). "
                f"Per-step overhead or low compression masks decode-call savings."
            )
        elif comp_d and comp_d["meets_3pct_quality_gate"] and comp_d["is_faster_than_vanilla"]:
            primary_bottleneck = "none_success"
            detailed_diagnosis = (
                f"SUCCESS: Condition D achieved quality within 3% (drop: {comp_d['relative_quality_drop_pct']}%) "
                f"and reduced real inference time by {comp_d['speedup_pct']}%."
            )

        summary["causal_diagnosis"] = {
            "primary_bottleneck": primary_bottleneck,
            "detailed_diagnosis": detailed_diagnosis,
        }

    return summary
