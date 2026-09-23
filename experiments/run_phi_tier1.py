"""Commit-pinned 12-prompt Phi comparison with exact run-scoped resume checks."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import run_quality_benchmark as benchmark  # noqa: E402
from experiments.benchmark_provenance import (  # noqa: E402
    build_generation_cache_key,
    canonical_sha256,
    create_or_verify_run_manifest,
    file_sha256,
    resolve_tested_commit,
    select_prompt_subset,
    validate_generation_cache_record,
    write_json_atomic,
)
from experiments.runtime_diagnostics import (  # noqa: E402
    add_runtime_derived_metrics,
    aggregate_runtime_metrics,
)


DEFAULT_DATA = REPO_ROOT / "data" / "cached_pure_pred_val_60.json"
DEFAULT_PROMPTS = (
    REPO_ROOT / "experiments" / "checkpoints" / "quality_benchmark" / "poc_12_prompt_ids.json"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "experiments" / "checkpoints" / "predictive_joint_pilot" / "checkpoint_step_100.pt"
)
DEFAULT_PREDICTOR = REPO_ROOT / "experiments" / "checkpoints" / "oracle_guided_predictor.pkl"
CONDITIONS = (
    "original_phi",
    "official_zip2zip",
    "predictive_step_100",
    "predictive_step_100_compressed_prompt",
    "predictive_step_100_compressed_prompt_gated_top16",
    "predictive_step_100_compressed_prompt_gated_top32",
    "predictive_step_150",
    "predictive_step_150_compressed_prompt",
)
EXPECTED_DOMAINS = {"code": 4, "reasoning": 4, "instruction": 4}


def _git_status(repo_root: Path) -> str:
    if not (repo_root / ".git").exists():
        return ""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runtime_identity(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "requested_device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["gpus"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_bytes": int(props.total_memory),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this runtime has no visible CUDA device")
    return info


def _prompt_text(sample: Mapping[str, Any]) -> str:
    return (
        benchmark.build_mbpp_prompt(sample)
        if sample["domain"] == "code"
        else str(sample["prompt"])
    )


def _condition_config(
    condition: str,
    base_revision: str,
    zip2zip_revision: str,
    checkpoint_path: Path,
    predictor_path: Path,
) -> dict[str, Any]:
    common = {
        "base_model_id": benchmark.PHI_MODEL_ID,
        "base_model_revision": base_revision,
        "tokenizer_id": benchmark.PHI_MODEL_ID,
        "tokenizer_revision": base_revision,
        "prompt_builder": benchmark.PROMPT_FORMATTER_VERSION,
        "evaluator_version": benchmark.EVALUATOR_VERSION,
    }
    if condition == "original_phi":
        return {**common, "condition": condition, "prompt_representation": "raw"}
    if condition == "official_zip2zip":
        return {
            **common,
            "condition": condition,
            "zip2zip_model_id": benchmark.ZIP2ZIP_MODEL_ID,
            "zip2zip_revision": zip2zip_revision,
            "prompt_representation": "official_zip2zip_tokenized",
        }
    step = "100" if "step_100" in condition else "150"
    checkpoint = checkpoint_path if step == "100" else checkpoint_path.with_name(
        "checkpoint_step_150.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required checkpoint for {condition} is missing: {checkpoint}")
    return {
        **common,
        "condition": condition,
        "zip2zip_model_id": benchmark.ZIP2ZIP_MODEL_ID,
        "zip2zip_revision": zip2zip_revision,
        "checkpoint_sha256": file_sha256(checkpoint),
        "predictor_sha256": file_sha256(predictor_path),
        "predictor_policy": {
            "kind": "capped_predictor",
            "budget": 32,
            "max_structural_slots": 0,
            "allow_numeric": True,
            "filter_bare_punctuation": True,
        },
        "prompt_representation": (
            "predictive_codebook_dp_segmented"
            if ("compressed_prompt" in condition)
            else "raw_base_token_ids"
        ),
        "emission_gate_top_n": (
            16 if "gated_top16" in condition
            else 32 if "gated_top32" in condition
            else None
        ),
    }


def _source_identity() -> tuple[dict[str, str], str]:
    source_paths = (
        REPO_ROOT / "experiments" / "run_phi_tier1.py",
        REPO_ROOT / "experiments" / "run_quality_benchmark.py",
        REPO_ROOT / "experiments" / "benchmark_provenance.py",
        REPO_ROOT / "experiments" / "runtime_diagnostics.py",
        REPO_ROOT / "experiments" / "mbpp_prompt.py",
        REPO_ROOT / "experiments" / "load_joint_checkpoint.py",
        REPO_ROOT / "experiments" / "load_oracle_predictor.py",
        REPO_ROOT / "src" / "zip2zip" / "model.py",
        REPO_ROOT / "src" / "zip2zip" / "tokenizer.py",
        REPO_ROOT / "src" / "zip2zip" / "static_codebook.py",
        REPO_ROOT / "src" / "zip2zip" / "predictor_policy.py",
        REPO_ROOT / "src" / "zip2zip" / "emission_gate.py",
    )
    hashes = {path.relative_to(REPO_ROOT).as_posix(): file_sha256(path) for path in source_paths}
    evaluator_source = {
        name: inspect.getsource(getattr(benchmark, name))
        for name in (
            "evaluate_mbpp_code",
            "evaluate_gsm8k_reasoning",
            "evaluate_alpaca_instruction",
            "severe_repetition_metrics",
            "generation_health_fields",
            "_answer_trace_positions",
            "TimingLogitsProcessor",
            "synchronize_device",
        )
    }
    return hashes, canonical_sha256(evaluator_source)


def _build_identity(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    tested_commit: str,
    runtime: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    evaluator_hash: str,
    condition_configs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    generation = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "dtype": "float16",
        "eos_rule": "last_generated_token_equals_tokenizer_eos_id",
    }
    cache_keys: dict[tuple[str, str], str] = {}
    prompt_identities = []
    for sample in samples:
        prompt = _prompt_text(sample)
        prompt_identities.append(
            {
                "id": sample["id"],
                "domain": sample["domain"],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "reference_sha256": hashlib.sha256(
                    str(sample.get("ground_truth_response", "")).encode("utf-8")
                ).hexdigest(),
            }
        )
        for condition, config in condition_configs.items():
            cache_keys[(sample["id"], condition)] = build_generation_cache_key(
                condition={
                    **dict(config),
                    "tested_commit": tested_commit,
                    "source_hashes": dict(source_hashes),
                },
                prompt_id=sample["id"],
                prompt_text=prompt,
                reference_text=str(sample.get("ground_truth_response", "")),
                generation=generation,
                evaluator_sha256=evaluator_hash,
                environment=runtime,
            )

    identity = {
        "schema": "phi_tier1_run_identity_v3",
        "runtime_diagnostics_schema": "phi_runtime_diagnostics_v1",
        "tested_commit": tested_commit,
        "tier": "phi_tier1_12",
        "prompt_ids_file_sha256": file_sha256(args.prompt_ids_file),
        "validation_data_sha256": file_sha256(args.validation_data),
        "prompt_ids": [sample["id"] for sample in samples],
        "prompts": prompt_identities,
        "conditions": {name: dict(config) for name, config in condition_configs.items()},
        "generation": generation,
        "evaluator_sha256": evaluator_hash,
        "source_hashes": dict(source_hashes),
        "runtime": dict(runtime),
        "record_cache_keys": {
            f"{prompt_id}::{condition}": key
            for (prompt_id, condition), key in sorted(cache_keys.items())
        },
    }
    return identity, cache_keys


def _load_current_run_records(
    raw_path: Path,
    cache_root: Path,
    expected_keys: Mapping[tuple[str, str], str],
    conditions: list[str],
    samples: list[dict[str, Any]],
) -> tuple[set[tuple[str, str]], list[dict[str, Any]]]:
    by_key = {
        (sample_id, condition): key
        for (sample_id, condition), key in expected_keys.items()
    }
    completed: set[tuple[str, str]] = set()
    records: dict[tuple[str, str], dict[str, Any]] = {}

    def accept(record: Any) -> None:
        if not isinstance(record, dict):
            return
        pair = (record.get("prompt_id"), record.get("condition"))
        expected = by_key.get(pair)
        if expected and validate_generation_cache_record(record, expected):
            records[pair] = record
            completed.add(pair)

    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    accept(json.loads(line))
                except json.JSONDecodeError:
                    continue

    for sample in samples:
        for condition in conditions:
            pair = (sample["id"], condition)
            if pair in completed:
                continue
            cache_path = cache_root / f"{by_key[pair]}.json"
            if not cache_path.is_file():
                continue
            try:
                accept(json.loads(cache_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue

    ordered_records = [
        records[(sample["id"], condition)]
        for sample in samples
        for condition in conditions
        if (sample["id"], condition) in records
    ]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in ordered_records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return completed, ordered_records


def _write_blind_review(
    run_dir: Path,
    samples: list[dict[str, Any]],
    records: list[dict[str, Any]],
    conditions: list[str],
) -> None:
    review_dir = run_dir / "instruction_review"
    private_dir = run_dir / "private"
    scoring_path = review_dir / "scoring_sheet.json"
    key_path = private_dir / "instruction_review_key.json"
    if scoring_path.exists() != key_path.exists():
        raise RuntimeError("Blind-review sheet and private key are out of sync; refusing to replace either")
    if scoring_path.exists():
        return

    by_prompt_condition = {
        (record["prompt_id"], record["condition"]): record for record in records
    }
    scoring_sheet = []
    unblind_key: dict[str, dict[str, str]] = {}
    randomizer = random.SystemRandom()
    for sample in samples:
        if sample["domain"] != "instruction":
            continue
        available = [
            condition
            for condition in conditions
            if (sample["id"], condition) in by_prompt_condition
        ]
        randomizer.shuffle(available)
        outputs = {}
        ratings = {}
        key = {}
        for index, condition in enumerate(available):
            label = f"Answer_{chr(ord('A') + index)}"
            outputs[label] = by_prompt_condition[(sample["id"], condition)].get("output_text", "")
            ratings[label] = {
                "task_adherence_1_to_5": None,
                "relevance_1_to_5": None,
                "completeness_1_to_5": None,
                "topic_drift_1_to_5": None,
                "truncation_or_eos_issue": None,
                "notes": "",
            }
            key[label] = condition
        scoring_sheet.append(
            {
                "prompt_id": sample["id"],
                "instruction": sample["prompt"],
                "outputs": outputs,
                "ratings": ratings,
            }
        )
        unblind_key[sample["id"]] = key

    write_json_atomic(scoring_path, scoring_sheet)
    write_json_atomic(key_path, unblind_key)


def _write_tier1_summary(
    run_dir: Path,
    samples: list[dict[str, Any]],
    records: list[dict[str, Any]],
    conditions: list[str],
    tested_commit: str,
) -> None:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        condition: {domain: [] for domain in EXPECTED_DOMAINS}
        for condition in conditions
    }
    for record in records:
        if record.get("condition") in grouped and record.get("domain") in EXPECTED_DOMAINS:
            grouped[record["condition"]][record["domain"]].append(record)

    summary: dict[str, Any] = {
        "status": "complete" if len(records) == len(samples) * len(conditions) else "incomplete",
        "tested_commit": tested_commit,
        "prompt_ids": [sample["id"] for sample in samples],
        "instruction_semantics": "manual_blind_review_pending",
        "conditions": {},
    }
    for condition in conditions:
        rows = [record for domain_rows in grouped[condition].values() for record in domain_rows]
        code = grouped[condition]["code"]
        reasoning = grouped[condition]["reasoning"]
        instruction = grouped[condition]["instruction"]
        total_expanded = sum(int(record.get("expanded_output_tokens", 0)) for record in rows)
        total_steps = sum(int(record.get("decode_steps", 0)) for record in rows)
        summary["conditions"][condition] = {
            "code": {
                "pass_count": sum(bool(record.get("problem_pass")) for record in code),
                "count": len(code),
                "syntax_valid_count": sum(bool(record.get("syntax_valid")) for record in code),
            },
            "reasoning": {
                "exact_correct_count": sum(bool(record.get("exact_correct")) for record in reasoning),
                "count": len(reasoning),
            },
            "instruction": {
                "mechanical_issue_count": sum(
                    bool(record.get("mechanical_instruction_failure")) for record in instruction
                ),
                "count": len(instruction),
                "eos_count": sum(bool(record.get("eos_reached")) for record in instruction),
                "hit_generation_cap_count": sum(bool(record.get("hit_max_length")) for record in instruction),
                "truncated_count": sum(bool(record.get("truncated")) for record in instruction),
                "severe_repetition_count": sum(
                    bool(record.get("severe_repetition_detected")) for record in instruction
                ),
                "semantic_score": None,
            },
            "compute": {
                "decode_steps": total_steps,
                "expanded_output_tokens": total_expanded,
                "micro_decode_reduction_pct": round(
                    100.0 * (1 - total_steps / max(total_expanded, 1)), 2
                ),
                "mean_wall_time_s": round(
                    sum(float(record.get("wall_time_s", 0)) for record in rows) / max(len(rows), 1),
                    3,
                ),
                "mean_ttft_s": round(
                    sum(float(record.get("ttft_s", 0)) for record in rows) / max(len(rows), 1),
                    3,
                ),
                "mean_model_prefill_tokens": round(
                    sum(int(record.get("model_prefill_tokens", record.get("base_prompt_tokens", 0))) for record in rows)
                    / max(len(rows), 1),
                    2,
                ),
                "mean_prompt_compression_pct": round(
                    sum(float(record.get("prompt_compression_pct", 0)) for record in rows)
                    / max(len(rows), 1),
                    2,
                ),
            },
        }

    vanilla_by_id = {
        record["prompt_id"]: record
        for record in records
        if record.get("condition") == "original_phi"
    }
    for condition in conditions:
        paired_rows = [
            add_runtime_derived_metrics(
                record,
                vanilla_record=vanilla_by_id.get(record.get("prompt_id")),
                quality_pass=record.get("quality_gate_pass"),
            )
            for record in records
            if record.get("condition") == condition
        ]
        summary["conditions"][condition]["runtime_diagnostics"] = aggregate_runtime_metrics(
            paired_rows
        )
        summary["conditions"][condition]["runtime_diagnostics"]["paired_prompt_ratios"] = [
            {
                "prompt_id": record.get("prompt_id"),
                "output_length_ratio_vs_vanilla": record.get("output_length_ratio_vs_vanilla"),
                "latency_ratio_vs_vanilla": record.get("latency_ratio_vs_vanilla"),
                "first_answer_decode_position": record.get("first_answer_decode_position"),
                "post_answer_decode_iterations": record.get("post_answer_decode_iterations"),
            }
            for record in paired_rows
        ]

    raw_condition = "predictive_step_100"
    compressed_condition = "predictive_step_100_compressed_prompt"
    raw_by_id = {
        record["prompt_id"]: record
        for record in records
        if record.get("condition") == raw_condition
    }
    compressed_by_id = {
        record["prompt_id"]: record
        for record in records
        if record.get("condition") == compressed_condition
    }
    paired = []
    for sample in samples:
        raw = raw_by_id.get(sample["id"])
        compressed = compressed_by_id.get(sample["id"])
        if not raw or not compressed:
            continue
        quality_fields = (
            ("problem_pass", "syntax_valid")
            if sample["domain"] == "code"
            else ("exact_correct",)
            if sample["domain"] == "reasoning"
            else (
                "mechanical_instruction_failure",
                "eos_reached",
                "hit_max_length",
                "truncated",
                "severe_repetition_detected",
                "response_length_base_tokens",
            )
        )
        paired.append(
            {
                "prompt_id": sample["id"],
                "codebook_match": raw.get("codebook_sha256") == compressed.get("codebook_sha256"),
                "raw_model_prefill_tokens": raw.get("model_prefill_tokens"),
                "compressed_model_prefill_tokens": compressed.get("model_prefill_tokens"),
                "prompt_compression_pct": compressed.get("prompt_compression_pct"),
                "raw_decode_steps": raw.get("decode_steps"),
                "compressed_decode_steps": compressed.get("decode_steps"),
                "raw_expanded_output_tokens": raw.get("expanded_output_tokens"),
                "compressed_expanded_output_tokens": compressed.get("expanded_output_tokens"),
                "raw_wall_time_s": raw.get("wall_time_s"),
                "compressed_wall_time_s": compressed.get("wall_time_s"),
                "quality": {
                    field: {
                        "raw": raw.get(field),
                        "compressed": compressed.get(field),
                    }
                    for field in quality_fields
                },
            }
        )
    matched_comparison_requested = (
        raw_condition in conditions and compressed_condition in conditions
    )
    if (
        matched_comparison_requested
        and len(paired) == len(samples)
        and not all(pair["codebook_match"] for pair in paired)
    ):
        raise RuntimeError(
            "Raw/compressed prompt comparison selected different codebooks; "
            "the matched A/B result is invalid."
        )
    summary["matched_prompt_representation"] = {
        "raw_condition": raw_condition,
        "compressed_condition": compressed_condition,
        "same_checkpoint_and_predictor_policy": matched_comparison_requested,
        "pair_count": len(paired),
        "all_codebooks_match": (
            bool(paired) and all(pair["codebook_match"] for pair in paired)
            if matched_comparison_requested
            else None
        ),
        "per_prompt": paired,
    }

    write_json_atomic(run_dir / "summary.json", summary)


def run(args: argparse.Namespace) -> Path:
    validation_path = Path(args.validation_data).resolve()
    prompt_ids_path = Path(args.prompt_ids_file).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    predictor_path = Path(args.predictor).resolve()
    with validation_path.open("r", encoding="utf-8") as source:
        all_samples = json.load(source)
    samples = select_prompt_subset(all_samples, prompt_ids_path, EXPECTED_DOMAINS)

    conditions = args.conditions or [
        "original_phi",
        "official_zip2zip",
        "predictive_step_100",
        "predictive_step_100_compressed_prompt",
    ]
    invalid = sorted(set(conditions) - set(CONDITIONS))
    if invalid:
        raise ValueError(f"Unknown conditions: {invalid}")
    if len(set(conditions)) != len(conditions):
        raise ValueError("conditions may not contain duplicates")
    if torch.device(args.device).type == "cuda":
        for label, revision in (
            ("--base-revision", args.base_revision),
            ("--zip2zip-revision", args.zip2zip_revision),
        ):
            if not revision or len(revision) != 40:
                raise ValueError(f"GPU runs require {label} as a full 40-character commit SHA")
        if _git_status(REPO_ROOT):
            raise ValueError("GPU runs require a clean, committed source worktree")

    tested_commit = args.tested_commit or resolve_tested_commit(REPO_ROOT)
    if len(tested_commit) != 40 or any(c not in "0123456789abcdef" for c in tested_commit.lower()):
        raise ValueError("--tested-commit must be a full 40-character Git SHA")
    runtime = _runtime_identity(args.device)
    source_hashes, evaluator_hash = _source_identity()
    condition_configs = {
        condition: _condition_config(
            condition,
            args.base_revision or "UNPINNED",
            args.zip2zip_revision or "UNPINNED",
            checkpoint_path,
            predictor_path,
        )
        for condition in conditions
    }
    identity, cache_keys = _build_identity(
        args,
        samples,
        tested_commit,
        runtime,
        source_hashes,
        evaluator_hash,
        condition_configs,
    )
    manifest_sha = canonical_sha256(identity)
    if args.output_dir:
        run_dir = Path(args.output_dir).resolve()
    else:
        writable_root = Path("/kaggle/working")
        if writable_root.is_dir():
            output_root = writable_root / "tokens_phi_tier1_runs"
        else:
            output_root = REPO_ROOT / "experiments" / "checkpoints" / "quality_benchmark" / "tier1_runs"
        run_dir = output_root / manifest_sha[:16]
    raw_path = run_dir / "raw_results.jsonl"
    manifest, resumed = create_or_verify_run_manifest(run_dir / "run_manifest.json", identity)
    cache_root = Path(args.cache_dir).resolve() if args.cache_dir else run_dir / "generation_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    completed_keys, records = _load_current_run_records(
        raw_path, cache_root, cache_keys, conditions, samples
    )
    manifest["status"] = "running"
    manifest["resumed"] = resumed
    manifest["cache_hit_count"] = len(completed_keys)
    manifest["started_or_resumed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json_atomic(run_dir / "run_manifest.json", manifest)

    predictive_bundles: dict[str, dict[str, Any]] = {}
    try:
        for step in ("100", "150"):
            step_conditions = [
                condition for condition in conditions
                if condition.startswith(f"predictive_step_{step}")
            ]
            needs_generation = any(
                (sample["id"], condition) not in completed_keys
                for condition in step_conditions
                for sample in samples
            )
            if step_conditions and needs_generation:
                checkpoint_for_step = (
                    checkpoint_path
                    if step == "100"
                    else checkpoint_path.with_name("checkpoint_step_150.pt")
                )
                predictive_bundles[step] = benchmark.load_predictive_model_bundle(
                    str(checkpoint_for_step),
                    device=args.device,
                    base_revision=args.base_revision,
                    model_revision=args.zip2zip_revision,
                    expected_step=int(step),
                )
                manifest.setdefault("checkpoint_load_reports", {})[step] = predictive_bundles[step][
                    "checkpoint_load_report"
                ]
                write_json_atomic(run_dir / "run_manifest.json", manifest)

        for condition in conditions:
            if condition == "original_phi":
                benchmark.run_condition_original_phi(
                    samples,
                    str(raw_path),
                    completed_keys,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    base_revision=args.base_revision,
                    generation_cache_keys=cache_keys,
                    cache_root=str(cache_root),
                    tested_commit=tested_commit,
                )
            elif condition == "official_zip2zip":
                benchmark.run_condition_official_zip2zip(
                    samples,
                    str(raw_path),
                    completed_keys,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    base_revision=args.base_revision,
                    model_revision=args.zip2zip_revision,
                    generation_cache_keys=cache_keys,
                    cache_root=str(cache_root),
                    tested_commit=tested_commit,
                )
            elif condition.startswith("predictive_step_100"):
                gate_n = condition_configs[condition].get("emission_gate_top_n")
                benchmark.run_condition_predictive(
                    str(checkpoint_path),
                    condition,
                    samples,
                    str(raw_path),
                    completed_keys,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    base_revision=args.base_revision,
                    model_revision=args.zip2zip_revision,
                    generation_cache_keys=cache_keys,
                    cache_root=str(cache_root),
                    tested_commit=tested_commit,
                    compress_prompt="compressed_prompt" in condition,
                    model_bundle=predictive_bundles.get("100"),
                    emission_gate_top_n=gate_n,
                )
            elif condition.startswith("predictive_step_150"):
                checkpoint_150 = checkpoint_path.with_name("checkpoint_step_150.pt")
                benchmark.run_condition_predictive(
                    str(checkpoint_150),
                    condition,
                    samples,
                    str(raw_path),
                    completed_keys,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    base_revision=args.base_revision,
                    model_revision=args.zip2zip_revision,
                    generation_cache_keys=cache_keys,
                    cache_root=str(cache_root),
                    tested_commit=tested_commit,
                    compress_prompt=condition.endswith("compressed_prompt"),
                    model_bundle=predictive_bundles.get("150"),
                )

        predictive_bundles.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _, records = _load_current_run_records(raw_path, cache_root, cache_keys, conditions, samples)
        _write_tier1_summary(run_dir, samples, records, conditions, tested_commit)
        expected_count = len(samples) * len(conditions)
        manifest["status"] = "complete" if len(records) == expected_count else "incomplete"
        manifest["completed_record_count"] = len(records)
        manifest["expected_record_count"] = expected_count
        manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json_atomic(run_dir / "run_manifest.json", manifest)
        if manifest["status"] == "complete":
            _write_blind_review(run_dir, samples, records, conditions)
        return run_dir
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json_atomic(run_dir / "run_manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the corrected 12-prompt Phi Tier-1 comparison.")
    parser.add_argument("--validation-data", default=str(DEFAULT_DATA))
    parser.add_argument("--prompt-ids-file", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--predictor", default=str(DEFAULT_PREDICTOR))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or an explicit device such as cuda:0")
    parser.add_argument(
        "--base-revision",
        default=benchmark.DEFAULT_PHI_REVISION,
        help="Pinned Phi/tokenizer hub commit SHA",
    )
    parser.add_argument(
        "--zip2zip-revision",
        default=benchmark.DEFAULT_ZIP2ZIP_REVISION,
        help="Pinned Zip2Zip hub commit SHA",
    )
    parser.add_argument("--tested-commit", help="Full project Git SHA; inferred when available")
    parser.add_argument("--output-dir", help="Explicit run directory; must be unused or have an identical manifest")
    parser.add_argument("--cache-dir", help="Run-generation cache directory")
    parser.add_argument("--max-new-tokens", type=int, default=benchmark.MAX_NEW_TOKENS)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    run_dir = run(args)
    print(f"Tier-1 run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
