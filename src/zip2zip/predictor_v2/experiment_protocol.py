"""Freeze gates and reproducibility helpers for canonical Predictor V2 runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.zip2zip.predictor_v2.canonical_dataset import (
    CanonicalDatasetError,
    CanonicalDatasetViews,
    FinalAccessError,
    FinalAccessPermit,
    sha256_file,
    sha256_json,
)


CANDIDATE_FREEZE_SCHEMA = "predictor_v2_candidate_freeze_v2"
ARCHITECTURE_FREEZE_SCHEMA = "predictor_v2_architecture_freeze_v2"
QUALITY_ATTRIBUTION_GATE_SCHEMA = "predictor_v2_quality_attribution_gate_v1"
CANDIDATE_PLAN_SCHEMA = "predictor_v2_candidate_plan_v1"
ARCHITECTURE_SHORTLIST_SCHEMA = "predictor_v2_architecture_shortlist_v1"
INTEGRATION_SUBSET_SCHEMA = "predictor_v2_integration_subset_v1"
LIVE_INTEGRATION_GATE_SCHEMA = "predictor_v2_live_integration_gate_v1"
MODEL_NAMES = ("Ridge", "PooledMLP", "CNNRanker", "GRURanker", "TransformerRanker")
CANDIDATE_STRATEGIES = (
    "baseline",
    "expanded_associations",
    "suffix_conditioned",
    "sparse_lexical",
)
SUPPORTED_POOL_SIZES = (256, 512, 1024, 2048)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def current_commit(cwd: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_manifest(cwd: str | Path = ".") -> dict[str, Any]:
    try:
        import numpy

        numpy_version: str | None = numpy.__version__
    except Exception:
        numpy_version = None
    try:
        import torch

        torch_version: str | None = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception:
        torch_version = None
        cuda_available = False
        cuda_device = None
    return {
        "git_commit": current_commit(cwd),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy_version,
        "torch": torch_version,
        "cuda_available": cuda_available,
        "cuda_device_0": cuda_device,
    }


def predictor_v2_source_hashes(cwd: str | Path = ".") -> dict[str, str]:
    """Hash the relevant source tree so dirty worktrees remain identifiable."""
    repo = Path(cwd).resolve()
    patterns = (
        "src/zip2zip/predictor_v2/**/*.py",
        "experiments/*predictor_v2*.py",
        "experiments/benchmark_candidate_recall.py",
        "experiments/benchmark_canonical_candidate_recall.py",
        "experiments/rebuild_train_association_index.py",
    )
    paths = sorted(
        {path.resolve() for pattern in patterns for path in repo.glob(pattern) if path.is_file()},
        key=lambda path: str(path).casefold(),
    )
    return {path.relative_to(repo).as_posix(): sha256_file(path) for path in paths}


def make_experiment_manifest(
    *,
    stage: str,
    views: CanonicalDatasetViews,
    model_revision: str,
    tokenizer_revision: str,
    candidate_config: Mapping[str, Any] | None = None,
    architecture_config: Mapping[str, Any] | None = None,
    seed: int | None = None,
    training_config: Mapping[str, Any] | None = None,
    output_artifacts: Mapping[str, str] | None = None,
    cwd: str | Path = ".",
) -> dict[str, Any]:
    source_hashes = predictor_v2_source_hashes(cwd)
    return {
        "schema": "predictor_v2_experiment_manifest_v1",
        "stage": stage,
        "created_at_utc": utc_now(),
        "dataset_sha256": views.dataset_sha256,
        "manifest_sha256": views.manifest_sha256,
        "provenance_sha256": views.provenance_sha256,
        "split_sha256": {
            "TRAIN": views.train_split_sha256,
            "DEV": views.dev_split_sha256,
            "FINAL": views.final_split_sha256,
        },
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "candidate_generator": dict(candidate_config or {}),
        "architecture": dict(architecture_config or {}),
        "seed": seed,
        "training_config": dict(training_config or {}),
        "runtime": runtime_manifest(cwd),
        "source_code": {
            "files_sha256": source_hashes,
            "tree_sha256": sha256_json(source_hashes),
        },
        "output_artifacts": dict(output_artifacts or {}),
    }


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> None:
    """Create a JSON artifact without replacing an existing file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(target, flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(payload)


def read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CanonicalDatasetError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CanonicalDatasetError(f"{label} is invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise CanonicalDatasetError(f"{label} must be a JSON object")
    return data


def validate_resume_pair(
    checkpoint_path: str | Path,
    run_metadata_path: str | Path,
    expected_run_config_sha256: str,
) -> dict[str, Any]:
    """Verify a completed seed run before a caller loads its checkpoint."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint is missing: {checkpoint}")
    metadata = read_json_object(run_metadata_path, "resume run metadata")
    if metadata.get("run_config_sha256") != expected_run_config_sha256:
        raise CanonicalDatasetError("resume configuration hash does not match")
    if metadata.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise CanonicalDatasetError("resume checkpoint hash does not match metadata")
    return metadata


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalDatasetError(f"{label} must be a non-empty string")
    return value.strip()


def load_quality_attribution_gate(
    path: str | Path, views: CanonicalDatasetViews, *, require_predictor_bottleneck: bool = True
) -> dict[str, Any]:
    gate = read_json_object(path, "broader live Phi attribution gate")
    evidence_path = gate.get("evidence_path")
    if not isinstance(evidence_path, str) or not Path(evidence_path).is_file() or sha256_file(evidence_path) != gate.get("evidence_sha256"):
        raise CanonicalDatasetError("quality attribution source evidence is missing or changed")
    if read_json_object(evidence_path, "quality attribution source evidence") != gate.get("evidence"):
        raise CanonicalDatasetError("quality attribution embedded evidence differs from its source artifact")
    gate = load_quality_attribution_gate_from_object(gate, views)
    status = gate.get("status")
    major = gate.get("predictor_codebook_major_bottleneck") is True
    if require_predictor_bottleneck and (status != "passed" or not major):
        raise CanonicalDatasetError("broader live attribution did not confirm Predictor/codebook as a major bottleneck; redirect before training")
    if status not in {"passed", "redirected"}:
        raise CanonicalDatasetError("quality attribution gate status must be passed or redirected")
    _require_nonempty_string(gate.get("primary_bottleneck"), "primary bottleneck")
    _require_nonempty_string(gate.get("rationale"), "quality attribution rationale")
    return gate


def make_quality_attribution_gate(
    *, evidence_path: str | Path, primary_bottleneck: str, rationale: str,
    views: CanonicalDatasetViews, output_path: str | Path,
) -> dict[str, Any]:
    evidence = read_json_object(evidence_path, "broader live Phi attribution evidence")
    allowed = {
        "candidate_generation", "candidate_ranking", "predictor", "codebook", "h_emission",
        "representation", "continuation_state", "eos", "serving", "other",
    }
    if primary_bottleneck not in allowed:
        raise CanonicalDatasetError(f"primary bottleneck must be one of {sorted(allowed)}")
    predictor_major = primary_bottleneck in {"candidate_generation", "candidate_ranking", "predictor", "codebook"}
    gate = {
        "schema": QUALITY_ATTRIBUTION_GATE_SCHEMA,
        "status": "passed" if predictor_major else "redirected",
        "created_at_utc": utc_now(),
        "dataset_sha256": views.dataset_sha256,
        "dev_split_sha256": views.dev_split_sha256,
        "evidence_path": str(Path(evidence_path).resolve()),
        "evidence_sha256": sha256_file(evidence_path),
        "evidence": evidence,
        "primary_bottleneck": primary_bottleneck,
        "predictor_codebook_major_bottleneck": predictor_major,
        "redirect_to": None if predictor_major else primary_bottleneck,
        "rationale": _require_nonempty_string(rationale, "quality attribution rationale"),
    }
    # Validate both passed and redirected artifacts, while allowing the latter to be reported.
    load_quality_attribution_gate_from_object(gate, views)
    write_json_exclusive(output_path, gate)
    return gate


def load_quality_attribution_gate_from_object(gate: Mapping[str, Any], views: CanonicalDatasetViews) -> dict[str, Any]:
    if gate.get("schema") != QUALITY_ATTRIBUTION_GATE_SCHEMA:
        raise CanonicalDatasetError("quality attribution gate has an unsupported schema")
    evidence = gate.get("evidence")
    if gate.get("dataset_sha256") != views.dataset_sha256 or gate.get("dev_split_sha256") != views.dev_split_sha256:
        raise CanonicalDatasetError("quality attribution gate dataset/DEV provenance mismatch")
    if not isinstance(evidence, Mapping) or evidence.get("scope") != "DEV":
        raise CanonicalDatasetError("quality attribution gate requires live DEV evidence")
    if evidence.get("dataset_sha256") != views.dataset_sha256 or evidence.get("dev_split_sha256") != views.dev_split_sha256:
        raise CanonicalDatasetError("quality attribution evidence provenance mismatch")
    if not isinstance(evidence.get("matched_prompt_count"), int) or evidence["matched_prompt_count"] <= 0 or not isinstance(evidence.get("domains"), list) or not evidence["domains"]:
        raise CanonicalDatasetError("quality attribution evidence requires matched prompts across domains")
    if not isinstance(evidence.get("failure_attribution_counts"), Mapping):
        raise CanonicalDatasetError("quality attribution evidence requires failure attribution counts")
    if not isinstance(evidence.get("baseline_conditions"), list) or set(evidence["baseline_conditions"]) < {"Vanilla", "Predictive Phi"}:
        raise CanonicalDatasetError("quality attribution evidence must compare Vanilla and Predictive Phi")
    required = {"candidate_generation", "candidate_ranking", "codebook", "h_emission", "representation", "continuation_state", "eos", "serving", "other"}
    if not required.issubset(evidence["failure_attribution_counts"]) or any(not isinstance(evidence["failure_attribution_counts"][name], int) or evidence["failure_attribution_counts"][name] < 0 for name in required):
        raise CanonicalDatasetError("quality attribution evidence omits one or more failure attribution categories")
    status = gate.get("status")
    major = gate.get("predictor_codebook_major_bottleneck") is True
    primary = gate.get("primary_bottleneck")
    if primary not in {"candidate_generation", "candidate_ranking", "predictor", "codebook", "h_emission", "representation", "continuation_state", "eos", "serving", "other"}:
        raise CanonicalDatasetError("quality attribution gate has an unsupported primary bottleneck")
    predictor_major = primary in {"candidate_generation", "candidate_ranking", "predictor", "codebook"}
    if status not in {"passed", "redirected"} or major != predictor_major or (status == "passed") != major:
        raise CanonicalDatasetError("quality attribution status does not agree with the primary bottleneck decision")
    expected_redirect = None if predictor_major else primary
    if gate.get("redirect_to") != expected_redirect:
        raise CanonicalDatasetError("quality attribution redirect does not agree with the primary bottleneck")
    _require_nonempty_string(gate.get("primary_bottleneck"), "primary bottleneck")
    _require_nonempty_string(gate.get("rationale"), "quality attribution rationale")
    return dict(gate)


def make_candidate_plan(
    *, benchmark_path: str | Path, strategy: str, pool_size: int, rationale: str,
    quality_attribution_gate_path: str | Path, views: CanonicalDatasetViews, output_path: str | Path,
) -> dict[str, Any]:
    load_quality_attribution_gate(quality_attribution_gate_path, views)
    if strategy not in CANDIDATE_STRATEGIES or pool_size not in SUPPORTED_POOL_SIZES:
        raise CanonicalDatasetError("candidate plan uses an unsupported strategy or pool size")
    results = read_json_object(benchmark_path, "candidate benchmark")
    if results.get("scope") != "DEV" or results.get("is_full_dev") is not True:
        raise CanonicalDatasetError("candidate plan requires a full DEV-only benchmark")
    if (results.get("dataset_sha256"), results.get("train_split_sha256"), results.get("dev_split_sha256")) != (views.dataset_sha256, views.train_split_sha256, views.dev_split_sha256):
        raise CanonicalDatasetError("candidate benchmark dataset/split hash mismatch")
    try:
        metrics = results["strategy_results"][strategy][str(pool_size)]
    except (KeyError, TypeError) as exc:
        raise CanonicalDatasetError("candidate plan selection has no DEV benchmark metrics") from exc
    plan = {
        "schema": CANDIDATE_PLAN_SCHEMA, "status": "proposed", "created_at_utc": utc_now(),
        "dataset_sha256": views.dataset_sha256, "train_split_sha256": views.train_split_sha256,
        "dev_split_sha256": views.dev_split_sha256,
        "quality_attribution_gate_sha256": sha256_file(quality_attribution_gate_path),
        "benchmark_sha256": sha256_file(benchmark_path),
        "benchmark_path": str(Path(benchmark_path).resolve()),
        "train_index_sha256": results.get("train_index_sha256"),
        "train_index_provenance_sha256": results.get("train_index_provenance_sha256"),
        "selection": {"strategy": strategy, "pool_size": pool_size, "config": metrics.get("candidate_config", {})},
        "dev_metrics": metrics, "rationale": _require_nonempty_string(rationale, "candidate plan rationale"),
    }
    if not plan["train_index_sha256"] or not plan["train_index_provenance_sha256"]:
        raise CanonicalDatasetError("candidate benchmark is missing TRAIN index provenance")
    write_json_exclusive(output_path, plan)
    return plan


def load_candidate_plan(path: str | Path, views: CanonicalDatasetViews, quality_attribution_gate_path: str | Path) -> dict[str, Any]:
    load_quality_attribution_gate(quality_attribution_gate_path, views)
    plan = read_json_object(path, "candidate plan")
    if plan.get("schema") != CANDIDATE_PLAN_SCHEMA or plan.get("status") != "proposed":
        raise CanonicalDatasetError("candidate plan is absent or not proposed")
    for key, expected in (("dataset_sha256", views.dataset_sha256), ("train_split_sha256", views.train_split_sha256), ("dev_split_sha256", views.dev_split_sha256), ("quality_attribution_gate_sha256", sha256_file(quality_attribution_gate_path))):
        if plan.get(key) != expected:
            raise CanonicalDatasetError(f"candidate plan {key} mismatch")
    benchmark_path = plan.get("benchmark_path")
    if not isinstance(benchmark_path, str) or not Path(benchmark_path).is_file() or sha256_file(benchmark_path) != plan.get("benchmark_sha256"):
        raise CanonicalDatasetError("candidate plan DEV benchmark is missing or changed")
    selection = plan.get("selection")
    if not isinstance(selection, Mapping) or selection.get("strategy") not in CANDIDATE_STRATEGIES or selection.get("pool_size") not in SUPPORTED_POOL_SIZES:
        raise CanonicalDatasetError("candidate plan has an invalid selection")
    benchmark = read_json_object(benchmark_path, "candidate benchmark")
    if benchmark.get("scope") != "DEV" or benchmark.get("is_full_dev") is not True or benchmark.get("dataset_sha256") != views.dataset_sha256 or benchmark.get("train_split_sha256") != views.train_split_sha256 or benchmark.get("dev_split_sha256") != views.dev_split_sha256:
        raise CanonicalDatasetError("candidate plan source benchmark has invalid DEV provenance")
    try:
        metrics = benchmark["strategy_results"][selection["strategy"]][str(selection["pool_size"])]
    except (KeyError, TypeError) as exc:
        raise CanonicalDatasetError("candidate plan selection is absent from its source benchmark") from exc
    if plan.get("dev_metrics") != metrics or plan.get("train_index_sha256") != benchmark.get("train_index_sha256") or plan.get("train_index_provenance_sha256") != benchmark.get("train_index_provenance_sha256"):
        raise CanonicalDatasetError("candidate plan metrics or TRAIN index hashes differ from its source benchmark")
    _require_nonempty_string(plan.get("rationale"), "candidate plan rationale")
    return plan


def _candidate_key(item: Mapping[str, Any]) -> tuple[str, int]:
    return str(item.get("architecture")), int(item.get("seed", -1))


def make_architecture_shortlist(
    *, bakeoff_path: str | Path, quality_attribution_gate_path: str | Path,
    candidate_plan_path: str | Path, selections: list[Mapping[str, Any]], rationale: str,
    views: CanonicalDatasetViews, output_path: str | Path,
) -> dict[str, Any]:
    load_quality_attribution_gate(quality_attribution_gate_path, views)
    plan = load_candidate_plan(candidate_plan_path, views, quality_attribution_gate_path)
    bakeoff = read_json_object(bakeoff_path, "architecture bakeoff")
    if bakeoff.get("scope") != "DEV" or bakeoff.get("is_full_dev") is not True or bakeoff.get("final_accessed") is not False:
        raise CanonicalDatasetError("shortlist requires a full DEV-only bakeoff that did not access FINAL")
    if bakeoff.get("candidate_plan_sha256") != sha256_file(candidate_plan_path):
        raise CanonicalDatasetError("bakeoff does not bind to the candidate plan")
    if len(selections) not in (1, 2):
        raise CanonicalDatasetError("architecture shortlist must contain one or two candidates")
    seen: set[tuple[str, int]] = set()
    chosen: list[dict[str, Any]] = []
    for item in selections:
        arch, seed = _candidate_key(item)
        if arch not in MODEL_NAMES or (arch, seed) in seen:
            raise CanonicalDatasetError("shortlist has an unsupported or duplicate architecture/seed")
        seen.add((arch, seed))
        try:
            run = next(r for r in bakeoff["architectures"][arch]["seed_runs"] if r.get("seed") == seed)
        except (KeyError, StopIteration, TypeError) as exc:
            raise CanonicalDatasetError("shortlisted architecture/seed is absent from the DEV bakeoff") from exc
        checkpoint = run.get("checkpoint_path")
        checkpoint_hash = run.get("checkpoint_sha256")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_file() or sha256_file(checkpoint) != checkpoint_hash:
            raise CanonicalDatasetError("shortlisted checkpoint is missing or changed")
        chosen.append({"architecture": arch, "seed": seed, "checkpoint_path": str(Path(checkpoint).resolve()), "checkpoint_sha256": checkpoint_hash, "architecture_config": run.get("architecture_config", {}), "dev_metrics": run})
    shortlist = {
        "schema": ARCHITECTURE_SHORTLIST_SCHEMA, "status": "complete", "created_at_utc": utc_now(),
        "dataset_sha256": views.dataset_sha256, "train_split_sha256": views.train_split_sha256, "dev_split_sha256": views.dev_split_sha256,
        "quality_attribution_gate_sha256": sha256_file(quality_attribution_gate_path),
        "candidate_plan_sha256": sha256_file(candidate_plan_path), "bakeoff_sha256": sha256_file(bakeoff_path),
        "bakeoff_path": str(Path(bakeoff_path).resolve()),
        "candidate_selection": plan["selection"], "candidates": chosen,
        "rationale": _require_nonempty_string(rationale, "shortlist rationale"),
    }
    write_json_exclusive(output_path, shortlist)
    return shortlist


def load_architecture_shortlist(path: str | Path, views: CanonicalDatasetViews, quality_attribution_gate_path: str | Path, candidate_plan_path: str | Path) -> dict[str, Any]:
    load_quality_attribution_gate(quality_attribution_gate_path, views)
    plan = load_candidate_plan(candidate_plan_path, views, quality_attribution_gate_path)
    shortlist = read_json_object(path, "architecture shortlist")
    if shortlist.get("schema") != ARCHITECTURE_SHORTLIST_SCHEMA or shortlist.get("status") != "complete":
        raise CanonicalDatasetError("DEV architecture shortlist is missing or incomplete")
    expected = {"dataset_sha256": views.dataset_sha256, "train_split_sha256": views.train_split_sha256, "dev_split_sha256": views.dev_split_sha256, "quality_attribution_gate_sha256": sha256_file(quality_attribution_gate_path), "candidate_plan_sha256": sha256_file(candidate_plan_path)}
    if any(shortlist.get(key) != value for key, value in expected.items()):
        raise CanonicalDatasetError("architecture shortlist provenance mismatch")
    if shortlist.get("candidate_selection") != plan.get("selection"):
        raise CanonicalDatasetError("architecture shortlist candidate selection differs from the candidate plan")
    bakeoff_path = shortlist.get("bakeoff_path")
    if not isinstance(bakeoff_path, str) or not Path(bakeoff_path).is_file() or sha256_file(bakeoff_path) != shortlist.get("bakeoff_sha256"):
        raise CanonicalDatasetError("architecture shortlist DEV bakeoff is missing or changed")
    bakeoff = read_json_object(bakeoff_path, "architecture bakeoff")
    if bakeoff.get("scope") != "DEV" or bakeoff.get("is_full_dev") is not True or bakeoff.get("final_accessed") is not False or bakeoff.get("candidate_plan_sha256") != sha256_file(candidate_plan_path):
        raise CanonicalDatasetError("architecture shortlist source bakeoff has invalid scope or provenance")
    candidates = shortlist.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 2:
        raise CanonicalDatasetError("architecture shortlist must contain one or two candidates")
    keys: set[tuple[str, int]] = set()
    for candidate in candidates:
        cp = candidate.get("checkpoint_path")
        if candidate.get("architecture") not in MODEL_NAMES or not isinstance(cp, str) or not Path(cp).is_file() or sha256_file(cp) != candidate.get("checkpoint_sha256"):
            raise CanonicalDatasetError("architecture shortlist contains a missing or changed checkpoint")
        key = _candidate_key(candidate)
        if key in keys:
            raise CanonicalDatasetError("architecture shortlist contains a duplicate architecture/seed")
        keys.add(key)
        try:
            run = next(item for item in bakeoff["architectures"][key[0]]["seed_runs"] if item.get("seed") == key[1])
        except (KeyError, StopIteration, TypeError) as exc:
            raise CanonicalDatasetError("architecture shortlist candidate is absent from its DEV bakeoff") from exc
        if run.get("checkpoint_sha256") != candidate.get("checkpoint_sha256") or run != candidate.get("dev_metrics"):
            raise CanonicalDatasetError("architecture shortlist candidate metrics/checkpoint differ from its DEV bakeoff")
    _require_nonempty_string(shortlist.get("rationale"), "shortlist rationale")
    return shortlist


def make_integration_subset_freeze(*, prompt_ids: list[str], views: CanonicalDatasetViews, output_path: str | Path) -> dict[str, Any]:
    dev_ids = {record.prompt_id for record in views.dev}
    if not prompt_ids or len(set(prompt_ids)) != len(prompt_ids) or not set(prompt_ids).issubset(dev_ids):
        raise CanonicalDatasetError("integration subset IDs must be unique non-empty DEV prompt IDs")
    subset = {"schema": INTEGRATION_SUBSET_SCHEMA, "status": "frozen", "created_at_utc": utc_now(), "dataset_sha256": views.dataset_sha256, "dev_split_sha256": views.dev_split_sha256, "prompt_ids": sorted(prompt_ids), "prompt_ids_sha256": sha256_json(sorted(prompt_ids))}
    write_json_exclusive(output_path, subset)
    return subset


def load_integration_subset(path: str | Path, views: CanonicalDatasetViews) -> dict[str, Any]:
    subset = read_json_object(path, "frozen integration subset")
    ids = subset.get("prompt_ids")
    if subset.get("schema") != INTEGRATION_SUBSET_SCHEMA or subset.get("status") != "frozen" or not isinstance(ids, list):
        raise CanonicalDatasetError("integration subset is missing or not frozen")
    if subset.get("dataset_sha256") != views.dataset_sha256 or subset.get("dev_split_sha256") != views.dev_split_sha256:
        raise CanonicalDatasetError("integration subset provenance mismatch")
    if not ids or len(set(ids)) != len(ids) or not set(ids).issubset({r.prompt_id for r in views.dev}) or subset.get("prompt_ids_sha256") != sha256_json(sorted(ids)):
        raise CanonicalDatasetError("integration subset contains invalid DEV prompt IDs")
    return subset


LIVE_REVIEW_CHECKS = ("h_emission", "continuation_state", "task_quality", "termination_health", "decode_step_savings")


def make_live_integration_gate(*, evidence_path: str | Path, review_path: str | Path, quality_attribution_gate_path: str | Path, candidate_plan_path: str | Path, shortlist_path: str | Path, subset_path: str | Path | None, views: CanonicalDatasetViews, output_path: str | Path) -> dict[str, Any]:
    load_quality_attribution_gate(quality_attribution_gate_path, views)
    plan = load_candidate_plan(candidate_plan_path, views, quality_attribution_gate_path)
    shortlist = load_architecture_shortlist(shortlist_path, views, quality_attribution_gate_path, candidate_plan_path)
    evidence = read_json_object(evidence_path, "live end-to-end integration evidence")
    review = read_json_object(review_path, "live integration review")
    if evidence.get("evaluation_mode") != "live_end_to_end" or evidence.get("dataset_sha256") != views.dataset_sha256 or evidence.get("dev_split_sha256") != views.dev_split_sha256:
        raise CanonicalDatasetError("live integration evidence must be end-to-end and bound to the canonical DEV split")
    expected_ids = sorted({r.prompt_id for r in views.dev}) if subset_path is None else load_integration_subset(subset_path, views)["prompt_ids"]
    if sorted(evidence.get("prompt_ids", [])) != expected_ids:
        raise CanonicalDatasetError("live integration prompt IDs do not match full DEV or the frozen integration subset")
    if subset_path is None and evidence.get("partition") != "DEV":
        raise CanonicalDatasetError("full DEV live integration evidence must declare partition DEV")
    if subset_path is not None and evidence.get("partition") != "FROZEN_INTEGRATION_SUBSET":
        raise CanonicalDatasetError("subset integration evidence must declare the frozen subset partition")
    rows = evidence.get("candidates")
    reviews = review.get("candidates")
    expected_keys = {_candidate_key(item) for item in shortlist["candidates"]}
    if not isinstance(rows, list) or len(rows) != len(expected_keys) or {_candidate_key(row) for row in rows} != expected_keys or not isinstance(reviews, list) or len(reviews) != len(expected_keys) or {_candidate_key(row) for row in reviews} != expected_keys:
        raise CanonicalDatasetError("live evidence and review must cover every shortlisted candidate exactly once")
    review_by_key = {_candidate_key(row): row for row in reviews}
    evidence_by_key = {_candidate_key(row): row for row in rows}
    for candidate in shortlist["candidates"]:
        key = _candidate_key(candidate)
        row = evidence_by_key[key]
        if row.get("checkpoint_sha256") != candidate["checkpoint_sha256"] or row.get("candidate_plan_sha256") != sha256_file(candidate_plan_path):
            raise CanonicalDatasetError("live integration candidate does not match the shortlisted checkpoint and candidate plan")
        result_path = row.get("results_path")
        if not isinstance(result_path, str) or not Path(result_path).is_file() or sha256_file(result_path) != row.get("results_sha256"):
            raise CanonicalDatasetError("live integration per-candidate result artifact is missing or changed")
        if row.get("prompt_count") != len(expected_ids) or not isinstance(row.get("domains"), list) or not row["domains"]:
            raise CanonicalDatasetError("live integration result has incomplete prompt or domain coverage")
        counts = ("h_emission_count", "continuation_failure_count", "eos_failure_count", "truncation_count", "repetition_count", "decode_steps", "vanilla_decode_steps")
        scores = ("task_quality", "matched_vanilla_task_quality")
        if any(not isinstance(row.get(field), int) or row[field] < 0 for field in counts) or any(not isinstance(row.get(field), (int, float)) or not math.isfinite(row[field]) for field in scores):
            raise CanonicalDatasetError("live integration result omits required emission, health, quality, or decode-step metrics")
        checks = review_by_key[key].get("checks")
        if not isinstance(checks, Mapping) or any(not isinstance(checks.get(name), Mapping) or checks[name].get("passed") is not True or not isinstance(checks[name].get("notes"), str) or not checks[name]["notes"].strip() for name in LIVE_REVIEW_CHECKS):
            raise CanonicalDatasetError(f"live integration review did not pass all checks for {key}")
    gate = {"schema": LIVE_INTEGRATION_GATE_SCHEMA, "status": "passed", "created_at_utc": utc_now(), "dataset_sha256": views.dataset_sha256, "dev_split_sha256": views.dev_split_sha256, "quality_attribution_gate_sha256": sha256_file(quality_attribution_gate_path), "candidate_plan_sha256": sha256_file(candidate_plan_path), "shortlist_sha256": sha256_file(shortlist_path), "integration_subset_sha256": sha256_file(subset_path) if subset_path else None, "prompt_ids_sha256": sha256_json(expected_ids), "evidence_path": str(Path(evidence_path).resolve()), "evidence_sha256": sha256_file(evidence_path), "review_path": str(Path(review_path).resolve()), "review_sha256": sha256_file(review_path), "candidate_keys": [list(key) for key in sorted(expected_keys)]}
    write_json_exclusive(output_path, gate)
    return gate


def load_live_integration_gate(path: str | Path, views: CanonicalDatasetViews, quality_attribution_gate_path: str | Path, candidate_plan_path: str | Path, shortlist_path: str | Path, subset_path: str | Path | None) -> dict[str, Any]:
    load_quality_attribution_gate(quality_attribution_gate_path, views)
    load_candidate_plan(candidate_plan_path, views, quality_attribution_gate_path)
    shortlist = load_architecture_shortlist(shortlist_path, views, quality_attribution_gate_path, candidate_plan_path)
    subset = load_integration_subset(subset_path, views) if subset_path else None
    gate = read_json_object(path, "live integration gate")
    expected = {"schema": LIVE_INTEGRATION_GATE_SCHEMA, "status": "passed", "dataset_sha256": views.dataset_sha256, "dev_split_sha256": views.dev_split_sha256, "quality_attribution_gate_sha256": sha256_file(quality_attribution_gate_path), "candidate_plan_sha256": sha256_file(candidate_plan_path), "shortlist_sha256": sha256_file(shortlist_path), "integration_subset_sha256": sha256_file(subset_path) if subset_path else None}
    if any(gate.get(key) != value for key, value in expected.items()):
        raise CanonicalDatasetError("live integration gate is absent, failed, or bound to different upstream evidence")
    source_paths: dict[str, str] = {}
    for path_field, hash_field in (("evidence_path", "evidence_sha256"), ("review_path", "review_sha256")):
        source = gate.get(path_field)
        if not isinstance(source, str) or not Path(source).is_file() or sha256_file(source) != gate.get(hash_field):
            raise CanonicalDatasetError(f"live integration {path_field} is missing or changed")
        source_paths[path_field] = source
    evidence = read_json_object(source_paths["evidence_path"], "live end-to-end integration evidence")
    review = read_json_object(source_paths["review_path"], "live integration review")
    expected_ids = sorted({record.prompt_id for record in views.dev}) if subset is None else subset["prompt_ids"]
    if evidence.get("evaluation_mode") != "live_end_to_end" or evidence.get("dataset_sha256") != views.dataset_sha256 or evidence.get("dev_split_sha256") != views.dev_split_sha256 or sorted(evidence.get("prompt_ids", [])) != expected_ids or gate.get("prompt_ids_sha256") != sha256_json(expected_ids):
        raise CanonicalDatasetError("live integration evidence prompt/split provenance is invalid")
    expected_partition = "DEV" if subset is None else "FROZEN_INTEGRATION_SUBSET"
    if evidence.get("partition") != expected_partition:
        raise CanonicalDatasetError("live integration evidence declares the wrong partition")
    expected_keys = {_candidate_key(item) for item in shortlist["candidates"]}
    rows, reviews = evidence.get("candidates"), review.get("candidates")
    key_list = [list(key) for key in sorted(expected_keys)]
    if gate.get("candidate_keys") != key_list or not isinstance(rows, list) or len(rows) != len(expected_keys) or {_candidate_key(row) for row in rows} != expected_keys or not isinstance(reviews, list) or len(reviews) != len(expected_keys) or {_candidate_key(row) for row in reviews} != expected_keys:
        raise CanonicalDatasetError("live integration gate does not cover the current shortlist exactly once")
    review_by_key = {_candidate_key(row): row for row in reviews}
    for row in rows:
        key = _candidate_key(row)
        candidate = next(item for item in shortlist["candidates"] if _candidate_key(item) == key)
        if row.get("checkpoint_sha256") != candidate.get("checkpoint_sha256") or row.get("candidate_plan_sha256") != sha256_file(candidate_plan_path) or row.get("prompt_count") != len(expected_ids) or not row.get("domains"):
            raise CanonicalDatasetError("live integration gate candidate checkpoint, plan, or coverage mismatch")
        result_source = row.get("results_path")
        if not isinstance(result_source, str) or not Path(result_source).is_file() or sha256_file(result_source) != row.get("results_sha256"):
            raise CanonicalDatasetError("live integration per-candidate results are missing or changed")
        counts = ("h_emission_count", "continuation_failure_count", "eos_failure_count", "truncation_count", "repetition_count", "decode_steps", "vanilla_decode_steps")
        scores = ("task_quality", "matched_vanilla_task_quality")
        if any(not isinstance(row.get(field), int) or row[field] < 0 for field in counts) or any(not isinstance(row.get(field), (int, float)) or not math.isfinite(row[field]) for field in scores):
            raise CanonicalDatasetError("live integration candidate lacks valid emission, health, quality, or decode metrics")
        checks = review_by_key[key].get("checks")
        if not isinstance(checks, Mapping) or any(not isinstance(checks.get(name), Mapping) or checks[name].get("passed") is not True or not isinstance(checks[name].get("notes"), str) or not checks[name]["notes"].strip() for name in LIVE_REVIEW_CHECKS):
            raise CanonicalDatasetError("live integration gate review no longer passes all required checks")
    return gate


def load_candidate_freeze(path: str | Path, views: CanonicalDatasetViews) -> dict[str, Any]:
    freeze = read_json_object(path, "candidate-generator freeze")
    if freeze.get("schema") != CANDIDATE_FREEZE_SCHEMA or freeze.get("status") != "frozen":
        raise CanonicalDatasetError("candidate-generator freeze is absent or not frozen")
    if freeze.get("dataset_sha256") != views.dataset_sha256:
        raise CanonicalDatasetError("candidate freeze dataset hash does not match canonical dataset")
    if freeze.get("dev_split_sha256") != views.dev_split_sha256:
        raise CanonicalDatasetError("candidate freeze DEV split hash does not match canonical dataset")
    if freeze.get("train_split_sha256") != views.train_split_sha256:
        raise CanonicalDatasetError("candidate freeze TRAIN split hash does not match canonical dataset")
    if not freeze.get("train_index_sha256") or not freeze.get("train_index_provenance_sha256"):
        raise CanonicalDatasetError("candidate freeze is missing the bound TRAIN index hashes")
    for field in ("quality_attribution_gate_sha256", "candidate_plan_sha256", "shortlist_sha256", "live_integration_gate_sha256"):
        if not freeze.get(field):
            raise CanonicalDatasetError(f"candidate freeze is missing required evidence hash: {field}")
    selection = freeze.get("selection")
    if not isinstance(selection, dict):
        raise CanonicalDatasetError("candidate freeze is missing selection")
    if selection.get("strategy") not in CANDIDATE_STRATEGIES:
        raise CanonicalDatasetError("candidate freeze uses an unknown retrieval strategy")
    if selection.get("pool_size") not in SUPPORTED_POOL_SIZES:
        raise CanonicalDatasetError("candidate freeze uses an unsupported pool size")
    if not isinstance(freeze.get("rationale"), str) or not freeze["rationale"].strip():
        raise CanonicalDatasetError("candidate freeze must record a rationale")
    if not freeze.get("benchmark_result_sha256"):
        raise CanonicalDatasetError("candidate freeze is missing benchmark evidence hash")
    return freeze


def load_architecture_freeze(
    path: str | Path,
    views: CanonicalDatasetViews,
    candidate_freeze_path: str | Path,
    *,
    quality_attribution_gate_path: str | Path,
    candidate_plan_path: str | Path,
    shortlist_path: str | Path,
    live_integration_gate_path: str | Path,
    integration_subset_path: str | Path | None = None,
) -> dict[str, Any]:
    candidate = load_candidate_freeze(candidate_freeze_path, views)
    load_quality_attribution_gate(quality_attribution_gate_path, views)
    plan = load_candidate_plan(candidate_plan_path, views, quality_attribution_gate_path)
    shortlist = load_architecture_shortlist(shortlist_path, views, quality_attribution_gate_path, candidate_plan_path)
    load_live_integration_gate(live_integration_gate_path, views, quality_attribution_gate_path, candidate_plan_path, shortlist_path, integration_subset_path)
    candidate_links = {
        "quality_attribution_gate_sha256": quality_attribution_gate_path,
        "candidate_plan_sha256": candidate_plan_path,
        "shortlist_sha256": shortlist_path,
        "live_integration_gate_sha256": live_integration_gate_path,
    }
    for field, source_path in candidate_links.items():
        if candidate.get(field) != sha256_file(source_path):
            raise CanonicalDatasetError(f"candidate freeze does not bind to {field}")
    if candidate.get("integration_subset_sha256") != (sha256_file(integration_subset_path) if integration_subset_path else None):
        raise CanonicalDatasetError("candidate freeze integration-subset binding mismatch")
    if candidate.get("selection") != plan.get("selection"):
        raise CanonicalDatasetError("candidate freeze selection differs from the current candidate plan")
    freeze = read_json_object(path, "architecture freeze")
    if freeze.get("schema") != ARCHITECTURE_FREEZE_SCHEMA or freeze.get("status") != "frozen":
        raise CanonicalDatasetError("architecture freeze is absent or not frozen")
    if freeze.get("dataset_sha256") != views.dataset_sha256:
        raise CanonicalDatasetError("architecture freeze dataset hash does not match canonical dataset")
    if freeze.get("train_split_sha256") != views.train_split_sha256:
        raise CanonicalDatasetError("architecture freeze TRAIN split hash does not match canonical dataset")
    if freeze.get("dev_split_sha256") != views.dev_split_sha256:
        raise CanonicalDatasetError("architecture freeze DEV split hash does not match canonical dataset")
    if freeze.get("candidate_freeze_sha256") != sha256_file(candidate_freeze_path):
        raise CanonicalDatasetError("architecture freeze does not bind to the loaded candidate freeze")
    bound_paths = {
        "quality_attribution_gate_sha256": quality_attribution_gate_path,
        "candidate_plan_sha256": candidate_plan_path,
        "shortlist_sha256": shortlist_path,
        "live_integration_gate_sha256": live_integration_gate_path,
    }
    for field, source_path in bound_paths.items():
        if freeze.get(field) != sha256_file(source_path):
            raise CanonicalDatasetError(f"architecture freeze does not bind to {field}")
    if freeze.get("integration_subset_sha256") != (sha256_file(integration_subset_path) if integration_subset_path else None):
        raise CanonicalDatasetError("architecture freeze integration-subset binding mismatch")
    if freeze.get("bakeoff_result_sha256") != shortlist.get("bakeoff_sha256"):
        raise CanonicalDatasetError("architecture freeze does not bind to the shortlisted DEV bakeoff")
    selection = freeze.get("selection")
    if not isinstance(selection, dict) or selection.get("architecture") not in MODEL_NAMES:
        raise CanonicalDatasetError("architecture freeze has an unknown model selection")
    if not isinstance(selection.get("architecture_config"), Mapping):
        raise CanonicalDatasetError("architecture freeze is missing the frozen model configuration")
    if not isinstance(freeze.get("rationale"), str) or not freeze["rationale"].strip():
        raise CanonicalDatasetError("architecture freeze must record a rationale")
    _validate_architecture_review(freeze.get("review"))
    if not freeze.get("bakeoff_result_sha256"):
        raise CanonicalDatasetError("architecture freeze is missing bakeoff evidence hash")
    checkpoint = selection.get("checkpoint_path")
    checkpoint_hash = selection.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or not isinstance(checkpoint_hash, str):
        raise CanonicalDatasetError("architecture freeze must name a checkpoint and its hash")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint_hash:
        raise CanonicalDatasetError("frozen architecture checkpoint is missing or has changed")
    if selection.get("candidate_config") != candidate.get("selection"):
        raise CanonicalDatasetError("architecture freeze candidate config differs from candidate freeze")
    if not any(
        entry.get("architecture") == selection.get("architecture")
        and entry.get("seed") == selection.get("seed")
        and entry.get("checkpoint_sha256") == selection.get("checkpoint_sha256")
        for entry in shortlist["candidates"]
    ):
        raise CanonicalDatasetError("architecture freeze selection was not included in the live-tested DEV shortlist")
    return freeze


def issue_final_access_permit(
    *,
    allow_final_eval: bool,
    quality_attribution_gate_path: str | Path,
    candidate_plan_path: str | Path,
    candidate_freeze_path: str | Path,
    shortlist_path: str | Path,
    live_integration_gate_path: str | Path,
    architecture_freeze_path: str | Path,
    integration_subset_path: str | Path | None = None,
    views: CanonicalDatasetViews,
) -> FinalAccessPermit:
    if not allow_final_eval:
        raise FinalAccessError("pass --allow-final-eval to open FINAL")
    load_quality_attribution_gate(quality_attribution_gate_path, views)
    load_candidate_plan(candidate_plan_path, views, quality_attribution_gate_path)
    load_candidate_freeze(candidate_freeze_path, views)
    load_architecture_shortlist(shortlist_path, views, quality_attribution_gate_path, candidate_plan_path)
    load_live_integration_gate(live_integration_gate_path, views, quality_attribution_gate_path, candidate_plan_path, shortlist_path, integration_subset_path)
    load_architecture_freeze(
        architecture_freeze_path, views, candidate_freeze_path,
        quality_attribution_gate_path=quality_attribution_gate_path,
        candidate_plan_path=candidate_plan_path,
        shortlist_path=shortlist_path,
        live_integration_gate_path=live_integration_gate_path,
        integration_subset_path=integration_subset_path,
    )
    return FinalAccessPermit(
        dataset_sha256=views.dataset_sha256,
        quality_attribution_gate_sha256=sha256_file(quality_attribution_gate_path),
        candidate_plan_sha256=sha256_file(candidate_plan_path),
        candidate_freeze_sha256=sha256_file(candidate_freeze_path),
        architecture_shortlist_sha256=sha256_file(shortlist_path),
        live_integration_gate_sha256=sha256_file(live_integration_gate_path),
        architecture_freeze_sha256=sha256_file(architecture_freeze_path),
        allow_final_eval=True,
    )


def claim_final_evaluation(
    *,
    claim_path: str | Path,
    result_path: str | Path,
    permit: FinalAccessPermit,
) -> dict[str, Any]:
    """Permanently claim the single FINAL attempt before the caller opens FINAL."""
    claim = {
        "schema": "predictor_v2_final_eval_claim_v1",
        "status": "claimed",
        "claimed_at_utc": utc_now(),
        "dataset_sha256": permit.dataset_sha256,
        "quality_attribution_gate_sha256": permit.quality_attribution_gate_sha256,
        "candidate_plan_sha256": permit.candidate_plan_sha256,
        "candidate_freeze_sha256": permit.candidate_freeze_sha256,
        "architecture_shortlist_sha256": permit.architecture_shortlist_sha256,
        "live_integration_gate_sha256": permit.live_integration_gate_sha256,
        "architecture_freeze_sha256": permit.architecture_freeze_sha256,
        "result_path": str(result_path),
    }
    write_json_exclusive(claim_path, claim)
    return claim


def make_candidate_freeze(
    *,
    candidate_plan_path: str | Path,
    quality_attribution_gate_path: str | Path,
    shortlist_path: str | Path,
    live_integration_gate_path: str | Path,
    integration_subset_path: str | Path | None = None,
    rationale: str,
    views: CanonicalDatasetViews,
    output_path: str | Path,
    cwd: str | Path = ".",
) -> dict[str, Any]:
    if not rationale.strip():
        raise CanonicalDatasetError("a selection rationale is required")
    gate = load_quality_attribution_gate(quality_attribution_gate_path, views)
    plan = load_candidate_plan(candidate_plan_path, views, quality_attribution_gate_path)
    shortlist = load_architecture_shortlist(shortlist_path, views, quality_attribution_gate_path, candidate_plan_path)
    live_gate = load_live_integration_gate(live_integration_gate_path, views, quality_attribution_gate_path, candidate_plan_path, shortlist_path, integration_subset_path)
    freeze = {
        "schema": CANDIDATE_FREEZE_SCHEMA,
        "status": "frozen",
        "created_at_utc": utc_now(),
        "dataset_sha256": views.dataset_sha256,
        "train_split_sha256": views.train_split_sha256,
        "dev_split_sha256": views.dev_split_sha256,
        "quality_attribution_gate_sha256": sha256_file(quality_attribution_gate_path),
        "candidate_plan_sha256": sha256_file(candidate_plan_path),
        "shortlist_sha256": sha256_file(shortlist_path),
        "live_integration_gate_sha256": sha256_file(live_integration_gate_path),
        "integration_subset_sha256": sha256_file(integration_subset_path) if integration_subset_path else None,
        "train_index_sha256": plan["train_index_sha256"],
        "train_index_provenance_sha256": plan["train_index_provenance_sha256"],
        "benchmark_result_sha256": plan["benchmark_sha256"],
        "selection": plan["selection"],
        "dev_metrics": plan["dev_metrics"],
        "live_integration_gate_status": live_gate["status"],
        "live_tested_candidates": live_gate["candidate_keys"],
        "quality_attribution_status": gate["status"],
        "rationale": rationale.strip(),
        "code_commit": current_commit(cwd),
    }
    write_json_exclusive(output_path, freeze)
    return freeze


def make_architecture_freeze(
    *,
    bakeoff_path: str | Path,
    architecture: str,
    seed: int,
    checkpoint_path: str | Path,
    candidate_freeze_path: str | Path,
    quality_attribution_gate_path: str | Path,
    candidate_plan_path: str | Path,
    shortlist_path: str | Path,
    live_integration_gate_path: str | Path,
    integration_subset_path: str | Path | None = None,
    review: Mapping[str, Any],
    rationale: str,
    views: CanonicalDatasetViews,
    output_path: str | Path,
    cwd: str | Path = ".",
) -> dict[str, Any]:
    if architecture not in MODEL_NAMES:
        raise CanonicalDatasetError(f"unknown architecture: {architecture}")
    if not rationale.strip():
        raise CanonicalDatasetError("a selection rationale is required")
    candidate = load_candidate_freeze(candidate_freeze_path, views)
    load_quality_attribution_gate(quality_attribution_gate_path, views)
    plan = load_candidate_plan(candidate_plan_path, views, quality_attribution_gate_path)
    shortlist = load_architecture_shortlist(shortlist_path, views, quality_attribution_gate_path, candidate_plan_path)
    live_gate = load_live_integration_gate(live_integration_gate_path, views, quality_attribution_gate_path, candidate_plan_path, shortlist_path, integration_subset_path)
    if candidate.get("candidate_plan_sha256") != sha256_file(candidate_plan_path):
        raise CanonicalDatasetError("candidate freeze is not bound to the current candidate plan")
    for field, source_path in (
        ("quality_attribution_gate_sha256", quality_attribution_gate_path),
        ("shortlist_sha256", shortlist_path),
        ("live_integration_gate_sha256", live_integration_gate_path),
    ):
        if candidate.get(field) != sha256_file(source_path):
            raise CanonicalDatasetError(f"candidate freeze does not bind to {field}")
    results = read_json_object(bakeoff_path, "architecture bakeoff")
    if results.get("scope") != "DEV" or results.get("is_full_dev") is not True:
        raise CanonicalDatasetError("architecture freeze requires DEV-only bakeoff evidence")
    if (
        results.get("dataset_sha256") != views.dataset_sha256
        or results.get("train_split_sha256") != views.train_split_sha256
        or results.get("dev_split_sha256") != views.dev_split_sha256
    ):
        raise CanonicalDatasetError("architecture bakeoff dataset/split hash mismatch")
    if results.get("candidate_plan_sha256") != sha256_file(candidate_plan_path):
        raise CanonicalDatasetError("architecture bakeoff does not bind to the candidate plan")
    if shortlist.get("bakeoff_sha256") != sha256_file(bakeoff_path):
        raise CanonicalDatasetError("architecture shortlist does not bind to the current DEV bakeoff")
    try:
        summary = results["architectures"][architecture]
        run = next(item for item in summary["seed_runs"] if item["seed"] == seed)
    except (KeyError, StopIteration, TypeError) as exc:
        raise CanonicalDatasetError("selected architecture/seed has no DEV metrics") from exc
    if architecture != "Ridge":
        required_seeds = {42, 43, 44}
        recorded_seeds = {item.get("seed") for item in summary.get("seed_runs", [])}
        if not required_seeds.issubset(recorded_seeds):
            raise CanonicalDatasetError("neural architecture freeze requires DEV results for seeds 42, 43, and 44")
    if not isinstance(run.get("inference_latency_ms"), Mapping):
        raise CanonicalDatasetError("selected architecture run is missing DEV inference latency")
    if not isinstance(run.get("dev_k16_by_domain"), Mapping) or not run["dev_k16_by_domain"]:
        raise CanonicalDatasetError("selected architecture run is missing per-domain DEV metrics")
    if not any(item.get("architecture") == architecture and item.get("seed") == seed and item.get("checkpoint_sha256") == run.get("checkpoint_sha256") for item in shortlist["candidates"]):
        raise CanonicalDatasetError("architecture selection was not included in the live-tested DEV shortlist")
    if [architecture, seed] not in live_gate.get("candidate_keys", []):
        raise CanonicalDatasetError("architecture selection did not pass the live integration gate")
    if plan["selection"] != candidate["selection"]:
        raise CanonicalDatasetError("candidate freeze selection differs from its proposed candidate plan")
    _validate_architecture_review(review)
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise CanonicalDatasetError(f"checkpoint not found: {checkpoint}")
    if run.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise CanonicalDatasetError("selected checkpoint does not match the DEV bakeoff run")
    recorded_path = run.get("checkpoint_path")
    if recorded_path and Path(recorded_path).resolve() != checkpoint.resolve():
        raise CanonicalDatasetError("selected checkpoint path differs from the DEV bakeoff run")
    freeze = {
        "schema": ARCHITECTURE_FREEZE_SCHEMA,
        "status": "frozen",
        "created_at_utc": utc_now(),
        "dataset_sha256": views.dataset_sha256,
        "train_split_sha256": views.train_split_sha256,
        "dev_split_sha256": views.dev_split_sha256,
        "quality_attribution_gate_sha256": sha256_file(quality_attribution_gate_path),
        "candidate_plan_sha256": sha256_file(candidate_plan_path),
        "candidate_freeze_sha256": sha256_file(candidate_freeze_path),
        "shortlist_sha256": sha256_file(shortlist_path),
        "live_integration_gate_sha256": sha256_file(live_integration_gate_path),
        "integration_subset_sha256": sha256_file(integration_subset_path) if integration_subset_path else None,
        "bakeoff_result_sha256": sha256_file(bakeoff_path),
        "selection": {
            "architecture": architecture,
            "seed": seed,
            "candidate_config": candidate["selection"],
            "architecture_config": run.get("architecture_config", {}),
            "training_config": results.get("training_config", {}),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "dev_metrics": run,
        },
        "rationale": rationale.strip(),
        "review": dict(review),
        "code_commit": current_commit(cwd),
    }
    write_json_exclusive(output_path, freeze)
    return freeze


def _validate_architecture_review(review: Any) -> None:
    required = (
        "multiseed_robustness",
        "latency_quality_pareto",
        "domain_regression",
        "ridge_baseline",
    )
    if not isinstance(review, Mapping):
        raise CanonicalDatasetError("architecture freeze requires an explicit review object")
    for item in required:
        value = review.get(item)
        if not isinstance(value, Mapping) or value.get("passed") is not True:
            raise CanonicalDatasetError(f"architecture review gate {item} has not passed")
        if not isinstance(value.get("notes"), str) or not value["notes"].strip():
            raise CanonicalDatasetError(f"architecture review gate {item} requires evidence notes")
