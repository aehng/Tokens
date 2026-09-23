"""Kaggle entrypoint: smoke first, then bounded timing and validation probes."""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("HF_HOME", "/kaggle/temp/tokens-hf-cache")
os.environ.setdefault("PIP_CACHE_DIR", "/kaggle/temp/tokens-pip-cache")

DATASET_ROOT = Path("/kaggle/input/tokens-step100-gpu-smoke")
OUTPUT_ROOT = Path("/kaggle/working/tokens-kaggle-output")
REPO_ROOT = Path("/kaggle/working/tokens-source")
PACKAGE_REPO = "https://github.com/aehng/Tokens.git"
PACKAGE_BRANCH = "codex/kaggle-gpu-enablement"
EXPECTED_CONDITIONS = ["original_phi", "predictive_step_100_compressed_prompt"]
MAJOR_TEXT_DIVERGENCE_THRESHOLD = 0.55


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
                log_path: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "COMMAND: " + " ".join(command) + "\n\nSTDOUT\n" + result.stdout
            + "\nSTDERR\n" + result.stderr,
            encoding="utf-8",
        )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout tail:\n{result.stdout[-5000:]}\nstderr tail:\n{result.stderr[-5000:]}"
        )
    return result


class NvidiaSampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi", "-i", "0", "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or f"nvidia-smi exit {result.returncode}")
                fields = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
                self.samples.append(
                    {
                        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                        "gpu_name": fields[0],
                        "driver_version": fields[1],
                        "utilization_pct": int(fields[2].replace("%", "")),
                        "memory_used_mib": int(fields[3].replace("MiB", "")),
                        "memory_total_mib": int(fields[4].replace("MiB", "")),
                    }
                )
            except Exception as error:  # retain telemetry errors for the report
                self.errors.append(f"{type(error).__name__}: {error}")
            self.stop_event.wait(0.5)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=15)


def python_environment(repo_root: Path) -> dict[str, Any]:
    code = r'''
import importlib.metadata as m, json, platform, torch, transformers
from pathlib import Path
packages = {}
for name in ("accelerate", "peft", "omegaconf", "datasets", "zip2zip-compression", "sentencepiece"):
    try: packages[name] = m.version(name)
    except m.PackageNotFoundError: packages[name] = None
device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
gpus = []
for index in range(device_count):
    props = torch.cuda.get_device_properties(index)
    free, total = torch.cuda.mem_get_info(index)
    gpus.append({"index": index, "name": torch.cuda.get_device_name(index), "memory_bytes": int(props.total_memory),
                 "idle_allocated_bytes": int(torch.cuda.memory_allocated(index)),
                 "idle_reserved_bytes": int(torch.cuda.memory_reserved(index)),
                 "idle_free_bytes": int(free), "idle_total_bytes": int(total),
                 "capability": list(torch.cuda.get_device_capability(index))})
print(json.dumps({"python": platform.python_version(), "torch": str(torch.__version__),
                  "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available(), "visible_gpu_count": device_count,
                  "gpus": gpus, "packages": packages}))
'''
    output = run_command([sys.executable, "-c", code], cwd=repo_root).stdout.strip()
    return json.loads(output)


def clone_source(repo_commit: str) -> None:
    if REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)
    run_command(
        ["git", "clone", "--depth", "1", "--branch", PACKAGE_BRANCH, PACKAGE_REPO, str(REPO_ROOT)],
        log_path=OUTPUT_ROOT / "logs" / "clone.log",
    )
    current = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).stdout.strip()
    if current != repo_commit:
        raise RuntimeError(f"Source branch HEAD {current} differs from artifact-pinned commit {repo_commit}")
    run_command(["git", "checkout", "--detach", repo_commit], cwd=REPO_ROOT)


def verify_artifacts() -> tuple[dict[str, Any], Path, Path]:
    artifact_manifest = json.loads((DATASET_ROOT / "artifact_manifest.json").read_text(encoding="utf-8"))
    if artifact_manifest.get("dataset_visibility") != "private" or not artifact_manifest.get("validation_split_only"):
        raise RuntimeError("The Kaggle artifact manifest must describe a private validation-only dataset")
    expected_files = artifact_manifest.get("files", {})
    for name, expected in expected_files.items():
        path = DATASET_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Required Kaggle input artifact is missing: {path}")
        if path.stat().st_size != int(expected["size_bytes"]) or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Kaggle input artifact size/hash mismatch: {name}")

    checkpoint_path = DATASET_ROOT / "checkpoint_step_100.pt"
    predictor_dataset_path = DATASET_ROOT / "oracle_guided_predictor.pkl"
    clone_source(artifact_manifest["repo_commit"])
    source_predictor = REPO_ROOT / "experiments/checkpoints/oracle_guided_predictor.pkl"
    if sha256(source_predictor) != sha256(predictor_dataset_path):
        raise RuntimeError("Private predictor artifact differs from the predictor pinned in the source commit")
    return artifact_manifest, checkpoint_path, predictor_dataset_path


def install_dependencies() -> None:
    torch_build = json.loads(
        run_command(
            [
                sys.executable,
                "-c",
                "import json, torch; print(json.dumps({'version': str(torch.__version__), 'cuda': torch.version.cuda}))",
            ],
        ).stdout.strip()
    )
    constraint_path = OUTPUT_ROOT / "logs" / "torch-build-constraints.txt"
    constraint_path.parent.mkdir(parents=True, exist_ok=True)
    constraint_path.write_text(f"torch=={torch_build['version']}\n", encoding="utf-8")
    run_command(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(REPO_ROOT)],
        cwd=REPO_ROOT,
        log_path=OUTPUT_ROOT / "logs" / "install_project.log",
    )
    run_command(
        [sys.executable, "-m", "pip", "install", "-c", str(constraint_path), "-r",
         str(REPO_ROOT / "experiments/kaggle/requirements_kaggle.txt")],
        cwd=REPO_ROOT,
        log_path=OUTPUT_ROOT / "logs" / "install_requirements.log",
    )
    torch_after = json.loads(
        run_command(
            [
                sys.executable,
                "-c",
                "import json, torch; print(json.dumps({'version': str(torch.__version__), 'cuda': torch.version.cuda}))",
            ],
        ).stdout.strip()
    )
    if torch_after != torch_build:
        raise RuntimeError(f"Dependency installation changed Kaggle's preinstalled Torch build: {torch_build} -> {torch_after}")


def run_tier1(output_dir: Path, artifact_manifest: dict[str, Any], checkpoint_path: Path,
              *, conditions: list[str]) -> None:
    prompt_ids = REPO_ROOT / "experiments/kaggle/smoke_prompt_ids.json"
    command = [
        sys.executable,
        "experiments/run_phi_tier1.py",
        "--validation-data", "data/cached_pure_pred_val_60.json",
        "--prompt-ids-file", str(prompt_ids),
        "--checkpoint", str(checkpoint_path),
        "--predictor", "experiments/checkpoints/oracle_guided_predictor.pkl",
        "--conditions", *conditions,
        "--device", "cuda:0",
        "--base-revision", artifact_manifest["phi_revision"],
        "--zip2zip-revision", artifact_manifest["zip2zip_revision"],
        "--tested-commit", artifact_manifest["repo_commit"],
        "--output-dir", str(output_dir),
        "--max-new-tokens", "300",
        "--prompts-per-domain", "1",
    ]
    run_command(command, cwd=REPO_ROOT, log_path=OUTPUT_ROOT / "logs" / f"{output_dir.name}.log")


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def normalized_similarity(left: str, right: str) -> float:
    norm = lambda value: " ".join(value.casefold().split())
    return difflib.SequenceMatcher(None, norm(left), norm(right), autojunk=False).ratio()


def outcome_for(record: dict[str, Any]) -> bool | None:
    for name in ("problem_pass", "exact_correct", "mechanical_instruction_pass"):
        if isinstance(record.get(name), bool):
            return record[name]
    return None


def cpu_comparison(records: list[dict[str, Any]], cpu_refs: dict[str, Any]) -> dict[str, Any]:
    references = {
        (record["prompt_id"], record["condition"]): record
        for record in cpu_refs["records"]
    }
    paired = []
    stop_reasons = []
    for record in records:
        key = (record["prompt_id"], record["condition"])
        reference = references.get(key)
        if reference is None:
            stop_reasons.append(f"CPU reference missing for {key}")
            continue
        similarity = normalized_similarity(reference.get("output_text", ""), record.get("output_text", ""))
        cpu_outcome, gpu_outcome = outcome_for(reference), outcome_for(record)
        outcome_mismatch = cpu_outcome is not None and gpu_outcome is not None and cpu_outcome != gpu_outcome
        row = {
            "prompt_id": record["prompt_id"],
            "condition": record["condition"],
            "cpu_record_schema": reference.get("record_schema"),
            "gpu_record_schema": record.get("record_schema"),
            "cpu_wall_time_s": reference.get("wall_time_s"),
            "gpu_wall_time_s": record.get("wall_time_s"),
            "cpu_decode_steps": reference.get("decode_steps"),
            "gpu_decode_steps": record.get("decode_steps"),
            "cpu_eos": reference.get("eos_reached"),
            "gpu_eos": record.get("eos_reached"),
            "normalized_text_similarity": similarity,
            "cpu_quality_outcome": cpu_outcome,
            "gpu_quality_outcome": gpu_outcome,
            "quality_outcome_mismatch": outcome_mismatch,
        }
        paired.append(row)
        if similarity < MAJOR_TEXT_DIVERGENCE_THRESHOLD:
            stop_reasons.append(
                f"major CPU/GPU text divergence for {key}: normalized similarity {similarity:.3f}"
            )
        if outcome_mismatch:
            stop_reasons.append(f"CPU/GPU task outcome changed for {key}: {cpu_outcome} -> {gpu_outcome}")

    timing = {}
    for condition in EXPECTED_CONDITIONS:
        cpu_rows = [row for row in cpu_refs["records"] if row["condition"] == condition]
        gpu_rows = [row for row in records if row["condition"] == condition]
        cpu_mean = statistics.mean(float(row["wall_time_s"]) for row in cpu_rows)
        gpu_mean = statistics.mean(float(row["wall_time_s"]) for row in gpu_rows)
        timing[condition] = {"cpu_mean_wall_time_s": cpu_mean, "gpu_mean_wall_time_s": gpu_mean,
                             "gpu_to_cpu_wall_ratio": gpu_mean / max(cpu_mean, 1e-9)}
        if gpu_mean >= cpu_mean:
            stop_reasons.append(
                f"GPU mean request wall time is not faster than paired CPU reference for {condition}: "
                f"{gpu_mean:.3f}s >= {cpu_mean:.3f}s"
            )
    return {"paired_rows": paired, "timing": timing, "stop_reasons": stop_reasons}


def validate_smoke(run_dir: Path, artifact_manifest: dict[str, Any], gpu_info: dict[str, Any],
                   sampler: NvidiaSampler) -> dict[str, Any]:
    raw_records = load_records(run_dir / "raw_results.jsonl")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    identity = manifest.get("identity", {})
    records_by_key = {(row.get("prompt_id"), row.get("condition")): row for row in raw_records}
    expected_ids = artifact_manifest["prompt_ids"]
    expected_keys = {(prompt_id, condition) for prompt_id in expected_ids for condition in EXPECTED_CONDITIONS}
    reasons = []
    if manifest.get("status") != "complete" or set(records_by_key) != expected_keys:
        reasons.append("Smoke run did not produce exactly six complete paired records")
    if identity.get("tested_commit") != artifact_manifest["repo_commit"]:
        reasons.append("Smoke manifest tested commit does not match the private artifact manifest")
    if identity.get("prompt_ids") != expected_ids:
        reasons.append("Smoke run prompt IDs or order differ from the fixed 3-prompt selection")
    predictive_config = identity.get("conditions", {}).get("predictive_step_100_compressed_prompt", {})
    if predictive_config.get("prompt_representation") != "predictive_codebook_dp_segmented":
        reasons.append("Predictive condition did not declare canonical compressed_prompt representation")
    if predictive_config.get("emission_gate_top_n") is not None:
        reasons.append("Smoke condition did not use the committed Phase-5 no-gate setting")
    if predictive_config.get("checkpoint_sha256") != artifact_manifest["checkpoint"]["sha256"]:
        reasons.append("Smoke condition checkpoint SHA256 does not match the private artifact manifest")
    if predictive_config.get("predictor_sha256") != artifact_manifest["predictor"]["sha256"]:
        reasons.append("Smoke condition predictor SHA256 does not match the private artifact manifest")
    predictor_report = manifest.get("runtime_resources", {}).get("predictor_report", {})
    policy = predictor_report.get("policy", {})
    expected_policy = {
        "kind": "capped_predictor",
        "budget": 32,
        "max_structural_slots": 0,
        "allow_numeric": True,
        "max_numeric_slots": None,
        "filter_bare_punctuation": True,
        "max_subtokens": 4,
    }
    if policy != expected_policy:
        reasons.append(f"Loaded predictor policy differs from the committed Step-100 policy: {policy}")

    load_report = manifest.get("checkpoint_load_reports", {}).get("100", {})
    if load_report.get("checkpoint_loader") != "joint_nested_v2":
        reasons.append("The centralized verified nested checkpoint loader was not used")
    if load_report.get("base_hash_status") != "verified":
        reasons.append("Frozen backbone hashes were not verified against checkpoint metadata")
    for component in ("lora", "input_encoder", "output_encoder"):
        component_report = load_report.get("components", {}).get(component, {})
        if int(component_report.get("changed_tensor_count", 0)) <= 0:
            reasons.append(f"No trained tensors were activated for {component}")
        if component_report.get("missing_keys") or component_report.get("unexpected_keys"):
            reasons.append(f"Checkpoint tensor keys did not match the model for {component}")

    if gpu_info.get("visible_gpu_count") != 1 or not gpu_info.get("cuda_available"):
        reasons.append("Runtime does not expose exactly one CUDA device")
    if not sampler.samples:
        reasons.append("nvidia-smi collected no utilization samples")
    elif max(sample["utilization_pct"] for sample in sampler.samples) <= 0:
        reasons.append("No non-zero CUDA utilization sample was observed on GPU 0")

    for key in sorted(expected_keys):
        row = records_by_key.get(key)
        if not row:
            continue
        if not row.get("output_text") or int(row.get("decode_steps", 0)) <= 0:
            reasons.append(f"Generation did not produce output for {key}")
        if row.get("record_schema") != "phi_generation_record_v3":
            reasons.append(f"Record {key} is not v3")
        if row.get("cuda_memory_model_loaded") is None or row.get("cuda_memory_generation") is None:
            reasons.append(f"CUDA VRAM measurements are missing for {key}")
        else:
            if row["cuda_memory_generation"]["peak_allocated_bytes"] <= 0:
                reasons.append(f"CUDA inference peak allocation was not recorded for {key}")
        placement = row.get("device_placement", {})
        if placement.get("all_parameter_devices") != ["cuda:0"]:
            reasons.append(f"Model parameters are not all on cuda:0 for {key}")
        if row.get("condition") == "predictive_step_100_compressed_prompt" and row.get("prompt_representation") != "predictive_codebook_dp_segmented":
            reasons.append(f"Predictive record {key} did not use the canonical prompt representation")
        generation = identity.get("generation", {})
        if generation.get("do_sample") is not False:
            reasons.append("Greedy generation was not configured")

    cpu_refs = json.loads((DATASET_ROOT / "cpu_reference_selected.json").read_text(encoding="utf-8"))
    comparison = cpu_comparison(raw_records, cpu_refs)
    reasons.extend(comparison["stop_reasons"])
    result = {
        "status": "pass" if not reasons else "stop",
        "stop_reasons": reasons,
        "device": "cuda:0",
        "gpu_name": gpu_info["gpus"][0]["name"] if gpu_info.get("gpus") else None,
        "visible_gpu_count": gpu_info.get("visible_gpu_count"),
        "checkpoint_load_report": load_report,
        "predictor_sha256": predictive_config.get("predictor_sha256"),
        "checkpoint_sha256": predictive_config.get("checkpoint_sha256"),
        "prompt_representation": predictive_config.get("prompt_representation"),
        "emission_gate_top_n": predictive_config.get("emission_gate_top_n"),
        "paired_cpu_comparison": comparison,
        "nvidia_smi_utilization": {
            "sample_count": len(sampler.samples),
            "max_gpu_utilization_pct": max((sample["utilization_pct"] for sample in sampler.samples), default=None),
            "max_memory_used_mib": max((sample["memory_used_mib"] for sample in sampler.samples), default=None),
            "samples": sampler.samples,
            "errors": sampler.errors,
        },
        "records": raw_records,
        "run_manifest": manifest,
    }
    return result


def timing_summaries(run_dirs: list[Path], cpu_refs: dict[str, Any]) -> dict[str, Any]:
    all_records = []
    for run_dir in run_dirs:
        all_records.extend(load_records(run_dir / "raw_results.jsonl"))
    summaries = {}
    for condition in EXPECTED_CONDITIONS:
        rows = [row for row in all_records if row.get("condition") == condition]
        cpu_rows = [row for row in cpu_refs["records"] if row["condition"] == condition]
        if not rows:
            raise RuntimeError(f"Timing repeats have no results for {condition}")
        steps_per_second = [row["decode_steps"] / max(row["generation_wall_time_s"], 1e-9) for row in rows]
        expanded_per_second = [row["expanded_output_tokens"] / max(row["wall_time_s"], 1e-9) for row in rows]
        seconds_per_step = [row["generation_wall_time_s"] / max(row["decode_steps"], 1) for row in rows]
        cpu_mean = statistics.mean(float(row["wall_time_s"]) for row in cpu_rows)
        gpu_mean = statistics.mean(float(row["wall_time_s"]) for row in rows)
        summaries[condition] = {
            "n": len(rows),
            "mean_gpu_request_wall_time_s": statistics.mean(float(row["wall_time_s"]) for row in rows),
            "median_gpu_request_wall_time_s": statistics.median(float(row["wall_time_s"]) for row in rows),
            "mean_generation_wall_time_s": statistics.mean(float(row["generation_wall_time_s"]) for row in rows),
            "mean_decode_steps": statistics.mean(float(row["decode_steps"]) for row in rows),
            "mean_expanded_output_tokens": statistics.mean(float(row["expanded_output_tokens"]) for row in rows),
            "mean_transformer_steps_per_second": statistics.mean(steps_per_second),
            "mean_seconds_per_transformer_step": statistics.mean(seconds_per_step),
            "mean_effective_expanded_tokens_per_second": statistics.mean(expanded_per_second),
            "mean_predictor_setup_s": statistics.mean(float(row.get("predictor_time_s", 0)) + float(row.get("hyper_setup_time_s", 0)) for row in rows),
            "gpu_to_cpu_mean_request_wall_ratio": gpu_mean / max(cpu_mean, 1e-9),
            "eos_count": sum(bool(row.get("eos_reached")) for row in rows),
            "truncation_count": sum(bool(row.get("truncated")) for row in rows),
            "severe_repetition_count": sum(bool(row.get("severe_repetition_detected")) for row in rows),
            "paired_runs": [
                {"prompt_id": row["prompt_id"], "repeat": index // 3 + 1,
                 "wall_time_s": row.get("wall_time_s"), "generation_wall_time_s": row.get("generation_wall_time_s"),
                 "decode_steps": row.get("decode_steps"), "expanded_output_tokens": row.get("expanded_output_tokens"),
                 "eos_reached": row.get("eos_reached"), "truncated": row.get("truncated")}
                for index, row in enumerate(rows)
            ],
        }
    return {"records": all_records, "conditions": summaries}


def smoke_failure_details(error: Exception, sampler: NvidiaSampler) -> dict[str, Any]:
    message = f"{type(error).__name__}: {error}"
    lowered = message.casefold()
    oom = "out of memory" in lowered or "outofmemory" in lowered or "cuda_error_out_of_memory" in lowered
    run_manifest = {}
    manifest_path = OUTPUT_ROOT / "smoke_tier1" / "run_manifest.json"
    if manifest_path.is_file():
        try:
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    memory = run_manifest.get("failure_cuda_memory", {})
    latest_nvidia = sampler.samples[-1] if sampler.samples else None
    peak_nvidia = max(sampler.samples, key=lambda row: row["memory_used_mib"]) if sampler.samples else None
    requested_match = re.search(r"(?:allocate|alloc)\s+([0-9.]+)\s*(GiB|MiB|GB|MB)", message, re.IGNORECASE)
    requested_bytes = None
    if requested_match:
        magnitude = float(requested_match.group(1))
        unit = requested_match.group(2).casefold()
        requested_bytes = int(magnitude * (1024 ** (3 if unit in ("gib", "gb") else 2)))
    estimated_missing_bytes = None
    if requested_bytes is not None and memory.get("free_bytes") is not None:
        estimated_missing_bytes = max(0, requested_bytes - int(memory["free_bytes"]))
    return {
        "error": message,
        "oom": oom,
        "run_manifest_status": run_manifest.get("status"),
        "failure_cuda_memory": memory or None,
        "nvidia_smi_latest": latest_nvidia,
        "nvidia_smi_peak_memory_sample": peak_nvidia,
        "requested_allocation_bytes_if_reported": requested_bytes,
        "estimated_missing_memory_bytes_if_requested_size_reported": estimated_missing_bytes,
        "likely_cause": (
            "Single-T4 model weights plus attention/KV cache, generation workspace, or loaded adapter overhead exceeded available device memory."
            if oom else None
        ),
        "reasonable_options": (
            [
                "Keep this run stopped; inspect the captured memory request and peak first.",
                "Consider a smaller context or lower generation cap, a documented memory-saving inference mode, or a single larger-memory accelerator in a later reviewed run.",
                "Do not combine both T4 devices or retry this package by brute force.",
            ] if oom else []
        ),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    session_started = datetime.now(timezone.utc).isoformat()
    phase_status = {"smoke": "not_started", "timing": "not_started", "funnel": "not_started", "continuation": "not_started"}
    smoke_result: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    sampler = NvidiaSampler()
    sampler.start()
    try:
        artifact_manifest, checkpoint_path, predictor_path = verify_artifacts()
        install_dependencies()
        environment = python_environment(REPO_ROOT)
        if not environment.get("cuda_available") or environment.get("visible_gpu_count") != 1:
            raise RuntimeError("CUDA device contract failed: expected exactly one visible GPU at cuda:0")
        if "T4" not in environment["gpus"][0]["name"]:
            raise RuntimeError(f"Kaggle supplied an unexpected accelerator: {environment['gpus'][0]['name']}")

        phase_status["smoke"] = "running"
        smoke_dir = OUTPUT_ROOT / "smoke_tier1"
        run_tier1(smoke_dir, artifact_manifest, checkpoint_path, conditions=EXPECTED_CONDITIONS)
        smoke_result = validate_smoke(smoke_dir, artifact_manifest, environment, sampler)
        phase_status["smoke"] = smoke_result["status"]
        write_json(OUTPUT_ROOT / "smoke_results.json", smoke_result)
        smoke_lines = [
            "# Single-T4 GPU smoke result",
            "",
            f"Status: **{smoke_result['status']}** on {smoke_result['gpu_name']} (`cuda:0`).",
            "",
            f"Checkpoint loader base hashes: `{smoke_result['checkpoint_load_report'].get('base_hash_status')}`; prompt mode: `{smoke_result['prompt_representation']}`; gate: `{smoke_result['emission_gate_top_n']}`.",
            "",
            f"Non-zero nvidia-smi utilization samples: {sum(sample['utilization_pct'] > 0 for sample in smoke_result['nvidia_smi_utilization']['samples'])}/{smoke_result['nvidia_smi_utilization']['sample_count']}; peak GPU utilization: {smoke_result['nvidia_smi_utilization']['max_gpu_utilization_pct']}%.",
            "",
            "| Condition | CPU mean request s | GPU mean request s | GPU/CPU |",
            "|---|---:|---:|---:|",
        ]
        for condition, timing in smoke_result["paired_cpu_comparison"]["timing"].items():
            smoke_lines.append(f"| {condition} | {timing['cpu_mean_wall_time_s']:.3f} | {timing['gpu_mean_wall_time_s']:.3f} | {timing['gpu_to_cpu_wall_ratio']:.3f} |")
        smoke_lines.extend(["", "## Stop conditions", ""])
        smoke_lines.extend([f"- {reason}" for reason in smoke_result["stop_reasons"]] or ["- None."])
        smoke_lines.extend(["", "Per-prompt output comparisons, v3 records, checkpoint report, and VRAM counters are in the JSON and raw run folder."])
        (OUTPUT_ROOT / "smoke_results.md").write_text("\n".join(smoke_lines) + "\n", encoding="utf-8")

        if smoke_result["status"] != "pass":
            raise StopAfterSmoke("Smoke gates failed; later GPU experiments were skipped")

        phase_status["timing"] = "running"
        cpu_refs = json.loads((DATASET_ROOT / "cpu_reference_selected.json").read_text(encoding="utf-8"))
        timing_dirs = []
        for repeat in (1, 2):
            repeat_dir = OUTPUT_ROOT / f"timing_repeat_{repeat}"
            run_tier1(repeat_dir, artifact_manifest, checkpoint_path, conditions=EXPECTED_CONDITIONS)
            timing_dirs.append(repeat_dir)
        timing = timing_summaries(timing_dirs, cpu_refs)
        timing_result = {
            "schema": "tokens_gpu_timing_sanity_v1",
            "device": "cuda:0",
            "gpu_name": environment["gpus"][0]["name"],
            "smoke_run_used_as_condition_warmup": True,
            "timing_repeats_per_prompt_condition": 2,
            "conditions": timing["conditions"],
            "records": timing["records"],
            "comparison_note": "CPU records are prior CPU runs; timing here excludes model-load time and includes predictive setup per request.",
        }
        write_json(OUTPUT_ROOT / "timing_sanity.json", timing_result)
        timing_lines = [
            "# Single-T4 timing sanity",
            "",
            f"The three-prompt smoke generated once per condition as warm-up; two repeats per prompt and condition followed on {timing_result['gpu_name']}.",
            "",
            "| Condition | Mean request s | Mean steps/request | Mean seconds/step | Steps/s | Expanded tok/s | GPU/CPU request ratio | EOS | Truncated |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for condition, row in timing_result["conditions"].items():
            timing_lines.append(
                f"| {condition} | {row['mean_gpu_request_wall_time_s']:.3f} | {row['mean_decode_steps']:.1f} | {row['mean_seconds_per_transformer_step']:.4f} | {row['mean_transformer_steps_per_second']:.2f} | {row['mean_effective_expanded_tokens_per_second']:.2f} | {row['gpu_to_cpu_mean_request_wall_ratio']:.3f} | {row['eos_count']}/{row['n']} | {row['truncation_count']}/{row['n']} |"
            )
        timing_lines.extend(["", "The seconds/step column isolates iteration cost from trajectory length. Predictive effective throughput includes setup; Vanilla has no predictor setup."])
        (OUTPUT_ROOT / "timing_sanity.md").write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
        if any(row["gpu_to_cpu_mean_request_wall_ratio"] >= 1.0 for row in timing_result["conditions"].values()):
            phase_status["timing"] = "stop"
            raise StopAfterSmoke("Repeated GPU timing was not faster than the paired CPU reference")
        phase_status["timing"] = "pass"

        phase_status["funnel"] = "running"
        phase5_dir = REPO_ROOT / "experiments/checkpoints/quality_benchmark/tier1_runs/3de046a1b858dc7a"
        run_command(
            [sys.executable, "-m", "experiments.kaggle.run_predictor_oracle_funnel",
             "--live-results", str(phase5_dir / "raw_results.jsonl"),
             "--phase5-manifest", str(phase5_dir / "run_manifest.json"),
             "--output-dir", str(OUTPUT_ROOT)],
            cwd=REPO_ROOT,
            log_path=OUTPUT_ROOT / "logs" / "predictor_oracle_funnel.log",
        )
        phase_status["funnel"] = "complete"

        phase_status["continuation"] = "running"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        run_command(
            [sys.executable, "-m", "experiments.kaggle.run_continuation_safety_gpu",
             "--checkpoint", str(checkpoint_path),
             "--predictor", "experiments/checkpoints/oracle_guided_predictor.pkl",
             "--expected-predictor-sha256", artifact_manifest["predictor"]["sha256"],
             "--probe-selection", str(OUTPUT_ROOT / "continuation_probe_selection.json"),
             "--funnel-json", str(OUTPUT_ROOT / "predictor_oracle_funnel.json"),
             "--output", str(OUTPUT_ROOT / "continuation_safety_gpu.json"),
             "--tested-commit", artifact_manifest["repo_commit"],
             "--expected-gpu-name", environment["gpus"][0]["name"]],
            cwd=REPO_ROOT,
            env=env,
            log_path=OUTPUT_ROOT / "logs" / "continuation_safety.log",
        )
        phase_status["continuation"] = "complete"

    except StopAfterSmoke as error:
        if smoke_result is not None:
            smoke_result.setdefault("stop_reasons", []).append(str(error))
            write_json(OUTPUT_ROOT / "smoke_results.json", smoke_result)
        print(str(error), flush=True)
    except Exception as error:
        phase_status["failed"] = f"{type(error).__name__}: {error}"
        (OUTPUT_ROOT / "logs" / "run_error.log").parent.mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "logs" / "run_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        active_phase = next(
            (name for name, status in phase_status.items() if status == "running"),
            None,
        )
        if active_phase in {"timing", "funnel", "continuation"}:
            failure = {
                "schema": f"tokens_kaggle_{active_phase}_failure_v1",
                "status": "failed",
                "phase": active_phase,
                "error_type": type(error).__name__,
                "error": str(error),
                "oom_detected": "out of memory" in str(error).casefold()
                    or "outofmemory" in str(error).casefold(),
                "traceback_log": "logs/run_error.log",
                "nvidia_smi_peak_memory_sample": max(
                    sampler.samples,
                    key=lambda sample: sample["memory_used_mib"],
                    default=None,
                ),
                "nvidia_smi_latest_sample": sampler.samples[-1] if sampler.samples else None,
                "nvidia_smi_errors": sampler.errors,
                "phase_status": dict(phase_status),
                "no_later_phase_started": True,
            }
            failure_stem = {
                "timing": "timing_sanity",
                "funnel": "predictor_oracle_funnel",
                "continuation": "continuation_safety_gpu",
            }[active_phase]
            write_json(OUTPUT_ROOT / f"{failure_stem}.json", failure)
            (OUTPUT_ROOT / f"{failure_stem}.md").write_text(
                "# " + active_phase.title() + " failure\n\n"
                + f"Status: **failed** during `{active_phase}`.\n\n"
                + f"- {type(error).__name__}: {error}\n"
                + f"- OOM detected: {failure['oom_detected']}\n"
                + f"- Peak GPU memory sample: `{failure['nvidia_smi_peak_memory_sample']}`\n"
                + "- No later phase was started. See `logs/run_error.log` for the traceback.\n",
                encoding="utf-8",
            )
        if smoke_result is None:
            failure_details = smoke_failure_details(error, sampler)
            smoke_result = {
                "status": "failed",
                "stop_reasons": [f"{type(error).__name__}: {error}"],
                "failure_details": failure_details,
                "records": [],
            }
            write_json(OUTPUT_ROOT / "smoke_results.json", smoke_result)
            failure_lines = [
                "# Single-T4 GPU smoke result",
                "",
                "Status: **failed before smoke completion**.",
                "",
                f"- {type(error).__name__}: {error}",
                "",
                f"CUDA memory at failure: `{failure_details['failure_cuda_memory']}`",
                f"Peak nvidia-smi sample: `{failure_details['nvidia_smi_peak_memory_sample']}`",
            ]
            if failure_details["oom"]:
                failure_lines.extend([
                    f"Estimated missing memory when the failed request size is available: `{failure_details['estimated_missing_memory_bytes_if_requested_size_reported']}` bytes.",
                    f"Likely cause: {failure_details['likely_cause']}",
                    "Reasonable options:",
                    *[f"- {option}" for option in failure_details["reasonable_options"]],
                ])
            failure_lines.extend(["", "No later phase was started. See `logs/run_error.log` and the smoke run manifest for full error details."])
            (OUTPUT_ROOT / "smoke_results.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    finally:
        sampler.stop()

    finished = datetime.now(timezone.utc).isoformat()
    if environment is not None:
        manifest = {
            "schema": "tokens_kaggle_gpu_environment_v1",
            "repo": "aehng/Tokens",
            "repo_commit": artifact_manifest["repo_commit"] if "artifact_manifest" in locals() else None,
            "repo_branch": PACKAGE_BRANCH,
            "model_id": "microsoft/Phi-3.5-mini-instruct",
            "phi_revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
            "tokenizer_id": "microsoft/Phi-3.5-mini-instruct",
            "tokenizer_revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
            "zip2zip_id": "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
            "zip2zip_revision": "11c461733a79d2a5de6b814585c3361ca2aacbe7",
            "checkpoint_sha256": artifact_manifest["checkpoint"]["sha256"] if "artifact_manifest" in locals() else None,
            "predictor_sha256": artifact_manifest["predictor"]["sha256"] if "artifact_manifest" in locals() else None,
            "torch_version": environment["torch"],
            "transformers_version": environment["transformers"],
            "cuda_version": environment["cuda_runtime"],
            "python_version": environment["python"],
            "driver_version": sampler.samples[-1].get("driver_version") if sampler.samples else None,
            "device": "cuda:0",
            "visible_gpu_count": environment["visible_gpu_count"],
            "gpu": environment["gpus"][0] if environment["gpus"] else None,
            "python_packages": environment["packages"],
            "idle_vram": environment["gpus"][0] if environment["gpus"] else None,
            "model_loaded_vram_by_condition": {},
            "peak_inference_vram": {"allocated_bytes": 0, "reserved_bytes": 0},
            "nvidia_smi_utilization": {
                "sample_count": len(sampler.samples),
                "max_gpu_utilization_pct": max((sample["utilization_pct"] for sample in sampler.samples), default=None),
                "max_memory_used_mib": max((sample["memory_used_mib"] for sample in sampler.samples), default=None),
                "samples": sampler.samples,
                "errors": sampler.errors,
            },
            "session_started_at_utc": session_started,
            "session_finished_at_utc": finished,
            "session_wall_time_s": round(time.monotonic() - start, 3),
            "phase_status": phase_status,
        }
        if smoke_result and smoke_result.get("records"):
            for row in smoke_result["records"]:
                condition = row.get("condition")
                loaded = row.get("cuda_memory_model_loaded") or {}
                generation = row.get("cuda_memory_generation") or {}
                manifest["model_loaded_vram_by_condition"][condition] = {
                    "allocated_bytes": loaded.get("allocated_bytes"),
                    "reserved_bytes": loaded.get("reserved_bytes"),
                }
                manifest["peak_inference_vram"]["allocated_bytes"] = max(
                    manifest["peak_inference_vram"]["allocated_bytes"],
                    int(generation.get("peak_allocated_bytes", 0)),
                )
                manifest["peak_inference_vram"]["reserved_bytes"] = max(
                    manifest["peak_inference_vram"]["reserved_bytes"],
                    int(generation.get("peak_reserved_bytes", 0)),
                )
        write_json(OUTPUT_ROOT / "environment_manifest.json", manifest)
    write_json(OUTPUT_ROOT / "run_status.json", {
        "phase_status": phase_status,
        "session_started_at_utc": session_started,
        "session_finished_at_utc": finished,
        "session_wall_time_s": round(time.monotonic() - start, 3),
    })


class StopAfterSmoke(Exception):
    pass


if __name__ == "__main__":
    main()
