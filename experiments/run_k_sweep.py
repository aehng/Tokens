"""Manifest-pinned, resumable K sweep for the fixed Phi Tier-1 prompt set."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch
from transformers import AutoTokenizer, LogitsProcessorList

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from zip2zip import StaticCodebookManager  # noqa: E402
from zip2zip.evidence_selector import EvidenceAwareSelector  # noqa: E402
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
from experiments.load_joint_checkpoint import CHECKPOINT_LOADER_ID, load_joint_checkpoint  # noqa: E402
from experiments.mbpp_prompt import build_mbpp_prompt  # noqa: E402
from experiments.run_quality_benchmark import (  # noqa: E402
    INITIAL_VOCAB,
    MAX_NEW_TOKENS,
    PHI_MODEL_ID,
    ZIP2ZIP_MODEL_ID,
    TimingLogitsProcessor,
    _load_zip2zip_model,
    evaluate_alpaca_instruction,
    evaluate_gsm8k_reasoning,
    evaluate_mbpp_code,
    sequence_reached_eos,
    synchronize_device,
)

DEFAULT_PROMPTS = (
    REPO_ROOT / "experiments" / "checkpoints" / "quality_benchmark" / "poc_12_prompt_ids.json"
)
DEFAULT_VALIDATION_DATA = REPO_ROOT / "data" / "cached_pure_pred_val_60.json"
DEFAULT_PREDICTOR = REPO_ROOT / "experiments" / "checkpoints" / "oracle_guided_predictor.pkl"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "experiments" / "checkpoints" / "predictive_joint_pilot" / "checkpoint_step_100.pt"
)
EXPECTED_DOMAINS = {"code": 4, "reasoning": 4, "instruction": 4}
MANIFEST_SCHEMA = "tokens_k_sweep_manifest_v1"
RECORD_SCHEMA = "k_sweep_generation_v2"


def codebook_hash(codebook_dict: Dict[Tuple[int, ...], int]) -> str:
    """Hash the complete ordered token-to-hypertoken mapping."""
    items = [
        {"subtokens": list(tokens), "token_id": token_id}
        for tokens, token_id in sorted(codebook_dict.items())
    ]
    return canonical_sha256(items)


def is_correct(record: Dict[str, Any]) -> bool:
    domain = record.get("domain")
    if domain == "code":
        return bool(record.get("problem_pass", False))
    if domain == "reasoning":
        return bool(record.get("exact_correct", False))
    if domain == "instruction":
        return not bool(record.get("instruction_failure", False))
    return False


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


def _git_status() -> str:
    if not (REPO_ROOT / ".git").exists():
        return ""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_identity() -> tuple[dict[str, str], str]:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "experiments" / "run_quality_benchmark.py",
        REPO_ROOT / "experiments" / "mbpp_prompt.py",
        REPO_ROOT / "experiments" / "load_joint_checkpoint.py",
        REPO_ROOT / "experiments" / "benchmark_provenance.py",
        REPO_ROOT / "src" / "zip2zip" / "evidence_selector.py",
        REPO_ROOT / "src" / "zip2zip" / "model.py",
        REPO_ROOT / "src" / "zip2zip" / "static_codebook.py",
    )
    hashes = {
        path.relative_to(REPO_ROOT).as_posix(): file_sha256(path)
        for path in paths
    }
    evaluator = {
        name: inspect.getsource(obj)
        for name, obj in (
            ("is_correct", is_correct),
            ("evaluate_mbpp_code", evaluate_mbpp_code),
            ("evaluate_gsm8k_reasoning", evaluate_gsm8k_reasoning),
            ("evaluate_alpaca_instruction", evaluate_alpaca_instruction),
            ("sequence_reached_eos", sequence_reached_eos),
        )
    }
    return hashes, canonical_sha256(evaluator)


def _validate_parameters(k_values: List[int], tau_values: List[float], max_new_tokens: int) -> None:
    if not k_values or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > 32
        for value in k_values
    ):
        raise ValueError("k_values must be a non-empty list of integers in [1, 32]")
    if len(set(k_values)) != len(k_values):
        raise ValueError("k_values may not contain duplicates")
    if len(set(tau_values)) != len(tau_values) or any(not math.isfinite(value) for value in tau_values):
        raise ValueError("tau_values must be finite and may not contain duplicates")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")


def _prompt_text(sample: Mapping[str, Any]) -> str:
    return build_mbpp_prompt(sample) if sample["domain"] == "code" else str(sample["prompt"])


def _append_report(out_md: Path, configs: Mapping[str, Mapping[str, Any]], identity_sha: str) -> None:
    lines = [
        "# Phi Tier-1 K Sweep",
        "",
        f"Run identity: `{identity_sha}`",
        "",
        "Fixed-K settings and optional adaptive thresholds are computed from this run only. Historical narrative claims are not carried forward as findings.",
        "",
        "| Configuration | Budget / threshold | Score | Micro decode reduction | Mean wall time |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in configs.items():
        setting = f"K={result['target_k']}" if result["min_tau"] is None else f"tau={result['min_tau']}"
        lines.append(
            f"| {name} | {setting} | {result['total_correct']} ({result['accuracy_pct']}%) | "
            f"{result['micro_reduction_pct']}% | {result['mean_wall_time_s']}s |"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_md.with_name(f".{out_md.name}.tmp")
    temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temp_path, out_md)


def run_sweep(
    prompts_path: str,
    k_values: List[int],
    tau_values: List[float],
    out_json: str,
    out_md: str,
    device_str: str = "cpu",
    *,
    validation_data_path: str = str(DEFAULT_VALIDATION_DATA),
    predictor_path: str = str(DEFAULT_PREDICTOR),
    checkpoint_path: str = str(DEFAULT_CHECKPOINT),
    base_revision: str | None = None,
    model_revision: str | None = None,
    tested_commit: str | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    cache_dir: str | None = None,
) -> Path:
    _validate_parameters(k_values, tau_values, max_new_tokens)
    device = torch.device(device_str)
    prompts_file = Path(prompts_path).resolve()
    validation_file = Path(validation_data_path).resolve()
    predictor_file = Path(predictor_path).resolve()
    checkpoint_file = Path(checkpoint_path).resolve()
    output_json = Path(out_json).resolve()
    output_md = Path(out_md).resolve()
    manifest_path = output_json.with_name(output_json.name + ".manifest.json")
    for required_path in (prompts_file, validation_file, predictor_file, checkpoint_file):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required K-sweep input is missing: {required_path}")

    if (output_json.exists() or output_md.exists()) and not manifest_path.exists():
        raise ValueError(
            "An output already exists without an exact run manifest. Choose a new output path; "
            "legacy results will not be reused or overwritten."
        )

    if device.type == "cuda":
        for label, revision in (("base revision", base_revision), ("Zip2Zip revision", model_revision)):
            if not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
                raise ValueError(f"GPU runs require {label} as a full 40-character commit SHA")
        if (REPO_ROOT / ".git").exists() and _git_status():
            raise ValueError("GPU runs require a clean, committed source worktree")
        if not (REPO_ROOT / ".git").exists() and not tested_commit:
            raise ValueError("Packaged GPU runs without Git metadata require --tested-commit")

    commit = tested_commit or resolve_tested_commit(REPO_ROOT)
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit.lower()):
        raise ValueError("tested_commit must be a full 40-character Git SHA")
    commit = commit.lower()
    runtime = _runtime_identity(device_str)
    source_hashes, evaluator_hash = _source_identity()

    with validation_file.open("r", encoding="utf-8") as source:
        all_samples = json.load(source)
    samples = select_prompt_subset(all_samples, prompts_file, EXPECTED_DOMAINS)

    tokenizer_kwargs = {"revision": base_revision} if base_revision else {}
    tokenizer = AutoTokenizer.from_pretrained(PHI_MODEL_ID, **tokenizer_kwargs)
    with predictor_file.open("rb") as source:
        raw_predictor = pickle.load(source)
    predictor_index = getattr(raw_predictor, "index", raw_predictor)
    selector = EvidenceAwareSelector(
        predictor_index=predictor_index,
        tokenizer=tokenizer,
        budget=32,
        max_structural_slots=0,
    )

    configurations = [(f"fixed_k_{value}", value, None) for value in k_values]
    configurations.extend(
        (f"adaptive_tau_{value:.1f}", 32, value) for value in tau_values
    )
    generation = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "dtype": "float16",
        "eos_rule": "last_generated_token_equals_tokenizer_eos_id",
    }
    checkpoint_sha = file_sha256(checkpoint_file)
    predictor_sha = file_sha256(predictor_file)
    model_revisions = {
        "base_model_id": PHI_MODEL_ID,
        "base_revision": base_revision or "UNPINNED",
        "zip2zip_model_id": ZIP2ZIP_MODEL_ID,
        "zip2zip_revision": model_revision or "UNPINNED",
    }

    plans: list[dict[str, Any]] = []
    selection_manifest = []
    cache_keys: dict[tuple[str, str], str] = {}
    for config_name, target_k, min_tau in configurations:
        for sample in samples:
            prompt = _prompt_text(sample)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            codebook, selection_meta = selector.select_codebook(
                prompt_ids,
                prompt_text=prompt,
                budget=target_k,
                min_score_threshold=min_tau,
            )
            cb_hash = codebook_hash(codebook)
            condition = {
                "policy": "EvidenceAwareSelector",
                "tested_commit": commit,
                "config": config_name,
                "target_k": target_k,
                "min_score_threshold": min_tau,
                "max_structural_slots": 0,
                "checkpoint_loader": CHECKPOINT_LOADER_ID,
                "checkpoint_sha256": checkpoint_sha,
                "predictor_sha256": predictor_sha,
                "codebook_sha256": cb_hash,
                "model_revisions": model_revisions,
                "source_hashes": source_hashes,
            }
            key = build_generation_cache_key(
                condition=condition,
                prompt_id=sample["id"],
                prompt_text=prompt,
                reference_text=str(sample.get("ground_truth_response", "")),
                generation=generation,
                evaluator_sha256=evaluator_hash,
                environment=runtime,
            )
            cache_keys[(sample["id"], config_name)] = key
            plans.append(
                {
                    "config": config_name,
                    "target_k": target_k,
                    "min_tau": min_tau,
                    "sample": sample,
                    "prompt": prompt,
                    "codebook": codebook,
                    "selection_meta": selection_meta,
                    "codebook_hash": cb_hash,
                    "cache_key": key,
                }
            )
            selection_manifest.append(
                {
                    "config": config_name,
                    "prompt_id": sample["id"],
                    "codebook_sha256": cb_hash,
                    "codebook_size": len(codebook),
                }
            )

    prompt_manifest = [
        {
            "id": sample["id"],
            "domain": sample["domain"],
            "prompt_sha256": hashlib.sha256(_prompt_text(sample).encode("utf-8")).hexdigest(),
            "reference_sha256": hashlib.sha256(
                str(sample.get("ground_truth_response", "")).encode("utf-8")
            ).hexdigest(),
        }
        for sample in samples
    ]
    identity = {
        "schema": "tokens_k_sweep_identity_v2",
        "tested_commit": commit,
        "prompt_ids_file_sha256": file_sha256(prompts_file),
        "validation_data_sha256": file_sha256(validation_file),
        "prompt_ids": [sample["id"] for sample in samples],
        "prompts": prompt_manifest,
        "configurations": [
            {"name": name, "target_k": k, "min_score_threshold": tau}
            for name, k, tau in configurations
        ],
        "selected_codebooks": selection_manifest,
        "generation": generation,
        "checkpoint_sha256": checkpoint_sha,
        "predictor_sha256": predictor_sha,
        "checkpoint_loader": CHECKPOINT_LOADER_ID,
        "model_revisions": model_revisions,
        "evaluator_sha256": evaluator_hash,
        "source_hashes": source_hashes,
        "runtime": runtime,
        "generation_cache_keys": {
            f"{prompt_id}::{config}": key
            for (prompt_id, config), key in sorted(cache_keys.items())
        },
    }
    manifest, resumed = create_or_verify_run_manifest(
        manifest_path, identity, schema=MANIFEST_SCHEMA
    )
    identity_sha = manifest["identity_sha256"]
    cache_root = Path(cache_dir).resolve() if cache_dir else output_json.with_name(
        output_json.stem + "_generation_cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    manifest.update(
        {
            "status": "running",
            "resumed": resumed,
            "started_or_resumed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    write_json_atomic(manifest_path, manifest)

    cache_records: dict[str, dict[str, Any]] = {}
    pending = []
    for plan in plans:
        cache_file = cache_root / f"{plan['cache_key']}.json"
        if not cache_file.is_file():
            pending.append(plan)
            continue
        try:
            record = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pending.append(plan)
            continue
        if (
            validate_generation_cache_record(record, plan["cache_key"], RECORD_SCHEMA)
            and record.get("prompt_id") == plan["sample"]["id"]
            and record.get("config") == plan["config"]
            and record.get("codebook_sha256") == plan["codebook_hash"]
        ):
            cache_records[plan["cache_key"]] = record
        else:
            pending.append(plan)

    model = None
    try:
        if pending:
            print(
                f"Exact cache hits: {len(cache_records)}; generations remaining: {len(pending)}. Loading model.",
                flush=True,
            )
            model = _load_zip2zip_model(
                ZIP2ZIP_MODEL_ID, base_revision, model_revision
            ).to(device)
            model.eval()
            model.output_encoder.to(torch.float32)
            load_report = load_joint_checkpoint(model, checkpoint_file)
            print(
                f"Loaded checkpoint ({load_report['lora_tensors']} LoRA tensors; "
                f"{load_report['input_encoder_tensors'] + load_report['output_encoder_tensors']} encoder tensors).",
                flush=True,
            )
            dim = model.zip2zip_config.encoder.hidden_size
            pad_id = tokenizer.pad_token_id or 32000
            disabled_ids = list(model.zip2zip_config.compression.disabled_ids)

            for plan in pending:
                sample = plan["sample"]
                codebook = plan["codebook"]
                prompt_ids = tokenizer.encode(plan["prompt"], add_special_tokens=False)
                setup_start = time.perf_counter()
                manager = StaticCodebookManager(
                    initial_vocab_size=INITIAL_VOCAB,
                    max_codebook_size=32,
                    max_subtokens=4,
                    embedding_dim=dim,
                    pad_token_id=pad_id,
                    disabled_ids=disabled_ids,
                )
                manager.set_seeded_codebook(codebook, batch_size=1, device=device)
                manager.attach_to_model(model)
                setup_time = time.perf_counter() - setup_start
                input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                synchronize_device(device)
                generation_start = time.perf_counter()
                timing = TimingLogitsProcessor(generation_start, static_mgr=manager)
                try:
                    with torch.no_grad():
                        output = model.generate(
                            input_ids=input_tensor,
                            max_new_tokens=max_new_tokens,
                            logits_processor=LogitsProcessorList([timing]),
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    synchronize_device(device)
                    generation_time = time.perf_counter() - generation_start
                finally:
                    manager.detach_from_model(model)
                    model.codebook_manager.reset()

                generated_ids = output[0, len(prompt_ids):].tolist()
                inverse_codebook = {token_id: list(tokens) for tokens, token_id in codebook.items()}
                expanded_ids: list[int] = []
                emitted = []
                for position, token_id in enumerate(generated_ids):
                    if token_id in inverse_codebook:
                        subtokens = inverse_codebook[token_id]
                        emitted.append(
                            {
                                "position": position,
                                "token_id": token_id,
                                "subtokens": subtokens,
                                "text": tokenizer.decode(subtokens),
                            }
                        )
                        expanded_ids.extend(subtokens)
                    else:
                        expanded_ids.append(token_id)
                output_text = tokenizer.decode(expanded_ids, skip_special_tokens=True)
                expanded_count = len(expanded_ids)
                decode_count = len(generated_ids)
                eos_reached = sequence_reached_eos(generated_ids, tokenizer.eos_token_id)
                record: dict[str, Any] = {
                    "record_schema": RECORD_SCHEMA,
                    "generation_cache_key": plan["cache_key"],
                    "generated_from_commit": commit,
                    "checkpoint_loader": CHECKPOINT_LOADER_ID,
                    "prompt_id": sample["id"],
                    "domain": sample["domain"],
                    "condition": plan["config"],
                    "config": plan["config"],
                    "target_k": plan["target_k"],
                    "min_tau": plan["min_tau"],
                    "codebook_sha256": plan["codebook_hash"],
                    "base_prompt_tokens": len(prompt_ids),
                    "decode_steps": decode_count,
                    "expanded_output_tokens": expanded_count,
                    "tokens_saved": max(0, expanded_count - decode_count),
                    "decode_reduction_pct": round(
                        100.0 * (1 - decode_count / max(expanded_count, 1)), 2
                    ) if expanded_count > decode_count else 0.0,
                    "wall_time_s": round(
                        float(plan["selection_meta"].get("latency_ms", 0)) / 1000.0
                        + setup_time
                        + generation_time,
                        3,
                    ),
                    "ttft_s": round(timing.ttft or 0.0, 3),
                    "hypertokens_count": len(emitted),
                    "codebook_size": len(codebook),
                    "used_slots": len({entry["token_id"] for entry in emitted}),
                    "dead_slots": len(codebook) - len({entry["token_id"] for entry in emitted}),
                    "hypertokens_emitted": emitted,
                    "eos_reached": eos_reached,
                    "hit_max_length": decode_count >= max_new_tokens,
                    "output_text": output_text,
                }
                used_slots = record["used_slots"]
                record["utilization_pct"] = round(
                    used_slots / len(codebook) * 100, 1
                ) if codebook else 0.0
                if sample["domain"] == "code":
                    asserts = [
                        line.strip()
                        for line in sample["ground_truth_response"].splitlines()
                        if line.strip().startswith("assert")
                    ]
                    record.update(evaluate_mbpp_code(output_text, asserts))
                elif sample["domain"] == "reasoning":
                    record.update(
                        evaluate_gsm8k_reasoning(output_text, sample["ground_truth_response"])
                    )
                else:
                    record.update(evaluate_alpaca_instruction(output_text, eos_reached))

                cache_records[plan["cache_key"]] = record
                write_json_atomic(cache_root / f"{plan['cache_key']}.json", record)
                manifest["completed_record_count"] = len(cache_records)
                write_json_atomic(manifest_path, manifest)
                print(
                    f"[{len(cache_records)}/{len(plans)}] {sample['id']} {plan['config']}: "
                    f"{decode_count} steps, {record['wall_time_s']:.2f}s",
                    flush=True,
                )

        configs_results: dict[str, Any] = {}
        for config_name, target_k, min_tau in configurations:
            records = [
                cache_records[cache_keys[(sample["id"], config_name)]]
                for sample in samples
            ]
            total_steps = sum(record["decode_steps"] for record in records)
            total_expanded = sum(record["expanded_output_tokens"] for record in records)
            correct_count = sum(is_correct(record) for record in records)
            total_slots = sum(record["codebook_size"] for record in records)
            used_slots = sum(record["used_slots"] for record in records)
            configs_results[config_name] = {
                "config": config_name,
                "target_k": target_k,
                "min_tau": min_tau,
                "total_correct": f"{correct_count}/{len(records)}",
                "accuracy_pct": round(100.0 * correct_count / len(records), 1),
                "micro_reduction_pct": round(
                    100.0 * (1 - total_steps / max(total_expanded, 1)), 2
                ),
                "tokens_saved": sum(record["tokens_saved"] for record in records),
                "mean_hypertokens": round(
                    sum(record["hypertokens_count"] for record in records) / len(records), 2
                ),
                "mean_codebook_size": round(total_slots / len(records), 1),
                "total_cb_slots": total_slots,
                "used_slots": used_slots,
                "dead_slots": total_slots - used_slots,
                "utilization_pct": round(100.0 * used_slots / max(total_slots, 1), 1),
                "mean_wall_time_s": round(
                    sum(record["wall_time_s"] for record in records) / len(records), 2
                ),
                "mean_ttft_s": round(
                    sum(record["ttft_s"] for record in records) / len(records), 3
                ),
                "records": records,
            }

        final_output = {
            "schema": "tokens_k_sweep_results_v2",
            "run_identity_sha256": identity_sha,
            "tested_commit": commit,
            "summary_table": {
                name: {key: value for key, value in result.items() if key != "records"}
                for name, result in configs_results.items()
            },
            "configs_detailed": configs_results,
            "cached_generations": cache_records,
        }
        write_json_atomic(output_json, final_output)
        _append_report(output_md, configs_results, identity_sha)
        manifest.update(
            {
                "status": "complete",
                "completed_record_count": len(cache_records),
                "expected_record_count": len(plans),
                "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        write_json_atomic(manifest_path, manifest)
        print(f"Manifest-verified K sweep complete: {output_json}", flush=True)
        return output_json
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        write_json_atomic(manifest_path, manifest)
        raise
    finally:
        if model is not None:
            del model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exact-manifest fixed-K evaluation on the Phi Tier-1 12-prompt set."
    )
    parser.add_argument("--prompts-file", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--validation-data", default=str(DEFAULT_VALIDATION_DATA))
    parser.add_argument("--predictor", default=str(DEFAULT_PREDICTOR))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 8, 16, 24, 32])
    parser.add_argument("--tau-values", nargs="*", type=float, default=[])
    parser.add_argument(
        "--out-json",
        default=str(REPO_ROOT / "experiments" / "checkpoints" / "quality_benchmark" / "k_sweep_tier1.json"),
    )
    parser.add_argument(
        "--out-md",
        default=str(REPO_ROOT / "experiments" / "checkpoints" / "quality_benchmark" / "k_sweep_tier1.md"),
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-revision", help="Full pinned Phi/tokenizer Hub commit SHA for GPU runs")
    parser.add_argument("--model-revision", help="Full pinned Zip2Zip Hub commit SHA for GPU runs")
    parser.add_argument("--tested-commit", help="Full project commit SHA; required for packaged runs without Git")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()
    run_sweep(
        prompts_path=args.prompts_file,
        k_values=args.k_values,
        tau_values=args.tau_values,
        out_json=args.out_json,
        out_md=args.out_md,
        device_str=args.device,
        validation_data_path=args.validation_data,
        predictor_path=args.predictor,
        checkpoint_path=args.checkpoint,
        base_revision=args.base_revision,
        model_revision=args.model_revision,
        tested_commit=args.tested_commit,
        max_new_tokens=args.max_new_tokens,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
