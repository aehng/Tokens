"""TRAIN-only / DEV-only multi-seed Predictor V2 architecture bakeoff.

This runner never opens FINAL. Its only selection metrics come from DEV, and it
requires the previously frozen candidate generator.
"""

from __future__ import annotations

import argparse
import os
import pickle
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.zip2zip.predictor_v2.candidate_retrieval import (
    ConfigurableCandidateGenerator,
    RetrievalStrategy,
    TrainOnlyAssociationIndex,
)
from src.zip2zip.predictor_v2.canonical_dataset import (
    load_canonical_dataset,
    load_manifest_tokenizer,
)
from src.zip2zip.predictor_v2.experiment_protocol import (
    MODEL_NAMES,
    load_candidate_plan,
    load_quality_attribution_gate,
    make_experiment_manifest,
    runtime_manifest,
    sha256_file,
    sha256_json,
    validate_resume_pair,
    write_json_exclusive,
)
from src.zip2zip.predictor_v2.metrics import evaluate_architecture_ranking
from src.zip2zip.predictor_v2.models.cnn_ranker import CNNRanker
from src.zip2zip.predictor_v2.models.gru_ranker import GRURanker
from src.zip2zip.predictor_v2.models.pooled_mlp import PooledMLPRanker
from src.zip2zip.predictor_v2.models.ridge import RidgeRanker
from src.zip2zip.predictor_v2.models.transformer_ranker import TransformerRanker
from src.zip2zip.predictor_v2.oracle_candidate_pool import CandidatePoolOracle
from src.zip2zip.predictor_v2.oracle_global import GlobalOccurrenceOracle
from src.zip2zip.predictor_v2.train_index import validate_train_index


NEURAL_SEEDS = (42, 43, 44)
K_VALUES = (8, 16, 32)


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _instantiate(name: str) -> Any:
    constructors = {
        "Ridge": RidgeRanker,
        "PooledMLP": PooledMLPRanker,
        "CNNRanker": CNNRanker,
        "GRURanker": GRURanker,
        "TransformerRanker": TransformerRanker,
    }
    return constructors[name]()


def _architecture_configuration(model: Any, name: str, epochs: int, learning_rate: float) -> dict[str, Any]:
    wrapper_parameters: dict[str, Any] = {}
    for key, value in vars(model).items():
        if isinstance(value, (str, int, float, bool)) or isinstance(value, torch.device):
            wrapper_parameters[key] = str(value) if isinstance(value, torch.device) else value
    return {
        "name": name,
        "wrapper_class": model.__class__.__name__,
        "wrapper_parameters": wrapper_parameters,
        "module_definition": repr(getattr(model, "model", None)) if hasattr(model, "model") else None,
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer": "closed_form_ridge" if name == "Ridge" else "AdamW",
            "weight_decay": None if name == "Ridge" else 1e-4,
            "early_stopping_split": "DEV" if name != "Ridge" else None,
        },
        "selection_split": "DEV",
    }


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50_ms": None, "p90_ms": None, "p99_ms": None}
    return {f"p{p}_ms": float(np.percentile(values, p)) for p in (50, 90, 99)}


def _bounds_by_split(
    records: list[Any],
    candidate_records: dict[str, list[Any]],
    tokenizer: Any,
    max_subtokens: int,
    time_limit_seconds: float,
) -> tuple[dict[int, dict[str, int]], dict[str, dict[int, dict[str, int]]]]:
    global_oracle = GlobalOccurrenceOracle(min_len=2, max_len=max_subtokens, time_limit_seconds=time_limit_seconds)
    pool_oracle = CandidatePoolOracle(tokenizer=tokenizer, time_limit_seconds=time_limit_seconds)
    totals = {k: {"global_lower": 0, "global_upper": 0, "candidate_lower": 0, "candidate_upper": 0, "global_exact": 0, "candidate_exact": 0, "prompts": 0} for k in K_VALUES}
    by_domain: dict[str, dict[int, dict[str, int]]] = {}
    for record in records:
        per_prompt: dict[int, tuple[Any, Any]] = {}
        for k in K_VALUES:
            global_result = global_oracle.solve(record.continuation_token_ids, k=k, tokenizer=tokenizer)
            candidate_result = pool_oracle.solve(
                candidate_records[record.prompt_id],
                record.continuation_token_ids,
                k=k,
                global_oracle_steps=global_result.steps_saved_lower_bound,
            )
            per_prompt[k] = (global_result, candidate_result)
            target = totals[k]
            target["global_lower"] += global_result.steps_saved_lower_bound
            target["global_upper"] += global_result.steps_saved_upper_bound
            target["candidate_lower"] += candidate_result.steps_saved_lower_bound
            target["candidate_upper"] += candidate_result.steps_saved_upper_bound
            target["global_exact"] += int(global_result.is_exact)
            target["candidate_exact"] += int(candidate_result.is_exact)
            target["prompts"] += 1
            domain = by_domain.setdefault(record.domain, {})
            dtarget = domain.setdefault(
                k,
                {"global_lower": 0, "global_upper": 0, "candidate_lower": 0, "candidate_upper": 0, "global_exact": 0, "candidate_exact": 0, "prompts": 0},
            )
            dtarget["global_lower"] += global_result.steps_saved_lower_bound
            dtarget["global_upper"] += global_result.steps_saved_upper_bound
            dtarget["candidate_lower"] += candidate_result.steps_saved_lower_bound
            dtarget["candidate_upper"] += candidate_result.steps_saved_upper_bound
            dtarget["global_exact"] += int(global_result.is_exact)
            dtarget["candidate_exact"] += int(candidate_result.is_exact)
            dtarget["prompts"] += 1
    return totals, by_domain


def _capture_interval(realized: int, lower: int, upper: int) -> dict[str, float | None]:
    if upper <= 0:
        return {"lower": None, "upper": None}
    return {
        "lower": realized / upper,
        "upper": min(1.0, realized / lower) if lower > 0 else 1.0,
    }


def _evaluate(
    model: Any,
    records: list[Any],
    candidates: dict[str, list[Any]],
    oracle_totals: dict[int, dict[str, int]],
    oracle_domains: dict[str, dict[int, dict[str, int]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    global_lower = {k: oracle_totals[k]["global_lower"] for k in K_VALUES}
    candidate_lower = {k: oracle_totals[k]["candidate_lower"] for k in K_VALUES}
    summary = evaluate_architecture_ranking(
        model,
        records,
        candidates,
        k_values=K_VALUES,
        global_oracle_steps_by_k=global_lower,
        candidate_oracle_steps_by_k=candidate_lower,
    )
    per_prompt_ms: list[float] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for record in records:
        start = time.perf_counter()
        model.rank_codebook(record.prompt_token_ids, candidates[record.prompt_id], domain=record.domain, k=32)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per_prompt_ms.append((time.perf_counter() - start) * 1000.0)
    inference = {
        "scoring_api": "rank_codebook over the frozen candidate pool, per DEV prompt",
        "latency_ms": _percentiles(per_prompt_ms),
    }
    domain_summary: dict[str, Any] = {}
    for domain in sorted({record.domain for record in records}):
        domain_records = [record for record in records if record.domain == domain]
        domain_bounds = oracle_domains[domain]
        domain_global = {k: domain_bounds[k]["global_lower"] for k in K_VALUES}
        domain_candidate = {k: domain_bounds[k]["candidate_lower"] for k in K_VALUES}
        domain_eval = evaluate_architecture_ranking(
            model,
            domain_records,
            candidates,
            k_values=K_VALUES,
            global_oracle_steps_by_k=domain_global,
            candidate_oracle_steps_by_k=domain_candidate,
        )
        domain_summary[domain] = domain_eval
    for k in K_VALUES:
        realized = summary["ranking_by_k"][k]["realized_dp_steps"]
        totals = oracle_totals[k]
        summary["ranking_by_k"][k]["candidate_oracle_capture_interval"] = _capture_interval(
            realized, totals["candidate_lower"], totals["candidate_upper"]
        )
        summary["ranking_by_k"][k]["global_oracle_capture_interval"] = _capture_interval(
            realized, totals["global_lower"], totals["global_upper"]
        )
    return {**summary, "by_domain": domain_summary}, inference


def _run_config_hash(
    *,
    dataset_hash: str,
    train_hash: str,
    dev_hash: str,
    candidate_plan_hash: str,
    architecture: str,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> str:
    return sha256_json(
        {
            "dataset_sha256": dataset_hash,
            "train_split_sha256": train_hash,
            "dev_split_sha256": dev_hash,
            "candidate_plan_sha256": candidate_plan_hash,
            "architecture": architecture,
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "runtime": runtime_manifest(),
        }
    )


def _save_model_exclusive(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"checkpoint already exists; pass --resume to reuse it: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def run_bakeoff(
    *,
    dataset_path: str,
    manifest_path: str,
    index_path: str,
    quality_attribution_gate_path: str,
    candidate_plan_path: str,
    output_dir: str,
    out_json: str,
    epochs: int = 12,
    learning_rate: float = 1e-3,
    seeds: tuple[int, ...] = NEURAL_SEEDS,
    time_limit_seconds: float = 10.0,
    resume: bool = False,
) -> dict[str, Any]:
    if not {42, 43, 44}.issubset(set(seeds)):
        raise ValueError("neural bakeoff must include seeds 42, 43, and 44")
    views, data_manifest = load_canonical_dataset(dataset_path, manifest_path)
    load_quality_attribution_gate(quality_attribution_gate_path, views)
    plan = load_candidate_plan(candidate_plan_path, views, quality_attribution_gate_path)
    candidate_plan_hash = sha256_file(candidate_plan_path)
    index = TrainOnlyAssociationIndex.load(index_path)
    validate_train_index(
        index,
        views,
        index_path=index_path,
        expected_index_sha256=plan["train_index_sha256"],
        expected_provenance_sha256=plan["train_index_provenance_sha256"],
    )
    tokenizer = load_manifest_tokenizer(data_manifest)
    strategy = RetrievalStrategy(plan["selection"]["strategy"])
    pool_size = int(plan["selection"]["pool_size"])
    generator = ConfigurableCandidateGenerator(index, tokenizer)

    train_records = [record.to_legacy_record(tokenizer) for record in views.train]
    dev_records = [record.to_legacy_record(tokenizer) for record in views.dev]
    candidates_by_prompt: dict[str, list[Any]] = {}
    train_candidates_by_prompt: dict[str, list[Any]] = {}
    dev_candidates_by_prompt: dict[str, list[Any]] = {}
    for record in train_records:
        candidates = generator.build_candidate_records(
            record, strategy=strategy, target_pool_size=pool_size
        )
        train_candidates_by_prompt[record.prompt_id] = candidates
        candidates_by_prompt[record.prompt_id] = candidates
    for record in dev_records:
        candidates = generator.build_candidate_records(
            record, strategy=strategy, target_pool_size=pool_size
        )
        dev_candidates_by_prompt[record.prompt_id] = candidates
        candidates_by_prompt[record.prompt_id] = candidates

    oracle_totals, oracle_domains = _bounds_by_split(
        dev_records,
        candidates_by_prompt,
        tokenizer,
        index.max_subtokens,
        time_limit_seconds,
    )
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if Path(out_json).exists():
        raise FileExistsError(f"bakeoff result already exists and is immutable: {out_json}")
    if not resume:
        existing = [path.name for path in target_dir.glob("*") if path.is_file()]
        if existing:
            raise FileExistsError(
                f"checkpoint directory already contains artifacts ({existing[:3]}); choose a new directory or pass --resume"
            )
    all_architectures: dict[str, Any] = {}
    training_config = {"epochs": epochs, "learning_rate": learning_rate, "neural_seeds": list(seeds)}
    cpu_rss_start = _process_rss_bytes()

    for architecture in MODEL_NAMES:
        architecture_seeds = (42,) if architecture == "Ridge" else seeds
        seed_runs: list[dict[str, Any]] = []
        for seed in architecture_seeds:
            config_hash = _run_config_hash(
                dataset_hash=views.dataset_sha256,
                train_hash=views.train_split_sha256,
                dev_hash=views.dev_split_sha256,
                candidate_plan_hash=candidate_plan_hash,
                architecture=architecture,
                seed=seed,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            stem = f"{architecture.lower()}_seed{seed}"
            checkpoint = target_dir / f"{stem}.pkl"
            run_json = target_dir / f"{stem}.json"
            have_checkpoint, have_metadata = checkpoint.exists(), run_json.exists()
            if resume and have_checkpoint and have_metadata:
                prior = validate_resume_pair(checkpoint, run_json, config_hash)
                with checkpoint.open("rb") as f:
                    model = pickle.load(f)
                dev_eval = prior["dev_evaluation"]
                inference = prior["inference_profile"]
                run = prior["seed_run"]
            else:
                if resume and (have_checkpoint or have_metadata):
                    raise FileNotFoundError(f"incomplete resume artifact pair for {stem}")
                if not resume and (have_checkpoint or have_metadata):
                    raise FileExistsError(f"seed artifacts already exist; pass --resume to reuse them: {stem}")
                _set_seed(seed)
                model = _instantiate(architecture)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                rss_before = _process_rss_bytes()
                start = time.perf_counter()
                model.fit(
                    train_records=train_records,
                    train_candidates=train_candidates_by_prompt,
                    dev_records=dev_records,
                    dev_candidates=dev_candidates_by_prompt,
                    epochs=epochs,
                    lr=learning_rate,
                )
                training_seconds = time.perf_counter() - start
                dev_eval, inference = _evaluate(
                    model, dev_records, candidates_by_prompt, oracle_totals, oracle_domains
                )
                cuda_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
                rss_after = _process_rss_bytes()
                parameter_count = int(model.get_parameter_count())
                model_bytes = int(model.get_model_size_bytes())
                _save_model_exclusive(model, checkpoint)
                k16 = dev_eval["ranking_by_k"][16]
                model_config = _architecture_configuration(model, architecture, epochs, learning_rate)
                run = {
                    "seed": seed,
                    "training_time_seconds": training_seconds,
                    "parameter_count": parameter_count,
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "serialized_model_bytes": model_bytes,
                    "cpu_rss_before_bytes": rss_before,
                    "cpu_rss_after_bytes": rss_after,
                    "cuda_peak_allocated_bytes": cuda_peak,
                    "dev_k16_realized_steps": k16["realized_dp_steps"],
                    "dev_k16_candidate_capture_interval": k16["candidate_oracle_capture_interval"],
                    "dev_k16_global_capture_interval": k16["global_oracle_capture_interval"],
                    "dev_k16_precision_at_k": k16["precision_at_k"],
                    "dev_k16_by_domain": {
                        domain: values["ranking_by_k"][16]["realized_dp_steps"]
                        for domain, values in dev_eval["by_domain"].items()
                    },
                    "inference_latency_ms": inference["latency_ms"],
                    "run_config_sha256": config_hash,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "architecture_config": model_config,
                }
                run_artifact = {
                    "schema": "predictor_v2_architecture_seed_run_v1",
                    "scope": "DEV",
                    "run_config_sha256": config_hash,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "dev_evaluation": dev_eval,
                    "inference_profile": inference,
                    "seed_run": run,
                }
                write_json_exclusive(run_json, run_artifact)
            seed_runs.append(run)
        k16_steps = [item["dev_k16_realized_steps"] for item in seed_runs]
        all_architectures[architecture] = {
            "architecture_config": seed_runs[0].get("architecture_config", {}),
            "seed_count": len(seed_runs),
            "dev_k16_steps_mean": float(statistics.mean(k16_steps)),
            "dev_k16_steps_std": float(statistics.pstdev(k16_steps)) if len(k16_steps) > 1 else 0.0,
            "dev_k16_capture_interval_mean": {
                "lower": float(statistics.mean(item["dev_k16_candidate_capture_interval"]["lower"] or 0.0 for item in seed_runs)),
                "upper": float(statistics.mean(item["dev_k16_candidate_capture_interval"]["upper"] or 0.0 for item in seed_runs)),
            },
            "seed_runs": seed_runs,
        }

    oracle_summary = {
        str(k): {key: value for key, value in oracle_totals[k].items()} for k in K_VALUES
    }
    artifact = {
        "schema": "predictor_v2_architecture_bakeoff_v1",
        "scope": "DEV",
        "is_full_dev": True,
        "final_accessed": False,
        "dataset_sha256": views.dataset_sha256,
        "manifest_sha256": views.manifest_sha256,
        "train_split_sha256": views.train_split_sha256,
        "dev_split_sha256": views.dev_split_sha256,
        "quality_attribution_gate_sha256": sha256_file(quality_attribution_gate_path),
        "candidate_plan_sha256": candidate_plan_hash,
        "candidate_config": plan["selection"],
        "training_config": training_config,
        "dev_oracle_bounds_by_k": oracle_summary,
        "dev_prompts": len(dev_records),
        "train_prompts": len(train_records),
        "architecture_metrics_use_lower_bound_denominators": True,
        "architectures": all_architectures,
        "run_manifest": make_experiment_manifest(
            stage="architecture_bakeoff",
            views=views,
            model_revision=data_manifest["model_revision"],
            tokenizer_revision=data_manifest["tokenizer_revision"],
            candidate_config=plan["selection"],
            training_config=training_config,
            output_artifacts={"json": out_json, "checkpoints": str(target_dir)},
        ),
        "process_rss_at_start_bytes": cpu_rss_start,
    }
    write_json_exclusive(out_json, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="TRAIN/DEV-only Predictor V2 architecture bakeoff")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--quality-attribution-gate", required=True)
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--output-dir", default="experiments/checkpoints/predictor_v2_canonical_bakeoff")
    parser.add_argument("--out-json", default="experiments/results/predictor_v2_architecture_bakeoff.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(NEURAL_SEEDS))
    parser.add_argument("--time-limit-seconds", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true", help="Reuse only hash-matching completed seed runs")
    args = parser.parse_args()
    run_bakeoff(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        index_path=args.index,
        quality_attribution_gate_path=args.quality_attribution_gate,
        candidate_plan_path=args.candidate_plan,
        output_dir=args.output_dir,
        out_json=args.out_json,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seeds=tuple(args.seeds),
        time_limit_seconds=args.time_limit_seconds,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
