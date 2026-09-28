"""One-time FINAL evaluation guarded by live and offline freezes plus explicit approval."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    claim_final_evaluation,
    issue_final_access_permit,
    load_architecture_freeze,
    load_architecture_shortlist,
    load_candidate_freeze,
    load_candidate_plan,
    load_live_integration_gate,
    load_quality_attribution_gate,
    make_experiment_manifest,
    sha256_file,
    utc_now,
    write_json_exclusive,
)
from src.zip2zip.predictor_v2.metrics import evaluate_architecture_ranking
from src.zip2zip.predictor_v2.oracle_candidate_pool import CandidatePoolOracle
from src.zip2zip.predictor_v2.oracle_global import GlobalOccurrenceOracle
from src.zip2zip.predictor_v2.train_index import validate_train_index


K_VALUES = (8, 16, 32)


def _capture_interval(realized: int, lower: int, upper: int) -> dict[str, float | None]:
    if upper <= 0:
        return {"lower": None, "upper": None}
    return {"lower": realized / upper, "upper": min(1.0, realized / lower) if lower > 0 else 1.0}


def run_final_once(
    *,
    dataset_path: str,
    manifest_path: str,
    index_path: str,
    quality_attribution_gate_path: str,
    candidate_plan_path: str,
    candidate_freeze_path: str,
    shortlist_path: str,
    live_integration_gate_path: str,
    architecture_freeze_path: str,
    integration_subset_path: str | None,
    claim_path: str,
    result_path: str,
    allow_final_eval: bool,
    time_limit_seconds: float = 10.0,
) -> dict[str, Any]:
    views, data_manifest = load_canonical_dataset(dataset_path, manifest_path)
    permit = issue_final_access_permit(
        allow_final_eval=allow_final_eval,
        quality_attribution_gate_path=quality_attribution_gate_path,
        candidate_plan_path=candidate_plan_path,
        candidate_freeze_path=candidate_freeze_path,
        shortlist_path=shortlist_path,
        live_integration_gate_path=live_integration_gate_path,
        architecture_freeze_path=architecture_freeze_path,
        integration_subset_path=integration_subset_path,
        views=views,
    )
    quality_gate = load_quality_attribution_gate(quality_attribution_gate_path, views)
    load_candidate_plan(candidate_plan_path, views, quality_attribution_gate_path)
    candidate_freeze = load_candidate_freeze(candidate_freeze_path, views)
    shortlist = load_architecture_shortlist(shortlist_path, views, quality_attribution_gate_path, candidate_plan_path)
    live_gate = load_live_integration_gate(live_integration_gate_path, views, quality_attribution_gate_path, candidate_plan_path, shortlist_path, integration_subset_path)
    architecture_freeze = load_architecture_freeze(
        architecture_freeze_path, views, candidate_freeze_path,
        quality_attribution_gate_path=quality_attribution_gate_path,
        candidate_plan_path=candidate_plan_path,
        shortlist_path=shortlist_path,
        live_integration_gate_path=live_integration_gate_path,
        integration_subset_path=integration_subset_path,
    )
    tokenizer = load_manifest_tokenizer(data_manifest)
    index = TrainOnlyAssociationIndex.load(index_path)
    validate_train_index(
        index,
        views,
        index_path=index_path,
        expected_index_sha256=candidate_freeze["train_index_sha256"],
        expected_provenance_sha256=candidate_freeze["train_index_provenance_sha256"],
    )
    generator = ConfigurableCandidateGenerator(index, tokenizer)
    candidate_selection = candidate_freeze["selection"]
    strategy = RetrievalStrategy(candidate_selection["strategy"])
    pool_size = int(candidate_selection["pool_size"])
    architecture_selection = architecture_freeze["selection"]
    checkpoint_path = Path(architecture_selection["checkpoint_path"])
    if sha256_file(checkpoint_path) != architecture_selection["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint changed after freeze validation")
    with checkpoint_path.open("rb") as f:
        model = pickle.load(f)
    result_file = Path(result_path)
    claim_file = Path(claim_path)
    if result_file.exists():
        raise FileExistsError(f"FINAL result already exists and is immutable: {result_file}")
    if claim_file.exists():
        raise FileExistsError(f"FINAL evaluation was already claimed: {claim_file}")
    claim_final_evaluation(
        claim_path=claim_file,
        result_path=result_file,
        permit=permit,
    )

    # FINAL becomes available only after the durable one-time claim is written.
    final_canonical = views.open_final(permit)
    final_records = [record.to_legacy_record(tokenizer) for record in final_canonical]
    candidates_by_prompt = {
        record.prompt_id: generator.build_candidate_records(
            record,
            strategy=strategy,
            target_pool_size=pool_size,
        )
        for record in final_records
    }

    global_oracle = GlobalOccurrenceOracle(min_len=2, max_len=index.max_subtokens, time_limit_seconds=time_limit_seconds)
    pool_oracle = CandidatePoolOracle(tokenizer=tokenizer, time_limit_seconds=time_limit_seconds)
    bounds: dict[int, dict[str, Any]] = {
        k: {"global_lower": 0, "global_upper": 0, "candidate_lower": 0, "candidate_upper": 0, "global_exact": 0, "candidate_exact": 0}
        for k in K_VALUES
    }
    bounds_by_domain: dict[str, dict[int, dict[str, Any]]] = defaultdict(
        lambda: {
            k: {"global_lower": 0, "global_upper": 0, "candidate_lower": 0, "candidate_upper": 0, "global_exact": 0, "candidate_exact": 0}
            for k in K_VALUES
        }
    )
    for record in final_records:
        for k in K_VALUES:
            global_result = global_oracle.solve(record.continuation_token_ids, k=k, tokenizer=tokenizer)
            candidate_result = pool_oracle.solve(
                candidates_by_prompt[record.prompt_id],
                record.continuation_token_ids,
                k=k,
                global_oracle_steps=global_result.steps_saved_lower_bound,
            )
            for dest in (bounds[k], bounds_by_domain[record.domain][k]):
                dest["global_lower"] += global_result.steps_saved_lower_bound
                dest["global_upper"] += global_result.steps_saved_upper_bound
                dest["candidate_lower"] += candidate_result.steps_saved_lower_bound
                dest["candidate_upper"] += candidate_result.steps_saved_upper_bound
                dest["global_exact"] += int(global_result.is_exact)
                dest["candidate_exact"] += int(candidate_result.is_exact)

    global_lower = {k: bounds[k]["global_lower"] for k in K_VALUES}
    candidate_lower = {k: bounds[k]["candidate_lower"] for k in K_VALUES}
    aggregate = evaluate_architecture_ranking(
        model,
        final_records,
        candidates_by_prompt,
        k_values=K_VALUES,
        global_oracle_steps_by_k=global_lower,
        candidate_oracle_steps_by_k=candidate_lower,
    )
    by_domain: dict[str, Any] = {}
    for domain in sorted(bounds_by_domain):
        records = [record for record in final_records if record.domain == domain]
        domain_bounds = bounds_by_domain[domain]
        domain_eval = evaluate_architecture_ranking(
            model,
            records,
            candidates_by_prompt,
            k_values=K_VALUES,
            global_oracle_steps_by_k={k: domain_bounds[k]["global_lower"] for k in K_VALUES},
            candidate_oracle_steps_by_k={k: domain_bounds[k]["candidate_lower"] for k in K_VALUES},
        )
        for k in K_VALUES:
            realized = domain_eval["ranking_by_k"][k]["realized_dp_steps"]
            domain_eval["ranking_by_k"][k]["candidate_oracle_capture_interval"] = _capture_interval(
                realized, domain_bounds[k]["candidate_lower"], domain_bounds[k]["candidate_upper"]
            )
            domain_eval["ranking_by_k"][k]["global_oracle_capture_interval"] = _capture_interval(
                realized, domain_bounds[k]["global_lower"], domain_bounds[k]["global_upper"]
            )
        by_domain[domain] = domain_eval
    for k in K_VALUES:
        realized = aggregate["ranking_by_k"][k]["realized_dp_steps"]
        aggregate["ranking_by_k"][k]["candidate_oracle_capture_interval"] = _capture_interval(
            realized, bounds[k]["candidate_lower"], bounds[k]["candidate_upper"]
        )
        aggregate["ranking_by_k"][k]["global_oracle_capture_interval"] = _capture_interval(
            realized, bounds[k]["global_lower"], bounds[k]["global_upper"]
        )

    artifact = {
        "schema": "predictor_v2_final_result_v1",
        "status": "FINAL_EVALUATED_ONCE",
        "scope": "FINAL",
        "created_at_utc": utc_now(),
        "dataset_sha256": views.dataset_sha256,
        "manifest_sha256": views.manifest_sha256,
        "final_split_sha256": views.final_split_sha256,
        "candidate_freeze_sha256": sha256_file(candidate_freeze_path),
        "architecture_freeze_sha256": sha256_file(architecture_freeze_path),
        "quality_attribution_gate_sha256": sha256_file(quality_attribution_gate_path),
        "candidate_plan_sha256": sha256_file(candidate_plan_path),
        "architecture_shortlist_sha256": sha256_file(shortlist_path),
        "live_integration_gate_sha256": sha256_file(live_integration_gate_path),
        "integration_subset_sha256": sha256_file(integration_subset_path) if integration_subset_path else None,
        "quality_attribution_status": quality_gate["status"],
        "gate_state": {
            "candidate_generator_frozen": True,
            "offline_architecture_shortlist_complete": True,
            "live_integration_gate_passed": live_gate.get("status") == "passed",
            "architecture_frozen": True,
            "final_evaluated": True,
        },
        "candidate_config": candidate_selection,
        "architecture_config": architecture_selection,
        "final_prompts": len(final_records),
        "final_oracle_bounds_by_k": {str(k): bounds[k] for k in K_VALUES},
        "aggregate_metrics": aggregate,
        "domain_metrics": by_domain,
        "run_manifest": make_experiment_manifest(
            stage="final_evaluation",
            views=views,
            model_revision=data_manifest["model_revision"],
            tokenizer_revision=data_manifest["tokenizer_revision"],
            candidate_config=candidate_selection,
            architecture_config=architecture_selection,
            seed=architecture_selection["seed"],
            output_artifacts={"result": str(result_file), "claim": str(claim_file)},
        ),
        "selection_or_tuning_after_final": False,
    }
    write_json_exclusive(result_file, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-time Predictor V2 FINAL evaluation")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--quality-attribution-gate", required=True)
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--candidate-freeze", required=True)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--live-integration-gate", required=True)
    parser.add_argument("--architecture-freeze", required=True)
    parser.add_argument("--integration-subset")
    parser.add_argument("--claim", default="experiments/results/predictor_v2_final_eval.claim.json")
    parser.add_argument("--out", default="experiments/results/predictor_v2_final_result.json")
    parser.add_argument("--allow-final-eval", action="store_true", required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=10.0)
    args = parser.parse_args()
    artifact = run_final_once(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        index_path=args.index,
        quality_attribution_gate_path=args.quality_attribution_gate,
        candidate_plan_path=args.candidate_plan,
        candidate_freeze_path=args.candidate_freeze,
        shortlist_path=args.shortlist,
        live_integration_gate_path=args.live_integration_gate,
        architecture_freeze_path=args.architecture_freeze,
        integration_subset_path=args.integration_subset,
        claim_path=args.claim,
        result_path=args.out,
        allow_final_eval=args.allow_final_eval,
        time_limit_seconds=args.time_limit_seconds,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
