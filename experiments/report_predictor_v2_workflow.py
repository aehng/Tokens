"""Summarize Predictor V2 gates in the approved attribution-first sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.zip2zip.predictor_v2.canonical_dataset import load_canonical_dataset, sha256_json
from src.zip2zip.predictor_v2.experiment_protocol import (
    load_architecture_freeze,
    load_architecture_shortlist,
    load_candidate_freeze,
    load_candidate_plan,
    load_live_integration_gate,
    load_quality_attribution_gate,
    read_json_object,
    sha256_file,
)


GATE_KEYS = (
    "candidate_generator_frozen",
    "offline_architecture_shortlist_complete",
    "live_integration_gate_passed",
    "architecture_frozen",
    "final_evaluated",
)


def _read_optional(path: str | None) -> dict[str, Any] | None:
    if not path or not Path(path).is_file():
        return None
    return read_json_object(path, path)


def _bottleneck_assessment(status: str, attribution: dict[str, Any] | None) -> str:
    if status == "not-yet-run":
        return "not yet assessed"
    if attribution is not None and status == "passed":
        return f"confirmed: {attribution.get('primary_bottleneck')}"
    if attribution is not None and status == "redirected":
        return f"not confirmed; redirected to {attribution.get('redirect_to')}"
    return f"assessment {status}"


def build_report(
    *, dataset_path: str, manifest_path: str,
    candidate_benchmark_path: str, quality_attribution_gate_path: str,
    sourcebook_benchmark_path: str | None = None,
    candidate_plan_path: str, architecture_bakeoff_path: str,
    architecture_shortlist_path: str, integration_subset_path: str | None,
    live_integration_gate_path: str, candidate_freeze_path: str,
    architecture_freeze_path: str, final_claim_path: str, final_result_path: str,
    historical_path: str | None = None,
) -> tuple[dict[str, Any], str]:
    views, manifest = load_canonical_dataset(dataset_path, manifest_path)
    candidate_benchmark = _read_optional(candidate_benchmark_path)
    sourcebook_benchmark = _read_optional(sourcebook_benchmark_path)
    attribution_raw = _read_optional(quality_attribution_gate_path)
    plan_raw = _read_optional(candidate_plan_path)
    bakeoff = _read_optional(architecture_bakeoff_path)
    shortlist_raw = _read_optional(architecture_shortlist_path)
    subset_raw = _read_optional(integration_subset_path)
    live_raw = _read_optional(live_integration_gate_path)
    candidate_freeze_raw = _read_optional(candidate_freeze_path)
    architecture_freeze_raw = _read_optional(architecture_freeze_path)
    final_claim = _read_optional(final_claim_path)
    final_result = _read_optional(final_result_path)
    historical = _read_optional(historical_path)
    blockers: list[str] = []

    def absent(message: str) -> str:
        blockers.append(message)
        return "not-yet-run"

    if candidate_benchmark is None:
        candidate_benchmark_status = absent("Full DEV candidate-recall and oracle benchmark has not been run.")
    elif candidate_benchmark.get("scope") != "DEV" or candidate_benchmark.get("is_full_dev") is not True or candidate_benchmark.get("dataset_sha256") != views.dataset_sha256 or candidate_benchmark.get("dev_split_sha256") != views.dev_split_sha256:
        candidate_benchmark_status = "invalid-provenance"
        blockers.append("Candidate-recall benchmark is not a valid full DEV artifact for this canonical dataset.")
    else:
        candidate_benchmark_status = "measured"

    if sourcebook_benchmark is None:
        sourcebook_benchmark_status = absent("The full DEV Phi-only/external/hybrid sourcebook comparison has not been run.")
    elif sourcebook_benchmark.get("scope") != "DEV" or sourcebook_benchmark.get("is_full_dev") is not True or sourcebook_benchmark.get("final_accessed") is not False or sourcebook_benchmark.get("dataset_sha256") != views.dataset_sha256 or sourcebook_benchmark.get("dev_split_sha256") != views.dev_split_sha256 or not {"phi_only", "external_only", "hybrid"}.issubset(sourcebook_benchmark.get("systems", {})) or sourcebook_benchmark.get("benchmark_sha256") != sha256_json({key: value for key, value in sourcebook_benchmark.items() if key != "benchmark_sha256"}):
        sourcebook_benchmark_status = "invalid-provenance"
        blockers.append("Sourcebook comparison is not a valid full DEV-only comparison for the current canonical dataset.")
    else:
        required_sizes = {"256", "512", "1024"}
        if any(not required_sizes.issubset(sourcebook_benchmark["systems"][name]) for name in ("phi_only", "external_only", "hybrid")):
            sourcebook_benchmark_status = "incomplete"
            blockers.append("Sourcebook comparison is missing one or more required candidate pool sizes.")
        else:
            sourcebook_benchmark_status = "measured"
    sourcebook_summary: dict[str, Any] | None = None
    if sourcebook_benchmark_status == "measured" and sourcebook_benchmark is not None:
        sourcebook_summary = {
            "manifest_sha256": sourcebook_benchmark.get("sourcebook", {}).get("manifest_sha256"),
            "database_sha256": sourcebook_benchmark.get("sourcebook", {}).get("database_sha256"),
            "database_bytes": sourcebook_benchmark.get("sourcebook", {}).get("database_bytes"),
            "process_rss_delta_bytes": sourcebook_benchmark.get("sourcebook", {}).get("process_rss_delta_bytes"),
            "retrieval_index_terms": sourcebook_benchmark.get("sourcebook", {}).get("retrieval_index_terms"),
            "source_examples": sourcebook_benchmark.get("sourcebook", {}).get("source_examples"),
            "unique_phrases_by_phi_token_length": sourcebook_benchmark.get("sourcebook", {}).get("unique_phrases_by_phi_token_length"),
            "systems": sourcebook_benchmark.get("systems"),
        }

    attribution = None
    if attribution_raw is None:
        attribution_status = absent("Broader live Vanilla/Predictive Phi attribution has not been recorded.")
    else:
        try:
            attribution = load_quality_attribution_gate(quality_attribution_gate_path, views, require_predictor_bottleneck=False)
            attribution_status = attribution["status"]
            if attribution_status == "redirected":
                blockers.append(f"Attribution redirects work to {attribution.get('redirect_to')}; Predictor V2 training remains gated off.")
        except Exception as exc:
            attribution_status = "invalid"
            blockers.append(f"Quality attribution gate is invalid: {exc}")

    plan = None
    if plan_raw is None:
        plan_status = absent("No provisional candidate plan has been selected from DEV evidence.")
    elif attribution is None or attribution.get("status") != "passed":
        plan_status = "blocked-by-attribution"
        blockers.append("Candidate planning and training require attribution to confirm a Predictor/codebook bottleneck.")
    else:
        try:
            plan = load_candidate_plan(candidate_plan_path, views, quality_attribution_gate_path)
            plan_status = "proposed"
        except Exception as exc:
            plan_status = "invalid"
            blockers.append(f"Candidate plan is invalid: {exc}")

    if bakeoff is None:
        bakeoff_status = absent("TRAIN-only Predictor V2 training and DEV comparison have not been run.")
    elif plan is None or bakeoff.get("scope") != "DEV" or bakeoff.get("is_full_dev") is not True or bakeoff.get("final_accessed") is not False or bakeoff.get("dataset_sha256") != views.dataset_sha256 or bakeoff.get("candidate_plan_sha256") != sha256_file(candidate_plan_path):
        bakeoff_status = "invalid-provenance"
        blockers.append("Architecture bakeoff is not a full DEV-only result bound to the current candidate plan.")
    else:
        bakeoff_status = "measured"

    shortlist = None
    if shortlist_raw is None:
        shortlist_status = absent("A one- or two-candidate DEV architecture shortlist has not been frozen.")
    elif plan is None:
        shortlist_status = "blocked-by-attribution"
        blockers.append("Architecture shortlist requires a valid quality gate and candidate plan.")
    else:
        try:
            shortlist = load_architecture_shortlist(architecture_shortlist_path, views, quality_attribution_gate_path, candidate_plan_path)
            shortlist_status = "complete"
        except Exception as exc:
            shortlist_status = "invalid"
            blockers.append(f"Architecture shortlist is invalid: {exc}")

    subset_status = "full-dev"
    if integration_subset_path:
        if subset_raw is None:
            subset_status = absent("The requested integration subset freeze is missing.")
        elif subset_raw.get("dataset_sha256") != views.dataset_sha256 or subset_raw.get("dev_split_sha256") != views.dev_split_sha256:
            subset_status = "invalid-provenance"
            blockers.append("Integration subset is not bound to the canonical DEV split.")
        else:
            subset_status = "frozen"

    live_gate = None
    if live_raw is None:
        live_status = absent("Small live end-to-end validation of the shortlisted candidates has not passed.")
    elif shortlist is None or plan is None:
        live_status = "blocked-by-upstream-gates"
        blockers.append("Live integration gate requires a complete DEV shortlist and candidate plan.")
    else:
        try:
            live_gate = load_live_integration_gate(live_integration_gate_path, views, quality_attribution_gate_path, candidate_plan_path, architecture_shortlist_path, integration_subset_path)
            live_status = "passed"
        except Exception as exc:
            live_status = "invalid"
            blockers.append(f"Live integration gate is invalid: {exc}")

    candidate_freeze = None
    if candidate_freeze_raw is None:
        candidate_freeze_status = absent("Candidate generator cannot freeze until the shortlisted system passes live integration.")
    elif live_gate is None:
        candidate_freeze_status = "blocked-by-live-gate"
        blockers.append("Candidate generator freeze is not accepted before a passed live integration gate.")
    else:
        try:
            candidate_freeze = load_candidate_freeze(candidate_freeze_path, views)
            if candidate_freeze.get("candidate_plan_sha256") != sha256_file(candidate_plan_path) or candidate_freeze.get("live_integration_gate_sha256") != sha256_file(live_integration_gate_path):
                raise ValueError("candidate freeze is bound to different upstream evidence")
            candidate_freeze_status = "frozen"
        except Exception as exc:
            candidate_freeze_status = "invalid"
            blockers.append(f"Candidate freeze is invalid: {exc}")

    architecture_freeze = None
    if architecture_freeze_raw is None:
        architecture_freeze_status = absent("Architecture cannot freeze until the live integration gate passes.")
    elif candidate_freeze is None or live_gate is None or plan is None or shortlist is None:
        architecture_freeze_status = "blocked-by-upstream-gates"
        blockers.append("Architecture freeze requires a passed live gate, candidate freeze, and complete shortlist.")
    else:
        try:
            architecture_freeze = load_architecture_freeze(
                architecture_freeze_path, views, candidate_freeze_path,
                quality_attribution_gate_path=quality_attribution_gate_path,
                candidate_plan_path=candidate_plan_path,
                shortlist_path=architecture_shortlist_path,
                live_integration_gate_path=live_integration_gate_path,
                integration_subset_path=integration_subset_path,
            )
            architecture_freeze_status = "frozen"
        except Exception as exc:
            architecture_freeze_status = "invalid"
            blockers.append(f"Architecture freeze is invalid: {exc}")

    gate_state = {
        "candidate_generator_frozen": candidate_freeze_status == "frozen",
        "offline_architecture_shortlist_complete": shortlist_status == "complete",
        "live_integration_gate_passed": live_status == "passed",
        "architecture_frozen": architecture_freeze_status == "frozen",
        "final_evaluated": False,
    }
    if final_result is not None:
        expected_hashes = {
            "dataset_sha256": views.dataset_sha256,
            "final_split_sha256": views.final_split_sha256,
            "quality_attribution_gate_sha256": sha256_file(quality_attribution_gate_path),
            "candidate_plan_sha256": sha256_file(candidate_plan_path),
            "candidate_freeze_sha256": sha256_file(candidate_freeze_path),
            "architecture_shortlist_sha256": sha256_file(architecture_shortlist_path),
            "live_integration_gate_sha256": sha256_file(live_integration_gate_path),
            "architecture_freeze_sha256": sha256_file(architecture_freeze_path),
        }
        if integration_subset_path:
            expected_hashes["integration_subset_sha256"] = sha256_file(integration_subset_path)
        result_state = final_result.get("gate_state", {})
        if architecture_freeze is None or final_result.get("scope") != "FINAL" or final_result.get("status") != "FINAL_EVALUATED_ONCE" or any(final_result.get(key) != value for key, value in expected_hashes.items()) or any(result_state.get(key) is not True for key in GATE_KEYS):
            final_status = "invalid-provenance"
            blockers.append("FINAL result is missing required freeze/live gate bindings or does not match this dataset.")
        else:
            final_status = "measured-once"
            gate_state["final_evaluated"] = True
    elif final_claim is not None:
        final_status = "claimed-without-result"
        blockers.append("FINAL attempt was claimed but has no immutable result; do not rerun silently.")
    else:
        final_status = "not-yet-run"
        if not all(gate_state[key] for key in GATE_KEYS[:-1]):
            blockers.append("FINAL remains unavailable until the candidate, shortlist, live, and architecture gates pass.")

    report = {
        "schema": "predictor_v2_workflow_report_v2",
        "dataset": {"status": "validated", "dataset_sha256": views.dataset_sha256, "manifest_sha256": views.manifest_sha256, "provenance_sha256": views.provenance_sha256, "model_revision": manifest["model_revision"], "tokenizer_revision": manifest["tokenizer_revision"], "counts": {"TRAIN": len(views.train), "DEV": len(views.dev), "FINAL": len(views.final_ids)}},
        "gate_state": gate_state,
        "stages": {
            "dev_sourcebook_comparison": {"status": sourcebook_benchmark_status, "path": sourcebook_benchmark_path, "summary": sourcebook_summary},
            "candidate_recall_oracle": {"status": candidate_benchmark_status, "path": candidate_benchmark_path},
            "broader_live_phi_attribution": {"status": attribution_status, "path": quality_attribution_gate_path, "redirect_to": attribution.get("redirect_to") if attribution else None},
            "candidate_plan": {"status": plan_status, "path": candidate_plan_path},
            "train_dev_architecture_bakeoff": {"status": bakeoff_status, "path": architecture_bakeoff_path},
            "offline_architecture_shortlist": {"status": shortlist_status, "path": architecture_shortlist_path},
            "integration_subset": {"status": subset_status, "path": integration_subset_path},
            "live_integration_gate": {"status": live_status, "path": live_integration_gate_path},
            "candidate_generator_freeze": {"status": candidate_freeze_status, "path": candidate_freeze_path},
            "architecture_freeze": {"status": architecture_freeze_status, "path": architecture_freeze_path},
            "final_evaluation": {"status": final_status, "path": final_result_path},
        },
        "historical_legacy_result": {"status": "historical-only" if historical is not None else "not-included", "path": historical_path, "used_for_selection": False},
        "selection_policy": {"final_participates_in_architecture_selection": False, "final_requires_explicit_allow_flag": True, "final_attempt_limit": 1},
        "blockers": blockers,
    }
    lines = [
        "# Predictor V2 Training and Evaluation Status", "",
        f"Canonical dataset: **validated** ({len(views.train)} TRAIN / {len(views.dev)} DEV / {len(views.final_ids)} FINAL)",
        f"Predictor/codebook bottleneck: **{_bottleneck_assessment(attribution_status, attribution)}**",
        "", "| Stage | Status | Artifact |", "|---|---|---|",
    ]
    for stage, details in report["stages"].items():
        lines.append(f"| {stage.replace('_', ' ').title()} | {details['status']} | `{details['path']}` |")
    lines += ["", "## Offline sourcebook comparison", ""]
    if sourcebook_summary is None:
        lines.append("The full DEV Phi-only/external-only/hybrid sourcebook comparison is not recorded yet.")
    else:
        lines += ["| Candidate source | Pool | K=32 capture interval | p50/p90/p99 retrieval (ms) | Code | Reasoning | Instruction |", "|---|---:|---:|---:|---:|---:|---:|"]
        for system, sizes in sourcebook_summary["systems"].items():
            for size, metrics in sizes.items():
                interval = metrics["candidate_oracle_capture_interval"]
                capture = "n/a" if interval.get("lower") is None else f"{interval['lower']:.3f}–{interval['upper']:.3f}"
                latency = metrics["retrieval_latency"]
                latency_text = f"{latency['p50_ms']:.3f}/{latency['p90_ms']:.3f}/{latency['p99_ms']:.3f}"
                domains = [metrics["domains"][domain]["capture_interval"] for domain in ("code", "reasoning", "instruction")]
                domain_text = ["n/a" if item.get("lower") is None else f"{item['lower']:.3f}–{item['upper']:.3f}" for item in domains]
                lines.append(f"| {system} | {size} | {capture} | {latency_text} | {domain_text[0]} | {domain_text[1]} | {domain_text[2]} |")
    lines += ["", "## Machine-readable gates", "", "```json", json.dumps(gate_state, indent=2, sort_keys=True), "```", "", "## Pending steps", ""]
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- None.")
    lines += ["", "FINAL is excluded from candidate and architecture selection and remains a one-time evaluation behind `--allow-final-eval`."]
    return report, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Predictor V2 evidence and gate state")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-benchmark", default="experiments/results/predictor_v2_candidate_recall.json")
    parser.add_argument("--sourcebook-benchmark", default="experiments/results/predictor_v2_sourcebook_dev.json")
    parser.add_argument("--quality-attribution-gate", default="experiments/results/predictor_v2_quality_attribution_gate.json")
    parser.add_argument("--candidate-plan", default="experiments/results/predictor_v2_candidate_plan.json")
    parser.add_argument("--architecture-bakeoff", default="experiments/results/predictor_v2_architecture_bakeoff.json")
    parser.add_argument("--shortlist", default="experiments/results/predictor_v2_architecture_shortlist.json")
    parser.add_argument("--integration-subset")
    parser.add_argument("--live-integration-gate", default="experiments/results/predictor_v2_live_integration_gate.json")
    parser.add_argument("--candidate-freeze", default="docs/predictor_v2_candidate_generator_freeze.json")
    parser.add_argument("--architecture-freeze", default="docs/predictor_v2_architecture_freeze.json")
    parser.add_argument("--final-claim", default="experiments/results/predictor_v2_final_eval.claim.json")
    parser.add_argument("--final-result", default="experiments/results/predictor_v2_final_result.json")
    parser.add_argument("--historical", default="docs/predictor_v2_bakeoff_results.json")
    parser.add_argument("--out-json", default="experiments/results/predictor_v2_workflow_report.json")
    parser.add_argument("--out-md", default="experiments/results/PREDICTOR_V2_WORKFLOW_REPORT.md")
    args = parser.parse_args()
    report, markdown = build_report(
        dataset_path=args.dataset, manifest_path=args.manifest,
        candidate_benchmark_path=args.candidate_benchmark,
        sourcebook_benchmark_path=args.sourcebook_benchmark,
        quality_attribution_gate_path=args.quality_attribution_gate,
        candidate_plan_path=args.candidate_plan,
        architecture_bakeoff_path=args.architecture_bakeoff,
        architecture_shortlist_path=args.shortlist,
        integration_subset_path=args.integration_subset,
        live_integration_gate_path=args.live_integration_gate,
        candidate_freeze_path=args.candidate_freeze,
        architecture_freeze_path=args.architecture_freeze,
        final_claim_path=args.final_claim, final_result_path=args.final_result,
        historical_path=args.historical,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
