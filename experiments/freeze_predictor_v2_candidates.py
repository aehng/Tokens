"""Freeze the candidate generator after the DEV shortlist passes live integration."""

from __future__ import annotations

import argparse
import json

from src.zip2zip.predictor_v2.canonical_dataset import load_canonical_dataset
from src.zip2zip.predictor_v2.experiment_protocol import make_candidate_freeze


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an immutable candidate-generator freeze artifact")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--quality-attribution-gate", required=True)
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--live-integration-gate", required=True)
    parser.add_argument("--integration-subset")
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--out", default="docs/predictor_v2_candidate_generator_freeze.json")
    args = parser.parse_args()
    views, _ = load_canonical_dataset(args.dataset, args.manifest)
    freeze = make_candidate_freeze(
        candidate_plan_path=args.candidate_plan,
        quality_attribution_gate_path=args.quality_attribution_gate,
        shortlist_path=args.shortlist,
        live_integration_gate_path=args.live_integration_gate,
        integration_subset_path=args.integration_subset,
        rationale=args.rationale,
        views=views,
        output_path=args.out,
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
