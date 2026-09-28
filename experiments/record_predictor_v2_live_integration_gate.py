"""Validate frozen-subset/full-DEV live end-to-end evidence and human gate review."""

from __future__ import annotations

import argparse
import json

from src.zip2zip.predictor_v2.canonical_dataset import load_canonical_dataset
from src.zip2zip.predictor_v2.experiment_protocol import make_live_integration_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a passed live integration gate")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--quality-attribution-gate", required=True)
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--integration-subset")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    views, _ = load_canonical_dataset(args.dataset, args.manifest)
    result = make_live_integration_gate(evidence_path=args.evidence, review_path=args.review, quality_attribution_gate_path=args.quality_attribution_gate, candidate_plan_path=args.candidate_plan, shortlist_path=args.shortlist, subset_path=args.integration_subset, views=views, output_path=args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
