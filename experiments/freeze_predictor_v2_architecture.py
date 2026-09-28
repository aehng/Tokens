"""Freeze a DEV-selected Predictor V2 architecture and verified checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.zip2zip.predictor_v2.canonical_dataset import load_canonical_dataset
from src.zip2zip.predictor_v2.experiment_protocol import make_architecture_freeze


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an immutable architecture freeze artifact")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bakeoff", required=True)
    parser.add_argument("--candidate-freeze", required=True)
    parser.add_argument("--quality-attribution-gate", required=True)
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--live-integration-gate", required=True)
    parser.add_argument("--integration-subset")
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--review-json", required=True, help="Passed multiseed/domain/latency/Ridge review notes")
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--out", default="docs/predictor_v2_architecture_freeze.json")
    args = parser.parse_args()
    views, _ = load_canonical_dataset(args.dataset, args.manifest)
    freeze = make_architecture_freeze(
        bakeoff_path=args.bakeoff,
        architecture=args.architecture,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        candidate_freeze_path=args.candidate_freeze,
        quality_attribution_gate_path=args.quality_attribution_gate,
        candidate_plan_path=args.candidate_plan,
        shortlist_path=args.shortlist,
        live_integration_gate_path=args.live_integration_gate,
        integration_subset_path=args.integration_subset,
        review=json.loads(Path(args.review_json).read_text(encoding="utf-8")),
        rationale=args.rationale,
        views=views,
        output_path=args.out,
    )
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
