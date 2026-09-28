"""Record the broader live Phi attribution gate, including any redirect decision."""

from __future__ import annotations

import argparse
import json

from src.zip2zip.predictor_v2.canonical_dataset import load_canonical_dataset
from src.zip2zip.predictor_v2.experiment_protocol import make_quality_attribution_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Record matched live Vanilla/Predictive Phi attribution")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--primary-bottleneck", required=True, choices=("candidate_generation", "candidate_ranking", "predictor", "codebook", "h_emission", "representation", "continuation_state", "eos", "serving", "other"))
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    views, _ = load_canonical_dataset(args.dataset, args.manifest)
    result = make_quality_attribution_gate(evidence_path=args.evidence, primary_bottleneck=args.primary_bottleneck, rationale=args.rationale, views=views, output_path=args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
