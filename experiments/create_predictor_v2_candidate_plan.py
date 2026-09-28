"""Propose the DEV-benchmarked candidate configuration after attribution."""

from __future__ import annotations

import argparse
import json

from src.zip2zip.predictor_v2.canonical_dataset import load_canonical_dataset
from src.zip2zip.predictor_v2.experiment_protocol import make_candidate_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a provisional candidate plan; this is not a freeze")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--quality-attribution-gate", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--pool-size", required=True, type=int)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    views, _ = load_canonical_dataset(args.dataset, args.manifest)
    result = make_candidate_plan(benchmark_path=args.benchmark, strategy=args.strategy, pool_size=args.pool_size, rationale=args.rationale, quality_attribution_gate_path=args.quality_attribution_gate, views=views, output_path=args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
