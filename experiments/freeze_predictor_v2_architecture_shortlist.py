"""Freeze the one- or two-candidate DEV architecture shortlist for live checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.zip2zip.predictor_v2.canonical_dataset import load_canonical_dataset
from src.zip2zip.predictor_v2.experiment_protocol import make_architecture_shortlist


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the DEV architecture shortlist")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--quality-attribution-gate", required=True)
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--bakeoff", required=True)
    parser.add_argument("--selection-json", required=True, help="JSON array of architecture/seed objects (one or two)")
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    views, _ = load_canonical_dataset(args.dataset, args.manifest)
    selections = json.loads(Path(args.selection_json).read_text(encoding="utf-8"))
    result = make_architecture_shortlist(bakeoff_path=args.bakeoff, quality_attribution_gate_path=args.quality_attribution_gate, candidate_plan_path=args.candidate_plan, selections=selections, rationale=args.rationale, views=views, output_path=args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
