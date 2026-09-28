"""Freeze an exact DEV-only prompt subset before small live integration runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.zip2zip.predictor_v2.canonical_dataset import load_canonical_dataset
from src.zip2zip.predictor_v2.experiment_protocol import make_integration_subset_freeze


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze exact DEV prompt IDs for live integration")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-ids-json", required=True, help="JSON array of DEV prompt IDs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    views, _ = load_canonical_dataset(args.dataset, args.manifest)
    prompt_ids = json.loads(Path(args.prompt_ids_json).read_text(encoding="utf-8"))
    result = make_integration_subset_freeze(prompt_ids=prompt_ids, views=views, output_path=args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
