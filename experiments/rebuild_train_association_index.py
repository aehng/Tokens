"""Build and save a provenance-bound index from canonical TRAIN records only."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.zip2zip.predictor_v2.canonical_dataset import (
    load_canonical_dataset,
    load_manifest_tokenizer,
)
from src.zip2zip.predictor_v2.experiment_protocol import write_json_exclusive
from src.zip2zip.predictor_v2.train_index import build_index_from_views


def build_train_association_index(
    manifest_path: str,
    continuations_path: str,
    max_subtokens: int = 4,
    min_cooccurrence: int = 2,
    top_candidates_per_token: int = 64,
) -> tuple[Any, dict[str, Any]]:
    """Validate the complete artifact, then pass only TRAIN to index construction."""
    views, manifest = load_canonical_dataset(continuations_path, manifest_path)
    tokenizer = load_manifest_tokenizer(manifest)
    index, summary = build_index_from_views(
        views,
        tokenizer=tokenizer,
        config={
            "max_subtokens": max_subtokens,
            "min_cooccurrence": min_cooccurrence,
            "top_candidates_per_token": top_candidates_per_token,
        },
    )
    return index, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a canonical TRAIN-only candidate index")
    parser.add_argument("--manifest", required=True, help="Canonical dataset provenance/split manifest")
    parser.add_argument("--continuations", required=True, help="Canonical continuation JSONL")
    parser.add_argument("--out-pkl", default="experiments/checkpoints/train_only_association_index.pkl")
    parser.add_argument("--out-summary", default="docs/train_only_association_index_summary.json")
    parser.add_argument("--max-subtokens", type=int, default=4)
    parser.add_argument("--min-cooccurrence", type=int, default=2)
    parser.add_argument("--top-candidates-per-token", type=int, default=64)
    args = parser.parse_args()

    if Path(args.out_pkl).exists() or Path(args.out_summary).exists():
        raise FileExistsError("index outputs already exist; choose new output paths to preserve prior evidence")

    index, summary = build_train_association_index(
        manifest_path=args.manifest,
        continuations_path=args.continuations,
        max_subtokens=args.max_subtokens,
        min_cooccurrence=args.min_cooccurrence,
        top_candidates_per_token=args.top_candidates_per_token,
    )
    Path(args.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    index.save(args.out_pkl)
    write_json_exclusive(args.out_summary, summary)
    print(f"Saved TRAIN-only index to {args.out_pkl}")
    print(f"Saved index provenance to {args.out_summary}")


if __name__ == "__main__":
    main()
