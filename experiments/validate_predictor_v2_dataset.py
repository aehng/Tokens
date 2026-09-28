"""Validate a canonical Predictor V2 JSONL artifact without running an experiment."""

from __future__ import annotations

import argparse
import json

from src.zip2zip.predictor_v2.canonical_dataset import load_canonical_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical continuation data and provenance")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    views, manifest = load_canonical_dataset(args.dataset, args.manifest)
    summary = {
        "status": "VALID",
        "dataset_sha256": views.dataset_sha256,
        "manifest_sha256": views.manifest_sha256,
        "provenance_sha256": views.provenance_sha256,
        "counts": {"TRAIN": len(views.train), "DEV": len(views.dev), "FINAL": len(views.final_ids)},
        "split_sha256": {
            "TRAIN": views.train_split_sha256,
            "DEV": views.dev_split_sha256,
            "FINAL": views.final_split_sha256,
        },
        "model": {"id": manifest["model_id"], "revision": manifest["model_revision"]},
        "tokenizer": {
            "id": manifest["tokenizer_id"],
            "revision": manifest["tokenizer_revision"],
            "vocab_size": manifest["tokenizer_vocab_size"],
        },
        "generation_config": manifest["generation_config"],
        "final_metrics_read": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
