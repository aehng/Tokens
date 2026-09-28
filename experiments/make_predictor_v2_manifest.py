"""Bind the canonical JSONL to an independently prepared prompt/split inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.zip2zip.predictor_v2.canonical_dataset import (
    create_canonical_manifest,
    write_canonical_manifest,
)


def _first_record(dataset_path: str) -> dict[str, Any]:
    with Path(dataset_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if isinstance(record, dict):
                    return record
                break
    raise ValueError("canonical dataset contains no JSON object records")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a hash-bound manifest using expected IDs/splits supplied independently"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-inventory", required=True, help="JSON with expected_prompt_ids, split_ids, allowed_domains")
    parser.add_argument("--provenance", required=True, help="JSON provenance object from the generation run")
    parser.add_argument("--tokenizer-vocab-size", required=True, type=int)
    parser.add_argument("--out", default="data/canonical_phi_continuations.manifest.json")
    args = parser.parse_args()
    inventory = json.loads(Path(args.split_inventory).read_text(encoding="utf-8"))
    provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    if not isinstance(inventory, dict) or not isinstance(provenance, dict):
        raise ValueError("split inventory and provenance must each be JSON objects")
    sample = _first_record(args.dataset)
    generation_config = sample.get("generation_config")
    if not isinstance(generation_config, dict):
        raise ValueError("first record must contain generation_config")
    manifest = create_canonical_manifest(
        args.dataset,
        expected_prompt_ids=inventory["expected_prompt_ids"],
        split_ids=inventory["split_ids"],
        allowed_domains=inventory["allowed_domains"],
        model_id=sample["model_id"],
        model_revision=sample["model_revision"],
        tokenizer_id=sample["tokenizer_id"],
        tokenizer_revision=sample["tokenizer_revision"],
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        generation_config=generation_config,
        provenance=provenance,
    )
    write_canonical_manifest(args.out, manifest)
    print(json.dumps({"manifest": args.out, "manifest_sha256": manifest["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
