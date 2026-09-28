"""Build and cache the canonical Predictor V2 dataset and fixed candidate pools.

Outputs:
- data/predictor_v2_split_manifest.json
- data/predictor_v2_dataset.pkl
- docs/predictor_v2_dataset_summary.json
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from src.zip2zip.predictor_v2.candidate_pool import PromptCandidateGenerator
from src.zip2zip.predictor_v2.dataset import (
    DEFAULT_SPLIT_MANIFEST_PATH,
    BakeoffDataset,
    create_deterministic_splits,
    save_split_manifest,
)
from src.zip2zip.predictor_v2.vanilla_labels import (
    CANONICAL_BASE_REVISION,
    CANONICAL_MODEL_ID,
    compute_dataset_manifest_hash,
    get_canonical_tokenizer,
    load_canonical_vanilla_records,
)

CACHED_PREDICTOR_PATH = "experiments/checkpoints/cached_predictor.pkl"
OUT_DATASET_PKL = "data/predictor_v2_dataset.pkl"
OUT_SUMMARY_JSON = "docs/predictor_v2_dataset_summary.json"


def main():
    parser = argparse.ArgumentParser(description="Build Predictor V2 Dataset & Candidate Pools")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument("--out-pkl", default=OUT_DATASET_PKL, help="Output path for serialized dataset")
    args = parser.parse_args()

    print("=" * 80)
    print("BUILDING PREDICTOR V2 CANONICAL DATASET & CANDIDATE POOLS")
    print("=" * 80)

    t0 = time.perf_counter()

    # 1. Load canonical Vanilla continuation records
    print("Loading canonical Vanilla continuation records...", flush=True)
    records = load_canonical_vanilla_records()
    dataset_hash = compute_dataset_manifest_hash(records)
    print(f"Loaded {len(records)} records. Dataset SHA-256: {dataset_hash}")

    # 2. Partition deterministically into Train, Dev, Frozen Test
    print("Creating deterministic Train / Dev / Frozen Test splits...", flush=True)
    manifest = create_deterministic_splits(records, seed=args.seed)
    save_split_manifest(manifest, DEFAULT_SPLIT_MANIFEST_PATH)
    print(f"Saved split manifest to {DEFAULT_SPLIT_MANIFEST_PATH}. Manifest SHA-256: {manifest.manifest_hash}")
    print(f"Domain breakdown: {manifest.domain_counts}")

    dataset = BakeoffDataset(records, manifest)
    dataset.verify_split_integrity()
    print("Verified zero prompt leakage between splits!")

    # 3. Load tokenizer and predictor index
    print("Loading canonical tokenizer and predictor index...", flush=True)
    tokenizer = get_canonical_tokenizer()
    with open(CACHED_PREDICTOR_PATH, "rb") as f:
        raw_pred = pickle.load(f)

    gen = PromptCandidateGenerator(raw_pred, tokenizer)

    # 4. Generate candidate pools for all 60 records
    print("Generating fixed candidate pools for all prompts...", flush=True)
    candidates_by_prompt: Dict[str, List[Any]] = {}
    total_candidates = 0
    total_occurring = 0
    candidate_counts = []

    for idx, r in enumerate(records, 1):
        cands = gen.build_candidate_records(r)
        candidates_by_prompt[r.prompt_id] = cands
        total_candidates += len(cands)
        n_occ = sum(1 for c in cands if c.occurs_in_vanilla)
        total_occurring += n_occ
        candidate_counts.append(len(cands))
        if idx % 15 == 0 or idx == len(records):
            print(f"  Processed [{idx}/{len(records)}] prompts... ({total_candidates} candidates accumulated)")

    mean_cands = total_candidates / len(records)
    print(f"\nCandidate pool generation complete:")
    print(f"  Total candidates: {total_candidates}")
    print(f"  Mean per prompt: {mean_cands:.1f} (min: {min(candidate_counts)}, max: {max(candidate_counts)})")
    print(f"  Total occurring in Vanilla: {total_occurring} ({total_occurring / total_candidates * 100:.2f}%)")

    # 5. Serialize dataset artifact
    dataset_bundle = {
        "schema": "predictor_v2_dataset_bundle_v1",
        "dataset_manifest_hash": dataset_hash,
        "split_manifest": manifest,
        "records": records,
        "candidates_by_prompt": candidates_by_prompt,
        "model_id": CANONICAL_MODEL_ID,
        "base_revision": CANONICAL_BASE_REVISION,
    }

    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(dataset_bundle, f)
    print(f"Saved dataset bundle to {args.out_pkl}")

    # 6. Save summary json
    summary = {
        "num_prompts": len(records),
        "dataset_hash": dataset_hash,
        "split_manifest_hash": manifest.manifest_hash,
        "train_prompts": len(dataset.train_records),
        "dev_prompts": len(dataset.dev_records),
        "frozen_test_prompts": len(dataset.frozen_test_records),
        "total_candidates": total_candidates,
        "mean_candidates_per_prompt": round(mean_cands, 2),
        "min_candidates": min(candidate_counts),
        "max_candidates": max(candidate_counts),
        "total_occurring_candidates": total_occurring,
        "overall_candidate_recall_rate": round(total_occurring / total_candidates, 4),
        "domain_counts": manifest.domain_counts,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }

    os.makedirs(os.path.dirname(OUT_SUMMARY_JSON), exist_ok=True)
    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved dataset summary to {OUT_SUMMARY_JSON}")
    print("Dataset build succeeded!")


if __name__ == "__main__":
    main()
