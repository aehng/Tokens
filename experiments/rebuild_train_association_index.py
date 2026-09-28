"""Rebuild TRAIN-Only Candidate Association Index with Zero Leakage.

Guarantees:
1. Mined EXCLUSIVELY from the 630 TRAIN canonical continuations.
2. Formally asserts zero DEV and zero FINAL prompts/continuations in training data.
3. Produces:
   - experiments/checkpoints/train_only_association_index.pkl
   - docs/train_only_association_index_summary.json
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from transformers import AutoTokenizer

from src.zip2zip.predictor_v2.candidate_retrieval import TrainOnlyAssociationIndex
from src.zip2zip.predictor_v2.vanilla_labels import (
    CANONICAL_MODEL_ID,
    CANONICAL_BASE_REVISION,
    VanillaContinuationRecord,
    get_canonical_tokenizer,
)

DEFAULT_MANIFEST_PATH = "docs/predictor_v2_scaled_dataset_manifest.json"
DEFAULT_CONTINUATIONS_PATH = "data/canonical_phi_continuations.jsonl"
DEFAULT_OUT_PKL = "experiments/checkpoints/train_only_association_index.pkl"
DEFAULT_OUT_SUMMARY = "docs/train_only_association_index_summary.json"


def extract_ngrams(
    tokens: List[int],
    disabled_ids: Set[int],
    min_len: int = 2,
    max_len: int = 4,
) -> Counter[Tuple[int, ...]]:
    counts: Counter[Tuple[int, ...]] = Counter()
    n = len(tokens)
    for length in range(min_len, max_len + 1):
        for i in range(n - length + 1):
            gram = tuple(tokens[i : i + length])
            if not any(t in disabled_ids for t in gram):
                counts[gram] += 1
    return counts


def build_train_association_index(
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    continuations_path: str = DEFAULT_CONTINUATIONS_PATH,
    max_subtokens: int = 4,
    min_cooccurrence: int = 2,
    top_candidates_per_token: int = 64,
) -> Tuple[TrainOnlyAssociationIndex, Dict[str, Any]]:
    print("=" * 80)
    print("BUILDING TRAIN-ONLY CANDIDATE ASSOCIATION INDEX")
    print("=" * 80)

    t0 = time.perf_counter()

    # 1. Load manifest and verify split prompt IDs
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_prompt_ids = set(manifest["train_prompt_ids"])
    dev_prompt_ids = set(manifest["dev_prompt_ids"])
    final_prompt_ids = set(manifest["final_prompt_ids"])

    assert len(train_prompt_ids & dev_prompt_ids) == 0, "TRAIN and DEV prompt IDs overlap!"
    assert len(train_prompt_ids & final_prompt_ids) == 0, "TRAIN and FINAL prompt IDs overlap!"
    assert len(dev_prompt_ids & final_prompt_ids) == 0, "DEV and FINAL prompt IDs overlap!"

    print(f"Manifest splits: Train={len(train_prompt_ids)}, Dev={len(dev_prompt_ids)}, Final={len(final_prompt_ids)}")

    # 2. Load continuations and filter strictly to TRAIN
    print(f"Reading continuations from {continuations_path}...", flush=True)
    all_records: List[VanillaContinuationRecord] = []
    with open(continuations_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            all_records.append(VanillaContinuationRecord.from_dict(d))

    print(f"Loaded total {len(all_records)} continuation records.")

    # Strict partition filtering
    train_records = [r for r in all_records if r.prompt_id in train_prompt_ids]
    dev_records = [r for r in all_records if r.prompt_id in dev_prompt_ids]
    final_records = [r for r in all_records if r.prompt_id in final_prompt_ids]

    print(f"Partitioned: Train={len(train_records)}, Dev={len(dev_records)}, Final={len(final_records)}")

    # STRICT ASSERTIONS:
    assert len(train_records) == len(train_prompt_ids), (
        f"Mismatch in train records count! Expected {len(train_prompt_ids)}, got {len(train_records)}"
    )

    # Formally verify zero presence of DEV or FINAL in train_records
    for r in train_records:
        assert r.prompt_id not in dev_prompt_ids, f"DEV prompt {r.prompt_id} leaked into train records!"
        assert r.prompt_id not in final_prompt_ids, f"FINAL prompt {r.prompt_id} leaked into train records!"

    tokenizer = get_canonical_tokenizer()
    disabled_ids = set([0, 1, 2] + list(range(32000, 32011)))

    # 3. Mine associations strictly on TRAIN records
    print("Mining phrase associations strictly from TRAIN records...", flush=True)
    global_counts: Counter[Tuple[int, ...]] = Counter()
    domain_counts: Dict[str, Counter[Tuple[int, ...]]] = defaultdict(Counter)
    prompt_to_phrase: Dict[int, Counter[Tuple[int, ...]]] = defaultdict(Counter)
    prompt_token_totals: Counter[int] = Counter()
    train_prompt_lexical_docs: List[Dict[str, Any]] = []

    for r in train_records:
        p_ids = r.prompt_token_ids
        c_ids = r.continuation_token_ids

        # Continuation n-grams
        c_ngrams = extract_ngrams(c_ids, disabled_ids, min_len=2, max_len=max_subtokens)
        p_unique = set(p_ids) - disabled_ids

        for p_tok in p_unique:
            prompt_token_totals[p_tok] += 1
            for gram, cnt in c_ngrams.items():
                prompt_to_phrase[p_tok][gram] += cnt

        for gram, cnt in c_ngrams.items():
            global_counts[gram] += cnt
            domain_counts[r.domain][gram] += cnt

        # Lexical doc representation for BM25 (top 32 phrases by occurrence count)
        top_c_phrases = {
            g: cnt for g, cnt in c_ngrams.most_common(32)
        }
        train_prompt_lexical_docs.append({
            "prompt_id": r.prompt_id,
            "prompt_text": r.prompt_text,
            "domain": r.domain,
            "continuation_phrases": top_c_phrases,
        })

    # 4. Compile index
    print("Compiling index structures...", flush=True)
    index = TrainOnlyAssociationIndex(
        disabled_ids=disabled_ids,
        train_prompt_lexical_docs=train_prompt_lexical_docs,
        train_prompt_ids=sorted(list(train_prompt_ids)),
        total_train_continuations=len(train_records),
        max_subtokens=max_subtokens,
    )

    # Precomputed global static
    sorted_global = sorted(
        global_counts.items(),
        key=lambda item: item[1] * (len(item[0]) - 1),
        reverse=True,
    )
    index.precomputed_global_static = [
        (gram, math.log1p(cnt) * (len(gram) - 1) * 0.25)
        for gram, cnt in sorted_global[:512]
    ]

    # Precomputed domain static
    for dom, d_counts in domain_counts.items():
        sorted_dom = sorted(
            d_counts.items(),
            key=lambda item: item[1] * (len(item[0]) - 1),
            reverse=True,
        )
        index.precomputed_domain_static[dom] = [gram for gram, _ in sorted_dom[:512]]

    # Normalized token associations
    for p_tok, phrases in prompt_to_phrase.items():
        tok_total = prompt_token_totals.get(p_tok, 1)
        norm_denom = tok_total + 25.0

        cand_list: List[Tuple[Tuple[int, ...], float]] = []
        for gram, co_cnt in phrases.items():
            if co_cnt < min_cooccurrence:
                continue
            weight = (co_cnt / norm_denom) * (len(gram) - 1) * 3.0
            cand_list.append((gram, weight))

        if len(cand_list) > top_candidates_per_token:
            cand_list.sort(key=lambda x: x[1], reverse=True)
            cand_list = cand_list[:top_candidates_per_token]
        else:
            cand_list.sort(key=lambda x: x[1], reverse=True)

        if cand_list:
            index.token_associations[p_tok] = cand_list

    elapsed = time.perf_counter() - t0
    print(f"Index built successfully in {elapsed:.2f}s:")
    print(f"  Indexed prompt tokens: {len(index.token_associations)}")
    print(f"  Global static candidates: {len(index.precomputed_global_static)}")
    print(f"  Domain banks: {list(index.precomputed_domain_static.keys())}")
    print(f"  Train lexical docs: {len(index.train_prompt_lexical_docs)}")

    summary = {
        "manifest_path": manifest_path,
        "manifest_hash": manifest.get("manifest_hash"),
        "total_train_continuations": len(train_records),
        "indexed_tokens_count": len(index.token_associations),
        "global_static_count": len(index.precomputed_global_static),
        "domain_static_counts": {k: len(v) for k, v in index.precomputed_domain_static.items()},
        "build_runtime_seconds": round(elapsed, 3),
        "leakage_verification": {
            "dev_prompts_checked": len(dev_prompt_ids),
            "final_prompts_checked": len(final_prompt_ids),
            "dev_prompts_leaked": 0,
            "final_prompts_leaked": 0,
            "zero_leakage_guaranteed": True,
        },
    }

    return index, summary


def main():
    parser = argparse.ArgumentParser(description="Rebuild TRAIN-Only Candidate Association Index")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH, help="Path to scaled dataset manifest")
    parser.add_argument("--continuations", default=DEFAULT_CONTINUATIONS_PATH, help="Path to canonical continuations JSONL")
    parser.add_argument("--out-pkl", default=DEFAULT_OUT_PKL, help="Output path for index pickle")
    parser.add_argument("--out-summary", default=DEFAULT_OUT_SUMMARY, help="Output path for summary JSON")
    args = parser.parse_args()

    index, summary = build_train_association_index(
        manifest_path=args.manifest,
        continuations_path=args.continuations,
    )

    index.save(args.out_pkl)
    print(f"Saved index to {args.out_pkl}")

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {args.out_summary}")


if __name__ == "__main__":
    main()
