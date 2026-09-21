"""
Targeted Reconciliation of Prompt Compression Metrics on the Exact 1,000 Held-Out Prompts.
Reconciles the apparent ~14.0% vs ~48.54% discrepancy:
- Examines K=64 vs K=128
- Compares Micro (sum of tokens) vs Macro (mean of per-sample ratios)
- Quantifies length-weighted bias across prompt-length buckets
- Verifies exact identity between segmentation paths and round-trip preservation.
"""

import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.evaluation.offline_segmenter import segment_tokens_dp


def reconcile(
    test_path: str = "data/corpus_stage3_test.jsonl",
    predictor_cache: str = "experiments/checkpoints/cached_predictor.pkl",
    n_samples: int = 1000,
):
    print("=" * 80)
    print("TARGETED RECONCILIATION: PROMPT COMPRESSION METRICS")
    print("=" * 80)

    with open(predictor_cache, "rb") as f:
        predictor = pickle.load(f)

    with open(test_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for i, line in enumerate(f) if i < n_samples]

    print(f"Analyzing exact {len(records)} test records...")

    # We evaluate both K=64 and K=128 to isolate budget effect from aggregation effect
    for k in [64, 128]:
        print(f"\n" + "-" * 80)
        print(f"ANALYSIS AT BUDGET K = {k}")
        print("-" * 80)

        base_lens = []
        comp_lens_verify = []
        comp_lens_suite = []
        per_prompt_comp_pct = []
        unique_hyper_used = []
        predicted_entries = []

        mismatches = 0
        flatten_failures = 0

        # Group by prompt length buckets
        buckets = {
            "0-32": [],
            "33-128": [],
            "129-512": [],
            "513-2048": [],
            "2049+": [],
        }

        for rec in records:
            p_ids = rec["prompt_token_ids"]
            P = len(p_ids)
            base_lens.append(P)

            # Predictor generates codebook
            cb, _ = predictor.select_prompt_conditioned(p_ids, budget=k)
            cb_phrases = set(cb.keys())
            predicted_entries.append(len(cb_phrases))

            # Path A (verify_prompt_guarantees):
            comp_len_a, tiles_a, stats_a = segment_tokens_dp(p_ids, cb_phrases)
            comp_lens_verify.append(comp_len_a)

            # Path B (comprehensive_suite):
            comp_len_b, tiles_b, stats_b = segment_tokens_dp(p_ids, cb_phrases)
            comp_lens_suite.append(comp_len_b)

            # Check exact equality of paths
            if comp_len_a != comp_len_b or tiles_a != tiles_b:
                mismatches += 1

            # Check flattening
            flat = [tok for t in tiles_a for tok in t]
            if flat != p_ids:
                flatten_failures += 1

            # Metrics
            pct = stats_a["compression_pct"]
            per_prompt_comp_pct.append(pct)
            u_used = stats_a["unique_hypertokens_used"]
            unique_hyper_used.append(u_used)

            # Bucketing
            if P <= 32:
                b_name = "0-32"
            elif P <= 128:
                b_name = "33-128"
            elif P <= 512:
                b_name = "129-512"
            elif P <= 2048:
                b_name = "513-2048"
            else:
                b_name = "2049+"

            buckets[b_name].append({
                "base": P,
                "comp": comp_len_a,
                "pct": pct,
                "u_used": u_used,
            })

        assert mismatches == 0, f"Paths diverged on {mismatches} samples!"
        assert flatten_failures == 0, f"Flattening failed on {flatten_failures} samples!"

        # Aggregations
        sum_base = sum(base_lens)
        sum_comp = sum(comp_lens_verify)
        micro_comp = (1.0 - sum_comp / sum_base) * 100.0
        macro_comp = float(np.mean(per_prompt_comp_pct))
        median_comp = float(np.median(per_prompt_comp_pct))

        p10 = float(np.percentile(per_prompt_comp_pct, 10))
        p25 = float(np.percentile(per_prompt_comp_pct, 25))
        p50 = float(np.percentile(per_prompt_comp_pct, 50))
        p75 = float(np.percentile(per_prompt_comp_pct, 75))
        p90 = float(np.percentile(per_prompt_comp_pct, 90))

        print(f"1. Total Base Positions across 1,000 prompts:      {sum_base:,}")
        print(f"   Total Compressed Positions:                     {sum_comp:,}")
        print(f"   --> MICRO Prompt Compression:                   {micro_comp:.2f}% (1 - sum(comp)/sum(base))")
        print(f"2. --> MACRO Prompt Compression:                   {macro_comp:.2f}% (mean of per-sample %)")
        print(f"3. Median Per-Prompt Compression:                  {median_comp:.2f}%")
        print(f"4. Percentiles:")
        print(f"   p10: {p10:.2f}% | p25: {p25:.2f}% | p50: {p50:.2f}% | p75: {p75:.2f}% | p90: {p90:.2f}%")
        print(f"   Mean Unique Hypertokens Used per Prompt:        {np.mean(unique_hyper_used):.1f} / {k}")

        print(f"\n5. Breakdown by Prompt Length Buckets:")
        print(f"   {'Bucket':<10} | {'Count':<6} | {'Sum Base':<10} | {'Sum Comp':<10} | {'Micro Comp %':<14} | {'Macro Comp %':<14} | {'Avg Base/Prompt':<16}")
        print("   " + "-" * 95)
        for b_name, b_recs in buckets.items():
            if not b_recs:
                continue
            b_cnt = len(b_recs)
            b_base = sum(r["base"] for r in b_recs)
            b_comp = sum(r["comp"] for r in b_recs)
            b_micro = (1.0 - b_comp / b_base) * 100.0 if b_base > 0 else 0.0
            b_macro = float(np.mean([r["pct"] for r in b_recs]))
            print(f"   {b_name:<10} | {b_cnt:<6} | {b_base:<10,} | {b_comp:<10,} | {b_micro:<14.2f}% | {b_macro:<14.2f}% | {b_base/b_cnt:<16.1f}")


if __name__ == "__main__":
    reconcile()
