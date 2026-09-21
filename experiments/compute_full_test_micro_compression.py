"""
Compute Full-Test-Set MICRO and MACRO Prompt Compression over all 7,512 samples.
Evaluates across major budget values: K in {16, 32, 64, 128, 256, 512}.

Metrics:
- MICRO prompt compression = (sum of all positions saved across corpus) / (sum of all base tokens across corpus)
- MACRO prompt compression = mean of (positions saved / base prompt tokens) per request
- Median prompt compression = median of (positions saved / base prompt tokens) per request
- Breakdown by prompt length bucket: <=32, 33-128, 129-512, 513-2048, >2048
"""

import json
import os
import pickle
import sys
import time
from typing import Dict, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.evaluation.offline_segmenter import segment_tokens_dp


def run_full_test_micro_compression(
    test_path: str = "data/corpus_stage3_test.jsonl",
    predictor_path: str = "experiments/checkpoints/cached_predictor.pkl",
    output_path: str = "experiments/full_test_micro_compression_results.json",
    k_values: List[int] = [16, 32, 64, 128, 256, 512],
):
    print("=" * 80)
    print("FULL TEST SET (7,512 SAMPLES) MICRO & MACRO PROMPT COMPRESSION AUDIT")
    print(f"Data: {test_path}")
    print(f"K values: {k_values}")
    print("=" * 80)

    # 1. Load Predictor
    t0 = time.time()
    print("Loading cached predictor...")
    with open(predictor_path, "rb") as f:
        predictor = pickle.load(f)
    print(f"Loaded predictor in {time.time() - t0:.2f}s")

    # 2. Load all test prompts
    t0 = time.time()
    print("Loading test samples...")
    samples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    total_samples = len(samples)
    print(f"Loaded {total_samples} samples in {time.time() - t0:.2f}s")

    total_base_tokens = sum(len(s["prompt_token_ids"]) for s in samples)
    print(f"Total Base Prompt Tokens: {total_base_tokens:,}")

    # Length buckets
    buckets = [
        ("<= 32 tokens", lambda l: l <= 32),
        ("33 - 128 tokens", lambda l: 33 <= l <= 128),
        ("129 - 512 tokens", lambda l: 129 <= l <= 512),
        ("513 - 2048 tokens", lambda l: 513 <= l <= 2048),
        ("> 2048 tokens", lambda l: l > 2048),
    ]

    results_by_k = {}

    for k in k_values:
        t_start = time.time()
        print(f"\nProcessing K = {k}...")

        total_compressed_tokens = 0
        per_sample_compression_pcts = []
        bucket_stats = {
            b_name: {"samples": 0, "base_tokens": 0, "comp_tokens": 0, "pcts": []}
            for b_name, _ in buckets
        }

        for s in samples:
            p_ids = s["prompt_token_ids"]
            base_len = len(p_ids)

            p_dict, _ = predictor.select_prompt_conditioned(p_ids, budget=k)
            comp_len, tiles, _ = segment_tokens_dp(p_ids, set(p_dict.keys()))

            total_compressed_tokens += comp_len
            ratio = (base_len - comp_len) / base_len if base_len > 0 else 0.0
            per_sample_compression_pcts.append(ratio * 100.0)

            for b_name, b_fn in buckets:
                if b_fn(base_len):
                    bucket_stats[b_name]["samples"] += 1
                    bucket_stats[b_name]["base_tokens"] += base_len
                    bucket_stats[b_name]["comp_tokens"] += comp_len
                    bucket_stats[b_name]["pcts"].append(ratio * 100.0)
                    break

        total_positions_saved = total_base_tokens - total_compressed_tokens
        micro_compression_pct = (total_positions_saved / total_base_tokens) * 100.0
        macro_compression_pct = float(np.mean(per_sample_compression_pcts))
        median_compression_pct = float(np.median(per_sample_compression_pcts))
        p25 = float(np.percentile(per_sample_compression_pcts, 25))
        p75 = float(np.percentile(per_sample_compression_pcts, 75))

        bucket_summary = {}
        for b_name, b_data in bucket_stats.items():
            if b_data["samples"] > 0:
                b_saved = b_data["base_tokens"] - b_data["comp_tokens"]
                b_micro = (b_saved / b_data["base_tokens"]) * 100.0
                b_macro = float(np.mean(b_data["pcts"]))
                bucket_summary[b_name] = {
                    "count": b_data["samples"],
                    "pct_of_samples": (b_data["samples"] / total_samples) * 100.0,
                    "base_tokens": b_data["base_tokens"],
                    "pct_of_tokens": (b_data["base_tokens"] / total_base_tokens) * 100.0,
                    "micro_compression_pct": b_micro,
                    "macro_compression_pct": b_macro,
                }

        results_by_k[str(k)] = {
            "k": k,
            "micro_compression_pct": micro_compression_pct,
            "macro_compression_pct": macro_compression_pct,
            "median_compression_pct": median_compression_pct,
            "p25": p25,
            "p75": p75,
            "total_base_tokens": total_base_tokens,
            "total_compressed_tokens": total_compressed_tokens,
            "total_positions_saved": total_positions_saved,
            "bucket_breakdown": bucket_summary,
            "eval_time_sec": time.time() - t_start,
        }

        print(f"K = {k:3d} | MICRO Compression: {micro_compression_pct:6.2f}% | MACRO Compression: {macro_compression_pct:6.2f}% (Median: {median_compression_pct:5.2f}%) [{time.time() - t_start:.2f}s]")

    # Save summary
    output_data = {
        "dataset": test_path,
        "total_samples": total_samples,
        "total_base_tokens": total_base_tokens,
        "results_by_k": results_by_k,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved full results to {output_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'K':>5} | {'MICRO % (Tokens Saved)':>22} | {'MACRO % (Per Request)':>22} | {'Median %':>10} | {'Saved Tokens':>14}")
    print("-" * 90)
    for k in k_values:
        r = results_by_k[str(k)]
        print(f"{k:5d} | {r['micro_compression_pct']:21.2f}% | {r['macro_compression_pct']:21.2f}% | {r['median_compression_pct']:9.2f}% | {r['total_positions_saved']:14,d}")
    print("=" * 90)


if __name__ == "__main__":
    run_full_test_micro_compression()
