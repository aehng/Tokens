"""Test and Benchmark FastPredictor vs Baseline StrictPredictor on Validation Data.

Measures:
- Latency (p50, p95, mean) on 100 validation prompts
- Vocabulary overlap (% of predicted phrases shared with baseline predictor)
- Compression ratio preservation on validation samples
"""

import sys
import os
import time
import pickle
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from transformers import AutoTokenizer
from experiments.dataset_loader import load_split
from experiments.heldout_predictor_benchmark import StrictHeldOutPhraseBank, StrictPredictor, evaluate_sample_codebook
from experiments.optimized_predictor import OptimizedPhraseIndex, FastPredictor


def benchmark_fast_predictor():
    print("Loading tokenizer and cached StrictHeldOutPhraseBank...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    disabled_ids = set(tokenizer.all_special_ids)
    if tokenizer.pad_token_id is not None:
        disabled_ids.add(tokenizer.pad_token_id)

    with open("experiments/strict_phrase_bank.pkl", "rb") as f:
        bank: StrictHeldOutPhraseBank = pickle.load(f)

    # Build optimized index
    opt_index = OptimizedPhraseIndex.build_from_phrase_bank(
        bank, top_candidates_per_token=64, min_cooccurrence=2
    )

    # Save compiled index for fast reuse
    with open("experiments/optimized_phrase_index.pkl", "wb") as f:
        pickle.dump(opt_index, f)
    print("Saved OptimizedPhraseIndex to experiments/optimized_phrase_index.pkl")

    fast_pred = FastPredictor(opt_index, initial_vocab_size=32011)
    base_pred = StrictPredictor(bank, initial_vocab_size=32011)

    val_samples = load_split("val")[:100]
    print(f"\nBenchmarking FastPredictor on {len(val_samples)} validation samples (budget=32)...")

    # Warmup
    for s in val_samples[:5]:
        p_ids = tokenizer.encode(s.prompt, add_special_tokens=False)
        fast_pred.select_prompt_conditioned(p_ids, budget=32)

    fast_latencies = []
    overlaps = []
    base_comp_ratios = []
    fast_comp_ratios = []

    for s in val_samples:
        p_ids = tokenizer.encode(s.prompt, add_special_tokens=False)
        r_ids = tokenizer.encode(s.response, add_special_tokens=False)

        # Fast predictor
        cb_fast, lat = fast_pred.select_prompt_conditioned(p_ids, budget=32)
        fast_latencies.append(lat)

        # Evaluate compression with fast predictor
        *_, fast_resp_comp, _, _, _ = evaluate_sample_codebook(
            p_ids, r_ids, cb_fast, disabled_ids, budget=32
        )
        fast_comp_ratios.append(fast_resp_comp)

    fast_latencies = np.array(fast_latencies)
    print("\n" + "=" * 60)
    print("FAST PREDICTOR BENCHMARK RESULTS (100 Validation Prompts)")
    print("=" * 60)
    print(f"Latency p50:  {np.percentile(fast_latencies, 50):.3f} ms")
    print(f"Latency p90:  {np.percentile(fast_latencies, 90):.3f} ms")
    print(f"Latency p95:  {np.percentile(fast_latencies, 95):.3f} ms")
    print(f"Latency p99:  {np.percentile(fast_latencies, 99):.3f} ms")
    print(f"Latency Mean: {np.mean(fast_latencies):.3f} ms ± {np.std(fast_latencies):.3f} ms")
    print(f"Average Response Compression (K=32): {np.mean(fast_comp_ratios):.2f}%")
    print("=" * 60)

    # Overlap test on first 10 samples against baseline predictor
    print("\nVerifying semantic consistency against unpruned baseline on 10 samples:")
    for i, s in enumerate(val_samples[:10]):
        p_ids = tokenizer.encode(s.prompt, add_special_tokens=False)
        cb_fast, _ = fast_pred.select_prompt_conditioned(p_ids, budget=32)
        cb_base, _ = base_pred.select_prompt_conditioned(p_ids, budget=32)
        common = set(cb_fast.keys()) & set(cb_base.keys())
        overlap_pct = (len(common) / 32.0) * 100.0
        overlaps.append(overlap_pct)
        print(f"  Sample {i+1}: {len(common)}/32 phrases match baseline ({overlap_pct:.1f}%)")

    print(f"\nMean Top-32 Overlap with unpruned baseline: {np.mean(overlaps):.1f}%")


if __name__ == "__main__":
    benchmark_fast_predictor()
