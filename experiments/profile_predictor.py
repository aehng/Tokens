"""Profile the baseline StrictPredictor to identify exact bottlenecks.

Measures:
- p50 and p95 latency
- Candidate phrases evaluated per prompt
- Exact time breakdown: repetition prior, association lookup, background prior, dictionary sorting
- Memory / index size
"""

import sys
import os
import time
import pickle
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from transformers import AutoTokenizer
from experiments.dataset_loader import load_split
from experiments.heldout_predictor_benchmark import StrictHeldOutPhraseBank, StrictPredictor, extract_ngrams

def profile_baseline_predictor():
    print("Loading tokenizer and cached StrictHeldOutPhraseBank...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    disabled_ids = set(tokenizer.all_special_ids)
    if tokenizer.pad_token_id is not None:
        disabled_ids.add(tokenizer.pad_token_id)

    cache_path = "experiments/strict_phrase_bank.pkl"
    with open(cache_path, "rb") as f:
        bank: StrictHeldOutPhraseBank = pickle.load(f)

    predictor = StrictPredictor(
        bank=bank,
        initial_vocab_size=32064,
        max_subtokens=3,
    )

    # Inspect index sizes
    num_global_phrases = len(bank.global_counts)
    num_prompt_tokens = len(bank.prompt_to_phrase)
    total_associations = sum(len(phrases) for phrases in bank.prompt_to_phrase.values())
    avg_assoc_per_token = total_associations / num_prompt_tokens if num_prompt_tokens else 0
    max_assoc = max(len(phrases) for phrases in bank.prompt_to_phrase.values()) if bank.prompt_to_phrase else 0

    print(f"Index Statistics:")
    print(f"  Unique global phrases: {num_global_phrases:,}")
    print(f"  Indexed prompt tokens: {num_prompt_tokens:,}")
    print(f"  Total association entries: {total_associations:,}")
    print(f"  Average associations per prompt token: {avg_assoc_per_token:.1f}")
    print(f"  Maximum associations for a single prompt token: {max_assoc:,}")

    val_samples = load_split("val")[:50]
    print(f"\nProfiling on {len(val_samples)} validation samples (budget=32)...")

    latencies_ms = []
    t_repetition_ms = []
    t_assoc_ms = []
    t_background_ms = []
    t_sort_ms = []
    candidates_considered = []

    for s in val_samples:
        prompt_ids = tokenizer.encode(s.prompt, add_special_tokens=False)

        t0 = time.perf_counter()
        scores = defaultdict(float)

        # 1. Repetition prior
        t_rep0 = time.perf_counter()
        p_ngrams = extract_ngrams(prompt_ids, disabled_ids, min_len=2, max_len=3)
        for gram, cnt in p_ngrams.items():
            savings = len(gram) - 1
            scores[gram] += cnt * savings * 6.0
        t_rep1 = time.perf_counter()

        # 2. Association lookup
        t_ass0 = time.perf_counter()
        p_tokens = set(prompt_ids) - disabled_ids
        for p_tok in p_tokens:
            if p_tok in bank.prompt_to_phrase:
                tok_total = bank.prompt_token_totals.get(p_tok, 1)
                for gram, co_cnt in bank.prompt_to_phrase[p_tok].items():
                    norm_score = (co_cnt / (tok_total + 25.0)) * (len(gram) - 1)
                    scores[gram] += norm_score * 3.0
        t_ass1 = time.perf_counter()

        # 3. Background prior
        t_bg0 = time.perf_counter()
        for gram, g_cnt in predictor.precomputed_global_static[: 32 * 2]:
            savings = len(gram) - 1
            scores[gram] += np.log1p(g_cnt) * savings * 0.25
        t_bg1 = time.perf_counter()

        # 4. Sorting
        t_srt0 = time.perf_counter()
        num_cands = len(scores)
        ranked = sorted(scores.items(), key=lambda it: it[1], reverse=True)[:32]
        t_srt1 = time.perf_counter()

        total_ms = (t_srt1 - t0) * 1000.0
        latencies_ms.append(total_ms)
        t_repetition_ms.append((t_rep1 - t_rep0) * 1000.0)
        t_assoc_ms.append((t_ass1 - t_ass0) * 1000.0)
        t_background_ms.append((t_bg1 - t_bg0) * 1000.0)
        t_sort_ms.append((t_srt1 - t_srt0) * 1000.0)
        candidates_considered.append(num_cands)

    latencies_ms = np.array(latencies_ms)
    print("\n--- Profiling Results (Baseline Predictor) ---")
    print(f"Latency p50: {np.percentile(latencies_ms, 50):.2f} ms")
    print(f"Latency p95: {np.percentile(latencies_ms, 95):.2f} ms")
    print(f"Latency Mean: {np.mean(latencies_ms):.2f} ms ± {np.std(latencies_ms):.2f} ms")
    print(f"Candidates considered per prompt (mean): {np.mean(candidates_considered):.0f} (max: {max(candidates_considered)})")
    print("\nWhere runtime is spent (mean per prompt):")
    print(f"  1. Repetition prior:    {np.mean(t_repetition_ms):.2f} ms ({np.mean(t_repetition_ms)/np.mean(latencies_ms)*100:.1f}%)")
    print(f"  2. Association lookup:  {np.mean(t_assoc_ms):.2f} ms ({np.mean(t_assoc_ms)/np.mean(latencies_ms)*100:.1f}%)")
    print(f"  3. Background prior:    {np.mean(t_background_ms):.2f} ms ({np.mean(t_background_ms)/np.mean(latencies_ms)*100:.1f}%)")
    print(f"  4. Dictionary sorting:  {np.mean(t_sort_ms):.2f} ms ({np.mean(t_sort_ms)/np.mean(latencies_ms)*100:.1f}%)")

if __name__ == "__main__":
    profile_baseline_predictor()
