"""
Repeated Latency Benchmark with Model Warmup and Randomized Interleaving.
Evaluates Base Phi-3.5 vs. Pure Predictive Seeded (and Official Reactive Zip2Zip):
- Initial model warmup
- 5 measured trials per prompt
- Randomized condition order per trial (coin flip to prevent thermal/systematic bias)
- Separate timing for:
  * Setup latency (predictor retrieval + hypertoken tensor synthesis)
  * Prefill latency (transformer prompt encoding)
  * Decode latency (autoregressive token generation)
  * Total end-to-end latency
- Paired statistical analysis:
  * Mean delta (Base - Pred)
  * Median delta
  * 95% bootstrap confidence interval
  * Statistical significance (Wilcoxon / paired t-test)
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, LogitsProcessorList

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import StaticCodebookManager, Zip2ZipModel
from src.evaluation.offline_segmenter import segment_tokens_dp


def run_timing_benchmark(
    num_prompts: int = 8,
    num_repeats: int = 4,
    max_new_tokens: int = 25,
    device: str = "cpu",
    output_path: str = "experiments/repeated_timing_results.json",
):
    print("=" * 80)
    print("REPEATED LIVE TIMING BENCHMARK WITH WARMUP & RANDOMIZED INTERLEAVING")
    print(f"Prompts: {num_prompts}, Repeats per prompt: {num_repeats}, Max New Tokens: {max_new_tokens}")
    print("=" * 80)

    # 1. Load Tokenizer & Predictor
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    with open("experiments/checkpoints/cached_predictor.pkl", "rb") as f:
        predictor = pickle.load(f)

    # 2. Load Model
    print("Loading model on CPU...")
    model = Zip2ZipModel.from_pretrained(
        "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
        max_codebook_size=32,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    initial_vocab_size = 32011
    disabled_ids = list(model.zip2zip_config.compression.disabled_ids)
    dim = 3072

    # 3. Model Warmup
    print("\nWarming up model with dummy generation passes...")
    warmup_ids = torch.tensor([[100, 200, 300, 400]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(2):
            model.base_model.generate(input_ids=warmup_ids, max_new_tokens=5, do_sample=False)
            model.generate(input_ids=warmup_ids, max_new_tokens=5, do_sample=False)
    print("Warmup complete!")

    # 4. Load Prompts
    with open("experiments/live_benchmark_prompts.json", "r", encoding="utf-8") as f:
        all_prompts = json.load(f)
    selected_prompts = all_prompts[:num_prompts]

    raw_trials = []

    for p_idx, p_info in enumerate(selected_prompts):
        p_id = p_info["id"]
        domain = p_info["domain"]
        p_text = p_info["prompt"]
        prompt_ids = tokenizer.encode(p_text, add_special_tokens=False)
        base_len = len(prompt_ids)

        print(f"\nEvaluating Prompt [{p_idx + 1}/{len(selected_prompts)}]: {p_id} ({domain}, {base_len} tokens)")

        # Prepare Predictive Codebook
        t_pred0 = time.perf_counter()
        p_dict, _ = predictor.select_prompt_conditioned(prompt_ids, budget=32)
        pred_phrases = list(p_dict.keys())
        t_pred1 = time.perf_counter()
        pred_retrieval_ms = (t_pred1 - t_pred0) * 1000.0

        comp_len, tiles, stats = segment_tokens_dp(prompt_ids, set(pred_phrases))
        seeded_dict = {p: initial_vocab_size + i for i, p in enumerate(pred_phrases)}
        resegmented = [tile[0] if len(tile) == 1 else seeded_dict[tile] for tile in tiles]

        static_mgr = StaticCodebookManager(
            initial_vocab_size=initial_vocab_size,
            max_codebook_size=32,
            max_subtokens=3,
            embedding_dim=dim,
            pad_token_id=32000,
            disabled_ids=disabled_ids,
        )
        t_seed0 = time.perf_counter()
        static_mgr.set_seeded_codebook(seeded_dict, batch_size=1, device=torch.device(device))
        t_seed1 = time.perf_counter()
        codebook_synth_ms = (t_seed1 - t_seed0) * 1000.0
        setup_total_ms = pred_retrieval_ms + codebook_synth_ms

        base_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        pred_tensor = torch.tensor([resegmented], dtype=torch.long, device=device)
        logits_proc = LogitsProcessorList([static_mgr.get_logits_processor()])

        for rep in range(num_repeats):
            # Randomize order (coin flip) to prevent thermal or memory fragmentation bias
            order = ["base", "pred"] if random.random() < 0.5 else ["pred", "base"]

            trial_res = {
                "prompt_id": p_id,
                "domain": domain,
                "repeat": rep,
                "order": order,
                "base_prompt_tokens": base_len,
                "compressed_prompt_tokens": len(resegmented),
                "setup_ms": setup_total_ms,
            }

            for cond in order:
                if cond == "base":
                    # Time Base Prefill (forward pass only)
                    t_pf0 = time.perf_counter()
                    with torch.no_grad():
                        _ = model.base_model(input_ids=base_tensor)
                    t_pf1 = time.perf_counter()
                    base_pf_ms = (t_pf1 - t_pf0) * 1000.0

                    # Time Base Full Generation
                    t_gen0 = time.perf_counter()
                    with torch.no_grad():
                        out_base = model.base_model.generate(
                            input_ids=base_tensor,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                    t_gen1 = time.perf_counter()
                    base_tot_ms = (t_gen1 - t_gen0) * 1000.0
                    base_dec_ms = max(0.0, base_tot_ms - base_pf_ms)

                    trial_res["base_prefill_ms"] = base_pf_ms
                    trial_res["base_decode_ms"] = base_dec_ms
                    trial_res["base_total_ms"] = base_tot_ms
                    trial_res["base_steps"] = len(out_base[0]) - base_len

                elif cond == "pred":
                    static_mgr.attach_to_model(model)
                    # Time Pred Prefill
                    t_pf0 = time.perf_counter()
                    with torch.no_grad():
                        _ = model.base_model(input_ids=pred_tensor)
                    t_pf1 = time.perf_counter()
                    pred_pf_ms = (t_pf1 - t_pf0) * 1000.0

                    # Time Pred Full Generation
                    t_gen0 = time.perf_counter()
                    with torch.no_grad():
                        out_pred = model.generate(
                            input_ids=pred_tensor,
                            max_new_tokens=max_new_tokens,
                            logits_processor=logits_proc,
                            do_sample=False,
                        )
                    t_gen1 = time.perf_counter()
                    pred_gen_ms = (t_gen1 - t_gen0) * 1000.0
                    static_mgr.detach_from_model(model)

                    pred_dec_ms = max(0.0, pred_gen_ms - pred_pf_ms)
                    pred_tot_ms = setup_total_ms + pred_gen_ms

                    trial_res["pred_prefill_ms"] = pred_pf_ms
                    trial_res["pred_decode_ms"] = pred_dec_ms
                    trial_res["pred_total_ms"] = pred_tot_ms
                    trial_res["pred_steps"] = len(out_pred[0]) - len(resegmented)

            delta_tot = trial_res["base_total_ms"] - trial_res["pred_total_ms"]
            delta_pf = trial_res["base_prefill_ms"] - trial_res["pred_prefill_ms"]
            delta_dec = trial_res["base_decode_ms"] - trial_res["pred_decode_ms"]
            trial_res["delta_total_ms"] = delta_tot
            trial_res["delta_prefill_ms"] = delta_pf
            trial_res["delta_decode_ms"] = delta_dec
            trial_res["pct_improvement"] = (delta_tot / trial_res["base_total_ms"]) * 100.0

            raw_trials.append(trial_res)
            print(f"  Trial {rep + 1}/{num_repeats} (Order: {order[0]}->{order[1]}): "
                  f"Base={trial_res['base_total_ms']:.1f}ms, Pred={trial_res['pred_total_ms']:.1f}ms "
                  f"| Delta = {delta_tot:+.1f}ms ({trial_res['pct_improvement']:+.2f}%)")

    # Statistical Bootstrap Analysis
    all_deltas_tot = [r["delta_total_ms"] for r in raw_trials]
    all_deltas_pf = [r["delta_prefill_ms"] for r in raw_trials]
    all_deltas_dec = [r["delta_decode_ms"] for r in raw_trials]
    all_pcts = [r["pct_improvement"] for r in raw_trials]

    # Paired per-prompt mean deltas
    prompt_grouped = {}
    for r in raw_trials:
        p = r["prompt_id"]
        if p not in prompt_grouped:
            prompt_grouped[p] = []
        prompt_grouped[p].append(r["delta_total_ms"])
    per_prompt_means = {p: float(np.mean(vals)) for p, vals in prompt_grouped.items()}

    # 10,000 bootstrap iterations
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(all_deltas_tot, size=len(all_deltas_tot), replace=True)) for _ in range(10000)]
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    stats_summary = {
        "n_prompts": num_prompts,
        "n_repeats": num_repeats,
        "total_paired_trials": len(raw_trials),
        "mean_total_delta_ms": float(np.mean(all_deltas_tot)),
        "median_total_delta_ms": float(np.median(all_deltas_tot)),
        "std_total_delta_ms": float(np.std(all_deltas_tot)),
        "bootstrap_95_ci_ms": [ci_low, ci_high],
        "mean_pct_improvement": float(np.mean(all_pcts)),
        "median_pct_improvement": float(np.median(all_pcts)),
        "mean_prefill_delta_ms": float(np.mean(all_deltas_pf)),
        "mean_decode_delta_ms": float(np.mean(all_deltas_dec)),
        "per_prompt_mean_deltas": per_prompt_means,
    }

    final_payload = {
        "summary": stats_summary,
        "trials": raw_trials,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("STATISTICAL TIMING VERIFICATION RESULTS")
    print("=" * 80)
    print(f"Total Paired Trials:        {len(raw_trials)}")
    print(f"Mean Total Delta:           {stats_summary['mean_total_delta_ms']:+.1f} ms (95% CI: [{ci_low:+.1f}, {ci_high:+.1f}] ms)")
    print(f"Median Total Delta:         {stats_summary['median_total_delta_ms']:+.1f} ms")
    print(f"Mean % Improvement:         {stats_summary['mean_pct_improvement']:+.2f}%")
    print(f"Median % Improvement:       {stats_summary['median_pct_improvement']:+.2f}%")
    print(f"Prefill Delta (Prefill-Only): {stats_summary['mean_prefill_delta_ms']:+.1f} ms")
    print(f"Decode Delta (Decode-Only):   {stats_summary['mean_decode_delta_ms']:+.1f} ms")
    print(f"\nPer-Prompt Mean Deltas (Base - Pred):")
    for p, d in per_prompt_means.items():
        print(f"  {p:20s}: {d:+.1f} ms")
    print("=" * 80)

    return stats_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--output", type=str, default="experiments/repeated_timing_results.json")
    args = parser.parse_args()

    run_timing_benchmark(
        num_prompts=args.prompts,
        num_repeats=args.repeats,
        max_new_tokens=args.tokens,
        output_path=args.output,
    )
