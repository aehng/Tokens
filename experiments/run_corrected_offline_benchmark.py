"""
Corrected Large-Scale Offline Benchmark for Dynamic Token Vocabularies.

Addresses all benchmark audit items:
1. Exact checkpoint LZW configuration (initial_vocab_size=32011, disabled_ids=[0,1,2,32000..32010], pad=32000).
2. Apples-to-apples comparison: LZW with max_subtokens=3 (matching predictive codebook) vs max_subtokens=4.
3. True sequential pre-seeded hybrid LZW simulation.
4. Controlled within-sample prefix-length scaling experiment on the SAME responses at [32, 64, 128, 256, 512, 1024, 2048].
5. Renamed to Answer-Aware Greedy Oracle.
6. Cluster-robust bootstrap by thread/conversation ID for all confidence intervals.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.evaluation.index_builder import build_fast_predictor_from_records
from src.evaluation.lzw_simulator import (
    OFFICIAL_DISABLED_IDS,
    OFFICIAL_INITIAL_VOCAB_SIZE,
    compute_lzw_span_compression,
    compute_preseeded_hybrid_lzw,
)
from src.evaluation.offline_segmenter import compute_oracle_codebook, segment_tokens_dp


def compute_cluster_bootstrap_ci(
    records: List[Dict],
    metric_name: str,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Cluster bootstrap resampling by thread_id to produce cluster-robust confidence intervals.
    """
    clusters = defaultdict(list)
    for r in records:
        clusters[r["thread_id"]].append(r[metric_name])

    all_vals = [val for vals in clusters.values() for val in vals]
    point_est = float(np.mean(all_vals)) if all_vals else 0.0

    unique_threads = list(clusters.keys())
    N_clusters = len(unique_threads)
    if N_clusters < 2:
        return point_est, point_est, point_est

    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sampled_threads = rng.choice(unique_threads, size=N_clusters, replace=True)
        sampled_vals = []
        for th in sampled_threads:
            sampled_vals.extend(clusters[th])
        boot_means.append(np.mean(sampled_vals))

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return point_est, low, high


def compute_cluster_bootstrap_diff_ci(
    records: List[Dict],
    metric_a: str,
    metric_b: str,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Cluster bootstrap for difference between two paired metrics (metric_a - metric_b).
    """
    clusters = defaultdict(list)
    for r in records:
        diff = r[metric_a] - r[metric_b]
        clusters[r["thread_id"]].append(diff)

    all_diffs = [val for vals in clusters.values() for val in vals]
    point_est = float(np.mean(all_diffs)) if all_diffs else 0.0

    unique_threads = list(clusters.keys())
    N_clusters = len(unique_threads)
    if N_clusters < 2:
        return point_est, point_est, point_est

    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sampled_threads = rng.choice(unique_threads, size=N_clusters, replace=True)
        sampled_diffs = []
        for th in sampled_threads:
            sampled_diffs.extend(clusters[th])
        boot_means.append(np.mean(sampled_diffs))

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return point_est, low, high


def run_corrected_benchmark(
    train_path: str = "data/corpus_stage3_train.jsonl",
    test_path: str = "data/corpus_stage3_test.jsonl",
    output_json: str = "experiments/corrected_stage3_results.json",
    plots_dir: str = "experiments/figures/corrected",
    budgets: List[int] = [32, 64, 128, 256],
    max_test_samples: Optional[int] = None,
) -> Dict:
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    print("=== 1. Loading Training Split & Mining Index ===")
    with open(train_path, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f]
    print(f"Loaded {len(train_records)} TRAIN records from {train_path}")

    t_idx_start = time.time()
    predictor = build_fast_predictor_from_records(train_records)
    print(f"Index training finished in {time.time() - t_idx_start:.2f}s")

    print("\n=== 2. Loading Held-Out Test Records ===")
    with open(test_path, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f]
    if max_test_samples:
        test_records = test_records[:max_test_samples]
    print(f"Loaded {len(test_records)} TEST records from {test_path}")

    unique_threads = set(r["thread_id"] for r in test_records)
    print(f"Total test conversations/threads: {len(unique_threads)} (Cluster units for bootstrap)")

    # 3. Main Budget Sweep Evaluation
    print(f"\n=== 3. Running Main Benchmark across budgets K = {budgets} ===")
    # budget -> list of per-sample results
    eval_by_budget = defaultdict(list)
    predictor_latencies = []

    t0 = time.time()
    for idx, rec in enumerate(test_records):
        if idx > 0 and idx % 1000 == 0:
            print(f"Evaluated {idx}/{len(test_records)} samples in {time.time() - t0:.1f}s ({idx / (time.time() - t0):.1f} samples/s)...")

        p_ids = rec["prompt_token_ids"]
        r_ids = rec["response_token_ids"]
        domain = rec.get("domain", "conversation")
        thread_id = rec["thread_id"]
        p_len = len(p_ids)
        r_len = len(r_ids)

        for k in budgets:
            row = {
                "id": rec["id"],
                "thread_id": thread_id,
                "domain": domain,
                "base_prompt": p_len,
                "base_response": r_len,
                "base_comp_pct": 0.0,
            }

            # A. Corrected LZW (Apples-to-apples: max_subtokens=3, checkpoint config)
            lzw_corr = compute_lzw_span_compression(
                p_ids, r_ids, budget=k, max_subtokens=3, initial_vocab_size=OFFICIAL_INITIAL_VOCAB_SIZE, disabled_ids=OFFICIAL_DISABLED_IDS
            )
            row["lzw_corr_comp_pct"] = lzw_corr["response_comp_pct"]

            # B. Upstream LZW (max_subtokens=4, to report old vs corrected)
            lzw_up = compute_lzw_span_compression(
                p_ids, r_ids, budget=k, max_subtokens=4, initial_vocab_size=OFFICIAL_INITIAL_VOCAB_SIZE, disabled_ids=OFFICIAL_DISABLED_IDS
            )
            row["lzw_upstream_comp_pct"] = lzw_up["response_comp_pct"]

            # C. Global Static Top-K
            g_dict = predictor.select_global_static(k)
            g_cb = set(g_dict.keys())
            _, _, stats_g = segment_tokens_dp(r_ids, g_cb)
            row["global_static_comp_pct"] = stats_g["compression_pct"]

            # D. Domain Static Top-K
            d_dict = predictor.select_domain_static(domain, k)
            d_cb = set(d_dict.keys())
            _, _, stats_d = segment_tokens_dp(r_ids, d_cb)
            row["domain_static_comp_pct"] = stats_d["compression_pct"]

            # E. Prompt Predictor
            p_dict, lat_ms = predictor.select_prompt_conditioned(p_ids, k)
            if k == 64:
                predictor_latencies.append(lat_ms)
            p_cb = set(p_dict.keys())
            _, _, stats_p = segment_tokens_dp(r_ids, p_cb)
            row["prompt_predictor_comp_pct"] = stats_p["compression_pct"]

            # F. Answer-Aware Greedy Oracle
            oracle_cb = compute_oracle_codebook(r_ids, k=k, max_length=3)
            _, _, stats_o = segment_tokens_dp(r_ids, oracle_cb)
            row["oracle_comp_pct"] = stats_o["compression_pct"]

            eval_by_budget[k].append(row)

    print(f"Main evaluation completed in {time.time() - t0:.2f}s!")

    # 4. Controlled Within-Sample Prefix Experiment
    print("\n=== 4. Running Controlled Length/Prefix Experiment on SAME Responses ===")
    prefix_lengths = [32, 64, 128, 256, 512, 1024, 2048]
    prefix_results = defaultdict(lambda: defaultdict(list))
    k_ctrl = 64

    # Filter samples supporting lengths
    for prefix_len in prefix_lengths:
        qualifying = [r for r in test_records if len(r["response_token_ids"]) >= prefix_len]
        print(f"Prefix length {prefix_len}: {len(qualifying)} qualifying samples")
        for rec in qualifying:
            p_ids = rec["prompt_token_ids"]
            r_prefix = rec["response_token_ids"][:prefix_len]
            domain = rec.get("domain", "conversation")

            # Corrected LZW on prefix
            lzw_res = compute_lzw_span_compression(
                p_ids, r_prefix, budget=k_ctrl, max_subtokens=3, initial_vocab_size=OFFICIAL_INITIAL_VOCAB_SIZE, disabled_ids=OFFICIAL_DISABLED_IDS
            )
            # Domain Static on prefix
            d_dict = predictor.select_domain_static(domain, k_ctrl)
            _, _, stats_d = segment_tokens_dp(r_prefix, set(d_dict.keys()))
            # Prompt Predictor on prefix
            p_dict, _ = predictor.select_prompt_conditioned(p_ids, k_ctrl)
            _, _, stats_p = segment_tokens_dp(r_prefix, set(p_dict.keys()))
            # Answer-Aware Greedy Oracle on prefix
            o_cb = compute_oracle_codebook(r_prefix, k=k_ctrl, max_length=3)
            _, _, stats_o = segment_tokens_dp(r_prefix, o_cb)

            row_pref = {
                "id": rec["id"],
                "thread_id": rec["thread_id"],
                "domain": domain,
                "prefix_len": prefix_len,
                "lzw_corr": lzw_res["response_comp_pct"],
                "domain_static": stats_d["compression_pct"],
                "prompt_predictor": stats_p["compression_pct"],
                "oracle": stats_o["compression_pct"],
                "pred_minus_lzw": stats_p["compression_pct"] - lzw_res["response_comp_pct"],
                "pred_minus_domain": stats_p["compression_pct"] - stats_d["compression_pct"],
            }
            prefix_results[prefix_len]["overall"].append(row_pref)
            prefix_results[prefix_len][domain].append(row_pref)

    # 5. True Sequential Pre-Seeded Hybrid LZW Simulation at K=128
    print("\n=== 5. Running True Pre-Seeded Sequential Hybrid LZW Simulation (K=128) ===")
    hybrid_configs = [
        {"name": "100% Domain Static (128 static, 0 reactive)", "k_dom": 128, "k_pred": 0, "k_lzw": 0},
        {"name": "100% Prompt Predictor (128 static, 0 reactive)", "k_dom": 0, "k_pred": 128, "k_lzw": 0},
        {"name": "100% Reactive LZW (0 static, 128 reactive)", "k_dom": 0, "k_pred": 0, "k_lzw": 128},
        {"name": "True Hybrid: 64 Dom Static + 64 Reactive LZW", "k_dom": 64, "k_pred": 0, "k_lzw": 64},
        {"name": "True Hybrid: 64 Prompt Predictor + 64 Reactive LZW", "k_dom": 0, "k_pred": 64, "k_lzw": 64},
        {"name": "True Hybrid: 32 Dom + 32 Pred + 64 Reactive LZW", "k_dom": 32, "k_pred": 32, "k_lzw": 64},
        {"name": "True Hybrid: 64 Dom + 32 Pred + 32 Reactive LZW", "k_dom": 64, "k_pred": 32, "k_lzw": 32},
    ]

    hybrid_out = []
    # Test on a representative subset of test records (1,000 samples) for fast exact simulation
    test_subset = test_records[:1000]
    for h_cfg in hybrid_configs:
        k_dom = h_cfg["k_dom"]
        k_pred = h_cfg["k_pred"]
        k_lzw = h_cfg["k_lzw"]

        sim_comps = []
        for rec in test_subset:
            p_ids = rec["prompt_token_ids"]
            r_ids = rec["response_token_ids"]
            domain = rec.get("domain", "conversation")

            preseeded = []
            if k_dom > 0:
                dom_dict = predictor.select_domain_static(domain, k_dom)
                preseeded.extend(dom_dict.keys())
            if k_pred > 0:
                pred_dict, _ = predictor.select_prompt_conditioned(p_ids, k_pred)
                preseeded.extend(pred_dict.keys())

            # De-duplicate while preserving order
            seen = set()
            clean_preseeded = []
            for phrase in preseeded:
                if phrase not in seen:
                    seen.add(phrase)
                    clean_preseeded.append(phrase)

            sim = compute_preseeded_hybrid_lzw(
                prompt_ids=p_ids,
                response_ids=r_ids,
                total_budget=128,
                preseeded_phrases=clean_preseeded,
                reactive_budget=k_lzw,
                max_subtokens=3,
                initial_vocab_size=OFFICIAL_INITIAL_VOCAB_SIZE,
                disabled_ids=OFFICIAL_DISABLED_IDS,
            )
            sim_comps.append(sim["response_comp_pct"])

        mean_val = float(np.mean(sim_comps))
        hybrid_out.append({
            "allocation": h_cfg["name"],
            "k_dom": k_dom,
            "k_pred": k_pred,
            "k_lzw": k_lzw,
            "mean_response_compression_pct": round(mean_val, 2),
        })
        print(f"Hybrid [{h_cfg['name']}]: {mean_val:.2f}%")

    # 6. Generate Clean Diagnostic Figures
    print("\n=== 6. Generating Corrected Diagnostic Plots ===")
    generate_corrected_plots(eval_by_budget, prefix_results, plots_dir)

    # 7. Compile Final Deliverable Tables with Cluster CIs
    print("\n=== 7. Compiling Deliverables with Cluster Bootstrap CIs ===")
    deliverables = compile_final_tables(
        eval_by_budget=eval_by_budget,
        prefix_results=prefix_results,
        hybrid_results=hybrid_out,
        predictor_latencies=predictor_latencies,
        budgets=budgets,
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(deliverables, f, indent=2)
    print(f"Corrected benchmark results saved to {output_json}!")

    return deliverables


def generate_corrected_plots(
    eval_by_budget: Dict,
    prefix_results: Dict,
    plots_dir: str,
) -> None:
    # Plot 1: Controlled Prefix-Length Scaling
    plt.figure(figsize=(9, 5), dpi=200)
    pref_lens = [32, 64, 128, 256, 512, 1024, 2048]
    pred_curve = []
    lzw_curve = []
    dom_curve = []
    oracle_curve = []

    for l in pref_lens:
        items = prefix_results[l]["overall"]
        pred_curve.append(np.mean([x["prompt_predictor"] for x in items]))
        lzw_curve.append(np.mean([x["lzw_corr"] for x in items]))
        dom_curve.append(np.mean([x["domain_static"] for x in items]))
        oracle_curve.append(np.mean([x["oracle"] for x in items]))

    x_indices = range(len(pref_lens))
    plt.plot(x_indices, oracle_curve, "k--", label="Answer-Aware Greedy Oracle", linewidth=2)
    plt.plot(x_indices, pred_curve, "b-o", label="Prompt Predictor (K=64)", linewidth=2.5, markersize=8)
    plt.plot(x_indices, dom_curve, "g-s", label="Domain Static (K=64)", linewidth=2, markersize=7)
    plt.plot(x_indices, lzw_curve, "r-d", label="Corrected LZW (K=64, sub=3)", linewidth=2, markersize=7)

    plt.xticks(x_indices, [str(l) for l in pref_lens])
    plt.title("Controlled Within-Sample Prefix Scaling (Same Responses)", fontsize=12, fontweight="bold")
    plt.xlabel("Exact Output Prefix Length (Base Tokens)", fontsize=11)
    plt.ylabel("Response Compression %", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "controlled_prefix_length_scaling.png"))
    plt.close()

    # Plot 2: Predictor vs LZW Advantage across Prefix Lengths
    plt.figure(figsize=(9, 5), dpi=200)
    deltas = [p - l for p, l in zip(pred_curve, lzw_curve)]
    plt.bar(x_indices, deltas, color=["teal" if d > 0 else "crimson" for d in deltas], alpha=0.85)
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
    plt.xticks(x_indices, [str(l) for l in pref_lens])
    plt.title("Predictor Advantage over LZW by Output Length (Within-Sample)", fontsize=12, fontweight="bold")
    plt.xlabel("Exact Output Prefix Length (Base Tokens)", fontsize=11)
    plt.ylabel("Predictor - LZW Delta (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "controlled_prefix_lzw_delta.png"))
    plt.close()

    # Plot 3: Budget K vs Compression
    plt.figure(figsize=(9, 5), dpi=200)
    budgets = [32, 64, 128, 256]
    b_pred = [np.mean([x["prompt_predictor_comp_pct"] for x in eval_by_budget[k]]) for k in budgets]
    b_lzw = [np.mean([x["lzw_corr_comp_pct"] for x in eval_by_budget[k]]) for k in budgets]
    b_dom = [np.mean([x["domain_static_comp_pct"] for x in eval_by_budget[k]]) for k in budgets]
    b_glob = [np.mean([x["global_static_comp_pct"] for x in eval_by_budget[k]]) for k in budgets]

    plt.plot(budgets, b_pred, "b-o", label="Prompt Predictor", linewidth=2.5, markersize=8)
    plt.plot(budgets, b_lzw, "r-d", label="Corrected LZW (sub=3)", linewidth=2, markersize=7)
    plt.plot(budgets, b_dom, "g-s", label="Domain Static", linewidth=2, markersize=7)
    plt.plot(budgets, b_glob, "y-^", label="Global Static", linewidth=2, markersize=7)

    plt.title("Response Compression across Codebook Budgets (K)", fontsize=12, fontweight="bold")
    plt.xlabel("Codebook Budget (K)", fontsize=11)
    plt.ylabel("Mean Response Compression %", fontsize=11)
    plt.xticks(budgets, [str(k) for k in budgets])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "budget_k_comparison.png"))
    plt.close()


def compile_final_tables(
    eval_by_budget: Dict,
    prefix_results: Dict,
    hybrid_results: List[Dict],
    predictor_latencies: List[float],
    budgets: List[int],
) -> Dict:
    out = {}

    # Table 1: Main Comparison with Cluster-Robust 95% CIs
    main_table = {}
    for k in budgets:
        records = eval_by_budget[k]
        main_table[str(k)] = {}

        for regime, key in [
            ("Base Tokenizer", "base_comp_pct"),
            ("Corrected LZW (max_sub=3)", "lzw_corr_comp_pct"),
            ("Upstream LZW (max_sub=4)", "lzw_upstream_comp_pct"),
            ("Global Static", "global_static_comp_pct"),
            ("Domain Static", "domain_static_comp_pct"),
            ("Prompt Predictor", "prompt_predictor_comp_pct"),
            ("Answer-Aware Greedy Oracle", "oracle_comp_pct"),
        ]:
            pt, low, high = compute_cluster_bootstrap_ci(records, key)
            main_table[str(k)][regime] = {
                "mean_pct": round(pt, 2),
                "cluster_ci_95": [round(low, 2), round(high, 2)],
            }

        # Deltas
        diff_lzw, low_lzw, high_lzw = compute_cluster_bootstrap_diff_ci(
            records, "prompt_predictor_comp_pct", "lzw_corr_comp_pct"
        )
        diff_dom, low_dom, high_dom = compute_cluster_bootstrap_diff_ci(
            records, "prompt_predictor_comp_pct", "domain_static_comp_pct"
        )

        main_table[str(k)]["Predictor minus Corrected LZW"] = {
            "mean_delta_pct": round(diff_lzw, 2),
            "cluster_ci_95": [round(low_lzw, 2), round(high_lzw, 2)],
        }
        main_table[str(k)]["Predictor minus Domain Static"] = {
            "mean_delta_pct": round(diff_dom, 2),
            "cluster_ci_95": [round(low_dom, 2), round(high_dom, 2)],
        }

    out["main_comparison_with_cluster_ci"] = main_table

    # Table 2: Controlled Prefix Experiment Results
    pref_table = {}
    for l in [32, 64, 128, 256, 512, 1024, 2048]:
        items = prefix_results[l]["overall"]
        diff_pt, diff_low, diff_high = compute_cluster_bootstrap_diff_ci(
            items, "prompt_predictor", "lzw_corr"
        )
        pref_table[str(l)] = {
            "sample_count": len(items),
            "lzw_corr_pct": round(float(np.mean([x["lzw_corr"] for x in items])), 2),
            "domain_static_pct": round(float(np.mean([x["domain_static"] for x in items])), 2),
            "prompt_predictor_pct": round(float(np.mean([x["prompt_predictor"] for x in items])), 2),
            "oracle_pct": round(float(np.mean([x["oracle"] for x in items])), 2),
            "predictor_minus_lzw_delta": round(diff_pt, 2),
            "cluster_ci_95_delta": [round(diff_low, 2), round(diff_high, 2)],
            "crossover_status": "Predictor Wins" if diff_pt > 0 else "LZW Wins",
        }
    out["controlled_prefix_results_k64"] = pref_table

    # Table 3: Per-Domain Results at K=64
    domains = ["code", "conversation", "reasoning"]
    domain_table = {}
    k64_recs = eval_by_budget[64]
    for dom in domains:
        dom_recs = [r for r in k64_recs if r["domain"] == dom]
        domain_table[dom] = {}
        for regime, key in [
            ("Corrected LZW (max_sub=3)", "lzw_corr_comp_pct"),
            ("Domain Static", "domain_static_comp_pct"),
            ("Prompt Predictor", "prompt_predictor_comp_pct"),
            ("Answer-Aware Greedy Oracle", "oracle_comp_pct"),
        ]:
            pt, low, high = compute_cluster_bootstrap_ci(dom_recs, key)
            domain_table[dom][regime] = {
                "mean_pct": round(pt, 2),
                "cluster_ci_95": [round(low, 2), round(high, 2)],
            }
    out["per_domain_results_k64"] = domain_table

    # Table 4: True Pre-Seeded Hybrid Results at K=128
    out["true_preseeded_hybrid_results_k128"] = hybrid_results

    # Table 5: Predictor Latency Profile
    if predictor_latencies:
        out["predictor_latency_profile"] = {
            "p50_ms": round(float(np.percentile(predictor_latencies, 50)), 3),
            "p90_ms": round(float(np.percentile(predictor_latencies, 90)), 3),
            "p95_ms": round(float(np.percentile(predictor_latencies, 95)), 3),
            "p99_ms": round(float(np.percentile(predictor_latencies, 99)), 3),
            "mean_ms": round(float(np.mean(predictor_latencies)), 3),
        }

    return out


if __name__ == "__main__":
    run_corrected_benchmark()
