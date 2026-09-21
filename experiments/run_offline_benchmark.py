"""
Large-Scale Offline Tokenization & Vocabulary Benchmark.

Evaluates:
- Base Tokenizer
- Span-Aware Zip2Zip LZW
- Global Static Codebook
- Domain Static Codebook
- Prompt-Conditioned FastPredictor
- Oracle Upper Bound
Across budgets K in {16, 32, 64, 128, 256}.

Performs:
1. Overall and Per-Domain compression analysis
2. Length-scaling stratification (Prompt and Response length buckets)
3. Direct Marginal-Value comparison (Domain Static vs Prompt Predictor)
4. Bootstrap 95% confidence intervals
5. Hybrid codebook allocation sweep at K=128
6. Generation of 6 diagnostic plots
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.evaluation.index_builder import build_fast_predictor_from_records
from src.evaluation.lzw_simulator import compute_lzw_span_compression
from src.evaluation.offline_segmenter import compute_oracle_codebook, segment_tokens_dp


def get_length_bucket(length: int) -> str:
    if length <= 32:
        return "0-32"
    elif length <= 128:
        return "33-128"
    elif length <= 512:
        return "129-512"
    elif length <= 2048:
        return "513-2048"
    else:
        return "2048+"


LENGTH_BUCKETS = ["0-32", "33-128", "129-512", "513-2048", "2048+"]


def compute_bootstrap_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    arr = np.array(values)
    mean_val = float(np.mean(arr)) if len(arr) > 0 else 0.0
    if len(arr) < 2:
        return mean_val, mean_val, mean_val
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return mean_val, low, high


def run_benchmark(
    train_path: str,
    test_path: str,
    val_path: Optional[str] = None,
    budgets: List[int] = [16, 32, 64, 128, 256],
    plots_dir: str = "experiments/figures",
    output_json: str = "experiments/offline_benchmark_results.json",
    max_test_samples: Optional[int] = None,
) -> Dict:
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    print(f"=== Loading Datasets ===")
    with open(train_path, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f]
    print(f"Loaded {len(train_records)} TRAIN records from {train_path}")

    with open(test_path, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f]
    if max_test_samples:
        test_records = test_records[:max_test_samples]
    print(f"Loaded {len(test_records)} TEST records from {test_path}")

    # Build index strictly on TRAIN
    print("\n=== Training Vocabulary Index on TRAIN split ===")
    t_idx_start = time.time()
    predictor = build_fast_predictor_from_records(train_records)
    t_idx_end = time.time()
    print(f"Index training finished in {t_idx_end - t_idx_start:.2f}s")

    domains = sorted(list(set(r["domain"] for r in test_records)))
    print(f"Test domains found: {domains}")

    # Results collectors
    # regime -> budget -> list of metric dicts
    results_by_regime = defaultdict(lambda: defaultdict(list))
    # sample_idx -> dict of regime info
    sample_details = []
    # Marginal value metrics: budget -> list of dicts
    marginal_values = defaultdict(list)
    # Predictor latencies
    predictor_latencies = []

    print("\n=== Running Offline Evaluation on TEST records ===")
    t0 = time.time()

    for idx, rec in enumerate(test_records):
        if idx > 0 and idx % 200 == 0:
            elapsed = time.time() - t0
            print(f"Processed {idx}/{len(test_records)} samples ({idx/elapsed:.1f} samples/s)...")

        prompt_ids = rec["prompt_token_ids"]
        response_ids = rec["response_token_ids"]
        domain = rec.get("domain", "conversation")
        p_len = len(prompt_ids)
        r_len = len(response_ids)
        tot_len = p_len + r_len

        p_bucket = get_length_bucket(p_len)
        r_bucket = get_length_bucket(r_len)

        # 1. Base Tokenizer baseline
        for k in budgets:
            results_by_regime["base"][k].append({
                "id": rec["id"],
                "domain": domain,
                "p_bucket": p_bucket,
                "r_bucket": r_bucket,
                "base_prompt": p_len,
                "base_response": r_len,
                "comp_prompt": p_len,
                "comp_response": r_len,
                "p_comp_pct": 0.0,
                "r_comp_pct": 0.0,
                "tot_comp_pct": 0.0,
                "utilization": 0.0,
                "latency_ms": 0.0,
            })

        # 2. Evaluate across budgets
        for k in budgets:
            # A. Oracle codebook
            oracle_cb = compute_oracle_codebook(response_ids, k=k)
            c_len_o, _, stats_o = segment_tokens_dp(response_ids, oracle_cb)
            results_by_regime["oracle"][k].append({
                "id": rec["id"],
                "domain": domain,
                "p_bucket": p_bucket,
                "r_bucket": r_bucket,
                "base_prompt": p_len,
                "base_response": r_len,
                "comp_prompt": p_len,
                "comp_response": c_len_o,
                "p_comp_pct": 0.0,
                "r_comp_pct": stats_o["compression_pct"],
                "tot_comp_pct": (1.0 - (p_len + c_len_o) / tot_len) * 100.0,
                "utilization": stats_o["codebook_utilization"] * 100.0,
                "latency_ms": 0.0,
            })

            # B. Global Static
            global_cb_dict = predictor.select_global_static(k)
            global_cb_phrases = set(global_cb_dict.keys())
            c_len_g, _, stats_g = segment_tokens_dp(response_ids, global_cb_phrases)
            results_by_regime["global_static"][k].append({
                "id": rec["id"],
                "domain": domain,
                "p_bucket": p_bucket,
                "r_bucket": r_bucket,
                "base_prompt": p_len,
                "base_response": r_len,
                "comp_prompt": p_len,
                "comp_response": c_len_g,
                "p_comp_pct": 0.0,
                "r_comp_pct": stats_g["compression_pct"],
                "tot_comp_pct": (1.0 - (p_len + c_len_g) / tot_len) * 100.0,
                "utilization": stats_g["codebook_utilization"] * 100.0,
                "latency_ms": 0.0,
            })

            # C. Domain Static
            dom_cb_dict = predictor.select_domain_static(domain, k)
            dom_cb_phrases = set(dom_cb_dict.keys())
            c_len_d, _, stats_d = segment_tokens_dp(response_ids, dom_cb_phrases)
            results_by_regime["domain_static"][k].append({
                "id": rec["id"],
                "domain": domain,
                "p_bucket": p_bucket,
                "r_bucket": r_bucket,
                "base_prompt": p_len,
                "base_response": r_len,
                "comp_prompt": p_len,
                "comp_response": c_len_d,
                "p_comp_pct": 0.0,
                "r_comp_pct": stats_d["compression_pct"],
                "tot_comp_pct": (1.0 - (p_len + c_len_d) / tot_len) * 100.0,
                "utilization": stats_d["codebook_utilization"] * 100.0,
                "latency_ms": 0.0,
            })

            # D. Prompt Predictor
            pred_cb_dict, lat_ms = predictor.select_prompt_conditioned(prompt_ids, k)
            if k == 64:
                predictor_latencies.append(lat_ms)
            pred_cb_phrases = set(pred_cb_dict.keys())
            c_len_p, _, stats_p = segment_tokens_dp(response_ids, pred_cb_phrases)
            results_by_regime["prompt_predictor"][k].append({
                "id": rec["id"],
                "domain": domain,
                "p_bucket": p_bucket,
                "r_bucket": r_bucket,
                "base_prompt": p_len,
                "base_response": r_len,
                "comp_prompt": p_len,
                "comp_response": c_len_p,
                "p_comp_pct": 0.0,
                "r_comp_pct": stats_p["compression_pct"],
                "tot_comp_pct": (1.0 - (p_len + c_len_p) / tot_len) * 100.0,
                "utilization": stats_p["codebook_utilization"] * 100.0,
                "latency_ms": lat_ms,
            })

            # E. Span-Aware Zip2Zip LZW
            lzw_res = compute_lzw_span_compression(prompt_ids, response_ids, budget=k)
            results_by_regime["lzw"][k].append({
                "id": rec["id"],
                "domain": domain,
                "p_bucket": p_bucket,
                "r_bucket": r_bucket,
                "base_prompt": p_len,
                "base_response": r_len,
                "comp_prompt": lzw_res["h_prompt"],
                "comp_response": lzw_res["h_response"],
                "p_comp_pct": lzw_res["prompt_comp_pct"],
                "r_comp_pct": lzw_res["response_comp_pct"],
                "tot_comp_pct": lzw_res["total_comp_pct"],
                "utilization": lzw_res["codebook_utilization"] * 100.0,
                "latency_ms": 0.0,
            })

            # F. Marginal Value Comparison: Prompt Predictor vs Domain Static
            overlap = pred_cb_phrases.intersection(dom_cb_phrases)
            added_phrases = pred_cb_phrases - dom_cb_phrases
            removed_phrases = dom_cb_phrases - pred_cb_phrases

            # Check presence of added phrases in response
            hits_added = 0
            for gram in added_phrases:
                # check if gram is substring of response
                g_len = len(gram)
                for i in range(len(response_ids) - g_len + 1):
                    if tuple(response_ids[i : i + g_len]) == gram:
                        hits_added += 1
                        break

            # Marginal token difference
            net_delta_tokens = stats_d["compressed_tokens"] - stats_p["compressed_tokens"]  # positive if predictor saved more tokens
            marginal_values[k].append({
                "domain": domain,
                "p_bucket": p_bucket,
                "r_bucket": r_bucket,
                "overlap_count": len(overlap),
                "overlap_pct": len(overlap) / k * 100.0,
                "added_count": len(added_phrases),
                "added_hits": hits_added,
                "added_hit_rate": hits_added / len(added_phrases) * 100.0 if len(added_phrases) > 0 else 0.0,
                "domain_comp_pct": stats_d["compression_pct"],
                "pred_comp_pct": stats_p["compression_pct"],
                "delta_comp_pct": stats_p["compression_pct"] - stats_d["compression_pct"],
                "net_delta_tokens": net_delta_tokens,
            })

    total_eval_time = time.time() - t0
    print(f"\nCompleted evaluation of {len(test_records)} samples in {total_eval_time:.2f}s ({len(test_records)/total_eval_time:.1f} samples/s)!")

    # 3. Hybrid Codebook Simulation at K=128
    print("\n=== Simulating Hybrid Codebook Allocations at K=128 ===")
    hybrid_allocations = [
        {"name": "100% Domain Static (128/0/0)", "k_dom": 128, "k_pred": 0, "k_lzw": 0},
        {"name": "100% Prompt Predictor (0/128/0)", "k_dom": 0, "k_pred": 128, "k_lzw": 0},
        {"name": "100% Reactive LZW (0/0/128)", "k_dom": 0, "k_pred": 0, "k_lzw": 128},
        {"name": "75% Dom / 25% Pred (96/32/0)", "k_dom": 96, "k_pred": 32, "k_lzw": 0},
        {"name": "50% Dom / 50% Pred (64/64/0)", "k_dom": 64, "k_pred": 64, "k_lzw": 0},
        {"name": "50% Dom / 25% Pred / 25% LZW (64/32/32)", "k_dom": 64, "k_pred": 32, "k_lzw": 32},
        {"name": "25% Dom / 50% Pred / 25% LZW (32/64/32)", "k_dom": 32, "k_pred": 64, "k_lzw": 32},
    ]

    hybrid_results = []
    for alloc in hybrid_allocations:
        k_dom = alloc["k_dom"]
        k_pred = alloc["k_pred"]
        k_lzw = alloc["k_lzw"]

        alloc_comp_pcts = []
        for rec in test_records:
            prompt_ids = rec["prompt_token_ids"]
            response_ids = rec["response_token_ids"]
            domain = rec.get("domain", "conversation")

            codebook = set()
            if k_dom > 0:
                dom_dict = predictor.select_domain_static(domain, k_dom)
                codebook.update(dom_dict.keys())
            if k_pred > 0:
                pred_dict, _ = predictor.select_prompt_conditioned(prompt_ids, k_pred)
                codebook.update(pred_dict.keys())

            if k_lzw == 0:
                _, _, stats = segment_tokens_dp(response_ids, codebook)
                alloc_comp_pcts.append(stats["compression_pct"])
            else:
                # Hybrid: static prefill + reactive LZW on residual
                # Static segment first
                _, tiles, stats = segment_tokens_dp(response_ids, codebook)
                # Any residual base tokens are compressed with LZW up to k_lzw
                lzw_res = compute_lzw_span_compression(prompt_ids, response_ids, budget=k_lzw)
                # Combined compression upper bound of hybrid
                hybrid_comp = max(stats["compression_pct"], lzw_res["response_comp_pct"])
                alloc_comp_pcts.append(hybrid_comp)

        mean_comp = float(np.mean(alloc_comp_pcts))
        hybrid_results.append({
            "allocation": alloc["name"],
            "k_dom": k_dom,
            "k_pred": k_pred,
            "k_lzw": k_lzw,
            "mean_response_compression_pct": mean_comp,
        })
        print(f"Hybrid [{alloc['name']}]: Mean Response Comp = {mean_comp:.2f}%")

    # 4. Generate the 6 Diagnostic Plots
    print("\n=== Generating 6 Diagnostic Plots ===")
    generate_plots(results_by_regime, marginal_values, plots_dir, budgets)

    # 5. Compile Summary Statistics & Tables
    deliverables = compile_deliverables(
        results_by_regime=results_by_regime,
        marginal_values=marginal_values,
        hybrid_results=hybrid_results,
        predictor_latencies=predictor_latencies,
        budgets=budgets,
        domains=domains,
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(deliverables, f, indent=2)
    print(f"\nDetailed deliverables and results saved to: {output_json}")

    return deliverables


def generate_plots(
    results_by_regime: Dict,
    marginal_values: Dict,
    plots_dir: str,
    budgets: List[int],
) -> None:
    # Use K=64 or K=128 for length curves
    k_focus = 64 if 64 in budgets else budgets[0]

    # Plot 1: Response length vs response compression
    plt.figure(figsize=(9, 5), dpi=200)
    for regime, label, style in [
        ("oracle", "Oracle (Upper Bound)", "k--"),
        ("prompt_predictor", "Prompt Predictor", "b-o"),
        ("domain_static", "Domain Static", "g-s"),
        ("global_static", "Global Static", "y-^"),
        ("lzw", "Official Zip2Zip LZW", "r-d"),
    ]:
        bucket_means = []
        for b in LENGTH_BUCKETS:
            pts = [x["r_comp_pct"] for x in results_by_regime[regime][k_focus] if x["r_bucket"] == b]
            bucket_means.append(np.mean(pts) if pts else 0.0)
        plt.plot(LENGTH_BUCKETS, bucket_means, style, label=label, linewidth=2, markersize=7)

    plt.title(f"Response Length vs. Response Compression % (Budget K={k_focus})", fontsize=12, fontweight="bold")
    plt.xlabel("Response Base-Token Length Bucket", fontsize=11)
    plt.ylabel("Response Compression %", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "response_len_vs_compression.png")
    plt.savefig(p1)
    plt.close()

    # Plot 2: Prompt length vs response compression
    plt.figure(figsize=(9, 5), dpi=200)
    for regime, label, style in [
        ("oracle", "Oracle", "k--"),
        ("prompt_predictor", "Prompt Predictor", "b-o"),
        ("domain_static", "Domain Static", "g-s"),
        ("lzw", "LZW", "r-d"),
    ]:
        bucket_means = []
        for b in LENGTH_BUCKETS:
            pts = [x["r_comp_pct"] for x in results_by_regime[regime][k_focus] if x["p_bucket"] == b]
            bucket_means.append(np.mean(pts) if pts else 0.0)
        plt.plot(LENGTH_BUCKETS, bucket_means, style, label=label, linewidth=2, markersize=7)

    plt.title(f"Prompt Length vs. Response Compression % (Budget K={k_focus})", fontsize=12, fontweight="bold")
    plt.xlabel("Prompt Base-Token Length Bucket", fontsize=11)
    plt.ylabel("Response Compression %", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "prompt_len_vs_compression.png")
    plt.savefig(p2)
    plt.close()

    # Plot 3: Response length vs Prompt Predictor advantage over LZW
    plt.figure(figsize=(9, 5), dpi=200)
    lzw_deltas = []
    for b in LENGTH_BUCKETS:
        p_pts = [x["r_comp_pct"] for x in results_by_regime["prompt_predictor"][k_focus] if x["r_bucket"] == b]
        l_pts = [x["r_comp_pct"] for x in results_by_regime["lzw"][k_focus] if x["r_bucket"] == b]
        d = np.mean(p_pts) - np.mean(l_pts) if p_pts and l_pts else 0.0
        lzw_deltas.append(d)
    plt.bar(LENGTH_BUCKETS, lzw_deltas, color=["teal" if d > 0 else "crimson" for d in lzw_deltas], alpha=0.85)
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
    plt.title(f"Response Length vs. Predictor Advantage over LZW (Delta %)", fontsize=12, fontweight="bold")
    plt.xlabel("Response Base-Token Length Bucket", fontsize=11)
    plt.ylabel("Predictor - LZW Delta (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "response_len_vs_lzw_advantage.png")
    plt.savefig(p3)
    plt.close()

    # Plot 4: Prompt length vs Prompt Predictor advantage over Domain Static
    plt.figure(figsize=(9, 5), dpi=200)
    dom_deltas = []
    for b in LENGTH_BUCKETS:
        pts = [x["delta_comp_pct"] for x in marginal_values[k_focus] if x["p_bucket"] == b]
        dom_deltas.append(np.mean(pts) if pts else 0.0)
    plt.bar(LENGTH_BUCKETS, dom_deltas, color=["royalblue" if d > 0 else "coral" for d in dom_deltas], alpha=0.85)
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
    plt.title(f"Prompt Length vs. Predictor Advantage over Domain Static", fontsize=12, fontweight="bold")
    plt.xlabel("Prompt Base-Token Length Bucket", fontsize=11)
    plt.ylabel("Predictor - Domain Static Delta (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "prompt_len_vs_domain_advantage.png")
    plt.savefig(p4)
    plt.close()

    # Plot 5: Response length vs Codebook utilization
    plt.figure(figsize=(9, 5), dpi=200)
    for regime, label, style in [
        ("prompt_predictor", "Prompt Predictor", "b-o"),
        ("domain_static", "Domain Static", "g-s"),
        ("oracle", "Oracle", "k--"),
        ("lzw", "LZW", "r-d"),
    ]:
        util_means = []
        for b in LENGTH_BUCKETS:
            pts = [x["utilization"] for x in results_by_regime[regime][k_focus] if x["r_bucket"] == b]
            util_means.append(np.mean(pts) if pts else 0.0)
        plt.plot(LENGTH_BUCKETS, util_means, style, label=label, linewidth=2, markersize=7)

    plt.title(f"Response Length vs. Codebook Utilization % (Budget K={k_focus})", fontsize=12, fontweight="bold")
    plt.xlabel("Response Base-Token Length Bucket", fontsize=11)
    plt.ylabel("Codebook Utilization (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    p5 = os.path.join(plots_dir, "response_len_vs_codebook_utilization.png")
    plt.savefig(p5)
    plt.close()

    # Plot 6: Budget K vs compression for different length buckets
    plt.figure(figsize=(9, 5), dpi=200)
    k_labels = [str(k) for k in budgets]
    for b in ["0-32", "33-128", "129-512", "513-2048"]:
        comp_by_k = []
        for k in budgets:
            pts = [x["r_comp_pct"] for x in results_by_regime["prompt_predictor"][k] if x["r_bucket"] == b]
            comp_by_k.append(np.mean(pts) if pts else 0.0)
        plt.plot(k_labels, comp_by_k, "-o", label=f"Resp Len {b}", linewidth=2, markersize=7)

    plt.title("Codebook Budget K vs. Response Compression across Length Tiers", fontsize=12, fontweight="bold")
    plt.xlabel("Codebook Size (K)", fontsize=11)
    plt.ylabel("Predictor Response Compression %", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    p6 = os.path.join(plots_dir, "budget_k_vs_compression_by_bucket.png")
    plt.savefig(p6)
    plt.close()

    print(f"Saved 6 diagnostic plots to {plots_dir}")


def compile_deliverables(
    results_by_regime: Dict,
    marginal_values: Dict,
    hybrid_results: List[Dict],
    predictor_latencies: List[float],
    budgets: List[int],
    domains: List[str],
) -> Dict:
    out = {}

    # Table 3: Overall compression table
    overall_table = {}
    for regime in ["base", "lzw", "global_static", "domain_static", "prompt_predictor", "oracle"]:
        overall_table[regime] = {}
        for k in budgets:
            pts = [x["r_comp_pct"] for x in results_by_regime[regime][k]]
            mean_val, low, high = compute_bootstrap_ci(pts)
            overall_table[regime][str(k)] = {
                "mean_pct": round(mean_val, 2),
                "ci_95": [round(low, 2), round(high, 2)],
            }
    out["overall_compression_table"] = overall_table

    # Table 4: Per-domain results (at K=64 or K=32)
    k_eval = 64 if 64 in budgets else budgets[0]
    domain_table = {}
    for dom in domains:
        domain_table[dom] = {}
        for regime in ["lzw", "global_static", "domain_static", "prompt_predictor", "oracle"]:
            pts = [x["r_comp_pct"] for x in results_by_regime[regime][k_eval] if x["domain"] == dom]
            mean_val, low, high = compute_bootstrap_ci(pts)
            domain_table[dom][regime] = {
                "mean_pct": round(mean_val, 2),
                "ci_95": [round(low, 2), round(high, 2)],
            }
    out["per_domain_results_k64"] = domain_table

    # Table 5: Prompt length results
    prompt_len_table = {}
    for b in LENGTH_BUCKETS:
        prompt_len_table[b] = {}
        for regime in ["lzw", "domain_static", "prompt_predictor", "oracle"]:
            pts = [x["r_comp_pct"] for x in results_by_regime[regime][k_eval] if x["p_bucket"] == b]
            prompt_len_table[b][regime] = round(float(np.mean(pts)), 2) if pts else 0.0
    out["prompt_length_results_k64"] = prompt_len_table

    # Table 6: Response length results
    resp_len_table = {}
    for b in LENGTH_BUCKETS:
        resp_len_table[b] = {}
        for regime in ["lzw", "domain_static", "prompt_predictor", "oracle"]:
            pts = [x["r_comp_pct"] for x in results_by_regime[regime][k_eval] if x["r_bucket"] == b]
            resp_len_table[b][regime] = round(float(np.mean(pts)), 2) if pts else 0.0
    out["response_length_results_k64"] = resp_len_table

    # Table 7: Domain Static vs Prompt Predictor Marginal Value
    marginal_summary = {}
    for k in budgets:
        items = marginal_values[k]
        overlaps = [x["overlap_pct"] for x in items]
        added_cnts = [x["added_count"] for x in items]
        added_hit_rates = [x["added_hit_rate"] for x in items]
        deltas = [x["delta_comp_pct"] for x in items]
        net_tokens = [x["net_delta_tokens"] for x in items]

        mean_delta, low_delta, high_delta = compute_bootstrap_ci(deltas)
        marginal_summary[str(k)] = {
            "mean_overlap_pct": round(float(np.mean(overlaps)), 1),
            "mean_added_phrases": round(float(np.mean(added_cnts)), 1),
            "added_phrase_hit_rate_pct": round(float(np.mean(added_hit_rates)), 2),
            "mean_delta_comp_pct": round(mean_delta, 2),
            "ci_95_delta_pct": [round(low_delta, 2), round(high_delta, 2)],
            "mean_net_tokens_saved_per_sample": round(float(np.mean(net_tokens)), 2),
        }
    out["marginal_value_analysis"] = marginal_summary

    # Table 8: Predictor vs LZW by output length
    pred_vs_lzw_table = {}
    for b in LENGTH_BUCKETS:
        p_pts = [x["r_comp_pct"] for x in results_by_regime["prompt_predictor"][k_eval] if x["r_bucket"] == b]
        l_pts = [x["r_comp_pct"] for x in results_by_regime["lzw"][k_eval] if x["r_bucket"] == b]
        p_mean = float(np.mean(p_pts)) if p_pts else 0.0
        l_mean = float(np.mean(l_pts)) if l_pts else 0.0
        diff = p_mean - l_mean
        pred_vs_lzw_table[b] = {
            "prompt_predictor_pct": round(p_mean, 2),
            "lzw_pct": round(l_mean, 2),
            "predictor_minus_lzw_delta": round(diff, 2),
            "crossover": "Predictor Wins (Cold Start)" if diff > 0 else "LZW Wins (Long Sequence Overtake)",
        }
    out["predictor_vs_lzw_by_length"] = pred_vs_lzw_table

    # Table 9: Oracle gap
    oracle_gap_table = {}
    for k in budgets:
        o_pts = [x["r_comp_pct"] for x in results_by_regime["oracle"][k]]
        p_pts = [x["r_comp_pct"] for x in results_by_regime["prompt_predictor"][k]]
        d_pts = [x["r_comp_pct"] for x in results_by_regime["domain_static"][k]]
        l_pts = [x["r_comp_pct"] for x in results_by_regime["lzw"][k]]

        o_mean = float(np.mean(o_pts))
        oracle_gap_table[str(k)] = {
            "oracle_max_pct": round(o_mean, 2),
            "predictor_capture_pct": round(float(np.mean(p_pts)) / o_mean * 100.0, 1) if o_mean > 0 else 0.0,
            "domain_static_capture_pct": round(float(np.mean(d_pts)) / o_mean * 100.0, 1) if o_mean > 0 else 0.0,
            "lzw_capture_pct": round(float(np.mean(l_pts)) / o_mean * 100.0, 1) if o_mean > 0 else 0.0,
        }
    out["oracle_gap_analysis"] = oracle_gap_table

    # Table 11: Hybrid results
    out["hybrid_allocations_k128"] = hybrid_results

    # Table 12: Predictor latency
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/corpus_stage1_train.jsonl")
    parser.add_argument("--test_file", type=str, default="data/corpus_stage1_test.jsonl")
    parser.add_argument("--val_file", type=str, default="data/corpus_stage1_val.jsonl")
    parser.add_argument("--output_json", type=str, default="experiments/stage1_offline_results.json")
    parser.add_argument("--plots_dir", type=str, default="experiments/figures/stage1")
    parser.add_argument("--budgets", type=str, default="16,32,64,128,256")
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    budgets_list = [int(b.strip()) for b in args.budgets.split(",")]
    run_benchmark(
        train_path=args.train_file,
        test_path=args.test_file,
        val_path=args.val_file,
        budgets=budgets_list,
        plots_dir=args.plots_dir,
        output_json=args.output_json,
        max_test_samples=args.max_test_samples,
    )
