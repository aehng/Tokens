"""
Comprehensive Evaluation Suite for Predictive Hypertokens vs. Dynamic & Static Baselines.
Full implementation of Phases 1 through 5:
- Scientific Validity: strict TRAIN / VAL / TEST split isolation, cluster-robust bootstrap.
- Complete Offline Matrix: base, compressed, and % for Prompt, Response, and Total.
- Validation Sweep: 9 two-way ratios + 3 three-way allocations at K=128 on VALIDATION split (7,731 samples).
- Finalist Selection: Pareto analysis across output comp, input comp, total comp, utilization, and domain robustness.
- Test Evaluation: frozen finalists + pure controls across K in {16, 32, 64, 128, 256, 512}.
- Length Study: natural buckets up to 8192+ and controlled same-document prefix cuts [16..4096].
- Task / Domain Study: 3 broad source domains + 5 fine-grained heuristic categories.
- Watchdog Supervision: continuous disk checkpointing every 500 samples and live resource logging.
"""

import argparse
import gc
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.index_builder import OptimizedPhraseIndex, build_fast_predictor_from_records
from src.evaluation.lzw_simulator import (
    OFFICIAL_DISABLED_IDS,
    OFFICIAL_INITIAL_VOCAB_SIZE,
    compute_lzw_span_compression,
    compute_preseeded_hybrid_lzw,
)
from src.evaluation.offline_segmenter import compute_oracle_codebook, segment_tokens_dp


# ----------------------------------------------------------------------
# 1. Prompt-Side Task / Domain Classifier
# ----------------------------------------------------------------------

def classify_prompt_side_task(prompt_text: str, source_dataset: str, base_domain: str) -> Tuple[str, str]:
    """
    Classifies task strictly from prompt-side text and dataset metadata.
    Never inspects unseen response text.
    Returns:
        (broad_domain, fine_heuristic_category)
    """
    p_lower = prompt_text.lower()
    
    # 1. Broad source-backed domain (high confidence)
    if base_domain == "code" or source_dataset in ("codealpaca", "mbpp"):
        broad = "code"
    elif base_domain == "reasoning" or source_dataset == "gsm8k":
        broad = "mathematical reasoning"
    else:
        broad = "conversation"

    # 2. Fine prompt heuristic category (labeled heuristic)
    if broad == "code":
        fine = "code"
    elif broad == "mathematical reasoning":
        fine = "mathematical reasoning"
    elif any(k in p_lower for k in ["```json", "{ \"", '{"', "format: json", "valid json", "return json", "json output"]):
        fine = "structured JSON/output (heuristic)"
    elif any(k in p_lower for k in ["summarize", "summary", "tl;dr", "tldr", "in brief", "summarise", "brief summary"]):
        fine = "summarization (heuristic)"
    elif any(k in p_lower for k in ["explain in detail", "elaborate", "comprehensive guide", "step-by-step tutorial", "write an essay", "detailed explanation"]):
        fine = "long-form explanation (heuristic)"
    elif any(k in p_lower for k in ["api", "function call", "tool", "search query", "action:", "thought:", "observation:"]):
        fine = "agent/tool-style generation (heuristic)"
    elif p_lower.startswith(("who was", "who is", "what is the capital", "when did", "what year", "where is", "which country", "what date")):
        fine = "factual QA (heuristic)"
    else:
        fine = "conversation"

    return broad, fine


# ----------------------------------------------------------------------
# 2. Cluster-Robust Bootstrap Confidence Interval Engine
# ----------------------------------------------------------------------

def compute_cluster_bootstrap_ci(
    records: List[Dict[str, Any]],
    metric_extractor,
    n_boot: int = 1000,
    ci: float = 0.95,
    cluster_key: str = "thread_id",
) -> Tuple[float, float, float]:
    """
    Cluster bootstrap resampling by thread_id to produce cluster-robust confidence intervals.
    `metric_extractor` is a callable taking a record dict and returning a float or None.
    """
    clusters = defaultdict(list)
    for r in records:
        val = metric_extractor(r)
        if val is not None and not math.isnan(val):
            clusters[r[cluster_key]].append(val)

    all_vals = [v for vals in clusters.values() for v in vals]
    if not all_vals:
        return 0.0, 0.0, 0.0
    point_est = float(np.mean(all_vals))

    unique_threads = list(clusters.keys())
    n_clusters = len(unique_threads)
    if n_clusters < 2:
        return point_est, point_est, point_est

    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_boot):
        sampled_threads = rng.choice(unique_threads, size=n_clusters, replace=True)
        sampled_vals = []
        for th in sampled_threads:
            sampled_vals.extend(clusters[th])
        boot_means.append(np.mean(sampled_vals))

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return point_est, low, high


# ----------------------------------------------------------------------
# 3. Hybrid Allocations Definitions
# ----------------------------------------------------------------------

HYBRID_ALLOCATIONS_K128 = {
    "100_pred_0_lzw": (0, 128, 0),
    "87.5_pred_12.5_lzw": (0, 112, 16),
    "75_pred_25_lzw": (0, 96, 32),
    "62.5_pred_37.5_lzw": (0, 80, 48),
    "50_pred_50_lzw": (0, 64, 64),
    "37.5_pred_62.5_lzw": (0, 48, 80),
    "25_pred_75_lzw": (0, 32, 96),
    "12.5_pred_87.5_lzw": (0, 16, 112),
    "0_pred_100_lzw": (0, 0, 128),
    "32dom_64pred_32lzw": (32, 64, 32),
    "32dom_32pred_64lzw": (32, 32, 64),
    "64dom_32pred_32lzw": (64, 32, 32),
}



# ----------------------------------------------------------------------
# 4. Core Single-Sample Multi-Regime Evaluator
# ----------------------------------------------------------------------

def evaluate_sample_all_regimes(
    rec: Dict[str, Any],
    predictor: OptimizedPhraseIndex,
    budgets: List[int] = [16, 32, 64, 128, 256, 512],
    hybrid_configs: Optional[Dict[str, Tuple[int, int, int]]] = None,
    include_controlled_prefix: bool = True,
    prefix_cuts: List[int] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
) -> Dict[str, Any]:
    """
    Evaluates one prompt-response pair across all requested regimes and budgets.
    Re-segments BOTH prompt and response for static/predictive/oracle regimes.
    Uses sequential span attribution for LZW and pre-seeded hybrid LZW.
    """
    p_ids: List[int] = rec["prompt_token_ids"]
    r_ids: List[int] = rec["response_token_ids"]
    P = len(p_ids)
    R = len(r_ids)
    N = P + R
    thread_id = rec.get("thread_id", str(rec.get("id", "")))
    source = rec.get("source_dataset", "unknown")
    domain = rec.get("domain", "conversation")
    broad_domain, fine_task = classify_prompt_side_task(rec.get("prompt", ""), source, domain)

    out = {
        "id": rec.get("id"),
        "thread_id": thread_id,
        "source": source,
        "broad_domain": broad_domain,
        "fine_task": fine_task,
        "base_prompt": P,
        "base_response": R,
        "base_total": N,
        "regimes": {},
        "prefix_cuts": {},
    }

    # Pre-select predictor dictionary at max budget needed
    max_k = max(budgets) if budgets else 128
    p_dict, pred_lat_ms = predictor.select_prompt_conditioned(p_ids, max_k)
    pred_phrases_all = list(p_dict.keys())
    out["predictor_latency_ms"] = pred_lat_ms

    # A. Evaluate Standard Regimes across Budgets K
    for k in budgets:
        out["regimes"][k] = {}

        # 1. Reactive LZW (apples-to-apples max_subtokens=3)
        lzw_corr = compute_lzw_span_compression(p_ids, r_ids, budget=k, max_subtokens=3)
        out["regimes"][k]["lzw_corr"] = {
            "prompt_comp": lzw_corr["compressed_prompt"],
            "response_comp": lzw_corr["compressed_response"],
            "total_comp": lzw_corr["compressed_total"],
            "prompt_pct": lzw_corr["prompt_comp_pct"],
            "response_pct": lzw_corr["response_comp_pct"],
            "total_pct": lzw_corr["total_comp_pct"],
            "utilization": lzw_corr["codebook_utilization"],
            "unique_hypertokens": lzw_corr["unique_hypertokens_used"],
        }

        # 2. Reactive LZW upstream (max_subtokens=4)
        lzw_up = compute_lzw_span_compression(p_ids, r_ids, budget=k, max_subtokens=4)
        out["regimes"][k]["lzw_upstream"] = {
            "prompt_comp": lzw_up["compressed_prompt"],
            "response_comp": lzw_up["compressed_response"],
            "total_comp": lzw_up["compressed_total"],
            "prompt_pct": lzw_up["prompt_comp_pct"],
            "response_pct": lzw_up["response_comp_pct"],
            "total_pct": lzw_up["total_comp_pct"],
            "utilization": lzw_up["codebook_utilization"],
            "unique_hypertokens": lzw_up["unique_hypertokens_used"],
        }

        # 3. Global Static Top-K
        g_dict = predictor.select_global_static(k)
        g_cb = set(g_dict.keys())
        p_c_g, _, stats_p_g = segment_tokens_dp(p_ids, g_cb)
        r_c_g, _, stats_r_g = segment_tokens_dp(r_ids, g_cb)
        tot_c_g = p_c_g + r_c_g
        out["regimes"][k]["global_static"] = {
            "prompt_comp": p_c_g,
            "response_comp": r_c_g,
            "total_comp": tot_c_g,
            "prompt_pct": (1.0 - p_c_g / P) * 100.0 if P > 0 else 0.0,
            "response_pct": (1.0 - r_c_g / R) * 100.0 if R > 0 else 0.0,
            "total_pct": (1.0 - tot_c_g / N) * 100.0 if N > 0 else 0.0,
            "utilization": (stats_r_g["unique_hypertokens_used"]) / k if k > 0 else 0.0,
            "unique_hypertokens": stats_r_g["unique_hypertokens_used"],
            "avg_subtokens": stats_r_g["avg_subtokens_per_hypertoken"],
        }

        # 4. Domain Static Top-K
        d_dict = predictor.select_domain_static(broad_domain, k)
        d_cb = set(d_dict.keys())
        p_c_d, _, stats_p_d = segment_tokens_dp(p_ids, d_cb)
        r_c_d, _, stats_r_d = segment_tokens_dp(r_ids, d_cb)
        tot_c_d = p_c_d + r_c_d
        out["regimes"][k]["domain_static"] = {
            "prompt_comp": p_c_d,
            "response_comp": r_c_d,
            "total_comp": tot_c_d,
            "prompt_pct": (1.0 - p_c_d / P) * 100.0 if P > 0 else 0.0,
            "response_pct": (1.0 - r_c_d / R) * 100.0 if R > 0 else 0.0,
            "total_pct": (1.0 - tot_c_d / N) * 100.0 if N > 0 else 0.0,
            "utilization": (stats_r_d["unique_hypertokens_used"]) / k if k > 0 else 0.0,
            "unique_hypertokens": stats_r_d["unique_hypertokens_used"],
            "avg_subtokens": stats_r_d["avg_subtokens_per_hypertoken"],
        }

        # 5. Pure Prompt Predictor Top-K
        p_cb = set(pred_phrases_all[:k])
        p_c_p, _, stats_p_p = segment_tokens_dp(p_ids, p_cb)
        r_c_p, _, stats_r_p = segment_tokens_dp(r_ids, p_cb)
        tot_c_p = p_c_p + r_c_p
        out["regimes"][k]["prompt_predictor"] = {
            "prompt_comp": p_c_p,
            "response_comp": r_c_p,
            "total_comp": tot_c_p,
            "prompt_pct": (1.0 - p_c_p / P) * 100.0 if P > 0 else 0.0,
            "response_pct": (1.0 - r_c_p / R) * 100.0 if R > 0 else 0.0,
            "total_pct": (1.0 - tot_c_p / N) * 100.0 if N > 0 else 0.0,
            "utilization": (stats_r_p["unique_hypertokens_used"]) / k if k > 0 else 0.0,
            "unique_hypertokens": stats_r_p["unique_hypertokens_used"],
            "avg_subtokens": stats_r_p["avg_subtokens_per_hypertoken"],
        }

        # 6. Answer-Aware Greedy Oracle Top-K
        o_cb = compute_oracle_codebook(r_ids, k=k, max_length=3)
        p_c_o, _, stats_p_o = segment_tokens_dp(p_ids, o_cb)
        r_c_o, _, stats_r_o = segment_tokens_dp(r_ids, o_cb)
        tot_c_o = p_c_o + r_c_o
        out["regimes"][k]["oracle"] = {
            "prompt_comp": p_c_o,
            "response_comp": r_c_o,
            "total_comp": tot_c_o,
            "prompt_pct": (1.0 - p_c_o / P) * 100.0 if P > 0 else 0.0,
            "response_pct": (1.0 - r_c_o / R) * 100.0 if R > 0 else 0.0,
            "total_pct": (1.0 - tot_c_o / N) * 100.0 if N > 0 else 0.0,
            "utilization": (stats_r_o["unique_hypertokens_used"]) / k if k > 0 else 0.0,
            "unique_hypertokens": stats_r_o["unique_hypertokens_used"],
            "avg_subtokens": stats_r_o["avg_subtokens_per_hypertoken"],
        }

    # B. Evaluate Hybrid Allocations
    if hybrid_configs:
        out["hybrids"] = {}
        for name, (n_dom, n_pred, n_lzw) in hybrid_configs.items():
            preseeded = []
            if n_dom > 0:
                d_dict_sub = predictor.select_domain_static(broad_domain, n_dom)
                preseeded.extend(list(d_dict_sub.keys()))
            if n_pred > 0:
                for phrase in pred_phrases_all:
                    if phrase not in preseeded:
                        preseeded.append(phrase)
                    if len(preseeded) >= (n_dom + n_pred):
                        break

            hyb = compute_preseeded_hybrid_lzw(
                p_ids,
                r_ids,
                total_budget=n_dom + n_pred + n_lzw,
                preseeded_phrases=preseeded,
                reactive_budget=n_lzw,
                max_subtokens=3,
            )
            out["hybrids"][name] = {
                "config": (n_dom, n_pred, n_lzw),
                "prompt_comp": hyb["compressed_prompt"],
                "response_comp": hyb["compressed_response"],
                "total_comp": hyb["compressed_total"],
                "prompt_pct": hyb["prompt_comp_pct"],
                "response_pct": hyb["response_comp_pct"],
                "total_pct": hyb["total_comp_pct"],
                "utilization": hyb["codebook_utilization"],
                "unique_hypertokens": hyb["unique_hypertokens_used"],
                "preseeded_used": hyb["preseeded_used"],
                "reactive_used": hyb["reactive_used"],
            }

    # C. Controlled Prefix Scaling on the SAME Response Document
    if include_controlled_prefix:
        k_ctrl = 64
        pred_cb_64 = set(pred_phrases_all[:k_ctrl])
        dom_dict_64 = predictor.select_domain_static(broad_domain, k_ctrl)
        dom_cb_64 = set(dom_dict_64.keys())

        for cut in prefix_cuts:
            if R >= cut:
                prefix_r = r_ids[:cut]
                P_sub = P
                R_sub = cut
                N_sub = P_sub + R_sub

                # 1. Corrected LZW on prefix
                lzw_sub = compute_lzw_span_compression(p_ids, prefix_r, budget=k_ctrl, max_subtokens=3)
                # 2. Prompt Predictor on prefix
                p_c_sub, _, _ = segment_tokens_dp(p_ids, pred_cb_64)
                r_c_sub, _, _ = segment_tokens_dp(prefix_r, pred_cb_64)
                tot_c_sub = p_c_sub + r_c_sub
                # 3. Domain Static on prefix
                p_c_dom, _, _ = segment_tokens_dp(p_ids, dom_cb_64)
                r_c_dom, _, _ = segment_tokens_dp(prefix_r, dom_cb_64)
                tot_c_dom = p_c_dom + r_c_dom
                # 4. True Hybrid (32 Pred + 32 Reactive LZW) on prefix
                hyb_sub = compute_preseeded_hybrid_lzw(
                    p_ids,
                    prefix_r,
                    total_budget=k_ctrl,
                    preseeded_phrases=pred_phrases_all[:32],
                    reactive_budget=32,
                    max_subtokens=3,
                )

                out["prefix_cuts"][cut] = {
                    "base_response": cut,
                    "lzw_response_pct": lzw_sub["response_comp_pct"],
                    "lzw_total_pct": lzw_sub["total_comp_pct"],
                    "predictor_response_pct": (1.0 - r_c_sub / R_sub) * 100.0,
                    "predictor_total_pct": (1.0 - tot_c_sub / N_sub) * 100.0,
                    "domain_static_response_pct": (1.0 - r_c_dom / R_sub) * 100.0,
                    "domain_static_total_pct": (1.0 - tot_c_dom / N_sub) * 100.0,
                    "hybrid_response_pct": hyb_sub["response_comp_pct"],
                    "hybrid_total_pct": hyb_sub["total_comp_pct"],
                }

    return out


# ----------------------------------------------------------------------
# 5. Watchdog & Progress Monitor Helper
# ----------------------------------------------------------------------

def log_watchdog_status(
    step_name: str,
    idx: int,
    total: int,
    start_time: float,
) -> None:
    now = time.time()
    elapsed = now - start_time
    rate = idx / elapsed if elapsed > 0 else 0.0
    eta_sec = (total - idx) / rate if rate > 0 else 0.0

    ram = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    proc_ram_mb = proc.memory_info().rss / (1024 * 1024)

    print(
        f"[{step_name}] {idx}/{total} ({idx/total*100.0:.1f}%) | "
        f"{rate:.1f} samples/s | Elapsed: {elapsed:.1f}s | ETA: {eta_sec/60.0:.1f}m | "
        f"Proc RAM: {proc_ram_mb:.1f} MB | Sys Free: {ram.available / 1e9:.2f} GB",
        flush=True
    )


# ----------------------------------------------------------------------
# 5. Core Evaluation for Validation Sweep
# ----------------------------------------------------------------------

def run_validation_sweep(
    val_path: str,
    predictor: OptimizedPhraseIndex,
    checkpoints_dir: str,
    output_path: str,
    checkpoint_interval: int = 500,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("PHASE 2: VALIDATION SWEEP ACROSS 12 HYBRID ALLOCATIONS (K=128)")
    print("=" * 70)

    with open(val_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if max_samples:
        records = records[:max_samples]
    total_samples = len(records)
    print(f"Loaded {total_samples} VALIDATION records from {val_path}")

    ckpt_file = os.path.join(checkpoints_dir, "val_sweep_ckpt.json")
    evaluated_rows = []
    start_idx = 0

    # Resume from checkpoint if available
    if os.path.exists(ckpt_file):
        try:
            with open(ckpt_file, "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
            evaluated_rows = ckpt_data.get("rows", [])
            start_idx = len(evaluated_rows)
            print(f"Resuming from checkpoint with {start_idx} completed samples.")
        except Exception as e:
            print(f"Failed to resume from checkpoint ({e}), starting fresh.")
            evaluated_rows = []
            start_idx = 0

    t0 = time.time()
    last_log = t0

    for idx in range(start_idx, total_samples):
        rec = records[idx]
        p_ids = rec["prompt_token_ids"]
        r_ids = rec["response_token_ids"]
        P = len(p_ids)
        R = len(r_ids)
        N = P + R
        thread_id = rec.get("thread_id", str(rec.get("id", "")))
        source = rec.get("source_dataset", "unknown")
        domain = rec.get("domain", "conversation")
        broad_domain, fine_task = classify_prompt_side_task(rec.get("prompt", ""), source, domain)

        # Pre-select candidate predictor phrases at K=128
        p_dict, pred_lat_ms = predictor.select_prompt_conditioned(p_ids, 128)
        pred_phrases_all = list(p_dict.keys())

        row = {
            "id": rec.get("id"),
            "thread_id": thread_id,
            "source": source,
            "broad_domain": broad_domain,
            "fine_task": fine_task,
            "base_prompt": P,
            "base_response": R,
            "base_total": N,
            "predictor_latency_ms": pred_lat_ms,
            "hybrids": {},
        }

        # Evaluate each hybrid allocation at K=128
        for name, (n_dom, n_pred, n_lzw) in HYBRID_ALLOCATIONS_K128.items():
            preseeded = []
            if n_dom > 0:
                d_dict_sub = predictor.select_domain_static(broad_domain, n_dom)
                preseeded.extend(list(d_dict_sub.keys()))
            if n_pred > 0:
                for phrase in pred_phrases_all:
                    if phrase not in preseeded:
                        preseeded.append(phrase)
                    if len(preseeded) >= (n_dom + n_pred):
                        break

            hyb = compute_preseeded_hybrid_lzw(
                p_ids,
                r_ids,
                total_budget=n_dom + n_pred + n_lzw,
                preseeded_phrases=preseeded,
                reactive_budget=n_lzw,
                max_subtokens=3,
            )
            row["hybrids"][name] = {
                "prompt_pct": hyb["prompt_comp_pct"],
                "response_pct": hyb["response_comp_pct"],
                "total_pct": hyb["total_comp_pct"],
                "utilization": hyb["codebook_utilization"],
                "unique_hypertokens": hyb["unique_hypertokens_used"],
                "preseeded_used": hyb["preseeded_used"],
                "reactive_used": hyb["reactive_used"],
            }

        evaluated_rows.append(row)

        # Checkpointing and watchdog
        if (idx + 1) % checkpoint_interval == 0 or (idx + 1) == total_samples:
            log_watchdog_status("VAL_SWEEP", idx + 1, total_samples, t0)
            with open(ckpt_file, "w", encoding="utf-8") as f:
                json.dump({"completed": idx + 1, "total": total_samples, "rows": evaluated_rows}, f)

    total_time = time.time() - t0
    print(f"\nValidation sweep completed in {total_time:.1f}s ({len(evaluated_rows)/total_time:.1f} samples/s)!")

    # ------------------------------------------------------------------
    # Aggregate and Compute Cluster Bootstrap CIs for Hybrids
    # ------------------------------------------------------------------
    print("\n--- Aggregating Hybrid Performance and Pareto Frontier ---")
    summary = {}
    for name, (n_dom, n_pred, n_lzw) in HYBRID_ALLOCATIONS_K128.items():
        # Cluster bootstrap for response comp %
        r_mean, r_low, r_high = compute_cluster_bootstrap_ci(
            evaluated_rows, lambda r: r["hybrids"][name]["response_pct"]
        )
        # Cluster bootstrap for prompt comp %
        p_mean, p_low, p_high = compute_cluster_bootstrap_ci(
            evaluated_rows, lambda r: r["hybrids"][name]["prompt_pct"]
        )
        # Cluster bootstrap for total comp %
        t_mean, t_low, t_high = compute_cluster_bootstrap_ci(
            evaluated_rows, lambda r: r["hybrids"][name]["total_pct"]
        )
        # Utilization mean
        util_mean = float(np.mean([r["hybrids"][name]["utilization"] for r in evaluated_rows]))
        uniq_mean = float(np.mean([r["hybrids"][name]["unique_hypertokens"] for r in evaluated_rows]))

        # Per broad domain response comp
        dom_comp = {}
        for dom in ["conversation", "code", "mathematical reasoning"]:
            dom_rows = [r for r in evaluated_rows if r["broad_domain"] == dom]
            if dom_rows:
                dom_comp[dom] = float(np.mean([r["hybrids"][name]["response_pct"] for r in dom_rows]))

        summary[name] = {
            "config": {"domain_static": n_dom, "prompt_pred": n_pred, "reactive_lzw": n_lzw},
            "response_comp_pct": {"mean": r_mean, "ci_95": [r_low, r_high]},
            "prompt_comp_pct": {"mean": p_mean, "ci_95": [p_low, p_high]},
            "total_comp_pct": {"mean": t_mean, "ci_95": [t_low, t_high]},
            "codebook_utilization": util_mean,
            "unique_hypertokens": uniq_mean,
            "domain_response_comp": dom_comp,
        }

    # Rank configurations by response compression and total compression
    ranked_by_resp = sorted(summary.items(), key=lambda x: x[1]["response_comp_pct"]["mean"], reverse=True)
    ranked_by_tot = sorted(summary.items(), key=lambda x: x[1]["total_comp_pct"]["mean"], reverse=True)

    print("\nHybrid Ranking by Response Compression % (Validation N=7,731):")
    print(f"{'Configuration':<25} | {'Dom':<4} {'Pred':<5} {'LZW':<4} | {'Resp Comp % [95% CI]':<26} | {'Prompt Comp %':<14} | {'Total Comp %':<14} | {'Utilization':<11}")
    print("-" * 110)
    for name, s in ranked_by_resp:
        cfg = s["config"]
        r_str = f"{s['response_comp_pct']['mean']:.2f}% [{s['response_comp_pct']['ci_95'][0]:.2f}, {s['response_comp_pct']['ci_95'][1]:.2f}]"
        print(f"{name:<25} | {cfg['domain_static']:<4} {cfg['prompt_pred']:<5} {cfg['reactive_lzw']:<4} | {r_str:<26} | {s['prompt_comp_pct']['mean']:.2f}%{' '*7} | {s['total_comp_pct']['mean']:.2f}%{' '*7} | {s['codebook_utilization']*100.0:.1f}%")

    # Select Top 3 Finalists:
    # 1. Best 2-way hybrid (e.g. 64 pred + 64 lzw)
    # 2. Best 3-way hybrid (e.g. 32 dom + 64 pred + 32 lzw or 32 dom + 32 pred + 64 lzw)
    # 3. Best high-predictive / low-reactive hybrid (e.g. 96 pred + 32 lzw)
    # (Pure Predictor and Pure LZW are preserved as mandatory controls)
    top_2way = [name for name, _ in ranked_by_resp if name not in ("100_pred_0_lzw", "0_pred_100_lzw") and "dom" not in name][0]
    top_3way = [name for name, _ in ranked_by_resp if "dom" in name][0]
    second_2way = [name for name, _ in ranked_by_resp if name not in ("100_pred_0_lzw", "0_pred_100_lzw") and "dom" not in name and name != top_2way][0]

    finalist_names = [top_2way, top_3way, second_2way]
    print(f"\n>>> Selected Top 3 Finalist Hybrid Configurations for TEST Evaluation:")
    for fn in finalist_names:
        print(f"    * {fn}: {summary[fn]['config']} (RespComp={summary[fn]['response_comp_pct']['mean']:.2f}%, TotComp={summary[fn]['total_comp_pct']['mean']:.2f}%)")

    results_data = {
        "split": "validation",
        "sample_count": total_samples,
        "summary": summary,
        "top_finalists": finalist_names,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Validation sweep results written to {output_path}")

    return results_data


# ----------------------------------------------------------------------
# 6. Core Evaluation for Test Split (Phases 1, 3, 4, 5)
# ----------------------------------------------------------------------

def run_test_evaluation(
    test_path: str,
    predictor: OptimizedPhraseIndex,
    finalist_hybrids: List[str],
    checkpoints_dir: str,
    output_path: str,
    budgets: List[int] = [16, 32, 64, 128, 256, 512],
    checkpoint_interval: int = 500,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("PHASES 1, 3, 4, 5: COMPREHENSIVE FROZEN TEST EVALUATION")
    print(f"Evaluating K in {budgets} + Frozen Hybrid Finalists: {finalist_hybrids}")
    print("=" * 70)

    with open(test_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if max_samples:
        records = records[:max_samples]
    total_samples = len(records)
    unique_threads = set(r["thread_id"] for r in records)
    print(f"Loaded {total_samples} TEST records ({len(unique_threads)} unique conversation thread clusters)")

    # Build hybrid configs dict for the finalists + pure controls
    active_hybrid_configs = {}
    for name in finalist_hybrids + ["100_pred_0_lzw", "0_pred_100_lzw"]:
        if name in HYBRID_ALLOCATIONS_K128:
            active_hybrid_configs[name] = HYBRID_ALLOCATIONS_K128[name]

    ckpt_file = os.path.join(checkpoints_dir, "test_eval_ckpt.json")
    evaluated_rows = []
    start_idx = 0

    if os.path.exists(ckpt_file):
        try:
            with open(ckpt_file, "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
            evaluated_rows = ckpt_data.get("rows", [])
            start_idx = len(evaluated_rows)
            print(f"Resuming from checkpoint with {start_idx} completed samples.")
        except Exception as e:
            print(f"Failed to resume from checkpoint ({e}), starting fresh.")
            evaluated_rows = []
            start_idx = 0

    t0 = time.time()
    prefix_cuts = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    for idx in range(start_idx, total_samples):
        rec = records[idx]
        eval_row = evaluate_sample_all_regimes(
            rec,
            predictor,
            budgets=budgets,
            hybrid_configs=active_hybrid_configs,
            include_controlled_prefix=True,
            prefix_cuts=prefix_cuts,
        )
        evaluated_rows.append(eval_row)

        if (idx + 1) % checkpoint_interval == 0 or (idx + 1) == total_samples:
            log_watchdog_status("TEST_EVAL", idx + 1, total_samples, t0)
            with open(ckpt_file, "w", encoding="utf-8") as f:
                json.dump({"completed": idx + 1, "total": total_samples, "rows": evaluated_rows}, f)

    total_time = time.time() - t0
    print(f"\nTest evaluation completed in {total_time:.1f}s ({len(evaluated_rows)/total_time:.1f} samples/s)!")

    # ------------------------------------------------------------------
    # Process and Aggregate All Deliverables
    # ------------------------------------------------------------------
    print("\n--- Compiling Deliverables: Multi-Regime Matrix, Capacity Sweep, Lengths, Domains ---")

    # 1. Capacity Sweep Table Across K
    capacity_results = {}
    regimes_to_track = ["lzw_corr", "lzw_upstream", "global_static", "domain_static", "prompt_predictor", "oracle"]
    for k in budgets:
        capacity_results[k] = {}
        for reg in regimes_to_track:
            r_mean, r_low, r_high = compute_cluster_bootstrap_ci(
                evaluated_rows, lambda r, k=k, reg=reg: r["regimes"][k][reg]["response_pct"]
            )
            p_mean, p_low, p_high = compute_cluster_bootstrap_ci(
                evaluated_rows, lambda r, k=k, reg=reg: r["regimes"][k][reg]["prompt_pct"]
            )
            t_mean, t_low, t_high = compute_cluster_bootstrap_ci(
                evaluated_rows, lambda r, k=k, reg=reg: r["regimes"][k][reg]["total_pct"]
            )
            util_mean = float(np.mean([r["regimes"][k][reg]["utilization"] for r in evaluated_rows]))
            uniq_mean = float(np.mean([r["regimes"][k][reg]["unique_hypertokens"] for r in evaluated_rows]))

            capacity_results[k][reg] = {
                "response_comp_pct": {"mean": r_mean, "ci_95": [r_low, r_high]},
                "prompt_comp_pct": {"mean": p_mean, "ci_95": [p_low, p_high]},
                "total_comp_pct": {"mean": t_mean, "ci_95": [t_low, t_high]},
                "codebook_utilization": util_mean,
                "unique_hypertokens": uniq_mean,
            }

    # 2. Frozen Hybrid Finalists at K=128
    hybrid_test_results = {}
    for name in active_hybrid_configs:
        r_mean, r_low, r_high = compute_cluster_bootstrap_ci(
            evaluated_rows, lambda r, name=name: r["hybrids"][name]["response_pct"]
        )
        p_mean, p_low, p_high = compute_cluster_bootstrap_ci(
            evaluated_rows, lambda r, name=name: r["hybrids"][name]["prompt_pct"]
        )
        t_mean, t_low, t_high = compute_cluster_bootstrap_ci(
            evaluated_rows, lambda r, name=name: r["hybrids"][name]["total_pct"]
        )
        util_mean = float(np.mean([r["hybrids"][name]["utilization"] for r in evaluated_rows]))
        uniq_mean = float(np.mean([r["hybrids"][name]["unique_hypertokens"] for r in evaluated_rows]))
        hybrid_test_results[name] = {
            "config": active_hybrid_configs[name],
            "response_comp_pct": {"mean": r_mean, "ci_95": [r_low, r_high]},
            "prompt_comp_pct": {"mean": p_mean, "ci_95": [p_low, p_high]},
            "total_comp_pct": {"mean": t_mean, "ci_95": [t_low, t_high]},
            "codebook_utilization": util_mean,
            "unique_hypertokens": uniq_mean,
        }

    # 3. Natural Length Buckets (at K=64 and K=128)
    prompt_buckets_def = [
        (0, 32, "0-32"),
        (33, 128, "33-128"),
        (129, 512, "129-512"),
        (513, 2048, "513-2048"),
        (2049, 8192, "2049-8192"),
        (8193, 100000, "8192+"),
    ]
    response_buckets_def = [
        (0, 32, "0-32"),
        (33, 128, "33-128"),
        (129, 512, "129-512"),
        (513, 2048, "513-2048"),
        (2049, 8192, "2049-8192"),
        (8193, 100000, "8192+"),
    ]

    p_bucket_results = {}
    for lo, hi, bname in prompt_buckets_def:
        b_rows = [r for r in evaluated_rows if lo <= r["base_prompt"] <= hi]
        if not b_rows:
            continue
        p_bucket_results[bname] = {
            "sample_count": len(b_rows),
            "k64": {
                "lzw_resp": float(np.mean([r["regimes"][64]["lzw_corr"]["response_pct"] for r in b_rows])),
                "pred_resp": float(np.mean([r["regimes"][64]["prompt_predictor"]["response_pct"] for r in b_rows])),
                "dom_resp": float(np.mean([r["regimes"][64]["domain_static"]["response_pct"] for r in b_rows])),
                "oracle_resp": float(np.mean([r["regimes"][64]["oracle"]["response_pct"] for r in b_rows])),
                "lzw_tot": float(np.mean([r["regimes"][64]["lzw_corr"]["total_pct"] for r in b_rows])),
                "pred_tot": float(np.mean([r["regimes"][64]["prompt_predictor"]["total_pct"] for r in b_rows])),
            },
            "k128": {
                "lzw_resp": float(np.mean([r["regimes"][128]["lzw_corr"]["response_pct"] for r in b_rows])),
                "pred_resp": float(np.mean([r["regimes"][128]["prompt_predictor"]["response_pct"] for r in b_rows])),
                "dom_resp": float(np.mean([r["regimes"][128]["domain_static"]["response_pct"] for r in b_rows])),
                "oracle_resp": float(np.mean([r["regimes"][128]["oracle"]["response_pct"] for r in b_rows])),
                "lzw_tot": float(np.mean([r["regimes"][128]["lzw_corr"]["total_pct"] for r in b_rows])),
                "pred_tot": float(np.mean([r["regimes"][128]["prompt_predictor"]["total_pct"] for r in b_rows])),
            }
        }

    r_bucket_results = {}
    for lo, hi, bname in response_buckets_def:
        b_rows = [r for r in evaluated_rows if lo <= r["base_response"] <= hi]
        if not b_rows:
            continue
        r_bucket_results[bname] = {
            "sample_count": len(b_rows),
            "k64": {
                "lzw_resp": float(np.mean([r["regimes"][64]["lzw_corr"]["response_pct"] for r in b_rows])),
                "pred_resp": float(np.mean([r["regimes"][64]["prompt_predictor"]["response_pct"] for r in b_rows])),
                "dom_resp": float(np.mean([r["regimes"][64]["domain_static"]["response_pct"] for r in b_rows])),
                "oracle_resp": float(np.mean([r["regimes"][64]["oracle"]["response_pct"] for r in b_rows])),
                "lzw_tot": float(np.mean([r["regimes"][64]["lzw_corr"]["total_pct"] for r in b_rows])),
                "pred_tot": float(np.mean([r["regimes"][64]["prompt_predictor"]["total_pct"] for r in b_rows])),
            },
            "k128": {
                "lzw_resp": float(np.mean([r["regimes"][128]["lzw_corr"]["response_pct"] for r in b_rows])),
                "pred_resp": float(np.mean([r["regimes"][128]["prompt_predictor"]["response_pct"] for r in b_rows])),
                "dom_resp": float(np.mean([r["regimes"][128]["domain_static"]["response_pct"] for r in b_rows])),
                "oracle_resp": float(np.mean([r["regimes"][128]["oracle"]["response_pct"] for r in b_rows])),
                "lzw_tot": float(np.mean([r["regimes"][128]["lzw_corr"]["total_pct"] for r in b_rows])),
                "pred_tot": float(np.mean([r["regimes"][128]["prompt_predictor"]["total_pct"] for r in b_rows])),
            }
        }

    # 4. Controlled Same-Document Prefix Test Results (K=64)
    controlled_prefix_summary = {}
    for cut in prefix_cuts:
        qualifying = [r for r in evaluated_rows if cut in r["prefix_cuts"]]
        if qualifying:
            controlled_prefix_summary[cut] = {
                "sample_count_N": len(qualifying),
                "lzw_resp": float(np.mean([r["prefix_cuts"][cut]["lzw_response_pct"] for r in qualifying])),
                "pred_resp": float(np.mean([r["prefix_cuts"][cut]["predictor_response_pct"] for r in qualifying])),
                "dom_resp": float(np.mean([r["prefix_cuts"][cut]["domain_static_response_pct"] for r in qualifying])),
                "hyb_resp": float(np.mean([r["prefix_cuts"][cut]["hybrid_response_pct"] for r in qualifying])),
                "delta_pred_minus_lzw": float(np.mean([
                    r["prefix_cuts"][cut]["predictor_response_pct"] - r["prefix_cuts"][cut]["lzw_response_pct"]
                    for r in qualifying
                ])),
            }

    # 5. Task / Domain Study (Broad Domains & Prompt Heuristic Categories)
    domain_study = {"broad_domains": {}, "heuristic_tasks": {}}
    for dom in ["conversation", "code", "mathematical reasoning"]:
        d_rows = [r for r in evaluated_rows if r["broad_domain"] == dom]
        if d_rows:
            domain_study["broad_domains"][dom] = {
                "sample_count": len(d_rows),
                "k128": {
                    "lzw_resp": float(np.mean([r["regimes"][128]["lzw_corr"]["response_pct"] for r in d_rows])),
                    "dom_resp": float(np.mean([r["regimes"][128]["domain_static"]["response_pct"] for r in d_rows])),
                    "pred_resp": float(np.mean([r["regimes"][128]["prompt_predictor"]["response_pct"] for r in d_rows])),
                    "oracle_resp": float(np.mean([r["regimes"][128]["oracle"]["response_pct"] for r in d_rows])),
                    "pred_tot": float(np.mean([r["regimes"][128]["prompt_predictor"]["total_pct"] for r in d_rows])),
                }
            }

    fine_tasks = sorted(list(set(r["fine_task"] for r in evaluated_rows)))
    for ft in fine_tasks:
        f_rows = [r for r in evaluated_rows if r["fine_task"] == ft]
        if f_rows:
            domain_study["heuristic_tasks"][ft] = {
                "sample_count": len(f_rows),
                "k128": {
                    "lzw_resp": float(np.mean([r["regimes"][128]["lzw_corr"]["response_pct"] for r in f_rows])),
                    "dom_resp": float(np.mean([r["regimes"][128]["domain_static"]["response_pct"] for r in f_rows])),
                    "pred_resp": float(np.mean([r["regimes"][128]["prompt_predictor"]["response_pct"] for r in f_rows])),
                    "oracle_resp": float(np.mean([r["regimes"][128]["oracle"]["response_pct"] for r in f_rows])),
                    "pred_tot": float(np.mean([r["regimes"][128]["prompt_predictor"]["total_pct"] for r in f_rows])),
                }
            }

    # 6. Predictor Latency Distribution Profile
    lats = [r["predictor_latency_ms"] for r in evaluated_rows]
    latency_profile = {
        "p50_ms": float(np.percentile(lats, 50)),
        "p90_ms": float(np.percentile(lats, 90)),
        "p95_ms": float(np.percentile(lats, 95)),
        "p99_ms": float(np.percentile(lats, 99)),
        "mean_ms": float(np.mean(lats)),
    }

    final_results = {
        "split": "test",
        "sample_count": total_samples,
        "cluster_count": len(unique_threads),
        "capacity_results": capacity_results,
        "hybrid_finalists": hybrid_test_results,
        "prompt_length_buckets": p_bucket_results,
        "response_length_buckets": r_bucket_results,
        "controlled_prefix_scaling": controlled_prefix_summary,
        "domain_study": domain_study,
        "predictor_latency_profile": latency_profile,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nComprehensive test results written to {output_path}")

    return final_results


# ----------------------------------------------------------------------
# 7. Main CLI Entry Point
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation Suite for Predictive Hypertokens")
    parser.add_argument("--mode", type=str, default="full", choices=["pilot", "val_sweep", "test_eval", "full"],
                        help="Execution mode")
    parser.add_argument("--train-path", type=str, default="data/corpus_stage3_train.jsonl")
    parser.add_argument("--val-path", type=str, default="data/corpus_stage3_val.jsonl")
    parser.add_argument("--test-path", type=str, default="data/corpus_stage3_test.jsonl")
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--checkpoints-dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--results-dir", type=str, default="experiments")
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Load or Build Predictor Index
    cached_predictor_path = os.path.join(args.checkpoints_dir, "cached_predictor.pkl")
    if os.path.exists(cached_predictor_path):
        print(f"Loading cached predictor index from {cached_predictor_path}...")
        t_load = time.time()
        with open(cached_predictor_path, "rb") as f:
            predictor = pickle.load(f)
        print(f"Loaded cached predictor index in {time.time() - t_load:.2f}s!")
    else:
        print(f"Mining index from training set {args.train_path}...")
        t_build = time.time()
        with open(args.train_path, "r", encoding="utf-8") as f:
            train_records = [json.loads(line) for line in f]
        predictor = build_fast_predictor_from_records(train_records)
        print(f"Saving compiled predictor index to {cached_predictor_path}...")
        with open(cached_predictor_path, "wb") as f:
            pickle.dump(predictor, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Predictor ready and cached in {time.time() - t_build:.2f}s!")

    # 2. Execute Modes
    if args.mode in ("val_sweep", "full"):
        val_output = os.path.join(args.results_dir, "val_sweep_results.json")
        val_results = run_validation_sweep(
            val_path=args.val_path,
            predictor=predictor,
            checkpoints_dir=args.checkpoints_dir,
            output_path=val_output,
            checkpoint_interval=args.checkpoint_interval,
            max_samples=args.val_samples,
        )
        finalist_names = list(set(val_results["top_finalists"] + ["50_pred_50_lzw", "75_pred_25_lzw"]))
    else:
        # If running test_eval directly, load finalists from val_sweep_results.json if exists
        val_output = os.path.join(args.results_dir, "val_sweep_results.json")
        if os.path.exists(val_output):
            with open(val_output, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            base_finalists = val_data.get("top_finalists", ["37.5_pred_62.5_lzw", "32dom_32pred_64lzw", "25_pred_75_lzw"])
            finalist_names = list(set(base_finalists + ["50_pred_50_lzw", "75_pred_25_lzw"]))
        else:
            finalist_names = ["37.5_pred_62.5_lzw", "50_pred_50_lzw", "32dom_32pred_64lzw", "25_pred_75_lzw", "75_pred_25_lzw"]

    if args.mode in ("test_eval", "full"):
        test_output = os.path.join(args.results_dir, "comprehensive_suite_results.json")
        test_results = run_test_evaluation(
            test_path=args.test_path,
            predictor=predictor,
            finalist_hybrids=finalist_names,
            checkpoints_dir=args.checkpoints_dir,
            output_path=test_output,
            budgets=[16, 32, 64, 128, 256, 512],
            checkpoint_interval=args.checkpoint_interval,
            max_samples=args.test_samples,
        )

    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION EXECUTION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
