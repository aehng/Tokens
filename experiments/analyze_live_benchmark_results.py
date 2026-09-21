"""
Comprehensive Analysis of Live Zero-Shot Autoregressive Benchmark Results.
Computes:
1. Aggregate performance across 15 stratified test prompts (75 condition runs).
2. Domain breakdown (Code vs. Reasoning).
3. Prefill vs. Decode vs. End-to-End latency and throughput.
4. Zero-Shot Realization Gap analysis.
5. Generates summary JSON and formatted tables.
"""

import json
from collections import defaultdict
import numpy as np

with open("experiments/live_zero_shot_results.json", "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"Loaded {len(records)} total evaluation records.")

# Aggregate by Condition
by_condition = defaultdict(list)
for r in records:
    by_condition[r["condition"]].append(r)

cond_summary = {}

for cond, rows in by_condition.items():
    step_reds = [r["realized_step_reduction_pct"] for r in rows]
    prefill_comps = [100.0 * (1.0 - r["compressed_prefill_tokens"] / r["base_prompt_tokens"]) for r in rows]
    total_comps = [
        100.0 * (1.0 - (r["compressed_prefill_tokens"] + r["generated_steps"]) / (r["base_prompt_tokens"] + r["base_equivalent_tokens"]))
        for r in rows
    ]
    eff_tok_s = [r["effective_tokens_per_sec"] for r in rows]
    raw_s_s = [r["raw_steps_per_sec"] for r in rows]
    lat_tot = [r["total_latency_ms"] for r in rows]
    lat_setup = [r["setup_latency_ms"] for r in rows]
    lat_dec = [r["decode_latency_ms"] for r in rows]
    ht_emitted = [r["hypertokens_emitted"] for r in rows]

    cond_summary[cond] = {
        "n_samples": len(rows),
        "mean_step_reduction_pct": float(np.mean(step_reds)),
        "std_step_reduction_pct": float(np.std(step_reds)),
        "mean_prefill_comp_pct": float(np.mean(prefill_comps)),
        "mean_total_comp_pct": float(np.mean(total_comps)),
        "mean_effective_tok_per_sec": float(np.mean(eff_tok_s)),
        "mean_raw_steps_per_sec": float(np.mean(raw_s_s)),
        "mean_total_latency_ms": float(np.mean(lat_tot)),
        "mean_setup_latency_ms": float(np.mean(lat_setup)),
        "mean_decode_latency_ms": float(np.mean(lat_dec)),
        "total_hypertokens_emitted": int(np.sum(ht_emitted)),
        "hypertokens_per_sample": float(np.mean(ht_emitted)),
    }

# Aggregate by Domain & Condition
by_domain_cond = defaultdict(lambda: defaultdict(list))
for r in records:
    by_domain_cond[r["domain"]][r["condition"]].append(r)

domain_summary = {}
for dom, cond_dict in by_domain_cond.items():
    domain_summary[dom] = {}
    for cond, rows in cond_dict.items():
        domain_summary[dom][cond] = {
            "n_samples": len(rows),
            "step_reduction_pct": float(np.mean([r["realized_step_reduction_pct"] for r in rows])),
            "prefill_comp_pct": float(np.mean([100.0 * (1.0 - r["compressed_prefill_tokens"] / r["base_prompt_tokens"]) for r in rows])),
            "effective_tok_per_sec": float(np.mean([r["effective_tokens_per_sec"] for r in rows])),
            "total_latency_ms": float(np.mean([r["total_latency_ms"] for r in rows])),
            "hypertokens_emitted": int(np.sum([r["hypertokens_emitted"] for r in rows])),
        }

final_report = {
    "aggregate": cond_summary,
    "by_domain": domain_summary,
}

with open("experiments/live_benchmark_summary.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2)

print("\n" + "=" * 100)
print(f"{'Condition':42s} | {'StepRed%':8s} | {'Prefill%':8s} | {'Total%':8s} | {'EffTok/s':8s} | {'Latency':9s} | {'Hypertokens'}")
print("=" * 100)
for cond, s in cond_summary.items():
    print(f"{cond:42s} | {s['mean_step_reduction_pct']:7.2f}% | {s['mean_prefill_comp_pct']:7.2f}% | {s['mean_total_comp_pct']:7.2f}% | {s['mean_effective_tok_per_sec']:8.2f} | {s['mean_total_latency_ms']:7.1f}ms | {s['total_hypertokens_emitted']:4d}")
print("=" * 100)

print("\n=== DOMAIN BREAKDOWN ===")
for dom in domain_summary:
    print(f"\n--- Domain: {dom.upper()} ---")
    for cond, s in domain_summary[dom].items():
        print(f"  {cond:40s} | StepRed: {s['step_reduction_pct']:5.2f}% | PrefillComp: {s['prefill_comp_pct']:5.2f}% | EffTok/s: {s['effective_tok_per_sec']:4.2f} | Latency: {s['total_latency_ms']:7.1f}ms | HT: {s['hypertokens_emitted']}")
