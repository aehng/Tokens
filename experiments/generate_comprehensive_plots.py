"""
Generates 6 high-resolution publication-quality diagnostic plots for the Comprehensive Evaluation Suite.
1. capacity_diminishing_returns.png
2. hybrid_ratio_pareto.png
3. controlled_prefix_crossover.png
4. response_len_vs_compression.png
5. prompt_len_vs_compression.png
6. task_domain_breakdown.png
"""

import json
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

out_dir = "experiments/figures/comprehensive"
os.makedirs(out_dir, exist_ok=True)
artifact_dir = r"C:\Users\elijk\.gemini\antigravity\brain\84e2a5d9-3835-4995-878c-2da317fb122e"

with open("experiments/comprehensive_suite_results.json", "r") as f:
    res = json.load(f)

with open("experiments/val_sweep_results.json", "r") as f:
    val_res = json.load(f)

# ----------------------------------------------------------------------
# 1. Capacity Diminishing Returns (K=16 to 512)
# ----------------------------------------------------------------------
plt.figure(figsize=(9, 5.5), dpi=300)
ks = [16, 32, 64, 128, 256, 512]
ks_str = [str(k) for k in ks]

pred_resp = [res["capacity_results"][k]["prompt_predictor"]["response_comp_pct"]["mean"] for k in ks_str]
lzw_resp = [res["capacity_results"][k]["lzw_corr"]["response_comp_pct"]["mean"] for k in ks_str]
dom_resp = [res["capacity_results"][k]["domain_static"]["response_comp_pct"]["mean"] for k in ks_str]
oracle_resp = [res["capacity_results"][k]["oracle"]["response_comp_pct"]["mean"] for k in ks_str]

pred_tot = [res["capacity_results"][k]["prompt_predictor"]["total_comp_pct"]["mean"] for k in ks_str]
lzw_tot = [res["capacity_results"][k]["lzw_corr"]["total_comp_pct"]["mean"] for k in ks_str]

plt.plot(ks, pred_resp, marker='o', linewidth=2.2, color='#1f77b4', label='Prompt Predictor (Response Comp %)')
plt.plot(ks, lzw_resp, marker='s', linewidth=2.2, color='#ff7f0e', label='Reactive LZW (Response Comp %)')
plt.plot(ks, dom_resp, marker='^', linewidth=1.8, linestyle='--', color='#2ca02c', label='Domain Static (Response Comp %)')
plt.plot(ks, pred_tot, marker='D', linewidth=1.8, linestyle=':', color='#9467bd', label='Prompt Predictor (Total Comp %)')
plt.plot(ks, lzw_tot, marker='v', linewidth=1.8, linestyle=':', color='#8c564b', label='Reactive LZW (Total Comp %)')

plt.title("Codebook Capacity Scaling: Response & Total Compression across Regimes", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Codebook Capacity $K$", fontsize=12)
plt.ylabel("Compression Percentage (%)", fontsize=12)
plt.xticks(ks)
plt.ylim(0, 36)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
plt.tight_layout()
p1 = os.path.join(out_dir, "capacity_diminishing_returns.png")
plt.savefig(p1)
plt.close()

# ----------------------------------------------------------------------
# 2. Hybrid Ratio Pareto Frontier (Validation N=7,731)
# ----------------------------------------------------------------------
plt.figure(figsize=(9, 5.5), dpi=300)
hybrid_sweep = [
    ("100/0", "100_pred_0_lzw"),
    ("87.5/12.5", "87.5_pred_12.5_lzw"),
    ("75/25", "75_pred_25_lzw"),
    ("62.5/37.5", "62.5_pred_37.5_lzw"),
    ("50/50", "50_pred_50_lzw"),
    ("37.5/62.5", "37.5_pred_62.5_lzw"),
    ("25/75", "25_pred_75_lzw"),
    ("12.5/87.5", "12.5_pred_87.5_lzw"),
    ("0/100", "0_pred_100_lzw"),
]

ratios_x = [lbl for lbl, _ in hybrid_sweep]
val_resp = [val_res["summary"][key]["response_comp_pct"]["mean"] for _, key in hybrid_sweep]
val_prompt = [val_res["summary"][key]["prompt_comp_pct"]["mean"] for _, key in hybrid_sweep]
val_tot = [val_res["summary"][key]["total_comp_pct"]["mean"] for _, key in hybrid_sweep]

plt.plot(ratios_x, val_resp, marker='o', linewidth=2.2, color='#d62728', label='Response Comp % (Peak at 37.5/62.5)')
plt.plot(ratios_x, val_tot, marker='s', linewidth=2.2, color='#2ca02c', label='Total Sequence Comp %')
plt.plot(ratios_x, val_prompt, marker='^', linewidth=1.8, linestyle='--', color='#1f77b4', label='Prompt Comp %')

plt.axvline(x=5, color='gray', linestyle=':', alpha=0.7, label='Optimal Hybrid (48 Pred + 80 LZW)')
plt.title("Hybrid Predictive/Reactive Allocation Sweep (Validation Split K=128)", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Allocation Ratio (Predictor % / Reactive LZW %)", fontsize=12)
plt.ylabel("Compression Percentage (%)", fontsize=12)
plt.xticks(rotation=30)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='center right')
plt.tight_layout()
p2 = os.path.join(out_dir, "hybrid_ratio_pareto.png")
plt.savefig(p2)
plt.close()

# ----------------------------------------------------------------------
# 3. Controlled Within-Sample Prefix Crossover (K=64)
# ----------------------------------------------------------------------
plt.figure(figsize=(9, 5.5), dpi=300)
cuts = [16, 32, 64, 128, 256, 512, 1024, 2048]
cuts_str = [str(c) for c in cuts]

c_lzw = [res["controlled_prefix_scaling"][c]["lzw_resp"] for c in cuts_str]
c_pred = [res["controlled_prefix_scaling"][c]["pred_resp"] for c in cuts_str]
c_dom = [res["controlled_prefix_scaling"][c]["dom_resp"] for c in cuts_str]
c_hyb = [res["controlled_prefix_scaling"][c]["hyb_resp"] for c in cuts_str]

plt.plot(cuts, c_lzw, marker='s', linewidth=2.2, color='#ff7f0e', label='Reactive LZW (Decays after saturation)')
plt.plot(cuts, c_pred, marker='o', linewidth=2.2, color='#1f77b4', label='Prompt Predictor (Pre-seeded global planning)')
plt.plot(cuts, c_hyb, marker='D', linewidth=2.2, color='#d62728', label='True Hybrid (32 Pred + 32 LZW)')
plt.plot(cuts, c_dom, marker='^', linewidth=1.8, linestyle='--', color='#2ca02c', label='Domain Static')

plt.axvline(x=128, color='black', linestyle='--', alpha=0.7, label='Crossover Point (128 Base Tokens)')
plt.xscale('log', base=2)
plt.xticks(cuts, [str(c) for c in cuts])
plt.title("Controlled Length Scaling on SAME Documents (K=64)", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Exact Response Prefix Length (Base Tokens, Log Scale)", fontsize=12)
plt.ylabel("Response Compression (%)", fontsize=12)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
plt.tight_layout()
p3 = os.path.join(out_dir, "controlled_prefix_crossover.png")
plt.savefig(p3)
plt.close()

# ----------------------------------------------------------------------
# 4. Response Length vs Compression (K=128)
# ----------------------------------------------------------------------
plt.figure(figsize=(9, 5.5), dpi=300)
r_buckets = ["0-32", "33-128", "129-512", "513-2048", "2049-8192"]
rb_lzw = [res["response_length_buckets"][b]["k128"]["lzw_resp"] for b in r_buckets]
rb_pred = [res["response_length_buckets"][b]["k128"]["pred_resp"] for b in r_buckets]
rb_dom = [res["response_length_buckets"][b]["k128"]["dom_resp"] for b in r_buckets]

x = np.arange(len(r_buckets))
w = 0.25

plt.bar(x - w, rb_lzw, width=w, label='Reactive LZW', color='#ff7f0e', alpha=0.9)
plt.bar(x, rb_pred, width=w, label='Prompt Predictor', color='#1f77b4', alpha=0.9)
plt.bar(x + w, rb_dom, width=w, label='Domain Static', color='#2ca02c', alpha=0.9)

plt.title("Response Compression by Output Length Tier (K=128)", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Natural Response Length Tier (Base Tokens)", fontsize=12)
plt.ylabel("Response Compression (%)", fontsize=12)
plt.xticks(x, r_buckets)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
plt.tight_layout()
p4 = os.path.join(out_dir, "response_len_vs_compression.png")
plt.savefig(p4)
plt.close()

# ----------------------------------------------------------------------
# 5. Prompt Length vs Compression (K=128)
# ----------------------------------------------------------------------
plt.figure(figsize=(9, 5.5), dpi=300)
p_buckets = ["0-32", "33-128", "129-512", "513-2048", "2049-8192"]
pb_lzw = [res["prompt_length_buckets"][b]["k128"]["lzw_resp"] for b in p_buckets]
pb_pred = [res["prompt_length_buckets"][b]["k128"]["pred_resp"] for b in p_buckets]
pb_dom = [res["prompt_length_buckets"][b]["k128"]["dom_resp"] for b in p_buckets]

x = np.arange(len(p_buckets))
plt.bar(x - w, pb_lzw, width=w, label='Reactive LZW', color='#ff7f0e', alpha=0.9)
plt.bar(x, pb_pred, width=w, label='Prompt Predictor', color='#1f77b4', alpha=0.9)
plt.bar(x + w, pb_dom, width=w, label='Domain Static', color='#2ca02c', alpha=0.9)

plt.title("Response Compression by Prompt Context Length (K=128)", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Prompt Length Tier (Base Tokens)", fontsize=12)
plt.ylabel("Response Compression (%)", fontsize=12)
plt.xticks(x, p_buckets)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
plt.tight_layout()
p5 = os.path.join(out_dir, "prompt_len_vs_compression.png")
plt.savefig(p5)
plt.close()

# ----------------------------------------------------------------------
# 6. Task / Domain Breakdown (K=128)
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 5.5), dpi=300)
doms = list(res["domain_study"]["broad_domains"].keys()) + ["factual QA*", "JSON*", "summarization*"]
dom_keys = list(res["domain_study"]["broad_domains"].keys())
d_lzw = [res["domain_study"]["broad_domains"][d]["k128"]["lzw_resp"] for d in dom_keys]
d_pred = [res["domain_study"]["broad_domains"][d]["k128"]["pred_resp"] for d in dom_keys]
d_dom = [res["domain_study"]["broad_domains"][d]["k128"]["dom_resp"] for d in dom_keys]

# Add heuristic tasks
h_tasks = [
    ("factual QA*", "factual QA (heuristic)"),
    ("JSON*", "structured JSON/output (heuristic)"),
    ("summarization*", "summarization (heuristic)")
]
for short_name, full_name in h_tasks:
    h_data = res["domain_study"]["heuristic_tasks"][full_name]["k128"]
    d_lzw.append(h_data["lzw_resp"])
    d_pred.append(h_data["pred_resp"])
    d_dom.append(h_data["dom_resp"])

xd = np.arange(len(doms))
plt.bar(xd - w, d_lzw, width=w, label='Reactive LZW', color='#ff7f0e', alpha=0.9)
plt.bar(xd, d_pred, width=w, label='Prompt Predictor', color='#1f77b4', alpha=0.9)
plt.bar(xd + w, d_dom, width=w, label='Domain Static', color='#2ca02c', alpha=0.9)

plt.title("Compression Across Task Domains & Prompt Heuristics (*Heuristic) (K=128)", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Domain / Task (* indicates prompt heuristic classification)", fontsize=11)
plt.ylabel("Response Compression (%)", fontsize=12)
plt.xticks(xd, doms, rotation=15)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
plt.tight_layout()
p6 = os.path.join(out_dir, "task_domain_breakdown.png")
plt.savefig(p6)
plt.close()

# Copy all 6 plots to artifact directory
for p in [p1, p2, p3, p4, p5, p6]:
    shutil.copy(p, artifact_dir)

print(f"Generated 6 high-resolution plots in {out_dir} and copied to {artifact_dir}!")
