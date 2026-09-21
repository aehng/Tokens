"""
Plot Live Zero-Shot Autoregressive Benchmark Results.
Generates publication-quality comparison figure comparing:
1. Realized Step Reduction vs. Prefill vs. Total Compression.
2. Wall-clock Latency and Effective Tokens/sec.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

with open("experiments/live_benchmark_summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)["aggregate"]

conditions = [
    "Base Phi-3.5 (K=0)",
    "Official Reactive Zip2Zip (Pure LZW, K=32)",
    "Pure Predictive Seeded (K=32)",
    "37.5% Pred / 62.5% LZW Hybrid (12/20)",
    "50/50 Hybrid (16/16)",
]

short_names = [
    "Base (K=0)",
    "Official LZW\n(K=32)",
    "Pure Predictor\n(K=32)",
    "Hybrid 37.5/62.5\n(12/20)",
    "Hybrid 50/50\n(16/16)",
]

prefill_comp = [summary[c]["mean_prefill_comp_pct"] for c in conditions]
step_red = [summary[c]["mean_step_reduction_pct"] for c in conditions]
total_comp = [summary[c]["mean_total_comp_pct"] for c in conditions]
latency = [summary[c]["mean_total_latency_ms"] for c in conditions]
eff_tok = [summary[c]["mean_effective_tok_per_sec"] for c in conditions]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Plot 1: Compression Breakdown
x = np.arange(len(conditions))
width = 0.25

rects1 = ax1.bar(x - width, prefill_comp, width, label="Prefill Compression (%)", color="#1f77b4")
rects2 = ax1.bar(x, step_red, width, label="Decode Step Reduction (%)", color="#2ca02c")
rects3 = ax1.bar(x + width, total_comp, width, label="Total Sequence Comp (%)", color="#ff7f0e")

ax1.set_ylabel("Compression Ratio (%)", fontsize=11)
ax1.set_title("Live Compression Breakdown by Condition\n(N=15 Stratified Prompts, Zero-Shot CPU)", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(short_names, fontsize=9.5)
ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
ax1.legend(fontsize=9.5)

# Add value labels
for rect in rects1:
    h = rect.get_height()
    if h > 0.1:
        ax1.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width() / 2, h), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
for rect in rects2:
    h = rect.get_height()
    if h > 0.1:
        ax1.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width() / 2, h), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
for rect in rects3:
    h = rect.get_height()
    if h > 0.1:
        ax1.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width() / 2, h), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, fontweight="bold")

# Plot 2: Latency & Effective Throughput
color_lat = "#d62728"
color_tok = "#9467bd"

ax2.set_xlabel("Condition", fontsize=11)
ax2.set_ylabel("Mean Total Latency (ms) [Lower is Better]", color=color_lat, fontsize=11)
bars_lat = ax2.bar(x - 0.15, latency, 0.3, color=color_lat, alpha=0.8, label="Latency (ms)")
ax2.tick_params(axis="y", labelcolor=color_lat)
ax2.set_xticks(x)
ax2.set_xticklabels(short_names, fontsize=9.5)
ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

# Secondary axis for throughput
ax2_twin = ax2.twinx()
ax2_twin.set_ylabel("Effective Throughput (tok/s) [Higher is Better]", color=color_tok, fontsize=11)
pts_tok = ax2_twin.plot(x, eff_tok, color=color_tok, marker="o", linewidth=2.5, markersize=8, label="Effective tok/s")
ax2_twin.tick_params(axis="y", labelcolor=color_tok)
ax2_twin.set_ylim(0, 4.0)

ax2.set_title("Wall-Clock Latency & Effective Throughput\n(Pure Predictor is Fastest: 8,264 ms vs. 8,750 ms Base)", fontsize=12)

# Value annotations for latency
for rect in bars_lat:
    h = rect.get_height()
    ax2.annotate(f"{int(h)}ms", (rect.get_x() + rect.get_width() / 2, h), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=color_lat)

for i, txt in enumerate(eff_tok):
    ax2_twin.annotate(f"{txt:.2f} t/s", (x[i], eff_tok[i]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=color_tok, fontweight="bold")

plt.tight_layout()
plt.savefig("experiments/figures/comprehensive/live_zero_shot_comparison.png", dpi=300)
plt.close()
print("Saved experiments/figures/comprehensive/live_zero_shot_comparison.png")
