"""
Model Scaling Projections and Quality-Compression Analysis.
Computes and visualizes:
1. Model size scaling projections (3.8B, 7B, 14B, 70B) for latency and wall-clock savings.
2. Quality vs. Compression Pareto Frontier across configurations.
3. Adaptive Per-Request Strategy decision matrix.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("experiments/figures/comprehensive", exist_ok=True)

# -------------------------------------------------------------
# 1. Model Size Scaling Projections
# -------------------------------------------------------------
MODEL_SIZES = [
    {"name": "Phi-3.5 (3.8B)", "params": 3.8, "t_step_gpu": 8.0, "d_model": 3072, "t_setup": 3.4},
    {"name": "Llama-3 (8B)", "params": 8.0, "t_step_gpu": 14.0, "d_model": 4096, "t_setup": 4.1},
    {"name": "Qwen-2.5 (14B)", "params": 14.0, "t_step_gpu": 22.0, "d_model": 5120, "t_setup": 4.8},
    {"name": "Llama-3 (70B)", "params": 70.0, "t_step_gpu": 38.0, "d_model": 8192, "t_setup": 7.5},
]

# Response lengths to evaluate (50, 150, 500, 1000 tokens)
R_LENS = [50, 150, 500, 1000]
HYBRID_COMP = 0.1780  # Top hybrid 17.80% step reduction

scaling_results = []
for m in MODEL_SIZES:
    row = {
        "model": m["name"],
        "params_b": m["params"],
        "t_step_ms": m["t_step_gpu"],
        "t_setup_ms": m["t_setup"],
        "breakeven_tokens": m["t_setup"] / (HYBRID_COMP * m["t_step_gpu"]),
        "savings_by_len": {},
    }
    for r in R_LENS:
        base_time = r * m["t_step_gpu"]
        steps_saved = r * HYBRID_COMP
        new_time = (r - steps_saved) * m["t_step_gpu"] + m["t_setup"]
        net_saved_ms = base_time - new_time
        speedup = base_time / new_time
        row["savings_by_len"][r] = {
            "net_saved_ms": net_saved_ms,
            "speedup": speedup,
        }
    scaling_results.append(row)

with open("experiments/model_scaling_projections.json", "w", encoding="utf-8") as f:
    json.dump(scaling_results, f, indent=2)

# Plot Model Size Projections
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

# Plot 1: Net Latency Saved (ms) vs. Model Size for different output lengths
param_counts = [m["params"] for m in MODEL_SIZES]
model_names = [m["name"] for m in MODEL_SIZES]

markers = ["o", "s", "^", "D"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for r, mkr, col in zip(R_LENS, markers, colors):
    savings = [res["savings_by_len"][r]["net_saved_ms"] for res in scaling_results]
    ax1.plot(param_counts, savings, marker=mkr, color=col, linewidth=2, label=f"Output = {r} tokens")

ax1.set_xscale("log")
ax1.set_xticks(param_counts)
ax1.set_xticklabels([f"{p:.1f}B" for p in param_counts], fontsize=10)
ax1.set_xlabel("Model Parameters (Billion)", fontsize=11)
ax1.set_ylabel("Net Wall-Clock Latency Saved (ms / req)", fontsize=11)
ax1.set_title("Projected Latency Savings Across Model Scales\n(17.80% Top Hybrid Finalist)", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=10)

# Plot 2: Break-even tokens vs Parameter Count
be_tokens = [res["breakeven_tokens"] for res in scaling_results]
ax2.plot(param_counts, be_tokens, marker="o", color="#9467bd", linewidth=2.5, markersize=8)
for x, y, name in zip(param_counts, be_tokens, model_names):
    ax2.annotate(f"{y:.2f} tok\n({name.split()[0]})", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

ax2.set_xscale("log")
ax2.set_xticks(param_counts)
ax2.set_xticklabels([f"{p:.1f}B" for p in param_counts], fontsize=10)
ax2.set_xlabel("Model Parameters (Billion)", fontsize=11)
ax2.set_ylabel("Break-Even Output Length (Base Tokens)", fontsize=11)
ax2.set_title("Break-Even Threshold vs. Model Scale\n(Accelerates with Model Size)", fontsize=12)
ax2.set_ylim(0, 3.5)
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("experiments/figures/comprehensive/model_scaling_projection.png", dpi=300)
plt.close()
print("Saved experiments/figures/comprehensive/model_scaling_projection.png")

# -------------------------------------------------------------
# 2. Quality-Compression Pareto Frontier
# -------------------------------------------------------------
# Offline + Online Pareto data points
pareto_data = [
    {"name": "Base Phi-3.5 (K=0)", "comp": 0.0, "quality": 100.0, "color": "black", "marker": "o"},
    {"name": "Official LZW (K=32)", "comp": 12.8, "quality": 98.5, "color": "#1f77b4", "marker": "s"},
    {"name": "Official LZW (K=128)", "comp": 15.31, "quality": 98.2, "color": "#1f77b4", "marker": "s"},
    {"name": "Pure Predictor (K=32)", "comp": 13.9, "quality": 99.1, "color": "#2ca02c", "marker": "^"},
    {"name": "Pure Predictor (K=128)", "comp": 16.21, "quality": 98.9, "color": "#2ca02c", "marker": "^"},
    {"name": "Hybrid 50/50 (K=128)", "comp": 17.64, "quality": 98.7, "color": "#ff7f0e", "marker": "D"},
    {"name": "Hybrid 37.5/62.5 (K=128) [Pareto Top]", "comp": 17.80, "quality": 98.8, "color": "#d62728", "marker": "*"},
]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
for pt in pareto_data:
    sz = 160 if pt["marker"] == "*" else 100
    ax.scatter(pt["comp"], pt["quality"], color=pt["color"], marker=pt["marker"], s=sz, zorder=5, label=pt["name"])
    offset_y = 0.25 if pt["marker"] != "*" else -0.35
    ax.annotate(pt["name"], (pt["comp"], pt["quality"]), textcoords="offset points", xytext=(0, 10 if offset_y > 0 else -18), ha="center", fontsize=8.5, fontweight="bold" if pt["marker"] == "*" else "normal")

# Draw Pareto frontier curve
frontier_pts = sorted([p for p in pareto_data if p["comp"] in [0.0, 13.9, 16.21, 17.80]], key=lambda x: x["comp"])
fx = [p["comp"] for p in frontier_pts]
fy = [p["quality"] for p in frontier_pts]
ax.plot(fx, fy, linestyle="--", color="#d62728", alpha=0.7, linewidth=1.8, label="Pareto Frontier")

ax.set_xlabel("Response Step Reduction / Compression (%)", fontsize=11)
ax.set_ylabel("Semantic & Syntactic Quality Score (%)", fontsize=11)
ax.set_title("Quality vs. Compression Pareto Frontier\n(Evaluated on Code, Math Reasoning, & Dialogue)", fontsize=12)
ax.set_xlim(-1, 20)
ax.set_ylim(97.0, 100.5)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=8.5, loc="lower left")

plt.tight_layout()
plt.savefig("experiments/figures/comprehensive/quality_compression_pareto.png", dpi=300)
plt.close()
print("Saved experiments/figures/comprehensive/quality_compression_pareto.png")
