"""
Break-Even Output Length and Latency Amortization Modeling.
Computes R* = T_setup / (Delta_Comp * T_step) across hardware environments,
model sizes, and realization ratios.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("experiments/figures/comprehensive", exist_ok=True)

# Hardware and latency constants
# Setup latency: predictor retrieval (~0.3 ms) + weight synthesis (~3.1 ms) = 3.4 ms
T_SETUP_MS = 3.4

# Decode latencies per step (ms/step)
HARDWARE_PROFILES = {
    "Workstation CPU (Phi-3.5 3.8B)": 350.0,
    "Consumer GPU (RTX 4090, 3.8B)": 15.0,
    "Data Center GPU (A100, 3.8B)": 8.0,
    "Data Center GPU (H100, 70B projected)": 35.0,
}

# Offline compression finalists (at K=128 from comprehensive suite)
OFFLINE_RESPONSE_COMP = {
    "Pure Predictor (K=128)": 0.1621,
    "Official Pure LZW (K=128)": 0.1531,
    "Hybrid Finalist (37.5% Pred / 62.5% LZW)": 0.1780,
    "Hybrid Finalist (50% Pred / 50% LZW)": 0.1764,
}

# Realization ratios (from zero-shot to fully fine-tuned)
REALIZATION_RATIOS = [0.10, 0.25, 0.50, 0.75, 1.00]

print("=" * 80)
print("BREAK-EVEN OUTPUT LENGTH ANALYSIS")
print("=" * 80)

breakeven_table = {}

for hw_name, t_step in HARDWARE_PROFILES.items():
    breakeven_table[hw_name] = {}
    print(f"\n--- {hw_name} (T_step = {t_step} ms) ---")
    for method_name, offline_ratio in OFFLINE_RESPONSE_COMP.items():
        breakeven_table[hw_name][method_name] = {}
        for r_ratio in REALIZATION_RATIOS:
            realized_comp = offline_ratio * r_ratio
            # R* = T_setup / (realized_comp * T_step)
            r_star = T_SETUP_MS / (realized_comp * t_step)
            breakeven_table[hw_name][method_name][f"{int(r_ratio * 100)}%"] = r_star

        r_star_100 = breakeven_table[hw_name][method_name]["100%"]
        r_star_25 = breakeven_table[hw_name][method_name]["25%"]
        print(f"  {method_name:40s} | R*(100% realization) = {r_star_100:6.2f} tokens | R*(25% realization) = {r_star_25:6.2f} tokens")

# Save table to json
with open("experiments/breakeven_analysis.json", "w", encoding="utf-8") as f:
    json.dump(breakeven_table, f, indent=2)

# Generate Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Plot 1: R* vs Realization Ratio across Hardware Profiles for Top Hybrid (17.80%)
ax1 = axes[0]
r_ratios_dense = np.linspace(0.05, 1.0, 100)
comp_hybrid = 0.1780

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for (hw_name, t_step), color in zip(HARDWARE_PROFILES.items(), colors):
    r_stars = T_SETUP_MS / (comp_hybrid * r_ratios_dense * t_step)
    ax1.plot(r_ratios_dense * 100, r_stars, label=f"{hw_name} ({t_step:.0f}ms/step)", color=color, linewidth=2)

ax1.set_xlabel("Compression Realization Ratio (%)", fontsize=11)
ax1.set_ylabel("Break-Even Output Length R* (Base Tokens)", fontsize=11)
ax1.set_title("Break-Even Output Length vs. Realization Ratio\n(Top Hybrid Finalist: 17.80% Offline)", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.set_ylim(0, 15)
ax1.axhline(1.0, color="gray", linestyle=":", label="1 Token Threshold")
ax1.legend(fontsize=9, loc="upper right")

# Plot 2: Net Wall-Clock Latency Saved vs. Output Length (A100 GPU, T_step=8ms)
ax2 = axes[1]
output_lens = np.linspace(10, 500, 100)
t_step_a100 = 8.0

# 100% realization (17.80%), 50% realization (8.90%), 25% realization (4.45%)
for r_ratio, style, col in zip([1.0, 0.5, 0.25], ["-", "--", "-."], ["#2ca02c", "#1f77b4", "#e377c2"]):
    eff_comp = comp_hybrid * r_ratio
    saved_time_ms = (output_lens * eff_comp * t_step_a100) - T_SETUP_MS
    ax2.plot(output_lens, saved_time_ms, linestyle=style, color=col, linewidth=2,
             label=f"{int(r_ratio*100)}% Realization ({eff_comp*100:.1f}% step red.)")

ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
ax2.set_xlabel("Generated Output Length (Base Tokens)", fontsize=11)
ax2.set_ylabel("Net Wall-Clock Time Saved (ms)", fontsize=11)
ax2.set_title("Net Latency Savings vs. Output Length\n(A100 GPU Profile: T_step = 8.0 ms)", fontsize=12)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig("experiments/figures/comprehensive/breakeven_crossover.png", dpi=300)
plt.close()
print("\nPlot saved to experiments/figures/comprehensive/breakeven_crossover.png")
