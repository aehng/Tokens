"""Generate concise, authoritative Phase 0 baseline artifacts for Step 100 benchmark comparison."""

import json
import os

COMP_PATH = "experiments/checkpoints/quality_benchmark/compression_by_domain.json"
AGG_PATH = "experiments/checkpoints/quality_benchmark/aggregate_results.json"
OUT_JSON = "experiments/checkpoints/quality_benchmark/baseline_step100_frozen.json"
OUT_MD = "experiments/checkpoints/quality_benchmark/BASELINE_STEP100_FROZEN.md"

def main():
    with open(COMP_PATH, "r", encoding="utf-8") as f:
        comp = json.load(f)
    with open(AGG_PATH, "r", encoding="utf-8") as f:
        agg = json.load(f)

    p100_comp = comp["predictive_step_100"]
    p100_agg = agg["predictive_step_100"]

    baseline = {
        "phase": 0,
        "model": "predictive_step_100",
        "checkpoint": "experiments/checkpoints/predictive_joint_pilot/checkpoint_step_100.pt",
        "predictor": "experiments/checkpoints/cached_predictor.pkl",
        "policy": "CappedPredictorPolicy(budget=32, max_structural_slots=0, allow_numeric=True, filter_bare_punctuation=True)",
        "validation_suite": "data/cached_pure_pred_val_60.json",
        "num_prompts": 60,
        "domains": {
            "code": {
                "dataset": "MBPP Python Code",
                "prompts_count": 20,
                "objective_quality": "0/20 Pass@1 (0.0%)",
                "valid_syntax_rate_pct": round(p100_agg["code_syntax_valid_rate"] * 100, 1),
                "failure_count": 20,
                "realized_micro_compression_pct": p100_comp["code"]["micro_decode_reduction_pct"],
                "macro_compression_pct": p100_comp["code"]["macro_decode_reduction_pct"],
                "mean_hypertokens_per_output": p100_comp["code"]["mean_hypertokens_per_prompt"],
                "quality_preserved_compression_pct": p100_comp["code"]["quality_preserved_decode_reduction_pct"],
                "tokens_saved": p100_comp["code"]["net_decode_steps_saved"],
                "decode_steps": p100_comp["code"]["total_decode_steps"],
                "expanded_tokens": p100_comp["code"]["total_expanded_tokens"],
            },
            "reasoning": {
                "dataset": "GSM8K Math Reasoning",
                "prompts_count": 20,
                "objective_quality": f"{p100_agg['gsm8k_accuracy']*100:.1f}% ({p100_agg['gsm8k_correct_count']})",
                "failure_count": 8,
                "realized_micro_compression_pct": p100_comp["reasoning"]["micro_decode_reduction_pct"],
                "macro_compression_pct": p100_comp["reasoning"]["macro_decode_reduction_pct"],
                "mean_hypertokens_per_output": p100_comp["reasoning"]["mean_hypertokens_per_prompt"],
                "quality_preserved_compression_pct": p100_comp["reasoning"]["quality_preserved_decode_reduction_pct"],
                "quality_preserved_tokens_saved": p100_comp["reasoning"]["quality_preserved_tokens_saved"],
                "tokens_saved": p100_comp["reasoning"]["net_decode_steps_saved"],
                "decode_steps": p100_comp["reasoning"]["total_decode_steps"],
                "expanded_tokens": p100_comp["reasoning"]["total_expanded_tokens"],
            },
            "instruction": {
                "dataset": "Alpaca Instruction Following",
                "prompts_count": 20,
                "objective_quality": f"{p100_agg['alpaca_failure_rate']*100:.1f}% failure rate ({p100_agg['alpaca_failure_count']})",
                "success_count": "16/20 (80.0%)",
                "failure_count": 4,
                "realized_micro_compression_pct": p100_comp["instruction"]["micro_decode_reduction_pct"],
                "macro_compression_pct": p100_comp["instruction"]["macro_decode_reduction_pct"],
                "mean_hypertokens_per_output": p100_comp["instruction"]["mean_hypertokens_per_prompt"],
                "quality_preserved_compression_pct": p100_comp["instruction"]["quality_preserved_decode_reduction_pct"],
                "quality_preserved_tokens_saved": p100_comp["instruction"]["quality_preserved_tokens_saved"],
                "tokens_saved": p100_comp["instruction"]["net_decode_steps_saved"],
                "decode_steps": p100_comp["instruction"]["total_decode_steps"],
                "expanded_tokens": p100_comp["instruction"]["total_expanded_tokens"],
            },
            "overall": {
                "dataset": "Full 60-Prompt Held-Out Suite",
                "prompts_count": 60,
                "overall_correct": "28/60 (46.7%)",
                "failure_count": 32,
                "realized_micro_compression_pct": p100_comp["overall"]["micro_decode_reduction_pct"],
                "macro_compression_pct": p100_comp["overall"]["macro_decode_reduction_pct"],
                "mean_hypertokens_per_output": p100_comp["overall"]["mean_hypertokens_per_prompt"],
                "quality_preserved_compression_pct": p100_comp["overall"]["quality_preserved_decode_reduction_pct"],
                "quality_preserved_tokens_saved": p100_comp["overall"]["quality_preserved_tokens_saved"],
                "tokens_saved": p100_comp["overall"]["net_decode_steps_saved"],
                "decode_steps": p100_comp["overall"]["total_decode_steps"],
                "expanded_tokens": p100_comp["overall"]["total_expanded_tokens"],
                "mean_wall_time_s": p100_agg["mean_wall_time_s"],
                "mean_ttft_s": p100_agg["mean_ttft_s"],
                "mean_throughput_tok_per_s": p100_agg["mean_throughput_tok_per_s"],
            },
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    md_lines = [
        "# Phase 0: Frozen Baseline Reference (Predictive Step 100)",
        "",
        "This artifact defines the exact, authoritative reference baseline for all subsequent optimization phases.",
        "Derived from the frozen 60-prompt validation run (`data/cached_pure_pred_val_60.json`) with `checkpoint_step_100.pt` and `CappedPredictorPolicy(K=32)`.",
        "",
        "## Summary Metrics by Domain",
        "",
        "| Domain | Objective Quality | Failures | Realized Micro Reduction % | Macro Reduction % | Hypertokens / Output | Quality-Preserved Reduction % | Quality-Preserved Tokens Saved | Total Tokens Saved |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **MBPP Code** | 0.0% Pass@1 (20% syntax) | 20 / 20 | **35.91%** | 26.32% | 88.15 | 0.00% | 0 | 3,132 |",
        f"| **GSM8K Math** | 60.0% Accuracy (12/20) | 8 / 20 | **12.96%** | 12.39% | 33.85 | **12.00%** | 491 | 893 |",
        f"| **Alpaca Instruction** | 20.0% Failure Rate (16/20 pass) | 4 / 20 | **3.70%** | 3.11% | 4.60 | **3.30%** | 70 | 125 |",
        f"| **Overall (60 prompts)** | 46.7% Correct (28/60) | 32 / 60 | **21.85%** | 13.94% | 42.20 | **9.03%** | 561 | 4,150 |",
        "",
        "## Key Baseline Baseline Takeaways",
        "1. **Code (MBPP)**: Highest raw compression (35.91%, 3,132 tokens saved, 75.5% of all savings), but 0% Pass@1 due to function signature and syntax errors.",
        "2. **Math Reasoning (GSM8K)**: Healthiest compression regime (12.96% micro, 12.00% quality-preserved reduction saving 491 steps with 60% accuracy).",
        "3. **Instruction (Alpaca)**: Strong quality (80% success, beating vanilla 70%), but compression is starved (only 3.70% micro, 125 tokens saved).",
        "4. **Economics**: Mean TTFT = 1.59s, Mean Latency = 49.61s (15.3% faster than Vanilla Phi's 58.59s).",
    ]

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Saved {OUT_JSON} and {OUT_MD}")

if __name__ == "__main__":
    main()
