# Phase 0: Frozen Baseline Reference (Predictive Step 100)

This artifact defines the exact, authoritative reference baseline for all subsequent optimization phases.
Derived from the frozen 60-prompt validation run (`data/cached_pure_pred_val_60.json`) with `checkpoint_step_100.pt` and `CappedPredictorPolicy(K=32)`.

## Summary Metrics by Domain

| Domain | Objective Quality | Failures | Realized Micro Reduction % | Macro Reduction % | Hypertokens / Output | Quality-Preserved Reduction % | Quality-Preserved Tokens Saved | Total Tokens Saved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MBPP Code** | 0.0% Pass@1 (20% syntax) | 20 / 20 | **35.91%** | 26.32% | 88.15 | 0.00% | 0 | 3,132 |
| **GSM8K Math** | 60.0% Accuracy (12/20) | 8 / 20 | **12.96%** | 12.39% | 33.85 | **12.00%** | 491 | 893 |
| **Alpaca Instruction** | 20.0% Failure Rate (16/20 pass) | 4 / 20 | **3.70%** | 3.11% | 4.60 | **3.30%** | 70 | 125 |
| **Overall (60 prompts)** | 46.7% Correct (28/60) | 32 / 60 | **21.85%** | 13.94% | 42.20 | **9.03%** | 561 | 4,150 |

## Key Baseline Baseline Takeaways
1. **Code (MBPP)**: Highest raw compression (35.91%, 3,132 tokens saved, 75.5% of all savings), but 0% Pass@1 due to function signature and syntax errors.
2. **Math Reasoning (GSM8K)**: Healthiest compression regime (12.96% micro, 12.00% quality-preserved reduction saving 491 steps with 60% accuracy).
3. **Instruction (Alpaca)**: Strong quality (80% success, beating vanilla 70%), but compression is starved (only 3.70% micro, 125 tokens saved).
4. **Economics**: Mean TTFT = 1.59s, Mean Latency = 49.61s (15.3% faster than Vanilla Phi's 58.59s).
