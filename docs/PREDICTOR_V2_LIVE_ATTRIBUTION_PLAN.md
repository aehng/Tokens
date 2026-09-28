# Predictor V2 Live Attribution Experiment Protocol

> [!IMPORTANT]
> **Execution Gate:** This plan is fully staged and validated. **Execution is paused pending explicit user approval.**

## 1. Experimental Arms (Controlled Head-to-Head)

| Arm ID | Configuration | Purpose | Expected Codebook Size K |
|---|---|---|---|
| **Arm A** | Vanilla Phi-3.5-mini-instruct | Absolute quality and baseline speed control | 0 |
| **Arm B** | Predictive Model (H Disabled) | Measures adapter/LoRA quality impact without token substitution | 0 |
| **Arm C1** | Predictive + Global Occurrence Oracle | Diagnostic upper bound with future knowledge | 32 |
| **Arm C2** | Predictive + Empirically Safe Oracle | Quality ceiling under conservative continuation safety | $\le 32$ |
| **Arm D** | Predictive + Legacy OracleGuidedPredictor | Historical baseline comparison | 32 |
| **Arm E** | Predictive + **Ridge** (Top-1) | Validates whether top offline ranker converts to live quality | 32 |
| **Arm F** | Predictive + **PooledMLP** (Top-2) | Validates Pareto-runner-up live behavior | 32 |

## 2. Metrics to Capture per Prompt
- **Task Quality:** Exact accuracy / test pass rate (MBPP code execution, GSM8K numeric match, Alpaca semantic score).
- **Divergence:** Token index of first divergence from matched Vanilla continuation.
- **Realized Decode Steps:** Net decode calls saved vs Vanilla.
- **Hypertoken Emissions:** Hit rate, dead slots, and emitted token IDs.
- **Termination Health:** EOS reached, repetition penalty triggers, length truncation.
- **Quality-Preserved Savings:** Decode-step reduction on prompts where task quality matches or exceeds Vanilla.

## 3. Decision Rules (Interpretation Gate)
- **Case 5 (Safe Oracle Fails):** If Arms C1 and C2 still degrade task quality compared to Arm A, Predictor V2 is *not* sufficient alone; HyperLinear/input encoder continuation representations are the primary bottleneck.
- **Case 6 (Safe Oracle Works, Predictors Fail):** If Arm C2 maintains 100% quality parity while Arms E and F fail, ranking/prediction remains the dominant blocker.
- **Case 7 (Ranker Succeeds Offline, No Emissions Live):** If Arm E ranks high-quality codebooks but emissions remain 0, output-head calibration is the next bottleneck.
