# Phi Quality/Speed Attribution Benchmark Report

- **Generated at**: `2026-09-29T12:05:40.524108+00:00`
- **Git Commit**: `d343a98`
- **Split**: `DEV` (Total Prompts: `9`)
- **Records Generated**: `36`

---

## 1. Executive Summary & Causal Diagnosis

- **Primary Bottleneck**: `checkpoint_lora`
- **Diagnosis**: Predictive checkpoint/LoRA without hypertokens (Condition B) loses 12.5% quality vs Vanilla. The fine-tuning or base model state is degraded before any hypertoken mechanism operates.

---

## 2. Condition Overview

| Condition | Prompts | Agg Quality | MBPP Pass | GSM8K Pass | Alpaca Pass | Mean Latency (s) | Total Steps | Compression | H Emissions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A_vanilla** | 9 | **88.9%** | 66.7% | 100.0% | 100.0% | 12.863 | 3015 | 0.0% | 0 |
| **B_h_disabled** | 9 | **77.8%** | 66.7% | 66.7% | 100.0% | 15.780 | 2515 | 0.0% | 0 |
| **C_oracle** | 9 | **77.8%** | 66.7% | 66.7% | 100.0% | 16.896 | 2684 | 5.4% | 114 |
| **D_real_predictor** | 9 | **66.7%** | 66.7% | 33.3% | 100.0% | 17.074 | 2688 | 2.6% | 48 |

---

## 3. Comparison vs Vanilla (<= 3% Quality Gate)

| Condition | Abs Quality Diff | Rel Quality Drop | Meets <=3% Gate? | Speedup vs Vanilla | Faster? | Steps Saved |
|---|---:|---:|:---:|---:|:---:|---:|
| **B_h_disabled** | -11.1% | 12.50% | **FAIL** | -22.7% | NO | 500 |
| **C_oracle** | -11.1% | 12.50% | **FAIL** | -31.4% | NO | 331 |
| **D_real_predictor** | -22.2% | 25.00% | **FAIL** | -32.7% | NO | 327 |

