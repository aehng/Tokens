# Phi Quality/Speed Attribution Benchmark Report

- **Generated at**: `2026-09-29T13:40:31.289256+00:00`
- **Git Commit**: `8af6589`
- **Split**: `DEV` (Total Prompts: `45`)
- **Records Generated**: `180`

---

## 1. Executive Summary & Causal Diagnosis

- **Primary Bottleneck**: `checkpoint_lora`
- **Diagnosis**: Predictive checkpoint/LoRA without hypertokens (Condition B) loses 28.57% quality vs Vanilla. The fine-tuning or base model state is degraded before any hypertoken mechanism operates.

---

## 2. Condition Overview

| Condition | Prompts | Agg Quality | MBPP Pass | GSM8K Pass | Alpaca Pass | Mean Latency (s) | Total Steps | Compression | H Emissions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A_vanilla** | 45 | **77.8%** | 53.3% | 86.7% | 93.3% | 30.445 | 15167 | 0.0% | 0 |
| **B_h_disabled** | 45 | **55.6%** | 26.7% | 53.3% | 86.7% | 18.499 | 14902 | 0.0% | 0 |
| **C_oracle** | 45 | **55.6%** | 26.7% | 60.0% | 80.0% | 19.775 | 15811 | 5.4% | 704 |
| **D_real_predictor** | 45 | **53.3%** | 33.3% | 40.0% | 86.7% | 16.962 | 13536 | 2.0% | 187 |

---

## 3. Comparison vs Vanilla (<= 3% Quality Gate)

| Condition | Abs Quality Diff | Rel Quality Drop | Meets <=3% Gate? | Speedup vs Vanilla | Faster? | Steps Saved |
|---|---:|---:|:---:|---:|:---:|---:|
| **B_h_disabled** | -22.2% | 28.57% | **FAIL** | +39.2% | YES | 265 |
| **C_oracle** | -22.2% | 28.57% | **FAIL** | +35.0% | YES | -644 |
| **D_real_predictor** | -24.4% | 31.43% | **FAIL** | +44.3% | YES | 1631 |

