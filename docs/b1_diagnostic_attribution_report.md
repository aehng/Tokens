# Phi Quality/Speed Attribution Benchmark Report

- **Generated at**: `2026-09-29T23:55:51.322634+00:00`
- **Git Commit**: `f59ef6319a1b9989c1154fd7349a9f6f6a41bb05`
- **Split**: `DEV` (Total Prompts: `12`)
- **Records Generated**: `36`

---

## 1. Executive Summary & Causal Diagnosis

- **Primary Bottleneck**: `unresolved_architecture_vs_checkpoint`
- **Diagnosis**: Causal attribution is blocked because one or more A/B0 token/logit parity or B0/B1 parameter-isolation gates did not pass. No downstream quality loss is assigned.

## Required Causal Gates

| Gate | Status |
|---|---|
| A/B0 token equality | PASS |
| A/B0 logit parity | DEFERRED |
| A/B1 matched-prefix diagnostic | MEASURED |
| A/B1 base-logit fidelity | MEASURED (top1=0.7083333134651184; mean KL(Vanilla || B1)=0.6822436451911926 nats) |
| B0/B1 upstream adapter isolation | PASS |
| B1/B2 Step-100 checkpoint isolation | FAIL |
| B0/B2 parameter isolation (legacy) | FAIL |
| Forced-H exact expansion | NOT_TESTED |
| Forced-H semantic position/cache state | NOT_TESTED |
| Forced-H immediate continuation stability | NOT_TESTED |

---

## 2. Condition Overview

| Condition | Prompts | Agg Quality | MBPP Pass | GSM8K Pass | Alpaca Pass | Mean Latency (s) | Total Steps | Live Realized Compression | H Emissions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A — PURE VANILLA** | 12 | **33.3%** | 0.0% | 0.0% | 100.0% | 14.873 | 4591 | 0.0% | 0 |
| **B0 — TOKENS ARCHITECTURE, VANILLA WEIGHTS** | 12 | **33.3%** | 0.0% | 0.0% | 100.0% | 15.546 | 4591 | 0.0% | 0 |
| **B1 — UPSTREAM EPFL PEFT ADAPTER, H DISABLED** | 12 | **66.7%** | 50.0% | 50.0% | 100.0% | 16.074 | 3431 | 0.0% | 0 |

---

## 3. Comparison vs Vanilla (<= 3% Quality Gate)

| Condition | Abs Quality Diff | Rel Quality Drop | Meets <=3% Gate? | Speedup vs Vanilla | Faster? | Steps Saved |
|---|---:|---:|:---:|---:|:---:|---:|
| **B0 — TOKENS ARCHITECTURE, VANILLA WEIGHTS** | +0.0% | 0.00% | PASS | -4.5% | NO | 0 |
| **B1 — UPSTREAM EPFL PEFT ADAPTER, H DISABLED** | +33.3% | -100.03% | PASS | -8.1% | NO | 1160 |

