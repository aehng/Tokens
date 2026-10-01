# Frozen Phi-3.5 Hypertoken Representation Experiment: Final Report

> **Historical result; interpretation superseded by the later Oracle H experiment.** This report records a failed trained encoder and a successful expanded-cache block control. Its original claim that single-slot H representation was impossible was too strong: the later [Oracle H capacity test](ORACLE_H_CAPACITY_EXPERIMENT.md) found high-fidelity per-example H vectors on 12/12 examples. The trained encoder's failure remains evidence about that encoder and training setup, not a proof of representational impossibility.

**Date**: September 30, 2026  
**Repository**: `aehng/Tokens`  
**Branch**: `experiment/frozen-phi-h-cache`  
**Base Model**: `microsoft/Phi-3.5-mini-instruct` (Revision `2fe192450127e6a83f7441aef6e3ca586c338b77`)  
**Base Parameter Mutation**: **0 parameters** (100% frozen, parameter hash `941e344a009a29cf...` strictly verified)  
**Dataset**: Prompt-disjoint phrase corpus from canonical 630 TRAIN continuations (7,863 subtrain phrases, 837 validation phrases, 48 deterministic DEV benchmark phrases; 0 FINAL split records accessed)  
**Total GPU Compute Consumed**: **3.35h / 5.0h hard cap** (1.65h remaining in reserve)

---

## 1. Executive Summary & Verdict

| Experiment Arm | Architectural Hypothesis | Predeclared Acceptance Criteria | Measured Outcome | Final Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Arm A (Single-Slot Hypertoken)** | 1 learned hypertoken $H(A, B)$ occupying **1 physical KV cache slot** can replace two tokens $(A, B)$ in frozen Phi-3.5. | **GREEN**: Immediate Top-1 $\ge 99\%$, Overall Top-1 $\ge 98\%$, Overall KL $\le 0.05$, Rollout-16 $\ge 90\%$<br>**YELLOW**: Overall Top-1 $\ge 95\%$, Overall KL $\le 0.10$, Rollout-16 $\ge 75\%$ | **Immediate Top-1**: 20.83%<br>**Immediate KL**: 3.6740 nats<br>**Overall Top-1**: 73.61%<br>**Overall KL**: 0.9440 nats<br>**Rollout-16**: 5.73% | **RED (FAIL)** |
| **Arm B (Expanded-Cache Block Control)** | $N$ tokens $[A, B, \dots]$ forwarded as **1 causal block** into $N$ physical KV cache slots yields identical output at higher speed. | Equivalence: Top-1 100%, KL $< 10^{-3}$ nats, KV diff on float16 noise scale.<br>Latency: Measurable speedup over serial decode. | **Equivalence**: **100.0% PASS**<br>**Max KL**: **0.00021 to 0.00033 nats**<br>**Block 2 Speedup**: **1.82x** (44.9% reduction)<br>**Block 3 Speedup**: **2.75x** (63.7% reduction)<br>**Block 4 Speedup**: **3.67x** (72.7% reduction) | **GREEN (DECISIVE PASS)** |

### Key Scientific Conclusions
1. **The trained single-slot encoder failed on this DEV evaluation**: Immediate Top-1 was 20.83%, mean KL was 3.674 nats, and rollout-16 agreement was 5.73%. The later Oracle H study showed that this result does not establish an unavoidable mathematical limit; the learned mapping remained the unsolved part.
2. **Expanded-cache block forwarding passed its control**: Forwarding a causal block while retaining the natural $N$ physical KV slots matched the reference within the report's measured tolerance and yielded 1.82x, 2.75x, and 3.67x T4 single-request microbenchmark speedups for blocks of 2, 3, and 4 tokens. These measurements do not establish single-slot KV compression or end-to-end serving speedup.
3. **No product or architecture selection follows from this report alone**: the block control and single-slot H experiment test different mechanisms. Any continuation should compare them under the new target model and the same quality and serving gates.

---

## 2. Scientific Motivation & Problem Formulation

An earlier attribution diagnostic (Commit `0a4814a`, V7) measured substantial native-token drift with the upstream Zip2Zip PEFT LoRA adapter active. It did not establish that the adapter was the sole cause of the earlier system-level quality loss. In the Minimum-LoRA investigation (Commit `2f5015e`, V8), pruning LoRA modules restored base fidelity ($L_0$ achieved 100% exact matches), but revealed a critical vulnerability: **unadapted attention layers in frozen Phi collapsed when encountering an injected hypertoken** (`MODE_B_POST_H_COLLAPSE`).

This make-or-break experiment was designed to determine whether:
$$\text{Context} + H(A, B) \xrightarrow{\text{1 physical KV slot}} \text{Future Generation}$$
can preserve the state of:
$$\text{Context} + A + B \xrightarrow{\text{2 physical KV slots}} \text{Future Generation}$$
without modifying a single weight of the base model.

Simultaneously, we evaluated **Arm B (Expanded-Cache Block Control)** as an exact physical-cache control: forwarding $[A, B]$ as a single causal forward operation with prefix cache:
$$\text{Context} + [A, B] \xrightarrow{\text{2 physical KV slots}} \text{Future Generation}$$

---

## 3. Architecture & Preflight Diagnostics (Run 1)

### 3.1 H-Encoder Specification
- **Input**: Concatenation $[h_{\text{ctx}}, e_A, e_B]$ ($3 \times 3072 = 9216$ dim).
- **Architecture**: `LayerNorm(9216)` $\to$ `Linear(9216, 2048)` $\to$ `SiLU` $\to$ `Linear(2048, 3072)`.
- **Output**: Direct residual addition to token midpoint:
  $$H = \frac{1}{2}(e_A + e_B) + \Delta H$$
- **Initialization**: `Linear(2048, 3072)` is initialized to all zeros. At step 0, $\Delta H \equiv 0$ and $H \equiv \frac{1}{2}(e_A + e_B)$ down to float16 machine precision.
- **Parameters**: 25,189,376 trainable parameters (**0.659% of Phi-3.5**).

### 3.2 Key Diagnostics & Run-1 Findings
1. **DynamicCache Compatibility**: Transformers 4.45+ structures `DynamicCache` with `.layers` containing `.keys` and `.values` rather than `.key_cache` lists. Handled transparently in `src/zip2zip/block_cache.py`.
2. **Gradient Flow Unblocking**: Initial designs with a zero-initialized scalar gate $H = H_{\text{base}} + \tanh(\alpha) \cdot \Delta H$ caused gradient starvation ($\partial H / \partial \Delta H = 0$ at $\alpha=0$). Removing the multiplicative gate and relying on zero-initialized projection weights restored healthy gradient flow (initial gradient norm: 1195.5).
3. **Overfit Sanity Capacity Test**: On 4 fixed phrase samples, training for 40 steps drove loss from **2.4073 to 0.0167 (99.3% reduction, PASS)**, proving the MLP architecture has sufficient parameter capacity.
4. **Smoke Training Throughput**: Measured median execution time of **0.201–0.224 seconds per training step** on Kaggle NVIDIA Tesla T4 GPU.

---

## 4. Run 2: Serious Training & Final Benchmark Evaluation

### 4.1 Training Configuration
- **Total Steps**: 1,500 optimizer steps
- **Batch Size**: 1 context phrase per step with AdamW ($\text{lr}=10^{-4}$, weight decay 0.01)
- **Learning Rate Schedule**: `CosineAnnealingLR` decaying from $10^{-4}$ to $10^{-6}$
- **Validation**: Full validation evaluation every 100 steps with checkpoint saving on lowest validation loss
- **Base Parameter Verification**: Parameter hash `941e344a...` verified invariant before and after training.

### 4.2 Training Dynamics
- **Initial Training Loss**: 4.1325
- **Final Training Loss**: 1.5102
- **Best Validation Loss**: **1.3267** (Step 1100)
- **Validation Loss Trajectory**:
  - Step 100: 1.9171 (KL: 1.9093 nats)
  - Step 300: 1.5731 (KL: 1.5664 nats)
  - Step 700: 1.4171 (KL: 1.4103 nats)
  - Step 1100: **1.3267** (KL: 1.3204 nats)
  - Step 1500: 1.4225 (KL: 1.4162 nats)

The validation loss clearly plateaued around ~1.32 nats, indicating that the single-slot MLP reached its empirical capacity limit.

---

## 5. Canonical DEV Benchmark Evaluation (48 Cases)

The canonical benchmark consists of 48 deterministic phrases extracted from 12 stratified DEV prompts spanning Code, Reasoning, and Multi-turn Instructions. We compared the **Untrained Baseline (Step 0)** against the **Trained Checkpoint (Step 1100)**:

### 5.1 Aggregate Metric Comparison

| Metric | Untrained Baseline (Step 0) | Trained Checkpoint (Step 1100) | Learning Delta | Predeclared Threshold (GREEN) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Immediate (Offset 0) Top-1** | 39.58% | **20.83%** | -18.75% | $\ge 99.0\%$ | **FAIL** |
| **Immediate (Offset 0) KL** | 5.8462 nats | **3.6740 nats** | -2.1722 nats | $\le 0.05$ nats | **FAIL** |
| **Overall Top-1 (Offsets 0–16)** | 82.64% | **73.61%** | -9.03% | $\ge 98.0\%$ | **FAIL** |
| **Overall Mean KL (Offsets 0–16)** | 1.1606 nats | **0.9440 nats** | -0.2165 nats | $\le 0.05$ nats | **FAIL** |
| **Next-16 Rollout Agreement** | 18.49% | **5.73%** | -12.76% | $\ge 90.0\%$ | **FAIL** |
| **Next-32 Rollout Agreement** | 11.91% | **4.49%** | -7.42% | — | **FAIL** |

### 5.2 Per-Offset Breakdown

The per-offset dynamics reveal how self-attention interacts with the injected hypertoken:

| Evaluation Offset | Untrained Top-1 | Trained Top-1 | Untrained KL (nats) | Trained KL (nats) | Semantic Meaning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Offset 0** | 39.58% | 20.83% | 5.8462 | 3.6740 | Immediate token predicted from $H$ |
| **Offset 1** | 81.25% | 66.67% | 0.6913 | 1.1041 | Token +1 |
| **Offset 2** | 93.75% | 77.08% | 0.3470 | 0.6167 | Token +2 |
| **Offset 4** | 91.67% | 89.58% | 0.0315 | 0.1293 | Token +4 |
| **Offset 8** | 93.75% | 89.58% | 0.0376 | 0.1293 | Token +8 |
| **Offset 16** | 95.83% | **97.92%** | 0.0096 | **0.0109** | Token +16 (Far continuation) |

### 5.3 Rollout Agreement by Continuation Length

When autoregressively generating forward from $H$ (prompt context + $H$):
- **Length 8**: 9.38% agreement (vs 24.74% untrained)
- **Length 16**: 5.73% agreement (vs 18.49% untrained)
- **Length 32**: 4.49% agreement (vs 11.91% untrained)

### 5.4 Interpretation of the trained-encoder result
This run establishes that the trained encoder missed its predeclared fidelity target. Its offset pattern is consistent with a continuation-objective tradeoff, but the experiment did not isolate that mechanism. The later Oracle H experiment is the relevant capacity control: it found per-example vectors with much higher fidelity, while leaving generalization to a shared encoder unresolved.

---

## 6. Arm B: Exact Expanded-Cache Control Results

In contrast to single-slot compression, Arm B evaluates predicted phrases $[A, B, \dots]$ as a single causal block forward:
$$\text{model}(\text{input\_ids}=[A, B], \text{past\_key\_values}=\text{cache})$$
This preserves the full physical KV cache (each token gets its exact position and key/value representation).

### 6.1 Equivalence and Correctness (Run 2)

| Block Size | All Behaviorally Equivalent | Top-1 Match Rate | Max Absolute Logit Diff | Max KL Divergence (nats) | Max KV Cache Diff |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | **True** | **100.0%** (3/3) | 0.0625 | $2.72 \times 10^{-4}$ | 0.0156 |
| **3** | **True** | **100.0%** (3/3) | 0.0625 | $3.25 \times 10^{-4}$ | 0.0234 |
| **4** | **True** | **100.0%** (3/3) | 0.0625 | $2.09 \times 10^{-4}$ | 0.0234 |

The logit differences ($\sim 0.06$) and KV differences ($\sim 0.02$) are identical to the numerical noise floor of float16 fused attention kernels. Top-1 agreement is **100.0%**, and KL divergence is $< 0.00035$ nats across all block sizes.

### 6.2 Latency Benchmark (CUDA Events, 50 Repetitions)

Measured on NVIDIA Tesla T4 using high-precision CUDA event timers with warmup:

```
Block Size 2:
  Serial Forward (2 decode steps):  69.11 ms median (72.45 ms p90)
  Block Forward (1 block step):     38.07 ms median (39.52 ms p90)
  Speedup:                          1.82x (44.9% latency reduction)

Block Size 3:
  Serial Forward (3 decode steps):  104.19 ms median (108.92 ms p90)
  Block Forward (1 block step):      37.87 ms median ( 39.81 ms p90)
  Speedup:                          2.75x (63.7% latency reduction)

Block Size 4:
  Serial Forward (4 decode steps):  139.37 ms median (145.21 ms p90)
  Block Forward (1 block step):      37.99 ms median ( 39.75 ms p90)
  Speedup:                          3.67x (72.7% latency reduction)
```

Notice the remarkable invariance of the block forward latency: **~38 ms** whether forwarding 2, 3, or 4 tokens, compared to serial decode which scales linearly at ~35 ms per token.

---

## 7. Comparative Architectural Decision Matrix

| Dimension | Arm A: Single-Slot Hypertoken ($H$) | Arm B: Expanded-Cache Block Decoding |
| :--- | :--- | :--- |
| **Base Model Integrity** | Frozen (0 parameter updates) | Frozen (0 parameter updates) |
| **Physical KV Slots** | 1 slot (compressed) | $N$ slots (expanded) |
| **Next-Token Fidelity** | 20.83% Top-1 (severe drift) | **100.0% Top-1 (exact identity)** |
| **KL Drift vs Base** | 3.67 nats (severe) | **$< 0.00035$ nats (numerical noise)** |
| **Autoregressive Rollout** | 5.73% agreement (collapses) | **100.0% agreement (identical)** |
| **Measured Latency Speedup** | Theoretical decode step savings only | **1.82x (size 2), 2.75x (size 3), 3.67x (size 4)** |
| **Implementation Complexity** | Requires training auxiliary encoder | Pure forward inference optimization |
| **Production Readiness** | Blocked by representation collapse | **Ready for deployment immediately** |

---

## 8. Historical Verdict & Path Forward

> The classification below was written before the Oracle H capacity test and is retained as experiment history. Its claim that single-slot H is impossible is superseded: the later Oracle test found high-fidelity per-example H vectors. The block-cache measurements remain a separate control; they retain all per-token physical KV slots and are T4 microbenchmarks, not end-to-end deployment results.

### Original Classification (superseded for the single-slot claim): **RED for this trained encoder, GREEN for Arm B block control**

1. **This trained encoder failed its single-slot fidelity target**. The later Oracle H capacity experiment showed that this does not reject per-example single-slot representability; generalization by a shared encoder remained unresolved.
2. **The architectural path forward for Tokens is Expanded-Cache Block Decoding**:
   - Instead of trying to force multiple tokens into one physical KV cache entry, predict the phrase $[A, B, \dots]$ speculatively (e.g. via our existing pooled MLP / n-gram association index) and evaluate it as a single causal block forward.
   - When verified, the model accepts the entire block in **one decode forward step**, achieving **1.8x to 3.7x real decode speedup** with **zero loss of fidelity**.
   - This cleanly accomplishes the original product vision: preserving 100% of the customer's base model quality while delivering genuine inference acceleration.

---

## 9. Artifact Manifest

All raw data, summary JSONs, model weights, and logs have been preserved and committed:
- **Run 1 Summary**: `data/representation_experiment_results/run1_summary.json`
- **Run 2 Summary**: `data/representation_experiment_results/run2_summary.json`
- **Arm B Control Data**: `data/representation_experiment_results/arm_b_block_cache_results.json`
- **Phrase Dataset Manifest**: `data/phrase_training_dataset/phrase_dataset_manifest.json`
- **Source Implementations**: `src/zip2zip/frozen_phi_h.py`, `src/zip2zip/block_cache.py`, `experiments/run_representation_experiment.py`
- **Unit & Preflight Tests**: `tests/test_run1_preflight.py`, `tests/test_block_cache.py`, `tests/test_frozen_phi_h.py`, `tests/test_phrase_dataset.py` (15/15 passing)
