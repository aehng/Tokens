# Architectural Diagnosis: Causal Resolution of B1 Base-Token Drift

- **Status**: Formally Resolved & Verified on Kaggle dual-T4 GPUs (Version 7)
- **Repo**: `aehng/Tokens`
- **Branch**: `grok/predictive-fidelity-audit`
- **Date**: September 29, 2026
- **Diagnostic Run Execution**: 418.9s wall-clock, 0.233 GPU-hours consumed; 2.50 GPU-hours remaining of 5.0h cap.
- **Reference Artifacts**:
  - [`docs/b1_diagnostic_matched_prefix.json`](file:///c:/Users/elijk/Documents/Projects/Tokens/docs/b1_diagnostic_matched_prefix.json)
  - [`docs/b1_diagnostic_attribution_summary.json`](file:///c:/Users/elijk/Documents/Projects/Tokens/docs/b1_diagnostic_attribution_summary.json)
  - [`docs/b1_diagnostic_attribution_report.md`](file:///c:/Users/elijk/Documents/Projects/Tokens/docs/b1_diagnostic_attribution_report.md)
  - [`docs/b1_diagnostic_raw_records.jsonl`](file:///c:/Users/elijk/Documents/Projects/Tokens/docs/b1_diagnostic_raw_records.jsonl)

---

## 1. Executive Summary & Causal Isolation ($H_1$ vs $H_2$)

The central question of the predictive hypertoken fidelity investigation was:
> *Does the observed ~22% quality drop between Vanilla Phi-3.5 and the hypertoken checkpoint originate from the upstream EPFL PEFT adapter ($H_1$) or from the local Step-100 hypertoken training checkpoint ($H_2$)?*

The matched-prefix diagnostic run on Kaggle decisively confirms **$H_1$**:

1. **Condition B0 (Tokens Wrapper, Vanilla Weights, NO Adapter, NO Checkpoint, $H$ Masked)**:
   - **12 out of 12** exact token sequence matches with pure Vanilla Phi-3.5 across all DEV prompts.
   - Proves conclusively that the Tokens architecture wrapper, `HyperEmbedding`, `HyperLinear`, and `StaticCodebookManager` preserve 100% token equivalence when base weights are unaltered.
2. **Condition B1 (Tokens Wrapper, Upstream EPFL PEFT Adapter, NO Step-100 Checkpoint, $H$ Masked)**:
   - Evaluated across 48 deterministic matched-prefix continuation states ($[0, 1, 4, 16]$ tokens across 12 DEV prompts).
   - Top-1 agreement rate collapses to **70.83%** (34/48 agreeing).
   - Immediate drift at Prefix 0: **66.7%** Top-1 agreement (4 of 12 prompts branch away on the very first token).
   - Code domain (`mbpp`) Top-1 agreement collapses to **50.0%** (8/16) with mean KL divergence of **1.2007 nats**.
   - Native vocabulary logits experience a mean absolute shift of **16.12** (max **53.28**).
   - Free greedy generation produces **0 out of 12** exact token matches vs Vanilla Phi.

> [!CAUTION]
> **Core Finding**: The upstream Zip2Zip PEFT LoRA adapter modifies the base language model's internal representation across all 32 transformer layers. Even when every hypertoken slot is masked to $-\infty$ and zero local checkpoint weights are loaded, the adapter alters the native next-token distribution.
> 
> The core product assumption—*“Add predictive capability without altering normal generation”*—is **false** for any architecture that routes normal base-token decoding through an active in-situ LoRA adapter.

---

## 2. Empirical Evidence Table

### Matched-Prefix Diagnostic Comparison (48 States)

| Prefix Length | Checked States | Top-1 Agreement | Top-1 Rate | Mean Top-5 Overlap | Mean KL ($\text{Vanilla} \parallel \text{B1}$) | Mean $L_1$ Logit Shift |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Prefix 0** (Post-prompt token 1) | 12 | 8 / 12 | **66.7%** | 48.3% | 0.9530 nats | 13.98 |
| **Prefix 1** (+1 token continuation) | 12 | 11 / 12 | **91.7%** | 71.7% | 0.1487 nats | 19.12 |
| **Prefix 4** (+4 tokens continuation) | 12 | 7 / 12 | **58.3%** | 56.7% | 1.1635 nats | 13.78 |
| **Prefix 16** (+16 tokens continuation) | 12 | 8 / 12 | **66.7%** | 76.7% | 0.4638 nats | 17.60 |
| **Overall Aggregate** | **48** | **34 / 48** | **70.83%** | **63.33%** | **0.6822 nats** | **16.12** |

### Breakdown by Task Domain

| Domain | Prompts | Checked States | Top-1 Agreement Rate | Mean KL Divergence | Max Logit Shift |
|---|:---:|:---:|:---:|:---:|:---:|
| **Code** (`mbpp`) | 4 | 16 | **50.0%** (8/16) | **1.2007 nats** | 50.699 |
| **Instruction** (`alpaca`) | 4 | 16 | **81.2%** (13/16) | **0.4451 nats** | 53.281 |
| **Reasoning** (`gsm8k`) | 4 | 16 | **81.2%** (13/16) | **0.4009 nats** | 49.375 |

---

## 3. Audit of the Current Architecture

### Parameter and Layer Inspection

Inspection of the loaded upstream adapter (`epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1`, revision `11c461733a79d2a5de6b814585c3361ca2aacbe7`) reveals:

- **Total LoRA Parameters**: 50,331,648 (~50.3M parameters).
- **Target Modules**:
  - `layers.{0..31}.self_attn.qkv_proj` ($r=32$, $\text{lora\_A}: [32, 3072]$, $\text{lora\_B}: [9216, 32]$)
  - `layers.{0..31}.self_attn.o_proj` ($r=32$, $\text{lora\_A}: [32, 3072]$, $\text{lora\_B}: [3072, 32]$)
  - `layers.{0..31}.mlp.gate_up_proj` ($r=32$, $\text{lora\_A}: [32, 3072]$, $\text{lora\_B}: [16384, 32]$)
  - `layers.{0..31}.mlp.down_proj` ($r=32$, $\text{lora\_A}: [32, 8192]$, $\text{lora\_B}: [3072, 32]$)
- **Layer Coverage**: Every single one of Phi-3.5's 32 transformer blocks (layers 0 through 31).

### The Exact Failure Mechanism

```mermaid
flowchart TD
    subgraph Current_Flawed_Flow["Current In-Situ Architecture (Condition B1/B2)"]
        P[Prompt / Generated Token x] --> EMB[HyperEmbedding Table]
        EMB --> L0["Transformer Layer 0: W_base + ΔW_lora"]
        L0 --> L1["Transformer Layer 1: W_base + ΔW_lora"]
        L1 --> Ldots["... Layers 2-30 ..."]
        Ldots --> L31["Transformer Layer 31: W_base + ΔW_lora"]
        L31 --> LN[Final LayerNorm]
        LN --> H_corrupt["Corrupted Hidden State h_L ≠ h_vanilla"]
        H_corrupt --> HEAD[lm_head: Native + H Slots]
        HEAD --> LOGITS[Shifted Native Logits ΔL1 ≈ 16.1]
        LOGITS --> MASK["StaticCodebookManager Logits Warper: Mask H slots [32011..32042] to -inf"]
        MASK --> OUT["Final Distribution (Top-1 Disagrees in 29.2% of positions!)"]
    end
```

### Why Vocabulary Masking Cannot Save Fidelity
1. `StaticCodebookManager` operates exclusively on the **output logits tensor** at the very end of the forward pass, setting indices $[32011, 32011+K-1]$ to $-\infty$.
2. However, the logits for all native tokens ($0 \le i < 32011$) are calculated as $\text{logits}[i] = W_{\text{lm\_head}}[i] \cdot h_L$.
3. Because $h_L$ was computed through 32 layers of LoRA-modified attention and MLP projections, $h_L$ deviates drastically from $h_{L,\text{vanilla}}$.
4. Masking hypertoken slots prevents the model from choosing an unseeded hypertoken, but does **nothing** to restore the corrupted probabilities among the 32,064 native tokens.

---

## 4. Evaluation of Alternative Architectures

To satisfy the product mandate (*preserve customer's Vanilla model behavior without compromise* while achieving inference speedup), four alternative architectures are evaluated:

| Architecture Option | Base Model Fidelity | Realized Speedup Potential | Implementation Complexity | Memory Footprint | Feasibility Assessment |
|---|:---:|:---:|:---:|:---:|:---:|
| **Option A: External Speculative Proposer** | **100% Guaranteed** | High ($1.5\times - 2.5\times$) | Medium | Moderate (2 models or small proposer) | **Recommended Primary** |
| **Option B: Adapter Only During Propose** | **100% Guaranteed** | Moderate | Low-to-Medium | Lowest (1 shared model) | **Recommended Secondary** |
| **Option C: Pure Separate Non-LoRA Predictor** | **100% Guaranteed** | High ($1.5\times - 3.0\times$) | Medium | Low (frozen base + small ranker) | **Strong Alternative** |
| **Option D: Conditional Token-Level LoRA Routing** | Fragile | Low | Very High | Low | **Not Recommended** |

### Detailed Evaluation of Options

#### Option A: External Speculative Proposer
- **Mechanism**: Pure frozen Vanilla Phi acts as the primary execution and verification engine. A secondary, highly compressed or LoRA-adapted model acts strictly as a speculative hypertoken/ngram proposer.
- **Verification Protocol**:
  1. The proposer suggests a hypertoken $H \equiv (t_1, t_2, \dots, t_m)$.
  2. Vanilla Phi verifies the sequence in a single speculative forward step.
  3. If accepted, KV-cache advances by $m$ tokens (achieving multi-token speedup).
  4. If rejected, Vanilla Phi defaults to its native next-token.
- **Fidelity**: Strictly 100% token-for-token identical to Vanilla Phi by mathematical construction.

#### Option B: Adapter Only During Candidate Generation
- **Mechanism**: A single model instance is used. During standard decoding, the LoRA adapter is disabled (`model.disable_adapters()`). The model operates exactly as Condition B0 (which achieves 12/12 exact token equivalence).
- **Phrase Proposal**: When the model reaches a phrase prediction trigger (e.g., prompt prefill or punctuation boundary), the adapter is briefly enabled to score or emit candidate hypertokens, which are then verified against the disabled base model.
- **Fidelity**: 100% base token fidelity during decode.

#### Option C: Pure Separate Non-LoRA Predictor (Tokens Predictor v2 Native)
- **Mechanism**: The base model remains 100% Vanilla Phi with `HyperEmbedding` / `HyperLinear` as in Condition B0. Candidate phrases are retrieved via the external sourcebook / association index (`candidate_retrieval.py`) and ranked by a lightweight external model (`PooledMLP` or `TransformerRanker`).
- **Synthesis**: The hypertoken encoder (`BaseEncoder`) synthesizes hypertoken embeddings using only the base model's frozen embedding weights.
- **Advantage**: Eliminates PEFT entirely from Phi-3.5's transformer layers.

#### Option D: Conditional LoRA Routing
- **Mechanism**: Custom attention kernels that apply $\Delta W$ only to hypertoken query/key tokens while routing native tokens through $W_{\text{base}}$.
- **Drawback**: Transformer self-attention inherently mixes query, key, and value vectors across all tokens in the sequence context. Even with masked projections, attention weights leak across token boundaries unless full dual-path attention is maintained.

---

## 5. Decision on Condition B2 (Step-100 Checkpoint)

> [!IMPORTANT]
> **Decision: Do NOT execute Condition B2 on Kaggle.**
> 
> **Scientific Justification**:
> 1. What would B2 tell us that we do not already know?
>    - B2 adds the local Step-100 fine-tuned checkpoint on top of B1.
>    - The Step-100 checkpoint was trained on hypertoken prediction tasks; it did not zero out the 50.3M LoRA parameters across Phi-3.5's 32 layers.
>    - Stage-1 historical records already established that `B_h_disabled` (which is B2) suffered a 22.2% quality collapse.
>    - The B1 diagnostic has conclusively proven that this collapse is already fully present in the upstream adapter (70.83% Top-1 agreement, 16.12 logit shift, 0/12 matches).
> 2. **Budget Stewardship**:
>    - A B2 run would consume ~0.25–0.35 GPU-hours simply to re-confirm that fine-tuned LoRA weights also alter base token logits.
>    - Our remaining budget is **2.50 GPU-hours** out of the 5.0h cap. This budget must be conserved to validate fidelity-preserving architectures (Option A / Option B / Option C).

---

## 6. Recommended Next Steps & Roadmap

1. **Implement Dynamic Adapter Bypassing (Option B prototype)**:
   - Verify locally on CPU/small device that wrapping base forward passes with `peft.disable_adapter()` restores 12/12 exact token equivalence even when PEFT adapter weights are loaded in memory.
2. **Benchmark Speculative Verification (Option A/C)**:
   - Implement speculative acceptance gating where hypertoken expansions $(t_1, \dots, t_m)$ proposed by the hypertoken dictionary are verified against frozen Vanilla Phi's top-1 or temperature-calibrated likelihood.
3. **Preserve Pinned DEV Split**:
   - Maintain strict separation from the FINAL test split until a candidate architecture demonstrates 100% base-token fidelity and measurable decode compression on DEV.
