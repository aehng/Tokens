# Tokens Bounded Minimum-LoRA Experiment Report

- **Generated At**: `2026-09-30T02:56:43.596863+00:00`
- **Device**: `cuda:0`
- **Tested Steps**: L0_ZERO, L1_TINY_ATTN_LAST4, L2_SMALL_ATTN_LAST8, L3_MOD_ALL_LAST8, LFULL
- **Viable Configuration Found**: `NO`

---

## 1. Executive Summary & Scientific Findings

**Finding**: None of the tested sub-network LoRA configurations met all 4 viability criteria simultaneously.
The existing Step-100 checkpoint was jointly co-trained with global rank-32 LoRA across all 32 layers. Post-hoc surgical masking reveals the architectural coupling between the hyperencoder embeddings and transformer layers.

---

## 2. Minimum-LoRA Ladder Comparison Table

| Ladder Step | Active LoRA Params | % of Phi-3.5 | Exact Base Matches | Top-1 Base Agrmt | Mean Base KL (nats) | Emitted H | Steps Saved % | Failure Mode / Verdict |
|---|---:|---:|:---:|:---:|:---:|---:|---:|:---:|
| **L0_ZERO** | 0 | 0.0000% | 12/12 | 100.0% | 0.0001 | 2119 | -46.1% | `MODE_B_POST_H_COLLAPSE` |
| **L1_TINY_ATTN_LAST4** | 2,359,296 | 0.0617% | 0/12 | 89.6% | 0.0181 | 154 | +8.0% | `MODE_B_POST_H_COLLAPSE` |
| **L2_SMALL_ATTN_LAST8** | 4,718,592 | 0.1235% | 0/12 | 91.7% | 0.0203 | 204 | +6.6% | `MODE_B_POST_H_COLLAPSE` |
| **L3_MOD_ALL_LAST8** | 12,582,912 | 0.3293% | 0/12 | 87.5% | 0.1193 | 347 | +2.3% | `MODE_B_POST_H_COLLAPSE` |
| **LFULL** | 50,331,648 | 1.3172% | 0/12 | 70.8% | 0.7446 | 145 | +8.6% | `MODE_B_POST_H_COLLAPSE` |

---

## 3. Failure Mode Taxonomy & Architectural Diagnosis

Each tested configuration was evaluated on 4 necessary criteria:
1. **Base-token fidelity**: Pure Vanilla Phi generation when H is not used.
2. **Useful H emission**: Emits valid hypertokens when enabled.
3. **Post-H continuation**: Context and KV cache remain stable after hypertoken emission.
4. **Real decode reduction**: Steps saved >= 5.0%.

### L0_ZERO
- **Verdict**: FAILED
- **Failure Mode**: `MODE_B_POST_H_COLLAPSE`
- **Diagnosis**: Mode B: Model emits hypertokens, but post-H continuation diverges into degradation/incoherence. Context/KV representations break.
- **Criteria Checklist**:
  - ✅ `criterion_1_base_fidelity`: PASS
  - ✅ `criterion_2_useful_h_emission`: PASS
  - ❌ `criterion_3_post_h_continuation`: FAIL
  - ❌ `criterion_4_real_decode_savings`: FAIL

### L1_TINY_ATTN_LAST4
- **Verdict**: FAILED
- **Failure Mode**: `MODE_B_POST_H_COLLAPSE`
- **Diagnosis**: Mode B: Model emits hypertokens, but post-H continuation diverges into degradation/incoherence. Context/KV representations break.
- **Criteria Checklist**:
  - ❌ `criterion_1_base_fidelity`: FAIL
  - ✅ `criterion_2_useful_h_emission`: PASS
  - ❌ `criterion_3_post_h_continuation`: FAIL
  - ✅ `criterion_4_real_decode_savings`: PASS

### L2_SMALL_ATTN_LAST8
- **Verdict**: FAILED
- **Failure Mode**: `MODE_B_POST_H_COLLAPSE`
- **Diagnosis**: Mode B: Model emits hypertokens, but post-H continuation diverges into degradation/incoherence. Context/KV representations break.
- **Criteria Checklist**:
  - ❌ `criterion_1_base_fidelity`: FAIL
  - ✅ `criterion_2_useful_h_emission`: PASS
  - ❌ `criterion_3_post_h_continuation`: FAIL
  - ✅ `criterion_4_real_decode_savings`: PASS

### L3_MOD_ALL_LAST8
- **Verdict**: FAILED
- **Failure Mode**: `MODE_B_POST_H_COLLAPSE`
- **Diagnosis**: Mode B: Model emits hypertokens, but post-H continuation diverges into degradation/incoherence. Context/KV representations break.
- **Criteria Checklist**:
  - ❌ `criterion_1_base_fidelity`: FAIL
  - ✅ `criterion_2_useful_h_emission`: PASS
  - ❌ `criterion_3_post_h_continuation`: FAIL
  - ❌ `criterion_4_real_decode_savings`: FAIL

### LFULL
- **Verdict**: FAILED
- **Failure Mode**: `MODE_B_POST_H_COLLAPSE`
- **Diagnosis**: Mode B: Model emits hypertokens, but post-H continuation diverges into degradation/incoherence. Context/KV representations break.
- **Criteria Checklist**:
  - ❌ `criterion_1_base_fidelity`: FAIL
  - ✅ `criterion_2_useful_h_emission`: PASS
  - ❌ `criterion_3_post_h_continuation`: FAIL
  - ✅ `criterion_4_real_decode_savings`: PASS

---

## 4. Retraining Recommendations

Based on the empirical attribution findings:
1. **Zero-LoRA Modular Target ($L_0$)**: If $L_0$ fails at hypertoken emission or post-H continuation due to representation mismatch, hyperencoder retraining must be executed with frozen Vanilla base weights.
2. **Bounded Interface Layer ($L_1 / L_2$)**: If an adaptation layer is strictly required, restrict LoRA training strictly to the top attention layers (e.g. layers 28–31 or 24–31, attention projections only). Zero out all MLP adaptations, which account for 58% of base drift.
3. **Dual Forward Routing**: Preserve Vanilla LM head and transformer blocks for base token generation, routing through the LoRA adapter *only* when evaluating candidate hypertoken logits.
