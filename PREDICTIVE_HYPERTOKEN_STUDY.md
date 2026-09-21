# Predictive Hypertoken Calibration Study — Experimental Design

> **For AI assistants**: This document describes the *forward-looking* experimental plan.
> For what has already been done, see `RESEARCH_LOG.md` first.

---

## Problem Statement

**EPFL's Zip2Zip** (reactive mode) compresses LLM decoding by building a token vocabulary on-the-fly during generation using LZW compression. It works, but requires per-step dictionary updates that add latency (~2× slower than baseline on CPU).

**This study** tests a different mode — **pure predictive** — where the vocabulary is built from the *prompt alone*, before any generation occurs. This eliminates per-step overhead entirely, but requires the model to have been calibrated to actually use the predicted vocabulary entries.

**The product we are building**: a datacenter plug-and-play accelerator. Load our runtime + sidecar onto a serving stack the customer already runs. Simplest changeover wins. Base weights stay frozen. LoRA / encoder calibration is an acceptable install step. Full retraining of their model is not.

**The product question this study answers**: Can we ship `frozen_base_model + our_small_adapter + predictor` so it attaches to models we have never trained on? What is the smallest adapter that still works?

Research on Phi-3.5 / EPFL Zip2Zip is the vehicle. See `docs/product.md` for the commercial bar.

---

## Experimental Design

### Fixed Parameters
- Base model: `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1` (Phi-3.5, 3.8B + EPFL LoRA r=32)
- Codebook budget: K=32 (32 active phrase→hypertoken slots per request)
- Validation set: `data/cached_pure_pred_val_60.json` — 60 prompts, frozen, never tuned on
- Validation set composition: 20 code (MBPP), 20 instruction (Alpaca), 20 reasoning (GSM8k)
- Training data: `data/cached_pure_pred_train_2k.pkl` — 2,000 samples, prompts only seeded
- max_new_tokens: 150 for validation generations
- Hardware: Intel Core CPU + Intel Arc 130V GPU (XPU), PyTorch 2.14.0+xpu

### Control Conditions (run for every adaptation level)

| Condition | Description | Purpose |
|---|---|---|
| **C1: Base Phi-3.5** | `base_model.generate()`, no hypertokens | Quality ceiling reference |
| **C2: Reactive Zip2Zip** | `model.generate()`, dynamic LZW codebook | Compression ceiling reference |
| **C3: Predictive Step 0** | Pure predictive, zero-shot (original weights) | Zero-shot baseline |
| **C4: Predictive Step 50** | Pure predictive, step-50 checkpoint | Training curve point |
| **C5: Predictive Step 100** | Pure predictive, step-100 checkpoint | Current training result |
| **C6+** | Additional checkpoints as training extends | Learning curve |

---

## Adaptation Ladder

Run in order. Stop at the first level that achieves first-stage success (≥5% MICRO decode reduction, quality preserved, base frozen).

### Level 0 — Zero-Shot
No training. Confirm that the original EPFL model does not spontaneously emit predictive hypertokens without calibration.

### Level 1 — Output Encoder Calibration *(in progress)*
**Trainable**: `output_encoder` (226.5M fp32 params)
**Frozen**: everything else (Phi-3.5 base, EPFL LoRA, input_encoder)
**Status**: 100 steps complete. 60-prompt validation not yet run.
**Training schedule**: extend to 200→300→500 steps with 60-prompt eval every 50 steps and early stopping.

**Rationale**: The `output_encoder` is the module that generates logit weights for hypertoken output positions. If it doesn't produce high-probability logits for the seeded hypertokens, the model won't emit them. Calibrating only this 226.5M module is the least invasive change.

**Deploy artifact**: `output_encoder_step{N}_deploy.pt` (~906 MB)

### Level 1b — Both Encoders
**Trainable**: `input_encoder` + `output_encoder` (453M fp32 params total)
**Rationale**: The `input_encoder` generates the embedding representation of hypertoken positions at prefill. A miscalibrated input encoding could make the model "not see" hypertokens correctly, preventing it from learning to predict them. Training both may unlock faster convergence.
**Only run if Level 1 at 300+ steps is insufficient.**

### Level 2 — Output Encoder + EPFL LoRA Unfrozen
**Trainable**: `output_encoder` + existing EPFL LoRA A/B matrices (226.5M + 50.33M = 276.8M fp32)
**Frozen**: Phi-3.5 base weights (hash must match before/after)
**Rationale**: The EPFL LoRA was trained for reactive Zip2Zip. Unfreezing it allows attention patterns across all 32 layers to adapt to the predictive use case.
**Risk**: may degrade reactive Zip2Zip behavior (must verify).
**Only run if Level 1b is insufficient. Requires user approval.**

### Level 3 — Output Encoder + Fresh Small LoRA
**Trainable**: `output_encoder` + **new** LoRA r=4 on last 4 transformer layers (`o_proj` only)
**New params**: ~800K (negligible)
**Rationale**: Adds targeted attention output adaptation at the layers most directly influencing token selection, without disturbing the full 50M-param EPFL LoRA.
**Only run if Level 1b is insufficient. Design requires user approval.**

### Level 4 — Broader LoRA
**Only run if Level 3 is insufficient. STOP and report before launching.**

### Level 5+ — Partial/Full Fine-Tuning
**DO NOT RUN without explicit approval. Report evidence, compute estimate, and commercial implications.**

---

## Per-Prompt Metrics (collected for every condition and every sample)

### Model Modification Metrics
*(reported once per level, not per prompt)*
- Total model params
- Trainable params, trainable %
- Adapter size on disk
- Base model hash: unchanged? (YES/NO)
- Extra runtime modules
- Extra runtime latency

### Compression Metrics
- `base_prompt_len` — token count of original prompt
- `compressed_prefill_positions` — positions after segmentation with codebook
- `prompt_compression_pct` — (base − compressed) / base × 100
- `n_predicted_hypertokens_available` — K slots seeded for this prompt
- `n_hypertokens_emitted` — how many the model actually generated
- `base_equiv_output_tokens` — token count after hypertoken expansion
- `actual_decode_steps` — transformer forward passes during generation
- `decode_step_reduction_pct` — (expanded − steps) / expanded × 100
- `offline_available_prompt_savings` — tokens saved in prompt segmentation
- `realization_ratio` — emitted / available (how much of the opportunity was captured)

### Timing Metrics
- `setup_ms` — predictor + codebook construction + segmentation
- `prefill_ms` — first transformer forward pass
- `decode_ms` — all decode steps combined
- `total_ms` — end-to-end wall time

### Quality Metrics
- **Code (MBPP)**: Python `ast.parse()` syntax validity; test case correctness where available
- **Reasoning (GSM8k)**: extract final answer from `#### <number>` pattern; exact match vs. ground truth
- **Instruction (Alpaca)**: check for truncation (<5 output words), repetition (trigram repeat ≥4×), corruption (>30% non-ASCII), and instruction adherence (manual label)
- `quality_score`: 0 or 1 (pass/fail)
- `quality_label`: descriptive string (e.g., "correct", "syntax_error", "truncated", "ok")

### Aggregate Statistics
Computed by domain (all / code / instruction / reasoning):
- Mean, median, bootstrap 95% CI (n=2000 resamples)
- MICRO compression: Σ(saved tokens) / Σ(base tokens) — **primary metric**
- MACRO compression: mean per-request % — secondary
- Hypertoken emission histogram: n_prompts with 0, 1+, 2+, 3+ hypertokens
- Quality pass rate

---

## Data Construction Rules

All training examples are constructed as follows. These rules are invariants — any deviation invalidates the experiment.

```
For each training sample:
1. Take the prompt text.
2. Tokenize with Phi-3.5 tokenizer.
3. Run predictor on prompt tokens only → select top-K phrases.
   INVARIANT: predictor CANNOT see the response at selection time.
4. Build seeded_dict: phrase_tuple → vocab_id (starting at 32011).
5. Segment the prompt tokens using seeded_dict → compressed prompt prefix.
6. Segment the response tokens using the SAME seeded_dict → compressed labels.
   (Phrases from prompt that also appear in response are hypertoken targets.)
7. Construct input_ids = [compressed_prompt] + [compressed_response]
   and labels = [-100 × prompt_len] + [compressed_response_labels]
8. Store: {input_ids, labels, seeded_dict}
```

The predictor may use:
- Prompt token contents ✅
- Global phrase frequency statistics from training corpus ✅
- Domain priors ✅

The predictor may NOT use:
- Response tokens ❌
- Future tokens ❌
- Test set data ❌

---

## Training Configuration

```python
# Current best config (used for 100-step run)
max_steps = 100            # extend to 500 for learning curve study
grad_accum_steps = 2
lr = 1e-4
scheduler = CosineAnnealingLR(T_max=max_steps, eta_min=1e-6)
optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
trainable = [model.output_encoder.parameters()]  # 226.5M params
frozen = [model.base_model, model.input_encoder]

# Precision setup (required for CPU backward)
model.base_model:    torch.float16  # fp16: saves RAM, no grad needed
model.input_encoder: torch.float16  # fp16: frozen
model.output_encoder: torch.float32 # fp32: must be fp32 for DNNL CPU backward

# PyTorch DNNL constraint: bf16/fp16 backward not supported on x86 AVX2 Windows
# Workaround: keep only trainable params in fp32
```

---

## Checkpoint Format

```python
# Full checkpoint (training resume) — ~2.718 GB
{
    'step': int,
    'output_encoder_state_dict': OrderedDict,  # 226.5M fp32 params = 0.906 GB
    'optimizer_state_dict': dict,              # AdamW m+v = 1.812 GB
    'loss': float,
}

# Deploy checkpoint (inference only) — ~906 MB
{
    'step': int,
    'loss': float,
    'output_encoder_state_dict': OrderedDict,  # same weights, no optimizer
    'config': {'K': 32, 'dtype': 'fp32'},
}
```

**Why 2.718 GB?** The checkpoint does NOT store the 3.8B base model. The full checkpoint is large only because of AdamW optimizer state (first + second moment estimates for every fp32 param = 2× the param count in bytes).

---

## Commercial Deployment Layout (datacenter)

The install target is a **serving cluster the customer already operates**, not a research notebook. If an adaptation level succeeds, the artifact they load is:

```
customer_serving_stack/
├── their_base_model/                  # already in the datacenter — UNCHANGED
│
├── our_runtime/                       # generate wrapper / serving hooks
│
├── zip2zip_modules/
│   ├── output_encoder_deploy.pt       ← ~906 MB (our trained module)
│   ├── [input_encoder_deploy.pt]      ← ~906 MB (if Level 1b)
│   └── [lora_adapter_deploy.pt]       ← small (if Level 3+); OK for changeover
│
├── predictor/
│   └── cached_predictor.pkl           ← 23 MB
│
└── zip2zip_config.json                ← K=32, max_subtokens=3, attach points
```

Changeover should be: place sidecar next to the existing model, load adapter, serve. Unload to roll back.

**Verification at datacenter install**:
- `hash(base_model)` before deployment == `hash(base_model)` after deployment
- Adapter loads and unloads at runtime without side effects
- Original model still produces identical output with adapter removed
- **Held-out model family**: the same install recipe can be repeated on a backbone we did not train (optional short LoRA/encoder calibration). That is the generality bar for the product.

Full write-up: `docs/product.md`.

---

## Decision Rules

### Continue training at current level if:
- Hypertoken emission is monotonically increasing across checkpoints
- Quality is stable (no degradation)
- Training loss is generally decreasing (even if noisy)
- Validation MICRO decode reduction hasn't plateaued

### Stop training at current level and escalate if:
- No improvement in hypertoken emission across 200 consecutive steps
- Quality degrades >5pp from base on any domain
- Loss diverges or spikes unrecoverably

### Escalate to next adaptation level if:
- Current level produces <2% MICRO decode reduction after 300+ steps
- Hypertoken emission remains near 0 on the 60-prompt set

### Report success and stop escalating if:
- ≥5% MICRO decode reduction (first-stage success)
- Quality preserved (≤1pp below base on all domains)
- Base model weights unchanged (hash verified)

### STOP and report (no approval to continue) if:
- All frozen-base approaches fail
- Full fine-tuning appears necessary
- Report: what failed, why, estimated compute for full training, commercial implications

---

## Open Questions (as of 2026-09-21)

1. **Starting point / datacenter attach**: The product must attach to **customer models we have never trained**, not require they serve EPFL’s Zip2Zip checkpoint. Research may keep using `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1`. Product work should prefer vanilla (or customer) backbones + our LoRA/encoders so changeover is “add sidecar,” not “swap their model.” LoRA is an acceptable install cost.

2. **XPU training**: Intel Arc 130V XPU is available with confirmed bf16 backward support. Should the `output_encoder` be moved to XPU (bf16) for training? This would require a 3-prompt correctness test to ensure custom SDPA kernels in `ResLatentAttnLayer` work correctly on Intel XPU.

3. **max_new_tokens**: 150 is recommended (covers all domains, limits wall time). Adjust if needed.

4. **Adapter portability**: Once one checkpoint works, test whether the sidecar generalizes across domains **and** across model families we did not train. Per-domain or per-family LoRA/encoder calibration is acceptable; requiring a new full train is not.

---

*This document is the forward-looking companion to `RESEARCH_LOG.md`.*
*Last updated: 2026-09-21.*
