# Predictive Hypertoken Calibration — Research Log

> **Intended audience**: Any AI assistant (ChatGPT, Grok, Claude, Gemini, etc.) picking up this project.
> This document is the authoritative record of what has been done, what was found, and what happens next.
> Read this before reading any code or asking questions.

---

## What This Project Is

This is a **product codebase** that uses research experiments on top of **[EPFL's zip2zip library](https://arxiv.org/abs/2506.01084)** — inference-time adaptive token vocabularies for LLMs. The current research vehicle is `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1` (Phi-3.5-mini, 3.8B, Zip2Zip hypertoken wrap). That checkpoint is how we learn; it is not the intended customer install.

**The product goal**: ship a datacenter plug-and-play inference accelerator. A customer keeps the base LLM they already serve. We add the smallest sidecar that still works (predictor + encoders, LoRA if needed) so decode uses fewer steps. Changeover should be: load adapter, serve, unload adapter. Base weights stay hash-identical. The same recipe should attach to **models we have never trained on**, with at most a short calibration job (LoRA / encoder fit)—not a full custom train. See `docs/product.md`.

**The scientific goal of this experiment**: find the minimum adaptation that makes predictive hypertokens reliable, so the datacenter artifact stays small and general.

---

## Glossary

| Term | Definition |
|---|---|
| **Hypertoken** | A single token ID ≥ 32011 that represents a multi-token phrase (e.g., `["the", "model"]` → hypertoken ID 32011). Decodes as 2+ base tokens but costs only 1 decode step. |
| **Reactive Zip2Zip** | Original EPFL mode: codebook is built dynamically during generation using LZW on the tokens already generated. No prediction needed. |
| **Predictive (Pure Predictive)** | Our mode: codebook is built from the **prompt only**, before any tokens are generated. Requires a predictor and a calibrated output encoder to work. |
| **Codebook** | The set of K active phrase→hypertoken mappings for a given request. At K=32, up to 32 phrases can be active. |
| **Predictor** | A pre-trained phrase-frequency model (`cached_predictor.pkl`, 23 MB) that selects the K most likely phrases given a prompt, without seeing the response. |
| **output_encoder** | A 2-layer, 226.5M-parameter transformer (part of Zip2Zip) that generates the logit weights for hypertoken positions. This is the component we are training. |
| **input_encoder** | A mirror 226.5M-parameter transformer that generates the embedding for hypertoken input positions. Frozen in our experiments so far. |
| **MICRO compression** | Total tokens saved / total base tokens across all samples. Primary metric for compute economics. |
| **MACRO compression** | Mean per-request compression percentage. Useful for characterizing individual samples. |
| **Realization ratio** | Actual live decode-step reduction / offline available reduction. Measures how much of the theoretical savings we actually capture. |
| **K / budget** | Number of codebook slots. We use K=32 for the pure predictive mode in these experiments. |

---

## Architecture

```
Zip2ZipModel
├── base_model: PeftModel (Phi-3.5-mini-instruct + EPFL LoRA r=32)
│   ├── embed_tokens [32011, 3072] fp16 — FROZEN
│   ├── layers[0..31] — FROZEN base weights
│   │   ├── self_attn.qkv_proj  — frozen + EPFL LoRA A/B (r=32)
│   │   ├── self_attn.o_proj    — frozen + EPFL LoRA A/B (r=32)
│   │   ├── mlp.gate_up_proj    — frozen + EPFL LoRA A/B (r=32)
│   │   └── mlp.down_proj       — frozen + EPFL LoRA A/B (r=32)
│   └── lm_head → HyperLinear   — wraps output_encoder at inference
│       (vocabulary is extended to 32011 + K slots during generation)
│
├── input_encoder: TransformerEncoder [2 layers, hidden=3072, heads=32]
│   └── 226.5M params — FROZEN in current experiments
│   └── Generates embeddings for hypertoken input positions
│
└── output_encoder: TransformerEncoder [2 layers, hidden=3072, heads=32]
    └── 226.5M params — OUR TRAINABLE COMPONENT
    └── Generates logit weights for hypertoken output positions
```

### EPFL LoRA Details (pre-existing, not added by us)

```
PeftType:       LoRA
r:              32  (rank-32, alpha=32 → effective scale = 1.0)
dropout:        0.0
bias:           none
Layers:         ALL 32 transformer blocks
Target modules: qkv_proj, o_proj, gate_up_proj, down_proj
Total params:   50.33M (fp32)
requires_grad:  False at load time (all frozen)
```

### Parameter Counts

| Component | Params | % of Total | Status |
|---|---|---|---|
| Phi-3.5 base weights | ~3.82B | 88.3% | Frozen — hash verified |
| EPFL LoRA A/B matrices | 50.33M | 1.16% | Frozen — requires_grad=False |
| `input_encoder` | 226.5M | 5.24% | Frozen in current experiments |
| **`output_encoder`** | **226.5M** | **5.24%** | **Trained by us** |
| **Total** | **4.324B** | 100% | — |

---

## Key Source Files

```
src/zip2zip/
  model.py                    Zip2ZipModel — wraps base model, installs HyperEmbedding/HyperLinear
  static_codebook.py          StaticCodebookManager — manages seeded codebook for pure predictive
  nn/embedding.py             HyperEmbedding — maps hypertoken IDs to input embeddings
  nn/linear.py                HyperLinear — extends LM head for hypertoken logits (PATCHED)
  nn/encoders/
    res_latent_attn.py        ResLatentAttnEncoder — the encoder architecture (wq/wk/wv/wo layout)

src/evaluation/
  offline_segmenter.py        segment_tokens_dp() — DP segmenter for compressing prompt with codebook

experiments/
  train_pure_predictive_calibration.py   Phase 1 training script (COMPLETED 100 steps)
  eval_60prompt_validation.py            60-prompt validation sweep (WRITTEN, NOT YET RUN)
  scratch/
    phase0_audit.py           Architecture audit script
    inspect_lora.py           LoRA structure inspection script

data/
  cached_pure_pred_train_2k.pkl    2,000 training samples (pickle) — prompt→codebook→labels
  cached_pure_pred_val_60.json     60 held-out validation prompts (JSON) — NEVER touch for tuning
  cached_pure_pred_val_60.json fields:
    id, domain, prompt, ground_truth_response, prompt_token_ids, base_prompt_len
    First 20: code (MBPP), records 20-39: instruction (Alpaca), 40-59: reasoning (GSM8k)

experiments/checkpoints/
  cached_predictor.pkl             23 MB — pre-trained phrase predictor
  pure_pred_k32_step50.pt          2.718 GB — output_encoder + AdamW optimizer at step 50
  pure_pred_k32_step100.pt         2.718 GB — output_encoder + AdamW optimizer at step 100
  pure_pred_k32_stageA_training_log.json   Full training log with per-step loss/grad/timing
```

---

## What Has Been Done

### 1. Offline Benchmark (Earlier Sessions)
Ran full-corpus offline compression audit over 7,512 samples from OASST1 at K∈{16,32,64,128,256,512}.

**Key result at K=32 (our target operating point):**
- MICRO prompt compression: **9.75%** (total tokens saved / total base tokens)
- MACRO prompt compression: 10.25% (mean per-request)
- Total corpus: 1,504,463 base tokens; 146,718 tokens saved

**Key finding**: The predictor selects good phrases, but without calibrated hypertoken emission, the model rarely produces them. This motivated calibration training.

### 2. Scientific Causality Verification
Confirmed via `verify_prompt_guarantees.py`:
- Codebook constructed from **prompt tokens only** — no response leakage
- Compressed prompt round-trips exactly (re-expansion matches original)
- Hypertokens can be synthesized before transformer prefill

### 3. Mixed Precision Fix (Critical Bug Fix)
**Problem**: PyTorch DNNL on x86 Windows CPU does NOT support fp16/bf16 backward. Loading entire model in float32 → 17+ GB RSS → severe Windows swap thrashing.

**Fix applied to two files:**
- `src/zip2zip/nn/linear.py` lines 36–48 (`HyperLinear.forward`): casts `h` to match `hyper_linear_weights.dtype` before bmm, then casts result back before concat
- `src/zip2zip/static_codebook.py` lines 317–325 (`get_hyper_linear_weights`): reallocates cache to match encoder output dtype

**Training precision**: backbone fp16 (frozen, no grad), `output_encoder` fp32 (trainable, backward works).

### 4. Training Data Preparation
- `data/cached_pure_pred_train_2k.pkl`: 2,000 training samples, each containing `input_ids`, `labels`, `seeded_dict`
- Training data construction: tokenize prompt → run predictor (prompt-only, no response) → build codebook → segment prompt and response → create training sequence with hypertoken labels
- `data/cached_pure_pred_val_60.json`: 60 held-out prompts, stratified (20 code, 20 instruction, 20 reasoning), **never touched during training**

### 5. Phase 1 Calibration Training — COMPLETE
Trained `output_encoder` for 100 steps on CPU.

```
Config:
  max_steps: 100
  grad_accum_steps: 2
  lr: 1e-4
  scheduler: CosineAnnealingLR (T_max=100, eta_min=1e-6)
  optimizer: AdamW (weight_decay=0.01)
  trainable: output_encoder only (226.5M fp32 params)
  frozen: base_model, input_encoder, EPFL LoRA
  total wall time: 2765.31s (~46 min)
  avg step time: 27.65s/step
```

**3-probe progression (1 GSM8k, 1 MBPP, 1 Alpaca sample):**

| Checkpoint | GSM8k Hypers Emitted | Decode Step Savings |
|---|---|---|
| Step 0 (zero-shot) | 1 | 1.54% |
| Step 50 | 2 | 3.03% |
| Step 100 | 3 | 4.48% |

Code/instruction: 0 hypertokens at all steps (prompts ~24 tokens, low phrase repetition → expected).

**Training loss**: oscillating 1.3–4.2, no clear plateau at step 100 → likely **undertrained**.
**Grad norms**: frequently spiking (up to 86×) → noisy training signal.

### 6. Phase 0 Architecture Audit — COMPLETE (This Session)
Ran `phase0_audit.py` and `inspect_lora.py`.

**Critical discovery**: The EPFL Zip2Zip checkpoint already contains a **rank-32 LoRA adapter** across all 32 transformer layers and all 4 projection types (qkv, o_proj, gate_up, down_proj). 50.33M params, all frozen at load time.

**Hash verification**: MD5 hashes of Phi-3.5 base weights and EPFL LoRA matrices confirmed **unchanged** before and after loading our step-100 checkpoint. Our training did not touch anything except `output_encoder`.

---

## Current Status

| Task | Status |
|---|---|
| Offline compression audit (7,512 samples) | ✅ Complete |
| Causality/round-trip verification | ✅ Complete |
| Mixed precision bug fix | ✅ Complete |
| Training data preparation (2k train, 60 val) | ✅ Complete |
| 100-step output_encoder calibration | ✅ Complete |
| Phase 0 architecture audit | ✅ Complete |
| 60-prompt full validation sweep | ⏸ Stopped after the first code prompt. Calibration emitted `\nassert` and truncated the answer. Do not resume the 300-generation sweep until a content codebook can commit a hypertoken inside an intact sentence. |
| Level 1b (input+output encoder training) | ❌ Not yet |
| Level 2 (unfreeze EPFL LoRA) | ❌ Not yet |
| Level 3 (fresh small LoRA) | ❌ Not yet |

---

## The Core Research Question

> **What is the minimum adapter we can ship so a datacenter can accelerate a frozen customer model (including families we never trained) with predictive hypertokens?**

Phi-3.5 Zip2Zip is the lab model for answering that. Product success is: same recipe, new base model, simple changeover (sidecar + optional LoRA).

### Adaptation Ladder (least → most invasive)

| Level | What is trained | New adapter size (deploy) | Status |
|---|---|---|---|
| 0 | Nothing | 0 | Reference baseline |
| 1 | `output_encoder` only (226.5M) | ~906 MB | **100 steps done; needs full 60-prompt eval** |
| 1b | `input_encoder` + `output_encoder` | ~1.81 GB | Not started |
| 2 | `output_encoder` + EPFL LoRA unfrozen | ~1.1 GB | Not started |
| 3 | `output_encoder` + fresh small LoRA r=4 | ~930 MB | Not started |
| 4 | `output_encoder` + broader LoRA | ~960 MB+ | Not started |
| 5 | Partial base unfreeze | — | STOP: requires approval |
| 6 | Full fine-tuning | — | DO NOT RUN |

**Decision rule**: proceed to the next level only if the current level fails to achieve ≥5% MICRO decode step reduction with quality preserved.

---

## Immediate Next Steps (ordered)

1. **Do not train the output encoder longer and do not resume the 60-prompt sweep.** On `mbpp_769`, step 50 and step 100 both emitted the codebook phrase `\nassert` and stopped after 35 tokens. Step 0 wrote a normal 192-token answer and emitted nothing. The K=32 codebook for that prompt was 31 punctuation, newline, and digit fragments, because the global frequency prior fills the budget. More steps of the same loss will practice that failure.

2. **Zip2Zip's own recipe is the consistent solution.** Paper: [arXiv:2506.01084](https://arxiv.org/abs/2506.01084). They train LoRA and both hyper-encoders together, on LZW-compressed documents, plus a reconstruction loss (λ=0.1) that converged to ~0. The codebook starts empty and only gains a code after the phrase has occurred. LZW, not a side predictor, decides when that code is the next symbol. The released Phi-3.5 checkpoint was trained that way (`max_codebook_size=2048`, `max_subtokens=4`).

3. **Forcing our predictor phrases into that model does not compress.** `experiments/scratch/commit_agreed_phrase.py` used the original encoders and content phrases from the prompt. On all 3 probes the expanded text matched the frozen greedy text (20/20 tokens) and committed 0 hypertokens. The two candidate phrases were rejected because the next token after the hyper-embedding was not the next token after the real phrase. The embedding is not a drop-in substitute unless the LM was trained on that code.

4. **The official path does both jobs.** `experiments/scratch/official_lzw_probe.py` compresses with `Zip2ZipTokenizer`, generates with the untouched checkpoint, and decompresses. At 40 decode steps: code 13 hypertokens / 24.5% fewer steps, instruction 19 hypertokens / 33.3%, GSM8K 4 hypertokens / 9.1%. All three completions are coherent. Prompt LZW savings were 4.2%, 0%, and 6.3%. Our earlier live bench fed raw token ids and set `max_codebook_size=32`, so it was not this procedure.

3. **Report results and decide** — if Level 1 at ~300–500 steps achieves first-stage success (≥5% decode reduction, quality preserved), report that as the candidate commercial architecture. If not, evaluate Level 2 or Level 3.

---

## Success Thresholds

| Tier | Criterion |
|---|---|
| Failure | <2% MICRO decode step reduction after 300 steps at any level |
| First success | ≥5% MICRO decode reduction, quality ≤1pp below base, frozen base weights |
| Strong success | ≥8% MICRO decode reduction, quality preserved, adapter ≤1 GB |
| Exceptional | >12% MICRO decode reduction approaching offline opportunity |

---

## Hardware

| Item | Value |
|---|---|
| OS | Windows 11 |
| RAM | 16.7 GB total, ~7.7 GB available |
| GPU | Intel Arc 130V (8.47 GB VRAM) |
| PyTorch | 2.14.0+xpu |
| `torch.xpu` | ✅ Available |
| XPU bf16 backward | ✅ Confirmed working |
| CUDA | ❌ Not available |
| OpenVINO | ❌ Not installed |
| bitsandbytes | ❌ Not installed |
| $0 budget | CPU-only by default; XPU usable if correctness confirmed |

**XPU memory constraint**: Phi-3.5 in bf16 = 3.87B × 2 bytes = ~7.74 GB. With overhead, likely OOM on 8.47 GB VRAM. Recommended strategy: keep backbone on CPU, optionally move `output_encoder` to XPU for training (~453 MB bf16, trivially fits).

---

## Constraints and Rules

- **No response leakage**: predictor uses only prompt tokens. This is a hard scientific requirement.
- **No tuning on val set**: the 60-prompt held-out set is never used for training decisions.
- **Quality is a hard gate**: step savings from producing worse/shorter output don't count.
- **No full fine-tuning without approval**: stop and report if all lighter approaches fail.
- **MICRO compression is primary**: total tokens saved / total base tokens (token-weighted). MACRO (per-request mean %) is secondary.
- **\$0 compute budget**: everything runs locally on CPU (or XPU if safe).

---

## Key Files to Read First (in order)

1. This file (`RESEARCH_LOG.md`) ← you are here
2. `PREDICTIVE_HYPERTOKEN_STUDY.md` ← the forward-looking experimental design
3. `experiments/train_pure_predictive_calibration.py` ← how training works
4. `experiments/eval_60prompt_validation.py` ← the validation sweep (run this next)
5. `src/zip2zip/static_codebook.py` ← the StaticCodebookManager (core runtime)
6. `src/zip2zip/nn/linear.py` ← HyperLinear (patched for mixed precision)
7. `experiments/checkpoints/pure_pred_k32_stageA_training_log.json` ← full training history

---

## Reproducibility

### Load the model and the step-100 checkpoint

```python
import torch, sys
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
from zip2zip import Zip2ZipModel

# Load model (fp16 backbone for RAM efficiency)
model = Zip2ZipModel.from_pretrained(
    'epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1',
    max_codebook_size=32,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Cast output_encoder to fp32 (required for CPU backward)
model.output_encoder.to(torch.float32)

# Load step-100 calibration weights
ckpt = torch.load(
    'experiments/checkpoints/pure_pred_k32_step100.pt',
    map_location='cpu',
    weights_only=False,
)
model.output_encoder.load_state_dict(ckpt['output_encoder_state_dict'])
```

### Run a predictive generation

```python
import pickle
from transformers import AutoTokenizer, LogitsProcessorList
from zip2zip import StaticCodebookManager
from src.evaluation.offline_segmenter import segment_tokens_dp

INITIAL_VOCAB = 32011
K = 32

tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct')
with open('experiments/checkpoints/cached_predictor.pkl', 'rb') as f:
    predictor = pickle.load(f)

prompt = "Write a Python function to compute the Fibonacci sequence."
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

# Build prompt-conditioned codebook (no response leakage)
p_dict, _ = predictor.select_prompt_conditioned(prompt_ids, budget=K)
pred_phrases = list(p_dict.keys())
comp_len, tiles, _ = segment_tokens_dp(prompt_ids, set(pred_phrases))
seeded_dict = {p: INITIAL_VOCAB + i for i, p in enumerate(pred_phrases)}
resegmented = [tile[0] if len(tile) == 1 else seeded_dict[tile] for tile in tiles]

# Setup static codebook manager
dim = model.zip2zip_config.encoder.hidden_size  # 3072
pad_id = 32000
disabled_ids = list(model.zip2zip_config.compression.disabled_ids)

static_mgr = StaticCodebookManager(
    initial_vocab_size=INITIAL_VOCAB,
    max_codebook_size=K,
    max_subtokens=3,
    embedding_dim=dim,
    pad_token_id=pad_id,
    disabled_ids=disabled_ids,
)
static_mgr.set_seeded_codebook(seeded_dict, batch_size=1, device=torch.device('cpu'))
static_mgr.attach_to_model(model)

# Generate
import torch
pred_tensor = torch.tensor([resegmented], dtype=torch.long)
logits_proc = LogitsProcessorList([static_mgr.get_logits_processor()])

with torch.no_grad():
    out = model.generate(
        input_ids=pred_tensor,
        max_new_tokens=150,
        logits_processor=logits_proc,
        do_sample=False,
    )
static_mgr.detach_from_model(model)

gen_tokens = out[0][len(resegmented):].tolist()
hypers = [t for t in gen_tokens if t >= INITIAL_VOCAB]
print(f"Generated {len(gen_tokens)} steps, {len(hypers)} hypertokens emitted")
print(tokenizer.decode(gen_tokens, skip_special_tokens=True))
```

---

*Last updated: 2026-09-21. Maintained by Antigravity (Google DeepMind) coding assistant.*
