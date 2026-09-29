# Architecture Audit: Predictive Tokens vs Legacy Zip2Zip

Date: 2026-09-29
Branch intent: `refactor/predictive-fidelity-architecture`
HEAD inspected: `5e0f328` on `codex/phi-quality-speed-attribution`, plus local uncommitted attribution WIP (not used as source of truth).
No GPU compute was used for this audit.

This document records what the repository actually does today. It is the Gate-0 research baseline for the fidelity refactor.

---

## 1. Verdict on the incoming audit

The incoming diagnosis is correct on every hard claim:

| Claim | Status |
|---|---|
| Predictive imports pull in `zip2zip-compression` via package `__init__` | Confirmed |
| `Zip2ZipModel` always constructs dynamic `CodebookManager`, then static H replaces it | Confirmed |
| Joint training is LM CE + reconstruction; no teacher KL | Confirmed |
| `compute_continuation_consistency_loss` is documented and not implemented | Confirmed |
| Training tokenizes raw prompt/response text and appends `tokenizer.eos_token_id` | Confirmed |
| Step-100 load path is Vanilla Phi + EPFL PEFT + our joint adapter | Confirmed |
| `src.zip2zip` / `zip2zip` / `sys.path` mixing is real | Confirmed |
| Kaggle expected `select_stratified_dev_prompts` from `attribution_harness.py`; it lives in the benchmark runner | Confirmed |
| Stage-1 quality collapse is upstream of the predictor | Confirmed by committed Stage-1 report |

Additional findings that change the refactor:

1. **B0 is not currently a clean identity path.** Even with LoRA disabled, `Zip2ZipModel` replaces the embedding and LM head with `HyperEmbedding` / `HyperLinear`, constructs a Rust LZW codebook manager, and `generate()` always allocates a `max_codebook_size` hyper-logit block (K=32 in the pilot). Wrapper fidelity is unproven.
2. **Training EOS is the tokenizer EOS (`<|endoftext|>` = 32000), not chat `<|end|>` (32007).** Canonical Vanilla generation stops on `[32007, 32001, 32000]` after `apply_chat_template`. This is a stronger termination-mismatch hypothesis than “add more EOS loss.”
3. **Training targets are benchmark gold answers in `data/train.jsonl`**, not the 630 canonical Phi TRAIN continuations. GSM8K rows contain calculator markup (`<<...>>`) and `####` answers. That is a teacher-distribution mismatch, not only a missing KL term.
4. **Each hyperencoder is 226,529,280 parameters.** Joint Step-100 trained LoRA 50,331,648 + input encoder 226,529,280 + output encoder 226,529,280 = 503,390,208 trainable parameters. The “small sidecar” is currently two 226.5M encoders plus the EPFL LoRA.
5. **Checkpoint `base_hashes` is an empty dict.** Frozen-base hashing was specified in the trainer and not recorded on Step-100.
6. **`Zip2ZipModel.forward` mutates `base_model.config.vocab_size` during labeled training** (`+= max_codebook_size`, then `-=`). Hidden global config mutation.
7. **Product docs still list predictor quality as the main Phi lever.** Stage-1 shows the opposite: B (H disabled) already loses 22.2 points.

---

## 2. What we are actually building vs what the code is

Product (from `docs/product.md` and the current brief):

```
frozen customer model
  + optional unloadable adapter
  + request-specific H representations from existing embeddings/LM head
  + prompt-conditioned predictor
  → fewer decode steps
  → Vanilla behavior preserved
```

Current implementation is still a Zip2Zip research fork:

```
epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1
  = microsoft/Phi-3.5-mini-instruct
  + EPFL LoRA (r=32, qkv/o/gate_up/down, all 32 layers)
  + EPFL input/output hyperencoders
  + dynamic Rust LZW CodebookManager (always constructed)
  + Zip2ZipTokenizer LZW path
then overwrite LoRA+encoders with joint Step-100
then optionally swap CodebookManager → StaticCodebookManager
```

The vLLM proof (`codex/vllm-predictive-poc`, known-good `23b2506`) is evidence that serving can own scheduler/KV/attention while a request-specific H vocabulary is mapped in. It is not evidence that the current HF training recipe preserves Vanilla.

---

## 3. Dependency graph (actual)

### 3.1 Direct `zip2zip_compression` imports in `src/`

Only three modules import the Rust package:

- `src/zip2zip/codebook.py` — `CompressionConfig`, `Codebook`, `CodebookManager as RustCodebookManager`
- `src/zip2zip/tokenizer.py` — `Codebook`, `LZWCompressor`
- `src/evaluation/lzw_simulator.py` — `LZWCompressor`

`src/zip2zip/static_codebook.py` does **not** import it. `src/zip2zip/predictor_v2/` does **not** import it. `src/tokens_vllm/` does **not** import it.

### 3.2 Why predictive still requires it

```
import zip2zip
  → src/zip2zip/__init__.py eager imports
      codebook.CodebookManager
      tokenizer.Zip2ZipTokenizer
      model.Zip2ZipModel
      ...

from zip2zip.predictor_v2.X import Y
  → still executes zip2zip/__init__.py first
  → codebook.py → zip2zip_compression

from zip2zip.nn.embedding import HyperEmbedding
  → embedding.py imports codebook.CodebookManager (type + runtime)
  → zip2zip_compression

Zip2ZipModel.__init__
  → CodebookManager.from_config(config)
      → AutoTokenizer.from_pretrained(base_model_name_or_path)
      → RustCodebookManager(...)
  → wraps embedding and lm_head
  → later StaticCodebookManager.attach_to_model() swaps the manager pointer
```

So the predictive/static path needs the Rust package because of **eager package init, type imports, and default model construction**, not because static H math uses LZW.

### 3.3 Training graph

```
experiments/train_predictive_zip2zip.py
  sys.path inserts repo root and src/
  Zip2ZipModel.from_pretrained("epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1")
      AutoModelForCausalLM(config.base_model_name_or_path)  # Vanilla Phi
      PeftModel.from_pretrained(epfl adapter)               # implicit
      load_pretrained_hyper_encoders()                      # EPFL encoders
  configure / inline: freeze base, train LoRA + both encoders
  data/train.jsonl  (GSM8K / MBPP / Alpaca gold)
  PredictivePipeline.process_sample
      tokenizer.encode(text, add_special_tokens=False)
      append tokenizer.eos_token_id   # Phi: 32000 <|endoftext|>
      prompt-only codebook, DP segment, mask prompt labels to -100
  DifferentiableTrainingManager.forward_step
      total_loss = lm_ce + 0.1 * reconstruction
      no KL, no continuation consistency, no chat-template contract
```

`trainable_mode: "encoders_only"` already exists in the trainer and YAML comments. The saved Step-100 checkpoint has `trainable_mode: "joint"`.

### 3.4 Stage-1 H-disabled (condition B) graph

```
experiments/run_phi_attribution_benchmark.py::load_predictive_bundle
  AutoModelForCausalLM(microsoft/Phi-3.5-mini-instruct @ 2fe19245)
  Zip2ZipModel.from_pretrained(epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1 @ 11c46173,
                               base_model=that_phi)
      → PEFT adapter from EPFL repo is loaded on top of Vanilla Phi
      → EPFL hyperencoders loaded
  load_joint_checkpoint(checkpoint_step_100.pt, expected_step=100)
      overwrites LoRA + input_encoder + output_encoder
      asserts frozen non-LoRA base tensors unchanged during the copy
  H disabled (K=0 / no static codebook / H logits masked)
```

Condition B is therefore:

```
Vanilla Phi
+ EPFL LoRA further trained 100 joint steps
+ EPFL-initialized encoders further trained 100 joint steps
+ Zip2Zip wrapper (HyperEmbedding, HyperLinear, CodebookManager)
```

It is **not** “Vanilla Phi + our LoRA.” There is no B0 (wrapper, no adapter, no H) in the committed Stage-1 matrix. Dirty local WIP started adding B0; it is not the published experiment.

### 3.5 Predictor-only intended graph (broken at import)

```
zip2zip.predictor_v2.{candidate_pool, models, train_index, ...}
  should be independent of LZW
  currently cannot be imported as zip2zip.predictor_v2 without package init
```

### 3.6 Kaggle runner graph

A Stage-1 kernel imported:

```
from src.zip2zip.predictor_v2.attribution_harness import select_stratified_dev_prompts
```

That symbol is defined in `experiments/run_phi_attribution_benchmark.py`, not in `attribution_harness.py`. This is a packaging/API provenance failure, independent of model quality.

---

## 4. Exact weights and trainable groups

Pinned IDs (already in the attribution harness):

| Piece | ID | Revision |
|---|---|---|
| Vanilla Phi | `microsoft/Phi-3.5-mini-instruct` | `2fe192450127e6a83f7441aef6e3ca586c338b77` |
| EPFL Zip2Zip | `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1` | `11c461733a79d2a5de6b814585c3361ca2aacbe7` |
| Step-100 | `experiments/checkpoints/predictive_joint_pilot/checkpoint_step_100.pt` | local artifact, 5761.2 MiB including Adam |

Step-100 file contents:

- `trainable_mode`: `"joint"`
- `lora_state_dict`: 256 tensors, 50,331,648 params, modules `{qkv_proj, o_proj, gate_up_proj, down_proj}`
- `input_encoder_state_dict`: 21 tensors, 226,529,280 params
- `output_encoder_state_dict`: 21 tensors, 226,529,280 params
- `loss` config: `lm_ce_weight=1.0`, `reconstruction_weight=0.1`
- `model.name_or_path`: `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1`
- `base_hashes`: `{}`
- optimizer state present (why the file is multi-GB)

What changed during Step-100: LoRA A/B on those four projections in all 32 layers, plus both hyperencoders. Frozen Phi backbone should have been untouched; the empty `base_hashes` means this checkpoint does not itself prove that.

Historical pure-predictive encoder-only files (`pure_pred_k32_step100_encoder.pt`) are a **different, retired recipe** and must not be mixed into A/B0/B1/B2.

---

## 5. Training format and termination contract

### Training (joint Step-100)

- Source: `data/train.jsonl` gold references (example GSM8K `gsm_8188` uses `<<10/100*700=70>>` and `#### 385000`).
- Prompt/response encoded with `add_special_tokens=False`.
- No `apply_chat_template`.
- Response target appends `tokenizer.eos_token_id` (Phi tokenizer EOS = 32000 `<|endoftext|>`).
- Labels: prompt = `-100`, response including that EOS is supervised.
- Codebook phrases selected from **prompt tokens only**, then both prompt and response are DP-segmented into H ids ≥ 32011.

### Canonical evaluation / Vanilla continuations

- `data/canonical_phi_continuations.jsonl` + `.manifest.json`
- Split: TRAIN 630 / DEV 135 / FINAL 135; FINAL is forbidden for tuning.
- Rendered prompt is native chat, e.g.

  `<|user|>\nInstruction: ...\nAnswer:<|end|>\n<|assistant|>\n`

- Generation: greedy, `eos_token_id=[32007, 32001, 32000]`, `pad_token_id=32000`, `max_new_tokens=1024` in the 900-row artifact.
- Observed Vanilla stop token on instruction rows: **32007 `<|end|>`**.

### Consequence

Training teaches: raw document completion + gold benchmark text + stop on 32000.
Evaluation scores: chat-template assistant turns + Phi’s own continuations + stop on 32007.

The Stage-1 GSM8K failure mode (correct answer, then another problem until the 1024 cap) is exactly what raw-completion Phi does when it is not in chat-turn mode. Condition B also has a drifted LoRA, so format mismatch and adapter drift are still confounded until B0/B1/B2 exist.

Do not add a giant 32007 loss until A vs B0 vs B1 vs B2 and the shared formatter are measured.

---

## 6. Why `zip2zip-compression` is imported, and whether predictive needs it

**Why today:** eager `zip2zip/__init__.py`, `CodebookManager` type imports in hyper modules, and `Zip2ZipModel.__init__` always constructing the Rust manager.

**Does static predictive inference need LZW?** No technical reason in the static codebook implementation. Phrase → slot mapping, encoder synthesis, logit concat, and span-based RoPE are all Python/PyTorch.

**What still needs LZW:** official reactive Zip2Zip generate/decompress, `Zip2ZipTokenizer` compression, `lzw_simulator`, historical reactive benches.

Legacy should become `zip2zip[legacy-lzw]` / `tokens[legacy-lzw]`. Predictive imports must succeed without that extra.

---

## 7. What can be isolated safely

Safe to isolate (keep on disk, optional import):

- `codebook.py` dynamic manager
- `tokenizer.py` LZW tokenizer
- `evaluation/lzw_simulator.py`
- reactive examples/benches/research SFT scripts
- official Zip2Zip eval harness

Required for predictive Tokens:

- `static_codebook.py`
- hyperencoders (`nn/encoders/*`, `nn/embedding.py`, `nn/linear.py`) after Protocol-typing
- predictor_v2 (minus package-init side effects)
- canonical dataset / Phi chat contract
- vLLM proof (`src/tokens_vllm`)
- joint checkpoint loader
- attribution schemas

Do not delete historical experiments, vLLM proof, sourcebook negative results, or canonical manifests.

---

## 8. Proposed target architecture (minimum, not a rewrite)

Keep the `zip2zip` package name for now so historical imports survive. Add a thin `tokens` façade for the product path.

```
tokens.model.TokensModel
    vanilla backbone (explicit from_pretrained)
    adapter: none | epfl | local (explicit, never implicit PEFT)
    h: off | static codebook
    encoders: attached only when h is enabled or training H

tokens.phi_chat          shared chat template + stop set
tokens.static_codebook   re-export, no LZW
tokens.predictive        predictor_v2 without package-init LZW
tokens.fidelity          teacher KL, continuation, reconstruction, H CE
```

Invariants:

1. Vanilla construction never loads EPFL or Step-100.
2. Loading EPFL LoRA is an explicit call.
3. Loading Step-100 is an explicit call.
4. H off means original embedding and LM head compute graph (no extra logit columns).
5. Predictor import does not import `zip2zip_compression`.
6. Reactive LZW is opt-in.

B0 is this identity wrapper. If A ≠ B0 on a deterministic smoke set, stop.

---

## 9. Minimum refactor sequence

1. Lazy `zip2zip/__init__.py`; Protocol instead of `CodebookManager` type imports; optional extra for LZW.
2. Explicit composition factory; stop implicit `PeftModel.from_pretrained` on the predictive path.
3. Shared Phi chat/termination utility; training and eval must call it.
4. B0 identity generate path + CPU token-equality tests (tiny stub model) and a documented GPU numerical tolerance.
5. Attribution conditions A / B0 / B1 (EPFL LoRA, H off) / B2 (Step-100, H off) / C / D.
6. Fidelity objective with frozen Vanilla teacher on **canonical TRAIN**, encoders-only first.
7. Kaggle pack → extract → dry-run before any GPU job.

---

## 10. Tests that protect useful existing work

Must stay green (or be explicitly marked historical):

- `tests/test_vllm_predictive_contract.py` and warmup/source-mount Kaggle tests
- `tests/test_static_codebook.py`
- `tests/test_joint_checkpoint_loader.py`
- `tests/test_predictor_v2_*` leakage/oracle/dataset tests
- `tests/test_quality_evaluator_contract.py`
- `tests/test_dataset_isolation.py` / FINAL prohibition
- `tests/test_official_zip2zip_regression.py` (legacy extra may be required)

Do **not** force joint-training smoke tests to remain the definition of correctness if they encode gold-SFT + LoRA + no KL.

New tests (this branch):

- package import boundaries without `zip2zip_compression`
- composition flags
- Phi chat/termination contract
- B0 identity wrapper
- KL masking / prompt masking
- Kaggle provenance dry-run

---

## 11. Experimental gates

| Gate | Question | Stop if fail |
|---|---|---|
| 0 Package | Predictive stack imports without LZW | Fix isolation |
| 1 Wrapper B0 | A token-equal B0 on smoke set | Fix wrapper; no training |
| 2 Adapter | B1/B2 with H off stay near Vanilla quality and EOS | Reject adapter; do not tune predictor |
| 3 Oracle H | C near chosen base, H actually used | Fix representation |
| 4 Utilization | Oracle H saves decode steps without quality collapse | Fix emission/positions |
| 5 Predictor | D approaches C | Then retrieval/K work is in scope |
| 6 Performance | Real time down under production-like serving | Runtime/vLLM, not more SFT |

Stage-1 already failed Gate 2 on the confounded B. Next measured experiment is Gate 1, then a disentangled Gate 2.

---

## 12. Current composition cheat-sheet

| Condition | Backbone | Adapter | Encoders | H | Codebook manager |
|---|---|---|---|---|---|
| A | Vanilla Phi | none | none | off | none |
| B0 (needed) | Vanilla Phi | none | present but inactive **or** unwrapped | off | identity / unused |
| B1 (needed) | Vanilla Phi | EPFL LoRA only | EPFL encoders, H masked | off | static empty / masked |
| B2 (today’s B) | Vanilla Phi | EPFL LoRA + Step-100 | Step-100 encoders, H masked | off | static empty / masked |
| C | chosen base from B* | same | same | oracle phrases | StaticCodebookManager |
| D | same | same | same | predictor phrases | StaticCodebookManager |

No GPU job until Gate 1 is proven on CPU stubs and a documented model-level plan. If B0 cannot be proven locally without weights, ship the identity-path unit tests and a Kaggle **smoke** only after packaging dry-run.

---

## 13. Decision already taken

We will not keep implicit PEFT load, eager LZW, or gold-reference CE as the predictive training default. Those are Zip2Zip research defaults. They fight the product invariant: **do not make the model faster by making it a different model.**
