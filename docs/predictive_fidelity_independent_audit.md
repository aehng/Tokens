# Independent Predictive Fidelity Audit

Date: 2026-09-29
Branch: `grok/predictive-fidelity-audit`
Inherited Luna HEAD: `24616f79b80d82d0665447054bf17cc706364aa6`
GPU compute used for this audit: none

Every statement is labeled **VERIFIED**, **INFERENCE**, or **UNKNOWN**.

---

## 1. Independent architecture diagram

```
A  Vanilla
   AutoModelForCausalLM(microsoft/Phi-3.5-mini-instruct @ 2fe19245)
   native lm_head (32064 rows) and embed_tokens
   tokenizer.apply_chat_template / encode of rendered Phi chat
   greedy generate, EOS {32007, 32001, 32000}

B0 Tokens wrapper / Vanilla weights
   same Phi load
   Zip2ZipModel.from_pretrained(epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1 @ 11c46173)
        PeftModel.from_pretrained(EPFL LoRA)     # present, then disable_adapter()
        load_pretrained_hyper_encoders()         # EPFL encoder init, unused when K=0
        HyperEmbedding / HyperLinear wrap embed and lm_head (shared .weight)
        codebook_backend=static StaticCodebookManager, empty codebook
        enable_base_token_positions()            # span RoPE; spans are 1 with H off
   Step-100 NOT loaded
   H logits masked

B1 same wrapper + load_joint_checkpoint(checkpoint_step_100.pt)
   overwrites LoRA A/B + input_encoder + output_encoder
   asserts non-LoRA base parameters unchanged during the copy
   adapter left enabled, H still masked

CF teacher-forced H
   codebook from DP phrases on the B1 base continuation
   force those phrases through H IDs, expand exactly
   compare next-token native logits after H vs after the equivalent base phrase

C  ORACLE CODEBOOK LIVE: hindsight phrases, model free to emit H or base tokens
D  real predictor: TRAIN association index + PooledMLP, K=32
```

Legacy LZW path (isolated):

```
Zip2ZipTokenizer + zip2zip_compression.LZWCompressor
CodebookManager -> RustCodebookManager
dynamic codebook updates during generate
```

**VERIFIED** from `src/zip2zip/model.py`, `static_codebook.py`, `nn/embedding.py`, `nn/linear.py`, `experiments/load_joint_checkpoint.py`, `experiments/run_phi_attribution_benchmark.py`.

---

## 2. Previous architecture audit

The committed audit at `docs/architecture_audit_predictive_fidelity.md` (HEAD inspected then: `5e0f328`) was **correct on the hard structural claims**:

- package `__init__` used to import `codebook` / `tokenizer` and therefore `zip2zip-compression`
- `Zip2ZipModel.__init__` used to always construct the Rust `CodebookManager`
- joint training is LM CE + 0.1 reconstruction, no Vanilla KL
- `compute_continuation_consistency_loss` is documented and not implemented
- training uses raw `tokenizer.encode` and tokenizer EOS 32000
- Stage-1 B is wrapper + EPFL PEFT + Step-100, not a clean B0
- `select_stratified_dev_prompts` lived only on the runner

It missed or understated:

1. **HyperLinear inserts H rows in the middle of the physical vocab**, at `initial_vocab_size=32011`, then concatenates the unused physical tail `32011..32063`. Luna's `normalize_wrapper_logits()` implements that layout correctly. **VERIFIED**
2. **Greedy A/B0 token IDs can still match** even with K extra rows, because ordinary Phi text lives in `0..32010` and unused H slots are masked to `-inf`. Tail IDs `32011..32063` are shifted by K in the expanded logit vector. **VERIFIED** layout; **INFERENCE** that smoke prompts will not sample the tail.
3. **Luna's CF schedule index used `input_ids.shape[1] - prompt_length`**. Cached HF generate passes a one-token `input_ids` tensor after the first step, so that formula is wrong. **VERIFIED**
4. **Luna's CF continuation check compared unforced top-1 to the next compressed schedule ID**. That is not hidden-state equivalence versus feeding the same base phrase. **VERIFIED**
5. **EPFL `position_mode` defaults to `compressed`**. Span-based RoPE is not active unless the wrapper installs the `base_token_end` hook. With H disabled, both conventions yield positions `0,1,2,...`. With H on, compressed mode cannot implement semantic spans. **VERIFIED**
6. **Importing the attribution runner pulled `Zip2ZipTokenizer` through `run_quality_benchmark`**, so a Kaggle bundle could fail even after package-init was cleaned. **VERIFIED**
7. **`base_phi_weight_sha256` can match A vs B0** after PEFT unwrap + `.base_layer.` canonicalization, and Luna's tiny fixture tests that. Whether it matches on real Phi is still **UNKNOWN** until A/B0 runs.

---

## 3. Why `zip2zip-compression` was imported, and whether predictive Tokens needs it

**VERIFIED.** Direct imports existed only in:

- `src/zip2zip/codebook.py`
- `src/zip2zip/tokenizer.py`
- `src/evaluation/lzw_simulator.py`

Predictive inference imported it because:

1. `pyproject.toml` listed it as a hard dependency
2. `zip2zip/__init__.py` eagerly imported `CodebookManager` and `Zip2ZipTokenizer`
3. `Zip2ZipModel.__init__` called `CodebookManager.from_config`, which constructs `RustCodebookManager`
4. `HyperEmbedding` / `HyperLinear` type-imported `CodebookManager`
5. `experiments/run_quality_benchmark.py` imported `Zip2ZipTokenizer` at module load, and the attribution runner imported that module for `TimingLogitsProcessor`

**Predictive/static Tokens does not need LZW.** Static codebook math, H expansion, predictor_v2, and the attribution gates never call the Rust compressor. **VERIFIED**

---

## 4. Exact model composition

### A — Vanilla

- `microsoft/Phi-3.5-mini-instruct` revision `2fe192450127e6a83f7441aef6e3ca586c338b77`
- native `AutoModelForCausalLM`, `trust_remote_code=False`
- native tokenizer, same revision
- no PEFT, no HyperEmbedding/HyperLinear, no codebook
- **VERIFIED** in `load_vanilla_model_and_tokenizer`

### B0 — wrapper, Vanilla weights

- same Phi revision as A
- EPFL Zip2Zip repo revision `11c461733a79d2a5de6b814585c3361ca2aacbe7` supplies PEFT adapter + encoder tensors
- Step-100 is not applied
- `disable_adapter()` context during generate
- empty static codebook, H slots masked
- after this audit: `codebook_backend="static"` so construction does not instantiate Rust LZW; `enable_base_token_positions()` so H-off RoPE is still `0,1,2,...`
- **VERIFIED** intended composition. Whether `disable_adapter()` is numerically Vanilla-equivalent remains the A/B0 live gate (**UNKNOWN** until GPU)

### B1 — wrapper + Step-100, H disabled

- same wrapper as B0
- `load_joint_checkpoint(..., expected_step=100)` copies
  - `lora_state_dict` (trainer filter: names containing `"lora"`)
  - `input_encoder_state_dict`
  - `output_encoder_state_dict`
- loader fingerprints non-LoRA `base_model` parameters and asserts they are unchanged during the copy
- checkpoint `base_hashes` is historically `{}` (**VERIFIED** by the previous audit of the file; this session did not reopen the 5.7 GiB blob — **UNKNOWN** here)
- **INFERENCE** from the loader and trainer: changed groups are LoRA on `qkv_proj` / `o_proj` / `gate_up_proj` / `down_proj` plus both hyperencoders. Exact names/shapes are recorded live by `components.*.changed_parameter_names`

### Adapter disable

`disabled_adapter_context` calls PEFT `disable_adapter()` and fails closed if `peft_config` exists without that API. **VERIFIED** in source. It does **not** by itself prove Vanilla logits; A/B0 must.

---

## 5. Review of Luna's gates

### A/B0 token gate — keep, with caveats

`token_equivalence_gate` requires exact generated IDs, termination, hashes, tokenizer, EOS, dtype, empty H, adapter-disabled flags. **VERIFIED** that it measures what it claims **if** the runtime dict is honest. It cannot see logit-row shift if greedy never samples a shifted tail ID.

### Logit normalization — correct for the real layout

```
cat(logits[..., :32011], logits[..., 32011+K:])
```

matches `HyperLinear.forward`. **VERIFIED**. Luna's TinyExpandedLogitModel appended H at the end; that fixture only matches Phi when `initial_vocab == physical_vocab`. The dedicated middle-insert unit test is the one that matters.

Tolerance: fp32 `1e-5`, fp16 `2e-3`, bf16 `2e-2`. **INFERENCE**: defensible for a shared-weight identity path; A/B0 should be near exact if the wrapper is neutral.

Same-prefix capture uses Vanilla generated IDs, indices `{0,1,4,16}` truncated before EOS, `use_cache=False`. **VERIFIED**. That is the right comparison, independent of generate caching.

### Base-model hash — plausible, not yet live-proven

Hashes unwrapped non-LoRA parameters, including `lm_head` / embeddings via shared weights, skips names containing `lora`, `input_encoder`, `output_encoder`, `.modules_to_save.`. Canonicalizes `.base_layer.`. **VERIFIED** in code. Tiny fixture A vs wrapped PEFT hashes match. Real Phi + HyperEmbedding name alignment is **UNKNOWN** until B0 load.

Skipping `.modules_to_save.` could hide an EPFL saved head if one exists. **UNKNOWN** without the adapter config.

### B0/B1 isolation — useful

Requires B0 did not load Step-100, B1 loader reports LoRA + both encoders with names and shapes, frozen-base hash unchanged during load, B0/B1 base hashes equal. **VERIFIED** as a loader/accounting gate. It does not hash buffers outside `named_parameters()`.

### CF — methodology was insufficient; now tightened

Keep:

- model-free DP tiling + exact expansion (**VERIFIED**, useful)
- semantic span positions from `StaticCodebookManager.prepare_input_ids` (**VERIFIED** math on full sequences)

Fixed:

- schedule index is now the logits-processor call count, not `input_ids.shape[1]`
- added teacher-forced native-logit comparison of `prompt+base_phrase` vs `prompt+H` at matched content

Still not a full KV-cache dump. It is a next-token-state comparison at equivalent semantic prefixes, which is the product question. **VERIFIED** intent of the new helper `h_vs_base_prefix_pair`.

### Offline opportunity analysis — keep

Disabled IDs `{0,1,2} ∪ {32000..32010}`, vocab `[0,32011)`, phrase length 2–4, K=32, DP tiling, overlapping occurrence counts, opportunity vs realized compression. **VERIFIED** against `segment_tokens_dp` / `OracleV2`. Historical C/D percentages remain useful **as offline reanalysis of Stage-1 records**, not as a new live result.

---

## 6. Training (review only, no retrain)

**VERIFIED** from `experiments/train_predictive_zip2zip.py`, `configs/predictive_joint_pilot.yaml`, `src/zip2zip/predictive_pipeline.py`, `training_objectives.py`:

- trainable: LoRA + both encoders (`trainable_mode: "joint"`); `encoders_only` exists but Step-100 is joint
- data: `data/train.jsonl` gold GSM8K/MBPP/Alpaca, not the 630 canonical Phi TRAIN continuations
- tokenization: raw encode, append `tokenizer.eos_token_id` (32000), no `apply_chat_template`
- loss: `lm_ce + 0.1 * reconstruction`
- no Vanilla teacher KL; continuation-consistency is a docstring only

This is misaligned with “preserve Vanilla, teach H”. Do not retrain until A/B0 and B0/B1 are measured.

---

## 7. Changes made on `grok/predictive-fidelity-audit`

1. Lazy LZW exports; HyperEmbedding/HyperLinear depend on a rust-free protocol
2. `Zip2ZipModel` constructs `StaticCodebookManager` when `codebook_backend="static"`
3. `zip2zip-compression` moved to optional extra `legacy-lzw`
4. Attribution runner uses static codebook + span positions
5. `select_stratified_dev_prompts` lives on the harness with pinned DEV12 IDs
6. CF call-count schedule + H-vs-base prefix logit comparison
7. Timing helpers split out of `run_quality_benchmark`
8. Extracted-archive packer + dry-run

No giant `tokens/` rename. Legacy LZW remains in place for official Zip2Zip reproduction.

---

## 8. Pinned 12-prompt DEV subset

In canonical DEV `split_ids` order, first four of each domain:

| Domain | IDs |
|---|---|
| Code | `mbpp_113`, `mbpp_168`, `mbpp_217`, `mbpp_225` |
| Reasoning | `gsm_2032`, `gsm_2044`, `gsm_2353`, `gsm_2491` |
| Instruction | `alpaca_1`, `alpaca_1024`, `alpaca_1029`, `alpaca_1132` |

Run order is Code, then Reasoning, then Instruction.

---

## 9. Proposed GPU sequence

1. Pack the source archive with `experiments/pack_attribution_source.py --dry-run`
2. Kaggle A + B0 only, `--prompt-limit 12 --k 32 --split DEV`
3. Stop if token or logit gate fails
4. Only then B1, then CF, then C/D

Do not spend the remaining 5-hour family budget until A/B0 is decided.

Conservative T4 bound for 12 greedy 1024-token generations × 2 arms: well under one hour if outputs terminate; worst case ~24k decode steps plus two model loads. **INFERENCE**
