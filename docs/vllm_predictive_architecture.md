# Predictive hypertokens on vLLM 0.30.0

This note is the contract for a narrow compatibility proof. It does not
change the predictive algorithm. It records how the existing Hugging Face
fast-inference semantics are reproduced inside vLLM, and what the vLLM
0.30.0 source actually does.

The proof stops after the checks in phases 0–9. It does not enable CUDA
graphs, automatic prefix caching, EAGLE, tensor or pipeline parallelism,
quantization, quality retraining, or a production server.

## Pinned versions

| Piece | Pin |
| --- | --- |
| Tokens implementation branch | `codex/vllm-predictive-poc` |
| Tokens base commit | `5ed767a6ea27743c7697fa707dbb9f6085c9b621` |
| vLLM tag | `v0.30.0` |
| vLLM commit | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Base model | `microsoft/Phi-3.5-mini-instruct` @ `2fe192450127e6a83f7441aef6e3ca586c338b77` |
| Zip2Zip reference | `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1` @ `11c461733a79d2a5de6b814585c3361ca2aacbe7` |
| Research checkpoint | Step-100 predictive checkpoint |

LoRA is merged before serving. Runtime LoRA is not used. The input and
output hyperencoders stay as separate modules and run once per request,
outside the decode loop.

Audit checkout used for the file and line references below:
`C:\Users\elijk\AppData\Local\Temp\vllm-0.30.0` at `ced6857`.

## What is being proved

One hypertoken is one transformer forward, one KV-cache entry, and several
semantic base-token positions. The speedup already measured on the 12-prompt
T4 run is not re-measured here. This proof only asks whether that compressed
physical sequence can share vLLM's scheduler, continuous batching, and paged
KV cache without mixing request-specific hypertoken state.

The scheduler, paged KV cache, attention kernels, continuous batching, and
sampler stay stock. The plugin changes three things:

- request-specific input embeddings for the 32 hypertoken slots
- semantic position ids passed to Phi RoPE
- request-specific hypertoken logits inserted into the trained layout

## Logical and physical vocabulary

The model was trained with an insertion at id 32011, not with hypertokens
appended after the physical Phi table.

| Range | Meaning |
| --- | --- |
| logical `0 .. 32010` | physical Phi ids `0 .. 32010` |
| logical `32011 .. 32042` | hypertokens `H0 .. H31` |
| logical `32043 .. 32095` | physical Phi ids `32011 .. 32063` |

```
if logical_id < 32011:
    physical_base_id = logical_id
elif logical_id < 32043:
    h_slot = logical_id - 32011
else:
    physical_base_id = logical_id - 32
```

Constants:

| Name | Value |
| --- | --- |
| `vocab_size` (what vLLM samples) | 32096 |
| `predictive_base_vocab_size` | 32064 |
| `predictive_initial_vocab_size` | 32011 |
| `predictive_codebook_size` | 32 |
| `predictive_max_subtokens` | 3 |

`ModelConfig.get_vocab_size()` must be 32096. The physical embedding and
LM head stay 32064 rows. A 32096-row checkpoint is not built.

This matches the prepared Hugging Face tables in
`StaticCodebookManager.prepare_inference_tables`: the effective input and
output tables are `cat(base[:32011], H, base[32011:])`. Output scoring
matches `HyperLinear.forward` when fast tables are not used:

```
cat(base_logits[..., :32011], h_logits, base_logits[..., 32011:])
```

The slow `HyperEmbedding.forward` does not remap the tail. The vLLM path
follows the prepared fast-inference mapping, which does.

Expansion outside vLLM, with `detokenize=False`:

- logical `< 32011` stays that base id
- `32011 <= logical < 32043` expands to that request's phrase
- logical `>= 32043` becomes `logical - 32`

The stock Phi tokenizer never sees hypertoken ids.

## Phase 0 source audit

Checked against vLLM `ced6857`. The ten planned hooks exist. No redesign
is required. Two constraints are mandatory and are called out below.

### 1. `get_model_state_cls()`

`vllm/v1/worker/gpu/model_states/__init__.py`, `resolve_model_state_cls`.

If the loaded module has `get_model_state_cls`, that class is used.
Otherwise vLLM picks encoder-decoder, encoder-only, Mamba, or
`DefaultModelState`.

`PredictivePhi3ForCausalLM.get_model_state_cls()` returns
`PredictiveModelState`. `init_model_state` constructs it and the state
sets `model.predictive_state = self` in its constructor.

This hook exists only on Model Runner V2
(`vllm/v1/worker/gpu/model_runner.py`). V2 is the default when Triton is
present and `_get_v2_model_runner_unsupported_features` is empty
(`vllm/config/vllm.py`, `use_v2_model_runner`). Custom logits processors
force V1. This proof must not register a logits processor, and it must
refuse to start unless `VllmConfig.use_v2_model_runner` is true. Set
`VLLM_USE_V2_MODEL_RUNNER=1` in the proof runner so a machine without
Triton fails loudly instead of silently using V1.

### 2. `prepare_inputs()` merge order

`GPUModelRunner.execute_model` builds:

```
model_inputs = {
    "input_ids": input_ids,
    "positions": input_batch.positions,
    "inputs_embeds": inputs_embeds,
    "intermediate_tensors": None,
    **self.model_state.prepare_inputs(input_batch, self.req_states),
}
```

Returned keys override the defaults. `DefaultModelState.prepare_inputs`
returns `{}` for ordinary 1D RoPE, which Phi-3.5 uses. Returning
`{"positions": semantic_positions}` is therefore the positions tensor
Phi receives.

KV slot mapping is computed earlier, in `GPUModelRunner.prepare_attn`,
from `input_batch.positions`. Those positions stay the physical
compressed positions. Overriding the model-input positions does not move
KV blocks, scheduler token counts, or physical sequence lengths.

Phi applies RoPE inside `LlamaAttention.forward`
(`vllm/model_executor/models/llama.py`) with
`self.rotary_emb(positions, q, k)`. `Phi3ForCausalLM` is a
`LlamaForCausalLM` subclass (`vllm/model_executor/models/phi3.py`). The
semantic tensor returned from `prepare_inputs` is the tensor that reaches
that call on the eager path (`model(**model_inputs)`).

### 3. `compute_logits` call site

`GPUModelRunner.sample` gathers `hidden_states[input_batch.logits_indices]`
and calls `self.model.compute_logits(sample_hidden_states)`. The dummy
sampler calls the same method. The override belongs on
`PredictivePhi3ForCausalLM.compute_logits`, not on a logits processor.

`LogitsProcessor` truncates gathered logits to `org_vocab_size`. The
physical head is 32064 rows, so the override computes those base logits
and then concatenates the 32 hypertoken logits itself. It must not return
the truncated physical tensor as the sampled tensor.

### 4. `idx_mapping`

`InputBatch.idx_mapping` maps a batch row to the stable
`req_state_idx`. It is not the transient batch position.
`query_start_loc` is the exclusive prefix sum of scheduled tokens per
batch row (`GPUModelRunner.prepare_inputs`).

Model-input token ownership is:

```
token_req_indices[token_row] = idx_mapping[batch_row]
```

for `token_row` in `[query_start_loc[batch_row], query_start_loc[batch_row + 1])`.

`prepare_inputs` copies that into the persistent GPU buffer before
`forward`. `embed_input_ids` does not receive `InputBatch`.

### 5. `expanded_idx_mapping`

Without speculative decoding, `expanded_idx_mapping` is `idx_mapping`
and has one entry per request. Logit rows are the last scheduled token
of each request (`logits_indices`), so there is one logit row per
request, not one per input token.

With speculative decoding, `expand_idx_mapping` widens it to
`total_num_logits`. This proof does not use speculative decoding, but
the code still reads `expanded_idx_mapping` for logit rows and
`idx_mapping` plus `query_start_loc` for input rows. The two buffers
are not assumed to have the same shape.

`prepare_inputs` copies `expanded_idx_mapping` into `logit_req_indices`
before the forward. `compute_logits` runs later, in `sample`, and reads
that staged buffer.

### 6. Add and remove

`GPUModelRunner.add_requests` walks `scheduler_output.scheduled_new_reqs`
and calls `model_state.add_request(req_index, new_req_data)` with
`req_index = req_states.req_id_to_index[req_id]`.

`_remove_request` calls `model_state.remove_request(req_id)` before
`req_states.remove_request`, so the slot can still be resolved.
`finish_requests` removes finished ids and preempted ids.

### 7. Preemption and resume

`Scheduler._preempt_request` frees KV blocks, sets
`num_computed_tokens = 0`, marks the request `PREEMPTED`, and puts it
back on the waiting queue. The id is added to `reset_preempted_req_ids`,
which the worker treats as a removal.

On Model Runner V2, a later schedule merges resumed requests into
`scheduled_new_reqs` and builds `NewRequestData` with
`prefill_token_ids=request._all_token_ids` (the full logical history).
`add_request` therefore runs again. Hypertoken vectors are rebuilt from
the same codebook. The semantic offset is reconstructed from that
history. Custom state is not kept across preemption.

### 8. `get_vocab_size()`

`ModelConfig.get_vocab_size` returns `model_arch_config.vocab_size`.
`ModelArchConfigConvertorBase.get_vocab_size` reads
`hf_text_config.vocab_size`. The runner stores that value as
`self.vocab_size` and the sampler uses it.

Stock `LlamaModel` also sizes `VocabParallelEmbedding` and `ParallelLMHead`
from `config.vocab_size`. Advertising 32096 through the Hugging Face
config would make those tables 32096 rows and the 32064-row checkpoint
would fail the loader check
(`loaded_weight.shape[output_dim] == org_vocab_size`).

The custom model therefore:

- leaves `hf_config.vocab_size` at 32096 so sampling and token-range
  checks see the logical vocabulary
- constructs the embedding and LM head with
  `predictive_base_vocab_size` (32064)

`hf_overrides` carries `vocab_size` and the `predictive_*` fields.
`architectures` is `PredictivePhi3ForCausalLM`, registered from the
`vllm.general_plugins` entry point via `ModelRegistry.register_model`.

### 9. `embed_input_ids` for Phi

Text Phi does not use `ModelState.prepare_inputs_embeds`. That path runs
only when `uses_inputs_embeds` is set (multimodal, or
`enable_prompt_embeds`). `LlamaModel.forward` calls `embed_input_ids`
when `inputs_embeds` is `None`. The override is
`PredictivePhi3Model.embed_input_ids`.

Logical ids are remapped before the physical embedding. Hypertoken
positions are first read as physical id 0, then overwritten with
`h_input[token_req_indices, h_slot]`. No logical id above 32063 is
passed to the physical embedding. No physical id `>= 32064` is passed
to it.

### 10. Dummy and CUDA-graph inputs

`execute_model` calls `model_state.prepare_inputs` for real and dummy
batches. Dummy batches from `InputBatch.make_dummy` use request indices
`0 .. num_reqs-1`, which can overlap live slots.

`prepare_inputs` may stage into the persistent mapping buffers with
`copy_`. It must not commit semantic offsets on a dummy or capture
batch, and it must not require a user codebook. Commit happens only in
`postprocess_state` after a successful real forward.

`postprocess_state` is reached from `postprocess_sampled` after
`sample` on the last pipeline rank, and from the non-last pipeline rank
when a chunk is not a final decode. This proof is pipeline-parallel
size 1, so the last-rank path is the one that commits.

The first proof runs with `enforce_eager=True`. Buffers still have
stable addresses so a later CUDA-graph pass does not have to redesign
them. The hot path does not call `tensor.item()`, `tensor.any()` from
Python, `bool(tensor)`, or other GPU-to-CPU syncs, and it does not
branch in Python on tensor contents.

## Request state

`PredictiveModelState` subclasses `DefaultModelState`. Fixed GPU buffers,
sized for `max_num_seqs` and the runner's token budget:

| Buffer | Shape |
| --- | --- |
| `h_input` | `[max_num_reqs, 32, hidden_size]` |
| `h_output` | `[max_num_reqs, 32, hidden_size]` |
| `h_spans` | `[max_num_reqs, 32]` |
| `semantic_offset` | `[max_num_reqs]` |
| `physical_accounted` | `[max_num_reqs]` |
| `token_req_indices` | `[max_num_batched_tokens]` |
| `logit_req_indices` | `[max_logits_rows]` |
| `pending_semantic_advance` | `[max_num_reqs]` |
| `pending_physical_advance` | `[max_num_reqs]` |

Per request this is `2 * 32 * 3072 * 2` bytes, about 0.375 MiB. Full
request-specific 32096-row tables are not allocated.

Updates use in-place `copy_` into those buffers.

### Codebook intake

The predictor stays outside vLLM. Offline proof requests pass the
codebook on `SamplingParams.extra_args`:

```
extra_args = {
    "predictive_codebook": {
        "version": 1,
        "phrases": [[...], ...],  # exactly 32, H-slot order
        "sha256": "...",
        "k": 32,
    }
}
```

`SamplingParams._verify_extra_args` allows nested dicts and lists. Token
ids fit in the MessagePack integer range. HTTP `vllm_xargs` is not part
of this proof; a later API can put one JSON string in that field.

`add_request` validates:

- exactly 32 phrases
- each phrase length is 2 or 3
- every source id is in `[0, 32011)`
- no disabled ids
- phrases are unique

It builds the padded codebook, runs the existing input hyperencoder
once and the existing output hyperencoder once, and writes
`h_input[req_index]`, `h_output[req_index]`, and `h_spans[req_index]`.
It stores the codebook hash. The legacy `semantic_offset`,
`physical_accounted`, `pending_semantic_advance`, and
`pending_physical_advance` buffers remain for compatibility and cleanup
audits, but they are not authoritative for RoPE progression. In
particular, a zero `semantic_offset` on preemption readmission is valid.

Tensor-parallel size is 1, so the encoders read the full physical
embedding and LM head. Setup time is recorded for time-to-first-token
accounting. The encoders do not run again during decode.

`remove_request` clears that slot's H vectors, spans, activity flag, and
legacy counters. The next `add_request` that receives the same `req_index`
rebuilds the codebook H state before the new request runs.

### Positions

`prepare_inputs` reads each request's `input_batch.num_computed_tokens_np`
row from the current vLLM scheduler batch. That count is the position
origin. In compressed mode, model positions are the contiguous interval
from that origin for the scheduled tokens. In `base_token_end` mode, the
state replays the logical history up to the scheduler count and rebuilds
the semantic origin from the current H spans, then advances by each
token's span. A base or shifted-tail id has span 1; a hypertoken has
`h_spans[req_idx, h_slot]`.

vLLM owns physical KV progress and supplies physical batch positions.
The predictive position contract supplies model/RoPE positions. They are
equal in compressed mode and can differ in `base_token_end` mode. Semantic
positions are written to the model-state position buffer; they are not
written into vLLM's physical KV addressing state.

The legacy shadow counters listed above are not read by `prepare_inputs`
to determine model positions and are not committed by `postprocess_state`.
`postprocess_state` is intentionally a no-op for position progression.
After preemption, vLLM 0.30 can reset `num_computed_tokens` to zero,
discard the KV blocks, and re-admit the request with its full logical
history. With prefix caching disabled, the worker rebuilds H state and
re-prefills that history at positions `0, 1, 2, ...`; no pre-preemption
shadow offset is preserved or required.

### RoPE and context length

Phi-3.5 Mini `max_position_embeddings` is 131072. `max_model_len` stays
the physical KV budget. It is not multiplied by 3.

`LlamaAttention` builds RoPE with `max_position=config.max_position_embeddings`,
not `max_model_len`. `Phi3LongRoPEScaledRotaryEmbedding` indexes a
cos/sin cache of that length. With `max_model_len` at or below the
original 4096 position limit, `use_long_rope` is false and the index is
the semantic position itself. A semantic position above a tiny
`max_model_len` and below 4096 therefore hits the short cache and does
not require a larger KV allocation.

Invariant: `semantic_position < 131072`. If a vLLM path rejects
`positions >= max_model_len` during the phase 9 eager forward, that
path is reported and patched only as far as the position/RoPE read.
KV allocation is not increased to make semantic positions legal.

### Logits

`compute_logits`:

1. Physical Phi logits, shape `[rows, 32064]`.
2. `req_idx = logit_req_indices[row]`.
3. `h_logits = hidden[row] @ h_output[req_idx].T`, shape `[rows, 32]`.
4. `cat(base[:, :32011], h_logits, base[:, 32011:])`, shape `[rows, 32096]`.

Sampling for the proof is greedy: temperature 0, no speculative
decoding, grammar, structured output, or logprobs. Sampled ids are in
`0 .. 32095`.

With H disabled, sampled ids are compared with stock Phi only after
`logical_output_to_base_ids`. Base ids stay themselves and shifted-tail
ids move back by K. An H id in that comparison is a failure.

## Proof limitations

These two paths are correct for this compatibility proof and are not
the production implementation.

`prepare_inputs` builds token ownership on CPU with NumPy, then copies
that index tensor to the GPU. A later version should generate and stage
ownership entirely on the GPU.

`add_request` synchronizes and uses `.item()` while it checks that a
reused slot was empty and while it times hyperencoder synthesis. That
work runs once per admission. It is time-to-first-token setup
instrumentation, not part of the serial decode loop.

## Product split (design only)

Not built in this proof.

Tokens frontend owns tokenization, the predictor, codebook construction,
prompt compression, hypertoken expansion, Phi detokenization, stop
strings, `max_tokens` semantics, usage accounting, and fallback to
vanilla vLLM.

vLLM owns scheduling, batching, KV, attention, the transformer, and
sampling.

Predictive mode falls back to ordinary vLLM for structured or grammar
decoding, prompt or completion logprobs, runtime LoRA, logit bias,
frequency or presence penalties, bad-word constraints, and speculative
decoding including EAGLE.

## Explicitly out of scope

- CUDA graphs. The code is written so buffers can be captured later.
  `enforce_eager=True` until phases 0–9 pass.
- Automatic prefix caching. Disabled. A later `cache_salt` would be
  SHA256 of the Tokens version, base and checkpoint version,
  hyperencoder version, and canonical codebook. The same logical ids
  with different hypertoken meanings must not share KV blocks.
- Tensor parallelism. Future TP must not all-gather the vocabulary.
  Each codebook needs at most 96 source rows. The future path gathers
  those rows, calls `forward_from_embeddings`, synthesizes 32 vectors,
  and stores them per request. That path is not implemented.
- Pipeline parallelism, quantization, EAGLE, quality training, K other
  than 32, and a concurrency benchmark.

## Implementation phases

Phase 0 is this audit. Phases 1–9 are the proof gates. Each gate is
recorded before the next one starts. GPU work is limited to those gates.
CPU tests cover the mapping, span, and isolation logic first.

| Phase | Gate |
| --- | --- |
| 1 | Custom model registered, H behavior off, logical vocab 32096, physical tables 32064, greedy base prompts match stock Phi |
| 2 | Boundary ids 0, 32010, 32011, 32042, 32043, 32095. No H id reaches the physical embedding. Concat and expansion match the HF layout |
| 3 | `gsm_2956` with the saved K=32 codebook. H vectors, spans, compressed ids, token owners, and semantic positions match HF |
| 4 | Teacher-forced logits, then greedy logical, expanded, and decoded trajectories |
| 5 | Two different codebooks, alone and together, in both orders |
| 6 | A finished request's slot is reused by a third codebook with no leftover state |
| 7 | Chunked prefill matches a single prefill |
| 8 | Preemption rebuilds H vectors and the semantic offset and matches the uninterrupted trajectory |
| 9 | Eager Phi forward with semantic position `> max_model_len` and `< 131072`, RoPE receiving that position, KV budget unchanged |

## Package

Implementation lives in `src/tokens_vllm/`. Tests live under `tests/`
and import that package. The vLLM plugin entry point is registered from
this repo's package metadata and only imports vLLM when the plugin loads.
