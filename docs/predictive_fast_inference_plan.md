# Predictive Fast Inference Plan

## Scope and safety boundary

This work targets the Step-100 static predictive inference hot path. It is a
CPU-only implementation and validation task. No Kaggle kernel, CUDA benchmark,
remote GPU, or GPU scheduling is authorized. The eventual GPU check described
below requires a separate explicit approval after review of the CPU results.

Work is isolated on `codex/predictive-fast-inference`; it must not be merged to
`main` as part of this task. Step-100 checkpoints, predictor files, datasets,
and existing V16 results are read-only.

## Baseline behavior captured before implementation

Let `V = initial_vocab_size` and `K = max_codebook_size`.

- Base token IDs occupy `[0, V)`.
- A seeded H token has absolute ID `V + j`, where `j` is its codebook slot.
  `set_seeded_codebook()` accepts either relative slots or absolute H IDs and
  stores each H token's base-token sequence.
- `HyperEmbedding.forward()` currently sends IDs below `V` through the base
  embedding and treats IDs at or above `V` as H slots. It obtains cached H
  input vectors, performs a base lookup and a flattened per-batch H lookup,
  masks both results, then adds them.
- `HyperLinear.forward()` computes the original base logits, computes H logits
  with a batched matrix multiply, and concatenates
  `base_logits[..., :V]`, H logits, and `base_logits[..., V:]`. Thus any
  original output rows after `V` are shifted by `K` in the expanded output ID
  space. H output rows currently have no bias; original base bias rows are
  preserved on either side of the insertion.
- The legacy input wrapper does not perform the corresponding shifted-tail
  lookup. It interprets every ID at or above `V` as an H slot, so it cannot
  reliably handle an original-tail ID shifted by `K`. The baseline tests
  preserve this finding; the fast table uses the output layer's verified
  ordering and maps shifted tail IDs to the original tail rows.
- For positions, each base ID spans one base position and each seeded H ID
  spans the number of constituent base tokens. With valid-token mask `m` and
  span `s`, positions are `offset + cumsum(s*m) - 1`, padding positions are
  returned as zero, and the per-row offset advances by `sum(s*m)`. This makes
  H2/H3 positions land at the end of their expanded phrase and carries that
  semantic offset across cached decode calls.
- `StaticCodebookManager.set_seeded_codebook()` builds the padded codebook
  updates and H-span table. `synthesize_hyper_vectors()` can explicitly
  synthesize both sides before generation. Otherwise the input and output
  vectors are lazily synthesized by the first respective wrapper call and
  cached. The output encoder is used when untied; the input encoder is reused
  when encoders are tied.
- `attach_to_model()` installs the static manager on the Zip2Zip wrapper and
  its HyperEmbedding/HyperLinear modules; `detach_from_model()` restores the
  previous manager. `Zip2ZipModel.generate()` resets positional request state
  before and after generation while the default static-manager reset preserves
  seeded definitions and vector caches. Replacing a codebook must invalidate
  every new effective table.
- The ordinary `Zip2ZipModel` initializes the dynamic `CodebookManager`; the
  static predictive path explicitly attaches a `StaticCodebookManager`.
  Training uses the legacy wrappers and checkpoint format. The joint
  checkpoint loader installs LoRA and encoder tensors separately; this task
  does not alter it or the saved checkpoint structure.
- At baseline, `decode_sequence()` expanded seeded H IDs but left output-tail
  IDs shifted. The implementation now restores original tail IDs by
  subtracting `K` only for IDs at or above `V + K`. Prepared inference inputs
  must map original prompt tail IDs upward by `K` before optional segmentation;
  `prepare_input_sequence(..., compress=True|False)` does this explicitly and
  is unavailable until effective tables are ready. This avoids collisions
  between raw tail IDs and H IDs without changing legacy/training behavior.

The requested `experiments/profile_gpu_decode_overhead.py` is not present in
this checkout. No GPU profiler or replacement GPU experiment will be run.

## Target inference lifecycle

1. Load the existing predictive checkpoint without changing its format.
2. Optionally merge PEFT/LoRA through
   `prepare_model_for_inference(model, merge_lora=True)`, using
   `merge_and_unload(safe_merge=True)`; never merge implicitly during model
   construction or training. If merging after table preparation, the helper
   invalidates those tables so they cannot retain pre-merge weights.
3. Seed and attach the request's static codebook.
4. During request setup, synthesize input and output H vectors once, then build
   effective input and output tables once.
5. Convert original tokenizer prompt IDs with
   `static_mgr.prepare_input_sequence(base_ids, compress=...)`; this shifts
   original tail IDs into the expanded ID space before segmentation. Generate
   using a trusted fast position path, one embedding lookup, Phi, and
   one output projection. No encoder call or effective-table rebuild belongs
   in the decode loop.
6. Reset request position state; invalidate prepared tables when the codebook
   changes or caches are explicitly cleared; detach to restore the prior
   manager.

For base embedding/head matrices with row counts `E` and `O`, the intended
expanded layouts are:

```text
effective input rows  = input_base[:V] + K input-H rows + input_base[V:]
effective output rows = output_base[:V] + K output-H rows + output_base[V:]
```

Original prompt IDs at or above `V` are shifted by `K` before they enter this
table; generated output IDs already use that expanded space. `decode_sequence()`
reverses the shift for tail IDs while expanding H tokens.

The corresponding effective output bias is original bias rows before `V`,
zero-valued H rows, then original bias rows from `V` onward. The fast embedding
uses the expanded input IDs directly, so shifted original-tail IDs address the
matching original tail row. Legacy/training behavior remains available and is
not switched to the fast path unless request setup has completed.

## Implementation and acceptance gates

The CPU implementation is recorded on the feature branch. Its reference suite,
CPU-only profiler, directional timing, measured memory, and current validation
status are summarized in
`experiments/reports/fast_inference_cpu_profile.md` and its adjacent JSON.
These synthetic measurements do not establish real-Phi correctness or GPU
performance; the future GPU review gate below remains unrun.

1. Add synthetic CPU reference tests for base/H embeddings, H2/H3 and mixed
   spans, base/H logits, output insertion order, output tail, masks, and
   repeated request offsets. Capture current legacy behavior before changing
   wrapper hot paths; report the known shifted-tail embedding limitation.
2. Add setup-time codebook validation and a trusted position path selected by
   Python manager state. Compare it exactly with the reference across random
   base/H2/H3 mixtures, masks, sequence lengths, and repeated calls. The fast
   path must not branch on tensor contents or call `.any()`, `.item()`, or
   `bool(tensor)`.
3. Add explicit request preparation, encoder-call counters, effective-table
   caches, build counters, and memory accounting. New codebooks and explicit
   cache-clearing resets must invalidate prepared state; normal generation
   resets must preserve it.
4. Add fast HyperEmbedding and HyperLinear routes using one lookup and one
   projection respectively. Verify weights, bias, vocabulary ordering, top-1,
   and greedy fixture IDs against the captured legacy references wherever the
   legacy path defines behavior. Verify prepared prompt tail-ID mapping and
   round-trip. Keep training and unprepared inference on the legacy routes.
5. Add and test an explicit inference LoRA-merge helper. It must reassign the
   merged base model, install the generation-position hook exactly once, set
   evaluation mode, and verify both Zip2Zip wrapper modules remain available.
6. Add a CPU-only equivalence harness, CPU operator profile, and small CPU
   timing checks for position, embedding, and projection paths. CPU timings are
   directional only and are not evidence of GPU speedup. A separate, gated
   real-Phi comparison harness is maintained for the eventual approved check;
   it is not run by default.
7. Run focused and existing tests, inspect changed files and diffs, and create
   small commits on this branch. Push only this feature branch; do not merge it.

The shifted-tail asymmetry is now explicit and covered: legacy logits preserve
tail rows, the fast input table maps shifted tail IDs to their source rows, and
output expansion reverses that shift. Stop and report if another token mapping
remains ambiguous, fast-vs-reference logits change materially, top-1 changes
unexpectedly, stale tables survive a reset, or correctness requires changing
training semantics or checkpoint loading.

## CPU implementation validation status

- Focused regression set: **46 passed** across fast-inference semantics,
  static codebook, position handling, checkpoint loading, segmentation, and
  quality-benchmark EOS tests. A separate synthetic backward test confirms
  unprepared HyperEmbedding/HyperLinear remain differentiable and do not select
  the fast path.
- CPU harness: all 15 synthetic checks passed, including identical greedy IDs,
  H2/H3 positions, output-tail order, prepared prompt-tail shifting and
  round-trip, second-codebook invalidation, and no encoder calls after setup.
- CPU operator profile: embedding changed from 2 embedding ops, 2 arange, 5
  multiply, and 1 add to 1 embedding op; output changed from 1 linear + 1 bmm
  + 1 cat to 1 linear; the position path has 0 `aten::any` calls (legacy: 2).
- Directional CPU medians (ms/op): positions **0.0563 → 0.0309**, embedding
  **0.0617 → 0.0036**, output projection **0.0743 → 0.0421**. These are small
  synthetic CPU tensors and do not predict or establish T4/GPU speedup.
- Synthetic 2,048-row, 128-hidden, K=32 float32 tables retain **2,142,352
  additional CPU bytes** in effective tables. Real-model memory depends on
  vocabulary, hidden width, and dtype; no real Phi model was loaded here.
- The full test suite was **not completed**. An exploratory broad CPU pytest
  invocation was interrupted after inspection showed it included full
  Phi-3.5-Mini generation and a full-model backward/optimizer smoke test. Its
  output was silent, so I cannot establish whether that optimizer-step test
  had begun before cancellation. No checkpoint files changed. This is not
  represented as a passing training test; the bounded synthetic gradient test
  above is the training-path evidence for this change.
- No live Phi generation, retraining, GPU/CUDA work, Kaggle job, or remote
  compute was run.

The detailed machine-readable and Markdown measurements are in
`experiments/reports/fast_inference_cpu_profile.json` and
`experiments/reports/fast_inference_cpu_profile.md`.

## Fast-inference integration and safety update (2026-09-23)

- The authoritative `experiments/run_quality_benchmark.py` remains on its
  legacy predictive path (`prepare_prompt_input_ids()` followed by
  `model.generate()`). Do not replace that path: it is the legacy control for
  the later A/B/C comparison.
- The separate `experiments/validate_predictive_fast_path.py` supports
  `vanilla`, `predictive_legacy_merged`, and `predictive_fast_merged`. Its
  default is a non-loading dry run. Real model execution is gated by all of
  `--execute --device cuda[:N] --allow-gpu`; the harness also enforces batch
  size 1 and a 256-token static KV-cache limit.
- Predictive B/C share a single codebook selected once from the same raw prompt
  token IDs. The harness checks prompt/tokenizer/decoding-setting identity and
  codebook identity before reporting the smoke comparison. The predictor keeps
  `max_subtokens=3`; the static inference manager remains at 4.
- The fast request order is explicit: merge LoRA with
  `prepare_model_for_inference(model, merge_lora=True)`, seed/attach the static
  manager, prepare tables, transform IDs with `prepare_input_sequence()`,
  generate, restore H/tail IDs with `decode_sequence()`, and detach. The legacy
  condition does not prepare the effective tables. Generation-critical IDs
  in model/generation configs and call-time overrides, plus
  `HyperEmbedding.padding_idx`, must remain below `V`; tail-ID remapping of HF
  generation settings is intentionally unsupported.
- Prepared fast inference is explicitly batch-size 1. Unsupported batch sizes
  fail before table state changes; direct prepared-position and generation
  paths enforce the same contract. Exact input, output, and optional output-bias
  rows are covered by tests. The trusted prepared position branch stays free of
  tensor-to-Python decisions (`aten::any`, `.item()`, and `bool(tensor)`), while
  unprepared legacy validation remains defensive.
- LoRA merge invalidates prepared tables. Tests rebuild against merged weights,
  verify the table-build counter, and confirm exactly one idempotent generation
  hook remains on the merged base.
- Table setup reports H-vector synthesis, effective input-table construction,
  effective output-table construction, and total preparation time. CUDA event
  timers synchronize only at the setup boundary, not per decode step. The
  future harness also records TTFT, end-to-end and decode wall time, mean /
  median / p95 cached-forward timing, VRAM before/after/peak during preparation,
  table bytes, output IDs, expanded IDs, H emissions, and B/C divergence/logit
  agreement diagnostics.
- Shape/dtype-only estimate for the pinned Phi-3.5 Mini dimensions
  (32,064 rows by 3,072 hidden, fp16, K=32) is **188.06 MiB per effective
  table**, **376.13 MiB total for the two full effective tables** held alongside
  the existing model weights. The newly inserted K rows themselves account for
  only about 0.38 MiB across both tables; copying the base rows is the dominant
  setup-memory cost. The harness recomputes the estimate from the actual loaded
  model's tensor shapes/dtypes before a future approved run. Persistent extended
  buffers that update only H rows remain a possible later optimization, not part
  of this change.
- Current bounded CPU validation: **62 focused tests passed** and the CPU-only
  harness passed all **15 synthetic checks**. Its refreshed component timings
  are directional CPU fixture results only. No real-Phi inference, GPU/CUDA,
  Kaggle, or remote-compute validation has been performed for the prepared path.

## Future GPU review gate (documented, not authorized to run)

The first GPU test remains blocked on the user's review and explicit approval.
After approval, use one T4, fp16, batch size 1, real Step-100, the canonical
predictor K=32 codebook, compressed prompt, no emission gate, merged LoRA, and
fixed static KV cache length 256. Use `gsm_2956` and approximately 100 cached
decode-forward calls (101 generated-token cap includes the first prefill
prediction; EOS may stop earlier). Run three conditions, in this order for
reporting:

1. Vanilla Phi.
2. Predictive Step-100, merged, legacy runtime.
3. Predictive Step-100, merged, prepared fast runtime.

The legacy and fast conditions must reuse the exact same codebook and original
prompt IDs. Compare one B/C smoke generation and save raw decode IDs, expanded
base IDs, text, H emissions, first divergence, top-1 agreement, top-5 overlap,
and finite-value max/mean logit differences. FP16 logits need not be bitwise
identical. Use CUDA Events for decode forwards; the first full-prompt forward is
prefill and is excluded from cached decode-step summaries. Run Vanilla
separately after the predictive pair is released so two full Phi models do not
need to remain resident together.

Proposed command (not run):

```powershell
python experiments/validate_predictive_fast_path.py --execute --device cuda:0 --allow-gpu --conditions vanilla predictive_legacy_merged predictive_fast_merged --prompt-id gsm_2956 --batch-size 1 --max-new-tokens 101 --kv-cache-length 256
```

Do not start with the 12-prompt benchmark, run a Kaggle job, or schedule any
GPU work as part of this gate. Historical V16 timings are context only and do
not establish a performance result for this new implementation.
