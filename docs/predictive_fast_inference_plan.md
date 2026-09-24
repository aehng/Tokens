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
  size 1. Its `generate()` run is explicitly a **behavioral smoke test**:
  `cache_implementation="static"` with capacity 256 is only a preallocation
  limit; the used context grows with the prompt and each generated token. Do
  not interpret those per-step timings as fixed-context measurements.
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
  behavioral smoke gets a separate two-token unmeasured warmup; manager request
  position state resets before the recorded generation, and unseeded H IDs are
  masked during warmup. The recorded smoke captures TTFT, end-to-end and decode wall time, output IDs,
  expanded IDs, H emissions, and B/C divergence/logit agreement diagnostics.
  The separate **true fixed-KV microbenchmark** prefills an actual 256-position
  cache, then measures one-token cached forwards from that same context length.
  It uses at least 20 warmups and exactly 100 measured iterations for Vanilla,
  predictive legacy, and predictive fast, in that order. A mutation probe runs
  against a copy: if one-token forward mutates the cache, every warmup and
  measured sample receives a fresh independently stored copy; otherwise the
  reference is reused only after the probe confirms it remains unchanged. Any
  copying/reconstruction occurs outside the timed interval. The harness records
  the cache class, layer/tensor metadata, mutation result, input sequence length
  for every sample, and asserts the reference cache is 256 positions before
  and after probing and timing. The installed Transformers 5.17.0
  `DynamicCache` is mutable and has no cache-copy method, so the harness uses
  validated `copy.deepcopy()` for it; CPU tests also cover legacy tuple caches.
  It reports mean / median / p95 / standard deviation / min / max forward time
  and steps per second, while excluding prefill and cache-copy work. The JSON
  and Markdown reports keep behavioral generation and fixed-KV measurements in
  separate sections. Derived comparisons include predictive overhead versus
  Vanilla, fast speedup versus legacy, and fast-minus-Vanilla residual
  per-step overhead. Historical V16 values—about 58 ms Vanilla, 100 ms legacy,
  and an approximate 66 ms break-even at the prior 12% decode-call reduction—
  are context only, never assertions or expected results. VRAM before/after/
  peak during fast table preparation and table bytes remain recorded separately.
- Shape/dtype-only estimate for the pinned Phi-3.5 Mini dimensions
  (32,064 rows by 3,072 hidden, fp16, K=32) is **188.06 MiB per effective
  table**, **376.13 MiB total for the two full effective tables** held alongside
  the existing model weights. The newly inserted K rows themselves account for
  only about 0.38 MiB across both tables; copying the base rows is the dominant
  setup-memory cost. The harness recomputes the estimate from the actual loaded
  model's tensor shapes/dtypes before a future approved run. Persistent extended
  buffers that update only H rows remain a possible later optimization, not part
  of this change.
- Previous fast-path integration CPU validation: **62 focused tests passed**
  and the CPU-only harness passed all **15 synthetic checks**. Its refreshed
  component timings are directional CPU fixture results only. No real-Phi
  inference, GPU/CUDA, Kaggle, or remote-compute validation has been performed
  for the prepared path.

- Fixed-KV harness correction CPU validation: **14 tests passed** in
  `tests/test_predictive_fast_inference_harness.py`; the harness compiled and
  `--dry-run` displayed both separate protocols without loading a model,
  checkpoint, or CUDA context. **No GPU work was performed.**

## Future GPU review gate (documented, not authorized to run)

The first GPU test remains blocked on the user's review and explicit approval.
After approval, use one T4, fp16, batch size 1, real Step-100, the canonical
predictor K=32 codebook, compressed prompt, no emission gate, and merged LoRA.
Use `gsm_2956`. Keep two distinct measurements, each covering vanilla,
predictive legacy, and predictive fast:

1. Behavioral smoke generation using a static cache with **capacity 256**.
   Its used KV context grows during generation; collect output, EOS, wall-time,
   TTFT, and legacy/fast equivalence diagnostics. These timings are not fixed-
   KV results and must not be directly compared with historical true-fixed-KV
   V16 figures.
2. True fixed-context microbenchmark: prefill exactly 256 physical KV
   positions, then run at least 20 warmup and exactly 100 measured one-token
   forwards. Probe mutation using a copy; if the cache is mutable, create an
   independently stored cache copy for every warmup and measured sample outside
   the timed interval. If it is proven immutable, reuse the untouched reference
   cache. Never feed a previous sample's returned cache into the next sample.
   Assert each sample starts at 256 and the reference cache remains at 256
   before and after probing and the full timing loop. Report cache class,
   layer/tensor metadata, mutation and copy strategy, and each sample's input
   sequence length. Use CUDA Events for the measured forward only; synchronize
   before the batch and after recording all events, not inside each sample.
   Report mean, median, p95, standard deviation, min/max, steps per second,
   warmup count, and measured count. Prefill and cache-copy time are excluded.

Run three conditions in the same order for both measurement sections:

1. Vanilla Phi.
2. Predictive Step-100, merged, legacy runtime.
3. Predictive Step-100, merged, prepared fast runtime.

The legacy and fast conditions must reuse the exact same codebook and original
prompt IDs. Compare one B/C smoke generation and save raw decode IDs, expanded
base IDs, text, H emissions, first divergence, top-1 agreement, top-5 overlap,
and finite-value max/mean logit differences. FP16 logits need not be bitwise
identical. Use CUDA Events for fixed-KV one-token forwards. Run Vanilla first
and release it before loading the predictive model, so two full Phi models do
not need to remain resident together. The two predictive conditions use one
merged model sequentially but each receives a fresh manager; after each
condition, assert the previous model and embedding/output manager bindings are
restored before continuing.

Proposed command (not run):

```powershell
python experiments/validate_predictive_fast_path.py --execute --device cuda:0 --allow-gpu --conditions vanilla predictive_legacy_merged predictive_fast_merged --prompt-id gsm_2956 --batch-size 1 --max-new-tokens 101 --static-cache-capacity 256
```

Do not start with the 12-prompt benchmark, run a Kaggle job, or schedule any
GPU work as part of this gate. Historical V16 timings are context only and do
not establish a performance result for this new implementation.

## Authoritative Tier-1 (12-Prompt) Validation Results (Version 7)

Executed on a single Tesla T4 GPU under commit `ea72dac3af7a6b5a3bbfcb2a5b6ab91167965ba2` using `disable_compile=True` and static cache capacity 512.

### 1. Fixed-KV Microbenchmark (Context Length = 256, 100 Measured Cached Forwards)
- **Vanilla:** 37.44 ms
- **Predictive Legacy Merged:** 37.79 ms (+0.93% overhead vs Vanilla)
- **Predictive Fast Merged:** 37.77 ms (+0.89% overhead vs Vanilla, +0.04% speedup vs Legacy)

### 2. 12-Prompt Aggregate Decode & Speedup
- **Total raw decode iterations:** 2,794
- **Total expanded output tokens:** 3,134
- **Net compression fraction:** 10.85%
- **Overall break-even decode latency:** 42.04 ms (well above the 37.77 ms per-forward step time)
- **Total Vanilla equivalent decode time:** 117,447.6 ms
- **Total Legacy decode time:** 106,016.9 ms (106,392.8 ms with setup)
- **Total Fast decode time:** 105,749.9 ms (106,243.6 ms with setup)
- **Fast decode speedup vs Vanilla:** +9.96% (+9.54% with setup)
- **Fast decode speedup vs Legacy:** +0.25% (+0.14% with setup)

### 3. Setup Costs & Memory
- **Fast Table Build Overhead:** Mean 3.48 ms (1.80 ms input table, 1.68 ms output table).
- **Fast Table Memory:** Exactly 394,395,648 bytes (~376 MiB VRAM) for the concatenated effective embedding and lm_head weight tables.
- **Legacy Eager Vector Synthesis:** Mean 31.33 ms (first prompt 81.22 ms).

### 4. Semantic Equivalence & Quality
- **Token Equivalence (Legacy vs Fast):** 12/12 prompts produced **100% identical token IDs** (`top1=1.0`). Finite logit differences were within normal FP16 precision limits (`max_abs_logit_diff <= 0.0039`).
- **MBPP Code Synthesis (4 prompts):** Vanilla 3/4 syntax valid, 1/4 problem pass. Legacy/Fast 1/4 syntax valid, 0/4 problem pass.
- **GSM8K Math Reasoning (4 prompts):** Vanilla 3/4 exact correct. Legacy/Fast 2/4 exact correct.
- **Alpaca Instruction Following (4 prompts):** Vanilla 2/4 mechanical pass. Legacy/Fast 3/4 mechanical pass.

