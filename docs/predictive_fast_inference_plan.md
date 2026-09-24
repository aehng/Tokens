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
   directional only and are not evidence of GPU speedup. Do not run a separate
   live-generation benchmark.
7. Run focused and existing tests, inspect changed files and diffs, and create
   small commits on this branch. Do not push or merge as part of this task.

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

## Future GPU review gate (documented, not authorized to run)

After the user has reviewed the completed CPU implementation and explicitly
approves a GPU run, validate on one T4 with one representative real K=32
codebook, fixed KV=256, and approximately 50–100 measured decode steps. Compare
only (A) the merged/base Vanilla reference and (B) optimized merged Predictive,
with a single smoke prompt for output equivalence. Do not start with a
12-prompt suite. The historical V16 references were approximately 58 ms/step
for Vanilla, 100 ms/step for merged Predictive, and roughly 12% decode-step
reduction; these motivate a later decision threshold near 66 ms/step but are
not a result for the new implementation.
