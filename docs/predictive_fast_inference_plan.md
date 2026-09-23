# Predictive Fast Inference Plan

## Scope and safety boundary

This work targets the Step-100 static predictive inference hot path. It is a
CPU-only implementation and validation task. No Kaggle kernel, CUDA benchmark,
remote GPU, or GPU scheduling is authorized. The eventual GPU check described
below requires a separate explicit approval after review of the CPU results.

Work is isolated on `codex/predictive-fast-inference`; it must not be merged to
`main` as part of this task. Step-100 checkpoints, predictor files, datasets,
and existing V16 results are read-only.

## Current behavior verified in source

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
- The input wrapper does not currently perform the corresponding shifted-tail
  lookup. It interprets every ID at or above `V` as an H slot. Therefore a
  shifted original-tail ID is not reliably handled by the legacy embedding
  path. This is a pre-existing asymmetry, not a behavior to hide: tests must
  capture ordinary/H legacy behavior and output-tail ordering, and separately
  prove the fast table's intended shifted-tail mapping. Greedy equivalence
  fixtures must make any remaining compatibility boundary explicit.
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

The requested `experiments/profile_gpu_decode_overhead.py` is not present in
this checkout. No GPU profiler or replacement GPU experiment will be run.

## Target inference lifecycle

1. Load the existing predictive checkpoint without changing its format.
2. Optionally merge PEFT/LoRA through an explicit inference-only helper using
   `merge_and_unload(safe_merge=True)`; never merge implicitly during model
   construction or training.
3. Seed and attach the request's static codebook.
4. During request setup, synthesize input and output H vectors once, then build
   effective input and output tables once.
5. Generate using a trusted fast position path, one embedding lookup, Phi, and
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

The corresponding effective output bias is original bias rows before `V`,
zero-valued H rows, then original bias rows from `V` onward. The fast embedding
uses the expanded input IDs directly, so shifted original-tail IDs address the
matching original tail row. Legacy/training behavior remains available and is
not switched to the fast path unless request setup has completed.

## Implementation and acceptance gates

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
   legacy path defines behavior. Keep training and unprepared inference on the
   legacy routes.
5. Add and test an explicit inference LoRA-merge helper. It must reassign the
   merged base model, install the generation-position hook exactly once, set
   evaluation mode, and verify both Zip2Zip wrapper modules remain available.
6. Add a CPU-only equivalence harness, CPU operator profile, and small CPU
   timing checks for position, embedding, and projection paths. CPU timings are
   directional only and are not evidence of GPU speedup. Do not run a separate
   live-generation benchmark.
7. Run focused and existing tests, inspect changed files and diffs, and create
   small commits on this branch. Do not push or merge as part of this task.

Stop and report if token mapping remains ambiguous, fast-vs-reference logits
change materially, top-1 changes unexpectedly, stale tables survive a reset,
or correctness requires changing training semantics or checkpoint loading.

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
