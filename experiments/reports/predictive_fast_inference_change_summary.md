# Predictive Fast Inference Change Summary

Branch: `codex/predictive-fast-inference`

## Findings

- The legacy predictive path performed multiple embedding operations and used
  a separate H-token output projection plus concatenation on each decode step.
- The trusted position path can avoid its tensor-dependent `aten::any` check.
- On the synthetic CPU fixture, the prepared path uses one embedding lookup,
  one output projection, and no `aten::any` call. All 15 harness checks and 46
  focused regression tests passed.
- Directional CPU medians were: positions 0.0563 to 0.0309 ms, embedding
  0.0617 to 0.0036 ms, output projection 0.0743 to 0.0421 ms. These are not
  real-Phi or GPU measurements and do not establish an end-to-end speedup.
- The synthetic effective tables held 2,142,352 additional CPU bytes.
- The final refreshed CPU-only run on 2026-09-23 reported position 0.0640 →
  0.0325 ms, embedding 0.0634 → 0.0037 ms, and output projection 0.0828 →
  0.0503 ms. These are small-fixture timings and remain directional only; do not infer GPU or
  real-Phi speedup from them.

## Changes

- Added request-scoped H-vector synthesis and effective input/output tables;
  added fast one-lookup/one-projection wrapper paths.
- Added setup-time codebook validation, table invalidation, explicit fast
  prompt-tail ID mapping/round-trip, and idempotent generation-hook setup.
- Added explicit inference-only PEFT/LoRA merge helper and CPU equivalence,
  differentiability, and lifecycle tests.
- Added the CPU-only profile harness and detailed results in
  `fast_inference_cpu_profile.md` and `fast_inference_cpu_profile.json`.
- Documented the inference lifecycle, findings, limitations, and a small
  future GPU validation gate in `docs/predictive_fast_inference_plan.md`.

## Fast-path integration and safety work

- The canonical quality benchmark remains on its legacy predictive route so it
  remains a controlled A/B reference. A separate dry-run-first harness at
  `experiments/validate_predictive_fast_path.py` supports Vanilla, predictive
  merged legacy, and predictive merged fast modes. The B/C pair reuses one
  exact K=32 codebook and the same original prompt token IDs. Actual execution
  requires `--execute --device cuda[:N] --allow-gpu`; no model weights or CUDA
  are loaded by the default dry run.
- Prepared table construction and generation now enforce batch size 1 before
  mutating request state. Generation-time special token overrides and model
  configs are checked so EOS/pad and other recognized generation IDs, as well
  as `HyperEmbedding.padding_idx`, remain below the H insertion point.
- Tests assert exact input/output/bias row layouts, shifted tail rows, LoRA
  table invalidation and successful rebuild, generation-hook idempotence after
  merge, and no `aten::any`/Python tensor decision in the trusted prepared
  position path. The duplicate `greedy_fixture()` definition was removed.
- Setup instrumentation reports H synthesis, input/output effective-table
  construction, and total prepare time. The gated GPU harness now separates
  behavioral `generate()` output from the true fixed-KV one-token microbenchmark.
- Shape-only estimate for Phi-3.5 Mini at 32,064 × 3,072, fp16, K=32: **188.06
  MiB per effective table / 376.13 MiB for both complete effective tables**.
  This is the full copied tables in addition to base weights, not just the
  roughly 0.38 MiB of newly inserted H rows. Actual loaded dimensions/dtypes
  are measured from metadata by the future run.
- Current bounded validation for this update: **63 focused CPU tests passed**;
  `experiments/validate_fast_inference_cpu.py` passed all **15** synthetic
  checks. The harness refreshed the CPU report files listed below. No real-Phi
  inference, GPU/CUDA, Kaggle, or remote compute was run.

## Files updated

- `docs/predictive_fast_inference_plan.md`
- `src/zip2zip/__init__.py`
- `src/zip2zip/inference.py`
- `src/zip2zip/model.py`
- `src/zip2zip/nn/embedding.py`
- `src/zip2zip/nn/linear.py`
- `src/zip2zip/static_codebook.py`
- `tests/test_predictive_fast_inference.py`
- `experiments/validate_fast_inference_cpu.py`
- `experiments/validate_predictive_fast_path.py`
- `tests/test_predictive_fast_inference_harness.py`
- `experiments/reports/predictive_fast_inference_change_summary.md`
- `experiments/reports/fast_inference_cpu_profile.md`
- `experiments/reports/fast_inference_cpu_profile.json`

## Validation and limits

- Earlier focused CPU regression tests: **46 passed**. This integration update's
  bounded suite: **63 passed** across nine explicitly selected test files.
- CPU harness: **15 synthetic checks passed** in the refreshed 2026-09-23 run.
- The full suite was not completed. A broad CPU pytest run was interrupted
  after discovering it included full-model generation and a backward/optimizer
  smoke test. The silent output leaves it uncertain whether that optimizer
  step had begun; no checkpoint files changed.
- No real Phi inference, retraining, GPU/CUDA work, Kaggle job, or remote
  compute was run. A future one-T4 comparison among Vanilla, predictive merged
  legacy, and predictive merged fast remains gated on explicit user review and
  approval; the exact proposed command and protocol are documented in
  `docs/predictive_fast_inference_plan.md`.

## Fixed-KV measurement correction (2026-09-23)

- The `generate()` path uses `cache_implementation="static"` and capacity
  256, but the actual KV sequence grows as tokens are generated. It is now
  labeled **behavioral smoke** in the JSON schema and is not described as or
  compared directly with a fixed-KV=256 timing result.
- A separate fixed-context section prefills exactly 256 physical positions for
  Vanilla, predictive legacy, and predictive fast. It runs 20 warmups and 100
  measured single-token cached forwards per condition.
- The installed Transformers 5.17.0 `DynamicCache` has no cache-copy method and
  mutates in place on one-token update. The harness probes a copied instance,
  then deep-copies and validates independent KV tensor storage for each
  warmup/measured sample outside the timed interval. For a representation that
  passes the probe as immutable, it reuses the unchanged reference object
  instead. CPU tests cover both this mutable cache behavior and an immutable
  legacy tuple representation; copy checks include object/storage identity,
  sequence length, layer count, shape, dtype, and device. Every sample's input
  cache length is asserted to be 256, and the reference cache is asserted to
  remain at 256 before/after the mutation probe and full timing loop.
- Warmups are 20 and measured forwards are 100, tracked separately. Copy and
  cache-prefill cost are outside the timed forward. CUDA timing is prepared
  with Events and synchronization only at batch boundaries. The JSON and
  human-readable Markdown output keep `behavioral_generation_smoke` separate
  from `fixed_kv_microbenchmark`; the latter reports next-position metadata,
  mean/median/p95/stddev/min/max, steps per second, request setup and VRAM
  metadata, plus legacy-vs-Vanilla, fast-vs-Vanilla and fast-vs-legacy derived
  comparisons. Predictive conditions use one predictor call, one serialized
  K=32 codebook SHA, identical raw prompt IDs and identical compressed IDs.
  Each gets a fresh manager attached to the same merged model object; restored
  model/embedding/output manager bindings are checked after detach. Vanilla is
  run first and released before loading the predictive model.
- Behavioral generation also gets two unmeasured warmup tokens before the
  recorded request; predictive warmup masks unseeded H IDs and resets manager
  position state afterward.
- Historical V16 references only (not assertions): about 58 ms/step Vanilla,
  100 ms/step predictive legacy, and roughly 66 ms/step break-even at the
  earlier 12% decode-call reduction. The harness does not require current data
  to match these values.
- CPU-only validation for this correction: `tests/test_predictive_fast_inference_harness.py`
  passed **14 tests**; Python compilation and the non-loading dry run passed.
  The dry run shows both protocols, reports behavioral static-cache capacity
  256 versus true active KV length 256, and loaded no model/checkpoint or CUDA
  context. No GPU, Kaggle, or remote compute was used.
