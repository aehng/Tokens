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
- `experiments/reports/fast_inference_cpu_profile.md`
- `experiments/reports/fast_inference_cpu_profile.json`

## Validation and limits

- Focused CPU regression tests: **46 passed**.
- CPU harness: **15 synthetic checks passed**.
- The full suite was not completed. A broad CPU pytest run was interrupted
  after discovering it included full-model generation and a backward/optimizer
  smoke test. The silent output leaves it uncertain whether that optimizer
  step had begun; no checkpoint files changed.
- No real Phi inference, retraining, GPU/CUDA work, Kaggle job, or remote
  compute was run. A real GPU comparison remains gated on explicit user review
  and approval.
