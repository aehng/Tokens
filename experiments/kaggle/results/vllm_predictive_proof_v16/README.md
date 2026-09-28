# vLLM predictive proof v16

## Launch provenance

- Branch: `codex/vllm-predictive-poc`
- Source commit: `d2794c19e6a7235dc10ad8d3fe2ef9b82c83f5f5`
- Source archive SHA-256: `5b9eeced7a666ed693278f14ddb66301b2620d1d6ba0fd90318dd25068bca8a7`
- Source archive size: `34,302,189` bytes
- Kaggle source dataset: `elikearl/tokens-vllm-predictive-source`, version 15 (`ready` before kernel launch)
- Kaggle kernel: `elikearl/tokens-vllm-predictive-poc`, version 16
- Kernel accelerator: `NvidiaTeslaT4` (configured)
- Proof mode: Phase 9 only; diagnostic parity disabled

## Result

The archive mounted with the expected commit and SHA-256. The bootstrap then
exited with code 1 after a Python 3.14 tar extraction deprecation warning and
an encoding error:

```text
mounted source dataset path: /kaggle/input/tokens-vllm-predictive-source
source dataset slug: elikearl/tokens-vllm-predictive-source
SOURCE_SHA: d2794c19e6a7235dc10ad8d3fe2ef9b82c83f5f5
proof_source.bin size: 34302189
archive SHA256: 5b9eeced7a666ed693278f14ddb66301b2620d1d6ba0fd90318dd25068bca8a7
/kaggle/src/script.py:135: DeprecationWarning: Python 3.14 will, by default, filter extracted tar archives and reject files or modify their metadata. Use the filter argument to control this behavior.
  bundle.extractall(root)
'charmap' codec can't encode characters in position 6-7: character maps to <undefined>
```

The error cause is unresolved. The process stopped before any Phase 9 output
was reported, so this is a bootstrap failure, not a Phase 9 acceptance result.
No v16 `phase_09_preemption.json`, `run_manifest.json`, or `environment.json`
was emitted. Phase 10 and the full Phases 1–10 proof did not run. The Kaggle
kernel file listing also contained older output files; those are not treated
as v16 evidence.

The environment did not reach the proof runner, so the actual GPU model,
vLLM/PyTorch/CUDA versions, and installed Transformers version were not
captured for this attempt. The next action is to identify the encoding error
before resubmitting the targeted Phase 9 proof.
