# vLLM predictive proof, kernel version 10

## Result

The proof failed in Phase 8 (chunked prefill). Phases 1–7 passed. The runner
stopped at the first failure, so Phases 9–10 did not run.

The generated token IDs matched the reference (16/16). The captured position
sequence did not: the three chunks reported positions `0–15`, `0–15`, and
`0–9`, while the reference expected one continuous sequence, `0–41`. The
failure therefore identifies a chunked-prefill position-accounting mismatch.
These artifacts do not isolate whether the incorrect sequence came from
position-offset reconstruction or from the position-capture path.

## Provenance

- Source commit: `c4c5091209ff8ebc33b0883d0719546496c8548b`
- Source archive SHA-256: `aa9982a71089e06ccc03b178090d86d2cf4a04dae50ae5a4c41dcfb1ea05e662`
- Kaggle kernel: `elikearl/tokens-vllm-predictive-poc`, version 10
- Diagnostic parity mode: disabled
- GPU: Tesla T4
- PyTorch / CUDA: `2.13.0+cu130` / `13.0`
- vLLM: `0.30.0` (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`)

`run_manifest.json` includes the failure traceback; `phase_08_chunked_prefill.json`
contains the compared token IDs and position sequences. The remaining phase
reports record the results through Phase 7. Large tensor dumps and the full
console log are not included.
