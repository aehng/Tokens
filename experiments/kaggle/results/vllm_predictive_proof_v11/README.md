# vLLM predictive proof, kernel version 11

## Result

This T4 run used source commit `379db49118d70095694efe31bbc4e5d0a8bbb31f`
with diagnostic parity disabled. Phases 1–7 passed; Phase 8 failed and the
runner stopped before Phases 9–10. Generated token IDs matched the HF
reference.

The state trace contains the target chunk calculations at `num_computed_tokens`
0, 16, and 32, producing positions 0–15, 16–31, and 32–41. The Phase 8
rotary capture assembled by the existing harness instead reports 0–15, 0–15,
and 0–9. The trace also contains four earlier batch entries without request IDs,
so it is not fully scoped to `phase8-chunk`. This result proves the harness
observed a mismatch but does not yet locate the first production handoff where
the positions diverge. No root cause is claimed from this run alone.

## Provenance

- Source commit: `379db49118d70095694efe31bbc4e5d0a8bbb31f`
- Source archive SHA-256: `bc0cbba3de3a9f971f564f5df2a9dccd29c821c425a5c776571d2052b28bbf70`
- Diagnostic parity mode: disabled
- GPU: Tesla T4
- PyTorch / CUDA: `2.13.0+cu130` / `13.0`
- vLLM: `0.30.0` (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`)

`phase_08_chunked_prefill.json` preserves the compared positions and progress
trace. `failure.json` and `run_manifest.json` preserve the traceback. Large
model and tensor outputs are not included.
