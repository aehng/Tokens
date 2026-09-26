# vLLM predictive proof, kernel v13

**Result: targeted Phase 8 PASS.** The diagnostic now captures the actual
`LlamaModel.forward` argument even though vLLM's eager call path bypasses
PyTorch module forward hooks.

## Provenance

- Kernel version: 13
- Source commit: `255e692d3612dcfc23f9c4122d66e8ad70f86d6d`
- Source dataset version: 12
- Source archive SHA-256: `c557ae5ece1b4ae60d0b35d7d1d65947a963aaace6b8c6b3ff470a238ebc6051`
- Archive size: 34,286,526 bytes
- Runtime: Python `3.12.13`, vLLM `0.30.0`, PyTorch `2.13.0+cu130`, CUDA `13.0`, Tesla T4
- Mode: targeted Phase 8 only; diagnostic parity disabled

The JSON files in this directory are the downloaded Kaggle reports for this
run.

## Acceptance results

- Scheduler progress for `phase8-chunk`: `0, 16, 32`.
- All six stages A–F recorded 42 positions and matched the reference sequence
  `0..41`.
- The actual layer-0 rotary call matched the HF/reference positions.
- Generated token IDs matched the reference `16/16`.
- The prefill trace was scoped to `phase8-chunk`; unrelated requests did not
  contribute events.
- The report status and phase-8 manifest entry are both `PASS`.

The semantic tensor remains separate from vLLM's physical input-batch
positions. Existing CPU contract tests cover the BTE distinction, H4
advancement, request-slot reuse, independent request origins, and position
reconstruction without a postprocess offset commit.

This run executes Phase 8 only. It does not count as the full Phases 1–10 GPU
proof. The next gate is a full proof using the same pushed source commit, with
diagnostic parity disabled and all ten phase reports retained.
