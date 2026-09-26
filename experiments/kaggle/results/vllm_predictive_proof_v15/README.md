# vLLM predictive proof v15

Kernel v15 used source commit
`3f2c95c9bb707a026075c3c429916f113b857719` and source archive SHA-256
`634518d9554382be6feabde5093dcbcc193d694c2ce25e3a236e7e9906d30bf5`.
The run used vLLM 0.30.0 on a T4 and diagnostic parity mode was disabled.

## Observed result

Phase 9 reached a real scheduler preemption:

- `pre-B` was recorded as preempted and as an admission add/remove/add cycle.
- Its scheduler preemption count was 1; `pre-A` remained at 0.
- At readmission, `pre-B` had `already=0`, `semantic_offset=0`, and
  `position_mode=compressed`.
- The available codebook identity checks matched on re-admission.
- The slot-clear check based on `h_spans` reported clear.
- Both concurrent token trajectories exactly matched their uninterrupted
  controls.

Phase 9 reported FAIL only because its old rebuild gate required
`already > 0` and a non-null semantic offset. That gate did not match the
vLLM 0.30 Model Runner V2 recompute-on-preemption path. With prefix caching
disabled, preemption discards KV state and resets `num_computed_tokens` to
zero. The scheduler re-admits the complete logical history as a new request;
the worker rebuilds predictive H state and recomputes positions from zero.
This is expected runtime behavior, not evidence of a production resume bug.

Phase 10 did not run because the runner stopped on the Phase 9 report.
The v15 run did not include the later proof-only audits for H-input and
H-output hashes, all slot tensors, admission-generation-bound A–F resumed
positions, or a source-captured full machine-readable Phase 9 report. The
corrected targeted Phase 9 proof adds those as explicit acceptance gates.

The next run should preserve `phase_09_preemption.json`,
`phase_10_semantic_rope.json`, `run_manifest.json`, `environment.json`, the
kernel log, and exact source provenance. Phase 10 remains unchanged and is
only run after targeted Phase 9 passes, followed by the full Phases 1–10
proof.
