# Kaggle runs

`experiments/kaggle/` did not exist before the vLLM proof. The established
inputs are still:

- dataset `elikearl/tokens-step100-gpu-smoke` version 1
- mount checks in `experiments/kaggle_step100_mount.py`
- TorchAO removal via `experiments/kaggle_dependency_preflight.py` before PEFT imports

`run_vllm_predictive_proof.py` is the one-session T4 proof for vLLM
`v0.30.0` (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`). It writes
`phase_*.json` under `/kaggle/working/vllm_predictive_proof/` and stops on
the first failed phase. It does not relaunch itself.

Set `VLLM_PROOF_PHASE8_ONLY=1` to run the targeted chunked-prefill proof
without executing Phases 1–7 or 9–10. It prepares one HF reference, runs the
real 16-token-chunk Phase 8 request, and exits. The report records scheduler
progress and request-scoped position snapshots at A–F: vLLM's physical batch
positions, calculated semantic positions, returned model positions, the
predictive wrapper, nested Llama model, and layer-0 RoPE. The run passes only
when the target request reports progress `0, 16, 32`, every captured stage
matches the reference positions, and generated token IDs match.

Semantic RoPE positions are written to the model-state position buffer. The
separate `input_batch.positions` tensor remains the physical position source
for vLLM's cache bookkeeping. In `base_token_end` mode those two position
sequences can differ. The current BTE history reconstruction copies token
history from GPU to CPU in the proof path; this is correct but not yet the
optimized decode implementation.

The kernel bootstrap is `bootstrap_kaggle.py`. `kaggle kernels push` uploads
only that script. The proof archive is the private dataset
`elikearl/tokens-vllm-predictive-source` (`source.tar.gz` and
`SOURCE_SHA.txt`). The bootstrap selects the input directory whose
`SOURCE_SHA.txt` matches the packed commit. Kaggle extracts an uploaded
`.tar.gz`, so the dataset also stores those bytes as `proof_source.bin`.
The archive is not stored in git.

## Latest targeted result

Kernel v13 passed the targeted Phase 8 proof on source commit
`255e692d3612dcfc23f9c4122d66e8ad70f86d6d`, archive SHA-256
`c557ae5ece1b4ae60d0b35d7d1d65947a963aaace6b8c6b3ff470a238ebc6051`.
The complete reports are in [the v13 result folder](results/vllm_predictive_proof_v13/README.md).
Stages A–F each matched reference positions `0..41`, scheduler progress was
`0, 16, 32`, and generated IDs matched `16/16`. Kernel v12's prior
instrumentation-only failure is preserved in
[the v12 result folder](results/vllm_predictive_proof_v12/README.md).

The full Phases 1–10 proof ran as kernel v14 on source commit
`28da08d370d3f5ac8a91ccd526b0935ae9fd551d` (archive SHA-256
`a35f001b7d2acc594742b82a716cf5234df5c977e498b5900f11675d17f084ab`). The
runner reached Phase 9, then failed because preemption was not exercised. The
budget requested seven output tokens for 42- and 74-token prompts with
16-token blocks. The final sampled output is not fed back into the KV cache,
so those requests only used 48 and 80 cached tokens: 3 + 5 blocks, exactly the
eight usable blocks. The test therefore did not force eviction. The recorded
solo and concurrent token trajectories matched, but no request was preempted
or rebuilt; Phase 10 did not run. See the [v14 run note](results/vllm_predictive_proof_v14/README.md).

The next full run will use eight output tokens in Phase 9, ignore EOS for both
the solo and concurrent requests, and record each request's scheduler
preemption counter while stepping. Phase 9 will pass only when the same request
has a positive scheduler preemption count, an admission add/remove/re-add
cycle, rebuilt predictive state, and an exact concurrent/uninterrupted token
trajectory match. The Phase 9 report will include cached-token and KV-block
counts explicitly. Phase 6 still requires distinct and correctly owned
codebook rows; Phase 10 still checks physical-KV and semantic-RoPE bounds
separately. For six H3 tokens, physical KV positions must stay at `0..5` within
`max_model_len=8`, while semantic RoPE positions must match `[2, 5, 8, 11,
14, 17]` and remain below Phi's exclusive limit of `131072`. The engine's
`max_model_len` constrains the physical KV sequence, not the semantic RoPE
coordinates.
