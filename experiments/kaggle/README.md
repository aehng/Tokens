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

Kernel v12 used source commit `03261da32e81981ef596403dd0f38e80f8fea45d`
and is recorded in [the v12 result folder](results/vllm_predictive_proof_v12/README.md).
The Phase 8 report was marked FAIL because the LlamaModel forward pre-hook
missed stage E; the captured layer-0 rotary positions and generated IDs matched
the reference. The probe now wraps the exact `LlamaModel.forward` boundary
and restores it after each request. A new targeted Phase 8 T4 run is required
to close that instrumentation gate. The full Phases 1–10 proof has not been
launched.
