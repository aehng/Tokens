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

The kernel bootstrap is `bootstrap_kaggle.py`. Push that script together with
a `source.tar.gz` archive of the proof commit. The archive is produced at
launch time and is not stored in git.
