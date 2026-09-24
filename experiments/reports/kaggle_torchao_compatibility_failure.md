# Kaggle T4 TorchAO Compatibility Failure

## Run record

- Kernel: `elikearl/tokens-fast-path-single-t4-b884-validation`, version 3
- Source: `becfe6787641d666376a471d5064b5db11f72df5`
- Accelerator: one Tesla T4
- Environment: Python 3.12.13; PyTorch 2.10.0+cu128; Transformers 5.17.0; PEFT 0.19.1; Accelerate 1.13.0; CUDA 12.8; TorchAO 0.10.0
- Step-100 mount, manifest, checkpoint SHA-256, and predictor SHA-256 checks passed.

## Failure and interpretation

PEFT failed during predictive model loading while dispatching the LoRA adapter:

```text
ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported
```

PEFT's actual version comparison rejects `torchao < 0.16.0`, so `0.16.0` itself satisfies the check. Upstream pairs TorchAO 0.16.0 with PyTorch 2.10.0. This benchmark does not use TorchAO, however, so the selected remedy is to remove the stale optional package—not add TorchAO or change PyTorch, Transformers, PEFT, Accelerate, CUDA, or the benchmark protocol.

The failure occurred during predictive model loading, before the predictive Legacy/Fast comparison. Any preceding Vanilla measurements existed only in memory: the prior harness wrote only its final report, so completed conditions were not durably preserved when the later condition failed. There is no complete three-condition comparison or benchmark conclusion from that run. The failure is an environment/import problem, not a benchmark finding.

## Required preflight for the replacement run

After the existing mount/hash/source checks and before importing PEFT, Transformers model-loading code, or the benchmark harness:

1. Record the installed TorchAO version without importing it.
2. Uninstall `torchao` if present; it is unused by this experiment.
3. In a fresh Python process, verify TorchAO is absent and import PyTorch, Transformers, PEFT, and Accelerate at the reviewed versions. Confirm that the single Tesla T4 remains usable.
4. Stop before benchmark/model loading if removal or import/device verification fails. Do not install TorchAO as an automatic fallback.

The replacement run must additionally atomically preserve each completed condition, including its behavioral result, fixed-KV measurements, cache checks, and environment provenance, before starting the next condition.
