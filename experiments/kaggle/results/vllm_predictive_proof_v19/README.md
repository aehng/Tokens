# vLLM predictive proof v19: full Phases 1–10

Kaggle kernel `elikearl/tokens-vllm-predictive-poc`, version 19, completed on a Tesla T4. It used source dataset version 17 and source commit `7fb0a21a78ae69f7b29c1a54f728493fefcb953a`. The committed source archive SHA-256 was `98f2aa9da0c8a73aa88b953cf4d9358ebb53c2338c43d2683d6532e749f072c8`.

Runtime: Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0, vLLM 0.30.0, Transformers 5.17.0. Diagnostic parity and prefix caching were disabled. Kaggle reported `KernelWorkerStatus.COMPLETE`. The run manifest records `PASS` for the full run and for every phase from 01 through 10. Phase 9 includes the scheduler-confirmed preemption and rebuild checks recorded for targeted kernel v18; Phase 10 semantic RoPE also reports `PASS`.

The environment, run manifest, and all ten machine-readable phase reports are preserved here. This is a research proof on the Phi-3.5-mini vehicle; it does not establish Qwen compatibility, production readiness, or broad quality parity.
