# vLLM predictive proof v18: targeted Phase 9

Kaggle kernel `elikearl/tokens-vllm-predictive-poc`, version 18, completed on a Tesla T4. It used source dataset version 17 and source commit `7fb0a21a78ae69f7b29c1a54f728493fefcb953a`. The committed source archive SHA-256 was `98f2aa9da0c8a73aa88b953cf4d9358ebb53c2338c43d2683d6532e749f072c8`.

Runtime: Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0, vLLM 0.30.0, Transformers 5.17.0. Diagnostic parity and prefix caching were disabled. The kernel status was `KernelWorkerStatus.COMPLETE`; the run manifest and targeted Phase 9 report both say `PASS`.

Phase 9 observed one scheduler preemption of `pre-B`, followed by add/remove/add. Readmission reset `num_computed_tokens` to zero and recomputed from position zero. The slot-clear checks passed before rebuilding H state; codebook identity, spans, H input/output, active H state, position mode, resumed A–F position trace, and concurrent-versus-uninterrupted token trajectories all matched.

Machine-readable environment, manifest, and Phase 9 report are preserved here. The full Phases 1–10 proof ran separately as kernel v19.
