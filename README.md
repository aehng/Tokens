# Tokens: Plug-and-Play LLM Inference Acceleration

**Status: Research archived (October 2026).** No active development is planned. Future work could reopen around a different target model, but the Phi results below do not establish that they transfer.

## Project idea

Tokens explored a datacenter inference accelerator that would work alongside models customers already serve. Its research idea was prompt-predictive **hypertokens**: before decoding, select likely short phrases from the prompt and assign them temporary, request-specific IDs. The model could represent or emit a phrase in one decode position, then expand it back into ordinary tokens. The goal was to reduce sequential decoding work while preserving the model's answers and continuation.

The customer would keep the same base-model weights, frozen and hash-identical. The intended design was model-agnostic: a small sidecar with a phrase predictor, input/output modules, and a serving runtime, plus only a short per-family calibration or small adapter if needed. It would attach to the customer's serving stack, with vLLM as the first integration target, and could be unloaded for rollback. Replacing or fully retraining the customer's model was outside the product goal. Success meant a quality-preserving service-level gain in latency, throughput, or GPU use under batching—not compression alone.

Unlike EPFL Zip2Zip's reactive mode, which builds a codebook from tokens already generated, this work focused on predicting a request's codebook from its prompt before decoding. The idea is also distinct from speculative decoding, which drafts and verifies continuations, and KV-cache compression, which reduces stored past state. **Phi-3.5-mini was the research vehicle, not the intended customer model.**

## What the research established

- **Serving compatibility:** A Phi-3.5 prototype passed all ten phases of a vLLM 0.30.0 proof on a Tesla T4, including checks for request-specific state, chunked prefill, slot reuse, and scheduler preemption/rebuild. This is a compatibility result, not a production deployment or quality-parity result.
- **A speed signal with a quality tradeoff:** In a separate 12-prompt T4 validation of the Step-100 predictive configuration, output compression was 10.85%; measured speedup was 9.96% for decoding and 9.54% including setup. Quality was mixed: code assertions passed on 0/4 prompts versus 1/4 for Vanilla, math was correct on 2/4 versus 3/4, and instruction checks passed on 3/4 versus 2/4. Quality-preserving acceleration was not established.
- **A one-position representation can exist for fixed contexts:** Directly optimizing a separate H vector for each of 12 prompts matched Vanilla's next-token top-1 on 12/12 prompts and achieved 95.31% agreement over 16-token rollouts. This was an oracle capacity test, not a reusable predictor. The earlier shared encoder scored 20.83% immediate top-1 agreement and 5.73% 16-token rollout agreement; the experiments did not isolate which training or architecture choices caused that gap.
- **The adapter-free wrapper preserved smoke-set generations:** Vanilla Phi and the Tokens wrapper produced identical greedy token sequences on the pinned 12-prompt set. The matched-prefix logit comparison did not complete, so full distribution-level equivalence remains unresolved.

## Takeaway

The work showed that phrase hypertokens can be represented in fixed contexts and that a prototype can be integrated with a modern serving engine. The main unresolved challenge is a shared, generalizable method that preserves model behavior while producing a reliable end-to-end serving gain. The project did not demonstrate a customer-ready or model-agnostic accelerator.

## Evidence

- [vLLM predictive proof, phases 1–10](experiments/kaggle/results/vllm_predictive_proof_v19/README.md)
- [12-prompt speed and quality validation](experiments/reports/tier1_12_validation_report.md)
- [Per-context hypertoken capacity experiment](docs/ORACLE_H_CAPACITY_EXPERIMENT.md)
- [A/B0 wrapper fidelity report](docs/AB0_GENERATION_FIDELITY_REPORT.md)
- [Historical product direction](docs/product.md)
