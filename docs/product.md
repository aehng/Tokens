# Product Goal: Datacenter Plug-and-Play Acceleration

> **Historical planning document.** The project's current status changed on 2026-10-01: research is paused while the target base model changes. The commercial goals and sequencing below record an earlier direction; they are not current product claims or an active roadmap.

This project is building a **commercial inference product**, not only a research result.

**First-class v1 serving target: vLLM.** The product should integrate as a runtime/plugin and small sidecar, not replace a customer's serving engine.

The Phi-3.5 proof has now established that the predictive architecture can coexist with stock vLLM scheduling, continuous batching, paged KV cache, attention, sampling, chunked prefill, slot reuse, and recompute-style preemption on the tested vLLM 0.30.0 stack. This is a compatibility proof, not a production deployment proof. Qwen3-8B remains the next model-family validation target, and realistic service-level performance still needs to be established.

The canonical execution order is maintained in the single [research roadmap](../experiments/RESEARCH_ROADMAP.md). The current order is:

1. Improve end-to-end quality on Phi, with predictor/codebook quality as the likely main lever.
2. Port and prove the architecture on Qwen3-8B.
3. Re-establish broad quality parity on Qwen.
4. Measure real vLLM performance, including continuous batching and realistic concurrency.
5. Turn the proof path into a generic plug-and-play serving integration.
6. Add production vLLM features such as CUDA graphs, codebook-safe prefix caching, high-concurrency batching/stress, tensor parallelism, quantization, and later speculative decoding.

Research experiments (Phi-3.5, Zip2Zip, calibration ladders, local CPU/XPU) exist to find the smallest adapter that works. The **end product** is something a datacenter operator can load onto serving stacks they already run, with the simplest possible changeover.

## What we are shipping toward

A customer keeps their existing base LLM. We add a small sidecar (encoders, optional LoRA, predictor, runtime) so decode uses fewer steps. The operator should be able to:

1. Keep the production model weights they already serve.
2. Load our adapter + runtime next to that model.
3. Flip traffic (or a fraction of traffic) onto the accelerated path.
4. Roll back by unloading the adapter. The original model still produces the same output.

LoRA (or a similarly small adapter) is an acceptable changeover cost. Full retraining of the customer’s base model is not the product we want.

## Datacenter changeover (target)

| Property | Target |
|---|---|
| Base model weights | Unchanged (hash-verified). Customer keeps their checkpoint. |
| What we install | Runtime + codebook/predictor + trained sidecar (encoders and/or LoRA). |
| Serving integration | vLLM first-class v1 target, through a runtime/plugin or thin supported integration; retain the customer's serving engine. |
| Rollout | Adapter on/off at runtime; original quality when off. |
| Models we have never trained on | First-class: attach to a new family with calibration (LoRA / encoder fit), not a full custom train. |
| Operator effort | As close to plug-and-play as we can get. Per-model LoRA or a short calibration job is OK. Asking them to replace the model is not. |

The base model stays unchanged. A small per-model adapter (hypermodules and/or LoRA) is acceptable; full retraining of an 8B customer model is not the desired product path. Preserve quantization, KV caching, batching, and EAGLE/speculative decoding wherever compatibility testing shows they can coexist. Do not claim compatibility in advance of testing the exact model and serving versions.

Production value is judged on measured **GPU-seconds per request, TTFT, TPOT, end-to-end latency, throughput, tail latency, batching efficiency, and task quality**. Position count or token reduction by itself is not a production result.

Ideal install shape:

```
customer_serving_stack/
├── their_base_model/          # already in the datacenter — UNCHANGED
├── our_runtime/               # generate wrapper, codebook, serving hooks
├── our_adapter/               # hyperencoders and/or LoRA, plus predictor
└── config.json                # K, max_subtokens, attach points
```

Verification at install:

- `hash(base_model)` before == after
- Adapter loads and unloads without side effects
- With adapter removed, outputs match the original model

## Generality (models we have never trained on)

Phi-3.5 + EPFL Zip2Zip is the **research vehicle**. It is not the product surface.

The product is **model-agnostic acceleration**:

- Works on models we did not pretrain.
- Prefer a recipe that transfers: freeze the customer backbone, attach our modules, optionally train a small LoRA / encoder on their stack (or a public proxy of that family).
- Measure success as: install on a held-out model family with only that calibration step, and get decode-step reduction without quality loss.

If zero-shot transfer is too weak, the fallback is **short per-model calibration** (LoRA / encoder), still leaving base weights frozen. That is still plug-and-play relative to replacing or fully finetuning their LLM.

## How research serves the product

Keep experiments cheap and local where possible. Every adaptation level is judged by **deployability**, not only compression:

| Prefer | Avoid (unless all lighter options fail) |
|---|---|
| Frozen customer base | Training their full model |
| Small LoRA / encoders we ship | Requiring they adopt EPFL’s specific checkpoint |
| Predictor + runtime around stock vLLM | Custom kernels they must rebuild the cluster around |
| One calibration job per new model family | Per-request or per-tenant retraining |
| Hash-stable rollback | Irreversible weight edits |

The research decision order is quality-preserved acceleration first: preserve answer quality and continuation, then reduce transformer work, then prove that the reduction survives real serving conditions such as continuous batching and concurrency. The later commercial target remains at least 5% MICRO decode reduction with quality held; compression from truncated or degraded output does not count.

## Batching and production throughput

Continuous/dynamic batching is a first-class production concern.

A single-request speedup is useful evidence, but vLLM is commonly deployed to maximize aggregate GPU utilization across many concurrent requests. Predictive inference therefore has to show that shorter physical sequences produce useful service-level gains under batching rather than merely moving a bottleneck elsewhere.

Production validation should include:

- concurrency sweeps
- requests/sec
- p50/p95/p99 latency
- TTFT and TPOT
- throughput-vs-latency curves
- scheduler fairness
- preemption behavior
- slot churn and codebook isolation
- GPU utilization and GPU-seconds/request

If a predictive speedup disappears or materially regresses tail latency under realistic batching, that is a product-level failure even if single-stream decode is faster.

## Non-goals for the v1 product

- Replacing the customer’s model card with ours.
- Requiring they serve only Zip2Zip-pretrained checkpoints.
- Research-only metrics that do not survive serving (quality regressions, non-unloadable adapters).

Research may still use EPFL Zip2Zip Phi-3.5 to answer scientific questions. Product work should keep asking: **can this land on a datacenter model we have never trained and deliver a quality-preserved service-level win?**
