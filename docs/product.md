# Product Goal: Datacenter Plug-and-Play Acceleration

This project is building a **commercial inference product**, not only a research result.

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
| Serving integration | Hugging Face / vLLM-class generate path, or a thin wrapper around it. |
| Rollout | Adapter on/off at runtime; original quality when off. |
| Models we have never trained on | First-class: attach to a new family with calibration (LoRA / encoder fit), not a full custom train. |
| Operator effort | As close to plug-and-play as we can get. Per-model LoRA or a short calibration job is OK. Asking them to replace the model is not. |

Ideal install shape:

```
customer_serving_stack/
├── their_base_model/          # already in the datacenter — UNCHANGED
├── our_runtime/               # generate wrapper, codebook, serving hooks
├── our_adapter/               # output_encoder and/or LoRA, plus predictor
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

Keep experiments cheap and local. Every adaptation level is judged by **deployability**, not only compression:

| Prefer | Avoid (unless all lighter options fail) |
|---|---|
| Frozen customer base | Training their full model |
| Small LoRA / encoders we ship | Requiring they adopt EPFL’s specific checkpoint |
| Predictor + runtime that wrap `generate()` | Custom kernels they must rebuild the cluster around |
| One calibration job per new model family | Per-request or per-tenant retraining |
| Hash-stable rollback | Irreversible weight edits |

Decision rule from the study still applies: climb the adaptation ladder only until we hit ≥5% MICRO decode reduction with quality held. Then productize that artifact for datacenter load.

## Non-goals for the v1 product

- Replacing the customer’s model card with ours.
- Requiring they serve only Zip2Zip-pretrained checkpoints.
- Research-only metrics that do not survive serving (quality regressions, non-unloadable adapters).

Research may still use EPFL Zip2Zip Phi-3.5 to answer scientific questions. Product work should keep asking: **can this land on a datacenter model we have never trained?**
