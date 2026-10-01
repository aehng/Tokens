# Tokens: frozen-model hypertoken research

**Project status: paused (October 2026).** This repository records experiments on reducing autoregressive steps by representing short token sequences with hypertokens while keeping a base model frozen. The target model changed, so work on the Phi-3.5 research vehicle is paused. The measurements below apply to `microsoft/Phi-3.5-mini-instruct`, revision `2fe192450127e6a83f7441aef6e3ca586c338b77`; they should not be assumed to transfer to another model.

## What we learned

The project separated three questions that earlier experiments had mixed together: whether one physical cache position has enough representational capacity, whether a shared learned encoder can find useful vectors, and whether the wrapper or an active adapter changes the base model. The strongest result is a positive capacity result paired with a clear learning gap: per-example optimized vectors worked very well on the fixed DEV set, while the trained shared encoder did not. This is a useful research diagnosis, not a solved general-purpose compression method.

| Experiment | Measured result | What it establishes |
|---|---|---|
| **Per-example Oracle H** | On 12 fixed DEV examples, immediate top-1 matched 12/12 with mean KL `0.000125` nats. Across offsets 0, 1, 2, 4, 8, and 16, top-1 matched 72/72 with mean KL `0.000288` nats. Greedy rollout agreement was 95.31% over 16 tokens and 91.93% over 32. | A separately optimized 3,072-dimensional vector at one physical cache position can closely reproduce the tested two-token phrase continuations on these contexts. This is an **existence/capacity test**: each example received its own optimized vector, so it does not demonstrate a shared encoder or generalization. See the [Oracle H report](docs/ORACLE_H_CAPACITY_EXPERIMENT.md). |
| **Trained single-slot H encoder** | On its 48-state DEV evaluation: 20.83% immediate top-1, mean KL about `3.674` nats, and 5.73% rollout-16 agreement. | This encoder and training setup did not learn a useful general mapping. The later Oracle result shows that this failure is not evidence that a one-slot representation is impossible. |
| **A vs. B0 wrapper generation** | On all 12 pinned prompts, rendered prompts, prompt token IDs, generated IDs, expanded IDs, and termination matched exactly. All ended with EOS `32007`. B0 reported no PEFT adapter, no Step-100 checkpoint, an empty H codebook, and masked H logits. | The adapter-free wrapper preserved deterministic greedy generations on this smoke set. The separate matched-prefix logit comparison did not complete, so full distribution-level wrapper fidelity remains unverified. See the [A/B0 report](docs/AB0_GENERATION_FIDELITY_REPORT.md). |
| **B1 active-adapter diagnostic** | With the upstream PEFT adapter active, H masked, and no Step-100 checkpoint: top-1 matched Vanilla at 34/48 prefix states (70.83%), mean KL was `0.6822` nats, and free greedy generation matched exactly on 0/12 prompts. | The adapter alone changed Phi's native predictions in this configuration. This isolates an adapter effect; it does not quantify how much of the earlier task-quality loss it caused. See the [B1 diagnosis](docs/B1_ARCHITECTURE_DIAGNOSIS.md). |
| **Earlier 45-prompt attribution benchmark** | Vanilla scored 77.78%; the historical H-disabled condition scored 55.56%, oracle-live scored 55.56%, and the real predictor scored 53.33%. | The original quality gate failed. Its H-disabled condition combined wrapper, adapter, and Step-100 checkpoint, so it could not attribute the drop to one component. The later B0 and B1 controls made that diagnosis more specific; they do not retroactively make the old quality result a controlled A/B0 comparison. See the [45-prompt report](docs/final_phi_attribution_report.md). |
| **Expanded-cache block control** | On a Tesla T4 single-request microbenchmark, block sizes 2, 3, and 4 measured 1.82×, 2.75×, and 3.67× forward speedups, while retaining the same number of physical KV positions. | Block forwarding can reduce serial calls in this control. It is neither one-slot cache compression nor an end-to-end serving speedup. See the [frozen-Phi report](docs/FROZEN_PHI_H_CACHE_EXPERIMENT.md). |
| **Earlier predictive prototype** | On a 12-prompt exploratory benchmark, the fast path reduced decode steps by 10.85% and measured 9.96% decode speedup (9.54% including setup). Quality was mixed: code assertions passed on 0/4 prompts and math was correct on 2/4, versus 1/4 and 3/4 for Vanilla. | There was a measurable small-run speed signal, but not quality-preserving deployment evidence. See the [Tier-1 report](experiments/reports/tier1_12_validation_report.md). |
| **vLLM integration proof** | The vLLM 0.30.0 T4 proof reports PASS for phases 1–10, including semantic positions and preemption checks. | This is an integration/compatibility result for the research path, not a quality, production-readiness, or broad-model claim. See the [vLLM proof record](experiments/kaggle/results/vllm_predictive_proof_v19/README.md). |

## How this work relates to prior approaches

This project builds on EPFL's [Zip2Zip](https://arxiv.org/abs/2506.01084), which combines dynamic LZW-based hypertokens, runtime embeddings, and a trained model path. The work here asks a narrower question: **can one input vector at one physical cache position stand in for a short phrase under a frozen decoder, and can a shared context-to-vector model learn that mapping?** Comparing per-example optimized vectors with a trained encoder separates representational capacity from the learned-mapping problem.

That question differs from **speculative decoding**, which uses a proposer and a verification procedure to preserve the target model's output distribution ([Leviathan et al., 2023](https://proceedings.mlr.press/v202/leviathan23a.html)). It also differs from **KV-cache sparsification**, such as Dynamic Memory Sparsification, which reduces retained cache state ([DMS](https://arxiv.org/abs/2506.05345)). The expanded-cache block control in this repository keeps the original physical KV positions and measures a separate mechanism. These are complementary lines of work; this repository does not claim that its mechanism is novel relative to all prior research or that it provides a formal exactness guarantee.

The distinctive contribution is the controlled diagnosis across the three questions: per-example capacity, learned mapping, and adapter-free wrapper behavior. The evidence also shows why those controls matter: earlier end-to-end quality numbers could not identify the responsible component, while the later tests demonstrated both strong per-example representation and measurable drift from the active adapter.

## Evidence and experiment history

- [A/B0 generation report](docs/AB0_GENERATION_FIDELITY_REPORT.md): per-prompt results, exact provenance, and limitations. [Recovered machine-readable evidence](experiments/results/phi_ab0_fidelity_dev12_20260929/offline_equivalence_evidence.json) and [raw generation records](experiments/results/phi_ab0_fidelity_dev12_20260929/raw_attribution_records.jsonl) preserve the prompts and token sequences.
- [Oracle H capacity report](docs/ORACLE_H_CAPACITY_EXPERIMENT.md): the strongest representation result.
- [Frozen-Phi H-cache report](docs/FROZEN_PHI_H_CACHE_EXPERIMENT.md): trained-encoder and expanded-cache control results; its earlier impossibility interpretation is superseded by Oracle H.
- [B1 adapter diagnosis](docs/B1_ARCHITECTURE_DIAGNOSIS.md): matched-prefix effect of the upstream adapter.
- [45-prompt Phi attribution report](docs/final_phi_attribution_report.md): the earlier quality gate and its attribution limits.
- [Tier-1 timing and quality report](experiments/reports/tier1_12_validation_report.md): small exploratory run with mixed task outcomes.
- [vLLM proof v19](experiments/kaggle/results/vllm_predictive_proof_v19/README.md): narrow integration/compatibility evidence.

## Limits and possible continuation

The key experiments use small DEV sets and one Phi revision. Oracle H gives each tested example its own optimized vector. The work did not produce a learned encoder that generalizes, establish broad task-quality parity, or finish A/B0 matched-prefix logit parity. Exact greedy sequence equality on 12 prompts is strong evidence for those tested generations, but it is not proof of full distribution identity. The recovered Version 3 output includes the archive SHA reported by the run, but the archive file itself was not present among the recovered artifacts, so that SHA could not be independently recomputed afterward.

Research could continue on a new target model by repeating the controls in order: establish adapter-free wrapper fidelity, measure per-example representation capacity, test a shared mapping, then evaluate quality and serving performance. A verified proposer-and-target method is another possible path. The existing Phi results provide hypotheses and a measurement framework; they do not establish that either approach will transfer to a changed base model.

## Upstream citation

```bibtex
@misc{geng2025zip2zipinferencetimeadaptivevocabularies,
  title={zip2zip: Inference-Time Adaptive Vocabularies for Language Models via Token Compression},
  author={Saibo Geng and Nathan Ranchin and Yunzhen Yao and Maxime Peyrard and Chris Wendler and Michael Gastpar and Robert West},
  year={2025},
  eprint={2506.01084},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2506.01084}
}
```
