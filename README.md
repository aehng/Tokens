# Tokens: frozen-model hypertoken research

**Status: paused (October 2026).** This research used Phi-3.5 as its test model. Work is paused while the target base model changes.

## What the project studies

Tokens is a research project exploring whether a frozen language model can use one “hypertoken” to represent a short sequence of ordinary tokens. The goal was to reduce serial decoding work while keeping the base model fixed and preserving answer quality.

The work builds on [EPFL Zip2Zip](https://arxiv.org/abs/2506.01084), but studies a narrower question: can one input vector at one cache position stand in for a short phrase, and can a shared encoder learn to produce that vector? This differs from [speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html), which drafts and verifies tokens, and [KV-cache sparsification](https://arxiv.org/abs/2506.05345), which reduces retained past state. These are related but distinct approaches; this is a scope distinction, not a broad novelty claim.

## What we found

- **A one-position representation can work in tested examples.** With a separately optimized vector for each of 12 fixed prompts, next-token top-1 matched Phi in 12/12 cases; across six continuation offsets it matched 72/72. Greedy continuation agreed 95.31% over 16 tokens and 91.93% over 32. This was an existence test, not a reusable encoder.
- **The shared encoder did not learn a reliable mapping.** It achieved 20.83% immediate top-1 agreement and 5.73% 16-token rollout agreement. That shows a learning and generalization gap; it does not show that one-position representation is impossible.
- **The adapter-free wrapper preserved generations on the smoke set.** Vanilla Phi and B0 produced exact greedy token sequences on the same 12 prompts. The matched-prefix logit comparison did not complete, so full distribution-level equality remains unresolved.
- **The upstream adapter changed Phi's predictions.** Top-1 matched Vanilla at 34/48 tested prefix states, and exact generated sequences matched on 0/12 prompts. This is why the adapter must be evaluated as a separate condition.
- **The prototype showed a speed signal with quality tradeoffs.** On a 12-prompt run, total time was 9.54% faster than Vanilla. Code assertions passed on 0/4 prompts versus 1/4 for Vanilla, math answers were correct on 2/4 versus 3/4, and instruction checks passed on 3/4 versus 2/4. Quality-preserving acceleration was not established.

## Research takeaway

The strongest result is evidence that a one-position representation can work when optimized for an individual context. The open problem is learning a shared mapping that generalizes while preserving quality and improving end-to-end performance. The project’s contribution is the separation of representation capacity, shared learning, wrapper fidelity, and adapter effects. It is not a production-ready accelerator.

If research resumes on a new base model, the first step is to repeat the wrapper-fidelity and capacity controls before training or performance claims. The Phi findings are hypotheses for that work, not proof they transfer.

## Key reports

- [Oracle H capacity experiment](docs/ORACLE_H_CAPACITY_EXPERIMENT.md)
- [A/B0 wrapper fidelity report](docs/AB0_GENERATION_FIDELITY_REPORT.md)
- [B1 adapter diagnosis](docs/B1_ARCHITECTURE_DIAGNOSIS.md)
- [12-prompt quality and speed report](experiments/reports/tier1_12_validation_report.md)
