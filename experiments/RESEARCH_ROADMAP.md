# Predictive Hypertoken Research Roadmap

**Status:** Canonical current research plan, updated 2026-09-28. This roadmap supersedes earlier active phase ordering in the study, research log, Qwen/vLLM planning snapshots, and the 2026-09-23 roadmap. It records the order of work; it does not authorize expensive experiments automatically.

## Current state

- The Phi-3.5 predictive path has a verified vLLM 0.30.0 compatibility proof. Targeted Phase 9 passed on Kaggle kernel v18 and the full Phases 1–10 proof passed on kernel v19 on a Tesla T4. The machine-readable reports were merged to `main` at `23b2506ab2837c76c1b181b1856360a5560a763f` and tagged `good`.
- That proof establishes that request-specific hypertoken embeddings/logits and semantic RoPE positions can coexist with stock vLLM scheduling, continuous batching, paged KV cache, attention, sampling, slot reuse, chunked prefill, and recompute-style preemption for the tested Phi stack.
- The proof does **not** establish broad task-quality parity, Qwen3-8B compatibility or performance, realistic production throughput, or compatibility with every vLLM production feature.
- The authoritative 12-prompt Phi T4 validation measured 10.85% net compression and +9.96% decode speedup versus Vanilla-equivalent timing (+9.54% including setup), but the quality sample remained mixed. Broad quality parity is therefore still the primary research risk.
- Predictor/codebook quality is expected to be the main improvement lever, but quality must be treated as an end-to-end property of the whole pipeline: candidate generation, ranking, codebook construction, H input/output representations, continuation state, termination/EOS behavior, decoding/serving, and fallback behavior.

Historical measurements remain historical. The [research log](../RESEARCH_LOG.md) records dated evidence, the [predictive study](../PREDICTIVE_HYPERTOKEN_STUDY.md) keeps research history, and the [benchmark methodology](QUALITY_BENCHMARK_METHODOLOGY.md) defines the live-run data contract.

## Decision principles

1. **Quality first, throughout the pipeline.** Do not optimize speed or raw compression by accepting unexplained quality loss.
2. **Improve hypertoken prediction first where the evidence points there, not by assumption.** Candidate generation, ranking, occurrence prediction, K, H representations, training objectives, and stopping/continuation are all valid suspects.
3. **Keep Vanilla as the control.** Every quality or performance claim must be paired against the same base model and serving contract.
4. **Preserve the verified vLLM proof point.** Do not casually change the known-good Phi/vLLM path while investigating unrelated quality issues.
5. **Separate correctness, quality, and performance.** Passing one does not imply the others.
6. **Measure production value, not token-count aesthetics.** Use TTFT, TPOT, time-to-EOS, words/sec, requests/sec, GPU-seconds/request, latency distributions, throughput, and task quality.
7. **Batching/concurrency is a first-class production concern.** vLLM's value comes largely from continuous batching and scheduler utilization; a speedup that disappears under realistic batched serving is not a production win.

## Canonical execution sequence

### Phase 1 — Broader Phi quality baseline and failure attribution

Establish a stronger Vanilla-vs-Predictive quality baseline before major architecture work.

Cover at least:
- reasoning/math
- code
- instruction following
- general knowledge / short factual generation
- longer-form generation / continuation

For each prompt, capture both final task quality and the trajectory needed to locate where divergence begins.

Attribute failures across the full pipeline:
- useful phrase absent from candidate generator
- useful phrase present but ranked out
- selected codebook phrase never occurs / dead slot
- useful hypertoken occurs but is not emitted
- emitted hypertoken changes continuation state
- H input/output representation error
- EOS / stopping / repetition / post-answer-tail issue
- serving or decoding contract mismatch

Do not classify every failure as a predictor problem merely because predictor quality is the leading hypothesis.

**Gate:** We have a repeatable quality suite and a ranked list of dominant failure modes.

### Phase 2 — Improve hypertoken prediction and end-to-end quality

Iterate on the dominant quality bottlenecks, with predictor/codebook quality as the likely primary workstream.

Potential work includes:
- better candidate generation
- better ranking / occurrence prediction
- quality attribution and oracle hierarchy (Global Occurrence Oracle vs Fixed Candidate-Pool Oracle vs Empirical Safety Oracle)
- predictor V2 architecture bake-off (Ridge, Pooled MLP, CNN+Suffix, GRU, 1-layer Transformer)
- top-two architecture live attribution test prior to final Predictor V2 selection
- improved supervision for the predictor (base model continuation labels instead of reference answers)
- better phrase filtering or codebook construction
- K recalibration only when supported by evidence
- output-head / HyperLinear training if useful H tokens are not emitted
- continuation-consistency or representation training if H emissions destabilize later decoding
- EOS / continuation training if Predictive is worse than matched Vanilla
- prompt/serving contract fixes if Vanilla and Predictive share the same stopping pathology

Every iteration must report:
1. quality vs Vanilla
2. quality-preserved decode-step reduction
3. H utilization / hit rate
4. termination and repetition health
5. per-step runtime
6. end-to-end latency

**Gate:** Phi quality is strong enough that moving to a larger, more product-relevant model is justified.

### Phase 3 — Qwen3-8B transfer and correctness

Port the verified architecture to Qwen3 8B while keeping the base model frozen.

Start with correctness, not speed:
- Vanilla Qwen HF reference
- Predictive Qwen HF
- Predictive Qwen vLLM
- logical/physical vocabulary mapping
- request-specific H embeddings and logits
- correct RoPE semantics
- HF-vLLM agreement
- multiple codebooks / request isolation
- slot reuse
- chunked prefill
- scheduler preemption/re-admission

Reuse the Phi proof design where it maps cleanly, but do not assume Phi-specific hooks generalize unchanged.

A small per-model calibration job / LoRA / encoder fit is acceptable. Full customer-base-model retraining is not the desired product path.

**Gate:** Qwen3-8B predictive inference is correct in HF and vLLM on the supported configuration.

### Phase 4 — Qwen quality parity

Repeat the quality program on Qwen rather than assuming the Phi result transfers.

Compare Vanilla Qwen vs Predictive Qwen on the same domains and diagnostics used in Phase 1.

Track:
- task accuracy / pass rate
- semantic instruction quality
- continuation divergence
- EOS / truncation / repetition
- codebook precision and H emission quality
- quality-preserved compression / decode-step reduction

**Gate:** The Qwen speed experiment is worth running at production-relevant scale.

### Phase 5 — Qwen vLLM performance, including batching and concurrency

Measure the system in the serving environment that matters.

Single-request metrics:
- TTFT
- TPOT / decode latency
- time-to-EOS
- end-to-end latency
- transformer decode calls
- expanded/base-token-equivalent throughput
- words/sec
- GPU utilization
- VRAM
- GPU-seconds/request
- setup/codebook synthesis overhead

Batched / concurrent metrics:
- continuous/dynamic batching behavior
- requests/sec
- tokens/sec and words/sec at the service level
- p50 / p95 / p99 latency
- concurrency 1, 2, 4, 8, and higher where hardware permits
- scheduler fairness / starvation
- request-specific codebook isolation under batching
- throughput-vs-latency curves
- whether predictive shortening improves batch turnover or is hidden by other bottlenecks

Batching is **production-critical**, not an optional polish item. A single-stream speedup is useful evidence, but vLLM production value depends heavily on batched throughput and scheduler efficiency.

**Gate:** Predictive Qwen provides a meaningful quality-preserved service-level advantage, not merely fewer logical decode steps.

### Phase 6 — Make the vLLM path genuinely plug-and-play

Turn the proof implementation into a supported serving surface.

Target shape:

```
Tokens frontend / sidecar
        ↓
predictor + codebook builder
        ↓
generic Tokens-vLLM adapter
        ↓
stock vLLM
```

Goals:
- normal OpenAI-compatible HTTP serving
- model-family adapter interface rather than a Phi-only wrapper
- clean install/load/unload path
- rollback to untouched Vanilla behavior
- request accounting and stop semantics
- observability and diagnostics
- safe fallback to Vanilla for unsupported request features

Desired operator experience should trend toward:

```bash
tokens serve Qwen/Qwen3-8B
```

rather than a custom research harness.

### Phase 7 — Production vLLM feature compatibility and stress

Add and validate production features in priority order based on customer value and performance impact.

Priority set:
1. CUDA graphs / compiled execution
2. codebook-safe automatic prefix caching
3. **continuous batching / dynamic batching stress and high-concurrency validation**
4. tensor parallelism
5. quantization
6. pipeline parallelism where relevant
7. production HTTP/API robustness, observability, fallback, and rollout controls
8. speculative decoding / EAGLE compatibility later, after the core product path is stable

For prefix caching, requests with the same logical H IDs but different codebook meanings must never incorrectly share KV state.

For batching, validate not only correctness but sustained throughput, tail latency, slot churn, preemption, fairness, and codebook isolation under load.

### Phase 8 — Product viability decision

At this point answer the product question directly:

- How much quality-preserved latency reduction do we get?
- How much throughput improvement do we get under realistic batching?
- What is the GPU-seconds/request improvement?
- What additional VRAM/setup cost do we pay?
- How much per-model calibration is required?
- How broad is model-family support?
- Which vLLM features remain unsupported?
- Is rollout/rollback simple enough for a datacenter operator?

Do not promote the project based on raw compression alone.

## Quality work continues through every phase

Quality is not a one-time gate that ends after Phase 2.

Each later phase must keep a Vanilla control and re-check the relevant quality suite whenever any of the following changes:
- model family
- predictor
- codebook algorithm
- K / phrase length
- H training or representation
- serving runtime
- batching/concurrency behavior
- quantization
- cache behavior
- compiled/CUDA-graph path
- speculative decoding

If a performance optimization changes output quality or continuation behavior, treat that as a correctness/quality regression, not an acceptable benchmark artifact.

## Stop and escalation conditions

Stop and investigate if:
- evaluator provenance is invalid
- checkpoint or predictor verification fails
- predictive quality materially regresses
- per-step overhead erases decode savings
- EOS/termination deteriorates badly
- batching causes request-state/codebook contamination
- throughput gains vanish at realistic concurrency
- a feature such as prefix caching or quantization violates the semantic-token contract
- a change improves raw compression but reduces quality-preserved compression
- an experiment does not justify its compute

## Required questions in future reports

Answer in order:
1. quality vs Vanilla
2. where any quality loss originates in the pipeline
3. transformer decode calls saved
4. quality-preserved savings
5. H utilization / predictor capture
6. whether answers arrive earlier/later
7. termination and wasted post-answer work
8. per-step cost
9. end-to-end latency
10. batched throughput and tail latency when applicable
11. GPU-seconds/request
12. dominant remaining bottleneck
