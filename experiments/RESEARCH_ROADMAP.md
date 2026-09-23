# Predictive Hypertoken Research Roadmap

**Status:** Canonical current research plan, updated 2026-09-23. This roadmap supersedes earlier active phase ordering in the study, research log, and the Qwen/vLLM planning snapshot. It records plans, not authorization to launch work. Do not start experiments, Kaggle jobs, K sweeps, or retraining without an explicit task.

## Current state

- The corrected 12-prompt Tier-1 result and Phase-5 contextual-emission-gate benchmark are complete. The authoritative Tier-1 report is [`checkpoints/quality_benchmark/tier1_authoritative.md`](checkpoints/quality_benchmark/tier1_authoritative.md); Phase-5 artifacts are under `checkpoints/quality_benchmark/tier1_runs/3de046a1b858dc7a/`.
- The canonical predictive prompt is `compressed_prompt` / `predictive_codebook_dp_segmented`; the verified Step-100 checkpoint and Oracle-Guided Predictor are the current predictive reference.
- For primary greedy decoding, **no gate is canonical**. Phase 5 found byte-identical output and identical decode trajectories across no-gate, top-16, and top-32 conditions, while the gates added substantial measured CPU decode cost. Gate tuning is closed for greedy decoding; reopen it only for a specific sampling-mode, constrained-serving, or safety hypothesis.
- The immediate next work, when authorized, is a small single-T4 Kaggle infrastructure smoke, followed by a paired GPU Vanilla-vs-Predictive runtime/stopping comparison. These are not yet reported as completed by this roadmap.

Historical measurements remain historical; do not backfill missing diagnostics or silently replace earlier plans/results. The [research log](../RESEARCH_LOG.md) records dated evidence, the [predictive study](../PREDICTIVE_HYPERTOKEN_STUDY.md) keeps research history, and the [benchmark methodology](QUALITY_BENCHMARK_METHODOLOGY.md) defines the live-run data contract.

## Phase-5 findings that changed the order

### Gate experiment

Across no gate, top-16, and top-32, greedy output was byte-identical for all 12 prompts. Every condition produced 2,794 transformer decode iterations, 3,128 expanded tokens, 334 steps saved (10.68% micro decode reduction), and 239 emitted hypertokens. Filtering roughly 79% of slots with top-16 or 70% with top-32 therefore did not change task quality, emitted phrases, or trajectory in this experiment.

The no-gate baseline cost 663.2 ms per decode step. Top-16 added 144.2 ms/step (+21.7% decode-time overhead); top-32 added 254.1 ms/step (+38.3%). The evidence supports treating most filtered candidates as **wasted capacity** under this greedy setup, not as demonstrated active quality damage. A **dead slot** is a predicted phrase that does not occur in the relevant target/generation; a **dangerous slot** is a phrase whose emission materially harms quality or continuation. These terms are not interchangeable. The gate remains available behind configuration for a separately justified future use, but is not part of the primary greedy path.

### Termination and post-answer tail

In the Phase-5 predictive run, 0/12 prompts emitted EOS and 7/12 reached `max_new_tokens=300`. All four GSM8K prompts had produced their deterministically extracted numerical answer by approximately decode step 41–183, yet continued generating. Across these prompts, 578/1,200 decode iterations (48.2%) occurred after the answer. For example, `gsm_6613` continued for 259 steps after its answer (86.3% of its trajectory), and `gsm_2956` continued for 204 steps (68.0%).

This is a measured stopping/continuation problem, not proof that hypertokens caused it. Earlier Vanilla Phi also had poor EOS behavior. The next meaningful comparison must be matched Vanilla Phi versus Predictive Step-100 under the same prompts and generation contract before assigning blame to predictive decoding, the serving contract, or EOS training.

### Predictor capture funnel

Current approximate measurements are 35.05% quality-aware oracle opportunity, 12.98% offline Oracle-Guided Predictor DP compression, and 10.68% realized live predictive micro decode reduction. Predictor codebook precision is about 27.14%, with about 72.86% dead slots. Quantify the loss at each funnel stage before tuning K. A dead slot is not evidence of a dangerous emission.

## Priority order

1. Task quality / parity with Vanilla.
2. Termination and continuation health.
3. Quality-preserved decode reduction.
4. Per-transformer-step runtime cost.
5. End-to-end latency and throughput.
6. Raw compression.
7. K tuning, only after the upstream bottlenecks are understood or improved.

Always distinguish raw compression, quality-preserved compression, decode-step reduction, and wall-clock speedup. Fewer transformer calls alone do not establish faster inference. Attribute latency among request setup, prefill/TTFT, decode-step count and cost, and generation trajectory.

## Canonical execution sequence and gates

No phase starts automatically. Keep long jobs asynchronous and commit-pinned: run from an immutable checkout; record the exact tested commit SHA, run manifest, configuration, environment, logs, outputs, and completion status; continue independent work elsewhere; serialize timing tests that share a physical GPU; and wait only at result-dependent decision gates. Every future live benchmark collects the v3 runtime/trajectory diagnostics in the same generations used for quality scoring. Do not add a duplicate large timing suite by default.

### Phase 6A — Single-T4 Kaggle smoke (infrastructure gate)

Use one Kaggle T4 and only three prompts (code, reasoning, instruction). Verify model fit, Step-100 loading and all expected trained tensors, unchanged frozen-backbone hashes, predictor hash, canonical `compressed_prompt`, no-gate path, actual CUDA use, peak VRAM, trustworthy GPU timing, and outputs broadly consistent with CPU greedy behavior. This is not the full benchmark. Stop on checkpoint/hash mismatch, CPU fallback, CUDA/runtime failure, OOM, or invalid timing.

### Phase 6B — Matched GPU Vanilla vs Predictive stopping/runtime test

Run the fixed Tier-1 12 prompts under matched generation settings:

- A: Vanilla Phi.
- B: verified Step-100 Predictive, canonical compressed prompt, Oracle-Guided Predictor, and no gate.

Collect domain quality (MBPP Pass@1 and syntax, GSM8K exact, Alpaca mechanical pass, and semantic instruction signal if available); trajectory (expanded output, decode iterations, EOS, token cap, repetition, deterministic answer position, post-answer tail); and runtime (TTFT, decode wall, iterations/sec, milliseconds/iteration, expanded tokens/sec, words/sec, total wall). Pair by prompt. Determine whether the difference is per-step cost, trajectory length, EOS/stopping, or setup/prefill. Specifically test whether Vanilla has the same post-answer tail behavior. This result selects the next engineering branch.

### Phase 7 — Predictor/oracle capture funnel

Quantify each stage: quality-aware oracle opportunity → good phrase present in candidate generator → good phrase ranked into top-K → top-K phrase occurs → phrase emitted → continuation-safe emission → quality-preserved realized savings. For every stage report phrase count, potential decode steps saved, retention versus prior stage, cumulative retention of oracle opportunity, domain, and phrase-length breakdown. Separate candidate-generation, ranking, K truncation, occurrence/dead-slot, model-emission, and continuation-safety loss. Never infer harm from non-occurrence.

### Phase 8 — Empirical continuation-safety probes

On a manageable, stratified subset compare the base constituent path `t1…tn` with the hypertoken path `H`: next-token KL, hidden-state cosine, top-1 agreement, top-k overlap, EOS-probability delta, and 3–5-token continuation agreement. Stratify by code/reasoning/instruction, phrase length, prompt-grounded/novel, numeric/ungrounded numeric, function/identifier, syntax/structural, grammatical glue, and whitespace/boundary type. Keep the probe cheap; do not enumerate every candidate. Test whether textually correct phrases can yield internally unsafe continuation state.

### Phase 9 — Evidence-based bottleneck decision

Do not automatically implement a predictor or training fix. Classify the dominant bottleneck using Phases 6B–8:

| Evidence | Next branch to consider |
|---|---|
| Candidate generator misses useful oracle phrases | Improve candidate generation |
| Ranker fails to prioritize available useful candidates | Predictor V2 / better supervision |
| Top-K mostly dead but harmless | Improve occurrence prediction or ranking efficiency |
| Useful hypertokens occur but are not emitted | Output-head / HyperLinear training |
| Correct hypertokens emit but continuation diverges | Continuation-consistency / representation training |
| Predictive EOS is worse than matched Vanilla | EOS-correct retraining / continuation training |
| Vanilla has the same long tails | Fix generation/stopping contract before attributing to hypertokens |
| Predictive GPU per-step cost is much higher | Profile/optimize HyperLinear or dynamic runtime |

### Phase 10 — K recalibration (downstream)

Only after the bottleneck is understood or improved, compare K = 4, 8, 16, 24, 32 using compressed prompts and no gate, with an improved predictor/configuration if Phase 9 justifies one. Measure quality, quality-preserved and raw compression, dead slots, emitted hypertokens, per-step and total runtime, EOS, and post-answer tails. K is downstream tuning, not the assumed architecture fix; do not pick the largest raw compression.

### Phase 11 — Short EOS-correct retraining (conditional)

Only after inference/predictor configuration is stable, warm-start Step-100 with compressed prompts, recommended predictor/K, no gate, and EOS-correct targets. Run +10 steps then Tier-1; continue to +25 total additional steps only if clearly improving; +50 additional is the maximum and requires continued improvement. Judge quality, EOS, truncation, repetition, post-answer tails, continuation, quality-preserved compression, and runtime—not loss alone.

### Phase 12 — Continuation consistency (only if needed)

If empirical evidence still indicates representation drift, compare the existing objective with the same objective plus next-token KL. Consider hidden-state matching only if cheap and justified. Success means near-Vanilla quality, better termination, quality-preserved compression, and reasonable runtime—not merely lower KL.

### Phase 13 — Tiered validation

- Tier 1: fixed 12 prompts, with the v3 diagnostics.
- Tier 2: 30 prompts (10 code, 10 reasoning, 10 instruction), only if Tier 1 clearly improves.
- Tier 3: 60 prompts, only if Tier 2 succeeds.

Do not promote tiers on training loss alone. Retain fresh holdout isolation and reuse cached baselines only when their full manifests match.

### After Phi is stable — Qwen 3 8B + vLLM

Only after Phi mechanics, continuation safety, predictor quality, and serving contract are resolved should work proceed to Qwen 3 8B with vLLM. Validate vanilla and EAGLE3 baselines, then staged Tokens prefill and predictive integration under the separate [production-validation plan](../docs/QWEN3_VLLM_PRODUCTION_VALIDATION.md). Do not migrate unresolved Phi bugs into Qwen/vLLM.

## Stop and escalation conditions

Stop and investigate/escalate if evaluator provenance is invalid; checkpoint or predictor verification fails; GPU smoke fails; predictive quality materially regresses; per-step overhead erases decode savings; EOS/termination deteriorates badly; a change improves raw compression but reduces quality-preserved compression; or the experiment does not justify its compute. Do not begin a larger tier, retraining, or another costly run to explain away a failed gate.

## Required questions in every future report

Answer in order: (1) quality vs Vanilla; (2) transformer decode calls saved; (3) quality-preserved savings; (4) whether answers arrive earlier/later; (5) termination; (6) wasted post-answer steps; (7) per-step cost; (8) total latency; (9) where oracle opportunity is lost; (10) dominant bottleneck. State uncertainty and denominators; association between H emissions and later behavior is not causation.
