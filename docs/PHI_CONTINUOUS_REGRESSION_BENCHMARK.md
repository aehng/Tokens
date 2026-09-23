# Continuous Phi Regression Benchmark Policy

**Status:** Evaluation policy. This document defines how future runs are selected, pinned, launched, and reported; it does not mean any test has run.

This policy is subordinate to the current phase gates and priorities in the
[canonical research roadmap](../experiments/RESEARCH_ROADMAP.md). In
particular, the next planned GPU comparison is the matched Vanilla-vs-
Predictive Tier-1 runtime/stopping test; tier escalation remains gated on its
evidence and on quality.

The standing question is whether the current predictive system improves on (1) Vanilla Phi, (2) Official Reactive Zip2Zip, and (3) the previous best verified predictive model. Answer it during development rather than waiting for a final project benchmark, while keeping expensive evaluations gated.

## Comparators and promotion rule

The routine three-way comparison is:

| Label | Reference |
|---|---|
| A | Vanilla Phi |
| B | Official Reactive Zip2Zip |
| C | Current experimental predictive stack |

Also compare C with the previous best verified predictive configuration. If that differs from C, it is a separate fourth run/configuration; if they are identical, reuse the same valid result. Never silently replace the previous-best reference. Tier 1 improvement alone does not promote a candidate to canonical/best-verified status; normally require Tier 2 first.

## Tiered benchmark ladder

### Tier 1 — fixed 12-prompt routine regression

Use the existing fixed subset: 4 MBPP/code, 4 GSM8K/reasoning, and 4 Alpaca/instruction prompts. After each meaningful model, predictor, policy, training, prompt-compression, EOS/stopping, or runtime change, evaluate the current candidate against A and B and the previous best. Reuse cached A/B results only when their full validity keys match (see below).

Record quality separately by domain:

- MBPP Pass@1 and syntax validity.
- GSM8K exact accuracy.
- Instruction mechanical failure rate and semantic score when available.

Record compression and compute:

- Transformer decode steps and expanded/base-equivalent output tokens.
- Net decode steps saved; MICRO and MACRO decode reduction.
- Quality-preserved decode reduction and predictive hypertokens emitted.

Record runtime:

- Predictor latency and codebook/setup latency.
- TTFT, decode wall time, total wall time, and expanded tokens/second.

For current vs. Vanilla Phi, Official Zip2Zip, and previous best predictive, report the quality delta, decode-step/compression delta, and wall-time delta. Do not collapse the three domains into one mixed quality percentage without also showing each domain.

Tier 1 is the normal regression, not a reason to rerun all references when valid cached generations already exist.

### Tier 2 — fixed 30-prompt development benchmark

Use a fixed 30-prompt suite: 10 code, 10 reasoning, and 10 instruction prompts. Run only when Tier 1 clearly passes and the candidate could plausibly replace the current best. Reuse valid Vanilla/Official generations. A Tier 1 win by itself does not establish a new best; normally require Tier 2 before promotion. Do not tune on a final held-out test set.

### Tier 3 — full 60-prompt milestone benchmark

The 60-prompt suite is an expensive development benchmark, not a routine per-change test and not a clean final holdout if it has informed tuning. Run only at meaningful milestones, such as after the evaluation harness is repaired, policy is finalized, a short retraining experiment succeeds, continuation-consistency training is adopted, before choosing a configuration for a longer/full training run, or after final training. Keep any fresh final holdout separate and unopened until settings are frozen.

Do not escalate to the next tier if the previous tier fails its quality/correctness gate or regresses without an understood cause.

## When not to run a tier

Do not run the model regression for documentation-only edits, analysis-only changes, unit-test-only changes, or refactors proven behavior-identical. Use unit tests and static checks as appropriate. A meaningful change includes predictor/ranker, K/adaptive-K policy, emission gate, hyperencoder, LoRA/training, objective, prompt-compression behavior, EOS/stopping behavior, or inference/runtime behavior.

## Baseline cache validity

Cache each baseline generation/result under a manifest hash containing, at minimum:

- Prompt IDs and prompt-text hash.
- Prompt/chat formatting or template version, plus raw/compressed prompt representation.
- Tokenizer ID/revision and model ID/revision or checkpoint hash.
- Evaluator version and relevant task-interface/test versions.
- Generation settings, `max_new_tokens`, and EOS/stopping rules.
- Baseline condition and checkpoint identity.

Reuse a Vanilla or Official result only when the relevant manifest keys match exactly. If a prompt, template, tokenizer/model/checkpoint, evaluator, generation setting, max token limit, prompt representation, or stopping rule changes, invalidate and regenerate the affected comparison conditions. The current MBPP, instruction-scoring, and prompt-representation repairs invalidate affected historical comparisons; do not relabel old results as corrected baselines.

## Asynchronous, commit-pinned execution

Long-running benchmark jobs should run asynchronously so independent project work can continue. Each run must test an immutable, exact Git commit:

1. Record the full 40-character `tested_commit` SHA, not just a branch name or dirty working-tree description. Run from a dedicated checkout/worktree pinned to that SHA; do not let edits in the active checkout alter a running benchmark.
2. If the candidate is not yet committed, create an explicit validation commit/ref containing the exact code/config being tested before dispatch. Do not amend, rebase, or otherwise change the tested commit while the job is running.
3. Assign a run ID such as `<short-sha>-tier1-<timestamp>`. Save the tier, tested SHA, model/checkpoint hashes, cache-key hashes, hardware/software versions, command/config, start/end time, job/process ID, exit status, logs, raw outputs, and summary in a run manifest.
4. Launch the run in the background, then continue work that does not depend on its answer: documentation, official API compatibility review, static harness review, workload preparation, or implementation in a separate commit/worktree. Do not modify the benchmark's pinned checkout.
5. GPU timing comparisons that share the same physical GPU must not compete concurrently; serialize those conditions or use genuinely isolated hardware. Asynchronous means the main work can continue while the job runs, not that contending runs produce valid latency numbers.
6. Wait for the result only at a decision gate: Tier 1 before deciding whether a candidate passes/promotes to Tier 2; Tier 2 before naming a new best; Tier 3 before a milestone/final claim. If a next action does not depend on the result, keep progressing in parallel.
7. Store the result under the exact `tested_commit` in the machine-readable scoreboard and a concise research-log entry. A later result-record commit should include both `tested_commit` and its own `result_record_commit`; do not alter the tested commit merely to attach a result after the fact.

If a run fails, is interrupted, or is invalidated by a manifest mismatch, record that status and reason; do not treat partial output as a completed benchmark. A completion notification should include the run ID, tested SHA, tier, pass/fail/invalid status, summary metrics, and artifact paths.

## Canonical scoreboard and evidence labels

Maintain one machine-readable scoreboard, for example `experiments/benchmark_scoreboard.json`, with entries for:

- Vanilla Phi.
- Official Zip2Zip.
- Best Verified Predictive.
- Current Experimental Predictive.

Each entry should link to its tested commit/checkpoint and applicable run manifests, show code/math/instruction quality separately, overall quality-gate count, MICRO decode reduction, quality-preserved decode reduction, mean wall time, throughput, and predictor overhead. Update the best-verified row only after the appropriate promotion tier passes; keep current experiments separate.

Label measurements explicitly:

- **MEASURED:** task quality, decode steps, and wall time on the stated hardware/runtime.
- **PROJECTED:** GPU/vLLM throughput or datacenter capacity that was not measured in that run.

Do not present CPU wall-time observations as measured GPU/vLLM performance. Every report should make the current candidate's result against Vanilla, Official, and previous best immediately visible.
