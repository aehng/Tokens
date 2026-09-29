# Broader Phi Live Quality Attribution Plan

**Status:** Prepared as the next live research phase. This plan was not executed during the offline candidate-retrieval run. It does not assume Predictor V2 is the root cause.

## Purpose and scope

Attribute end-to-end quality loss across the existing Phi predictive pipeline before training a Predictor V2 architecture. Use the corrected canonical dataset and its pinned native chat-template, model/tokenizer revision, greedy decoding, EOS, and 1024-token ceiling contract. Use DEV only; do not open or score FINAL.

The primary comparison uses matched DEV prompts and one fixed predictive checkpoint/adapter. Keep prompts, base revision, tokenizer, rendered prompt, generation settings, evaluator, and runtime identical across arms. Freeze the prompt list and arm configuration before live execution. The preferred scope is all 135 DEV prompts (45 per domain). A nine-prompt wiring check (three per domain) may precede it, but is plumbing-only and cannot decide the bottleneck.

## Controlled arms

| Arm | Live condition | Purpose |
|---|---|---|
| A | Vanilla Phi | Matched task-quality, termination, and decode-step baseline. |
| B | Predictive checkpoint/adapter with H emission disabled | Isolates adapter, checkpoint, and serving-path quality without token substitution. |
| C1 | Same predictive condition with a K≤32 occurrence-oracle codebook derived from the matched Vanilla model continuation | Tests an occurrence-only hindsight upper bound. It may select phrases that are unsafe for predictive continuation. |
| C2 | Same predictive condition with a K≤32 continuation-safe oracle codebook derived from that Vanilla model continuation | Tests whether an idealized safer codebook can preserve task quality and continuation. |
| D | Same predictive condition with the current learned predictor/codebook | Measures the current predictor/codebook against C1 and C2. |

Oracle codebook construction may inspect the matched Vanilla model continuation only. It must not use human reference answers or FINAL data. Preserve the oracle construction artifacts and hashes. C1/C2 are diagnostic hindsight conditions, not deployable predictors.

## Frozen measurements

For every prompt and arm, retain the prompt-list hash, arm/checkpoint/codebook hashes, runtime and generation contract, raw generated IDs/text, termination reason/token, and evaluator output. Measure:

- task-specific quality: MBPP tests, GSM8K numeric answer, and a frozen Alpaca rubric;
- first divergence from matched Vanilla output and the failure location;
- H opportunities, attempted and successful H emissions, selected IDs, and dead codebook slots;
- continuation-state validity after each H emission, fallback/replay events, and repeated or malformed output;
- EOS reached, cap reached, repetition/pathology indicators, and output truncation;
- generated/decode steps, useful steps saved, wall time, and GPU time per request;
- quality-preserved savings, reported only for prompts meeting the frozen task-quality criterion.

Compare conditions prompt-by-prompt and by Code, Reasoning, and Instruction. Keep aggregate, per-domain, and per-prompt evidence; do not hide regressions in an average.

## Execution size and compute estimate

The full matched study is 135 prompts × 5 arms = **675 generations**, with a per-generation ceiling of 1024 new tokens. The hard upper bound is **691,200 generated tokens** if every arm reaches the cap; actual token count and GPU time must be measured from the completed run. The Vanilla outputs are reused to construct and score C1/C2, so oracle construction does not add another live Vanilla generation arm. Before committing to the full run, record a bounded nine-prompt-per-arm estimate for elapsed time, GPU-seconds, memory, and failures. Do not extrapolate a wall-clock promise from the offline candidate benchmark.

## Interpretation gate

- **B materially below A:** The adapter/checkpoint/training/serving path is already a quality problem; fix or isolate it before Predictor V2 training.
- **B near A, C1 below A, C2 near A:** Occurrence alone is unsafe; prioritize codebook safety and continuation-state behavior.
- **B near A, C2 near A, D below C2:** The learned predictor/codebook is a supported bottleneck. Predictor V2 TRAIN fitting and DEV comparison may proceed.
- **C2 below A:** Even a continuation-safe hindsight codebook cannot preserve the baseline; representation, H handling, continuation, EOS, or another subsystem must be addressed before ranking architecture.
- **Offline capture is strong but D emits few/no H tokens:** H emission/calibration or serving is the immediate bottleneck; offline ranking gains do not pass the training gate by themselves.
- **B near A and D near C2 with low capture/savings:** Candidate generation/ranking may be a bottleneck, but require the DEV retrieval evidence and live observations to distinguish them.
- **EOS, continuation-state, or task-quality failures dominate:** Redirect to that subsystem rather than assuming Predictor V2 ranking will fix it.

Record the observed primary category as one of `candidate_generation`, `candidate_ranking`, `predictor`, `codebook`, `h_emission`, `representation`, `continuation_state`, `eos`, `serving`, or `other`, with evidence and an explicit pass/redirect decision. Do not select an architecture from this attribution run.

## Downstream order

Only a passed attribution gate permits Predictor V2 architecture training on TRAIN and comparison/shortlisting on DEV. Then take at most two shortlisted candidates through a small live end-to-end DEV integration check. Combine offline DEV and live DEV evidence to freeze the candidate generator, architecture, checkpoint, and configuration. FINAL remains a one-time evaluation after those gates, with explicit `--allow-final-eval`; it cannot participate in selection. Larger live end-to-end validation follows the freeze.
