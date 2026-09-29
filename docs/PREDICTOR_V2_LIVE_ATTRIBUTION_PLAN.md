# Broader Phi Live Quality Attribution Plan

**Status:** Prepared as the live phase after the full offline DEV comparison of Phi-only, external-only, and hybrid sourcebook retrieval. It does not assume Predictor V2 is the root cause.

## Purpose and scope

Attribute end-to-end quality loss after the offline sourcebook DEV screen and before training a new Predictor V2 architecture. The completed DEV comparison did not justify external or hybrid retrieval, so the current Arm D uses the Phi-only TRAIN retrieval/codebook path with the existing frozen scorer. Reconsider the external-sourcebook arm only after a later full DEV result shows a material opportunity gain at acceptable latency. This is a retrieval/codebook attribution check, not a new neural calibration bakeoff. Use the corrected canonical dataset and its pinned native chat-template, model/tokenizer revision, greedy decoding, EOS, and 1024-token ceiling contract. Use DEV only; do not open or score FINAL.

The primary comparison uses matched DEV prompts and one fixed predictive checkpoint/adapter. Keep prompts, base revision, tokenizer, rendered prompt, generation settings, evaluator, and runtime identical across arms. Freeze the prompt list and arm configuration before live execution. The preferred scope is all 135 DEV prompts (45 per domain). A nine-prompt wiring check (three per domain) may precede it, but is plumbing-only and cannot decide the bottleneck.

## Controlled arms

| Arm | Live condition | Purpose |
|---|---|---|
| A | Pure Vanilla Phi | Matched task-quality, termination, logits, and decode-step baseline. |
| B0 | Tokens wrapper around the same Vanilla Phi weights; no Step-100 checkpoint, adapters disabled, empty H codebook | Isolates wrapper/plumbing. This arm must pass exact output and same-prefix logit parity against A before any downstream causal result is interpreted. |
| B1 | Tokens wrapper plus the pinned Step-100 checkpoint; H disabled | Isolates trained-checkpoint effects after B0/B1 base-weight fingerprints and the exact changed-parameter inventory are verified. |
| C | **ORACLE CODEBOOK LIVE:** B1 plus a hindsight codebook derived from A's matched DEV continuation; generation remains free to emit H or base tokens | Measures live H realization from an occurrence-oracle codebook. It is not an offline compression ceiling. |
| CF | **FORCED ORACLE:** force an exact DP tiling of a known B1 continuation through H IDs, then expand it | Tests exact representation, semantic positions/cache advancement, and immediate continuation stability independently of the model's willingness to emit H. C and D remain blocked unless CF passes. |
| D | B1 plus the current Phi-only TRAIN retrieval/codebook path (`expanded_associations` at the provisional 1024-pool setting) | Measures current candidate/predictor/codebook behavior after the architecture, checkpoint, and forced-H gates pass. The completed sourcebook DEV screen did not justify carrying external or hybrid retrieval into live evaluation; revisit only after a later full DEV result shows a material opportunity gain at acceptable latency. |

Oracle codebook construction may inspect the matched Vanilla model continuation only. It must not use human reference answers or FINAL data. Preserve the oracle construction artifacts and hashes. C and CF are diagnostic hindsight conditions, not deployable predictors.

## Compression terminology and mandatory gates

- **ORACLE CEILING** is Oracle V2/DP compression of a known realized sequence.
- **CODEBOOK OPPORTUNITY** is DP savings possible on a sequence using the exact supplied phrase list.
- **ORACLE CODEBOOK LIVE** is C: the model sees hindsight phrases but chooses whether to emit H.
- **FORCED ORACLE** is CF: exact matching phrases are explicitly represented by H and expanded afterward.
- **LIVE REALIZED COMPRESSION** counts actual H spans emitted during generation. **END-TO-END SPEED** is measured separately from decode-call savings.

Before trusting GPU conclusions, CPU fixtures must pass for Oracle V2/DP, exact codebook recovery, opportunity-versus-realization accounting, generation/config routing, H masking, forced expansion, semantic-position advancement, split isolation, resume behavior, and compute-budget accounting. Then the small matched DEV run must pass all of these live gates:

1. **A/B0 token equality:** same tokenizer and native rendered Phi chat prompt, EOS IDs, dtype, greedy config, and token ceiling; exact generated token IDs, lengths, and termination must match.
2. **A/B0 logit parity:** compare native-vocabulary logits on deterministic prompt/prefix states. Record max/mean absolute difference, top-1 and top-k agreement, top-1 margin difference, and EOS/special-token differences. Top-1 must agree at every checked state and numerical differences must stay within the dtype-specific tolerance.
3. **B0/B1 parameter isolation:** record equal base-Phi fingerprints, verify B0 did not load the Step-100 checkpoint, and serialize every changed tensor name and shape for LoRA and H encoders in B1.
4. **CF exact expansion and state:** exact token round-trip; H semantic position and cache offset advance by the phrase's base-token span; compare unforced next-token choices immediately after each forced H. Any failure redirects to wrapper/H representation/state diagnosis and blocks C/D interpretation.

Use split-isolated DEV inputs. Reject mixed-split canonical artifacts before parsing continuation rows so this phase cannot read FINAL responses. No GPU continuation starts until the focused CPU suite, compile/import checks, and `git diff --check` pass. A failed A/B0 gate stops checkpoint/H attribution; a failed B0/B1 isolation gate stops checkpoint claims; a failed CF gate stops live H/predictor claims.

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

The full matched study has 135 prompts and five autoregressive arms (A, B0, B1, C, D), or **675 autoregressive generations**. CF adds up to 135 forced schedules based on the already-recorded B1 continuations. At a 1024-token ceiling, the conservative total bound is **829,440 decode calls** across those paths. The A outputs are reused to build C's hindsight codebook; the B1 outputs are reused as CF targets. Before expanding beyond the initial 12-prompt subset (4 per domain), record elapsed time, GPU-seconds, memory, failures, and cumulative project-budget use. Do not extrapolate a wall-clock promise from the offline candidate benchmark.

## Interpretation gate

- **A/B0 token or logit gate fails:** Stop causal attribution. Locate the wrapper/plumbing discrepancy before interpreting B1, C, CF, or D.
- **A and B0 pass, then B1 quality drops:** The trained checkpoint is implicated only if the B0/B1 changed-parameter inventory passes and the paired serving contract is equal.
- **B1 is healthy, but C degrades:** Active H use is implicated only after CF establishes exact expansion, semantic position/cache accounting, and immediate continuation stability.
- **CF fails:** Redirect to H representation, position, cache, or continuation-state mechanics. Do not blame the predictor.
- **CF passes and C has low realization relative to exact supplied-codebook opportunity:** H emission is weak; distinguish this from candidate coverage using the offline opportunity/realization ratio.
- **C behaves safely but D loses opportunity or quality:** Candidate generation/ranking becomes a supported hypothesis only after comparing D's exact supplied-codebook opportunity, live realization, and paired quality.
- **EOS, continuation-state, task-quality, or serving failures dominate:** Redirect work to that subsystem. Predictor V2 remains a leading hypothesis, not an assumed root cause.
- **EOS, continuation-state, or task-quality failures dominate:** Redirect to that subsystem rather than assuming Predictor V2 ranking will fix it.

Record the observed primary category as one of `candidate_generation`, `candidate_ranking`, `predictor`, `codebook`, `h_emission`, `representation`, `continuation_state`, `eos`, `serving`, or `other`, with evidence and an explicit pass/redirect decision. Do not select an architecture from this attribution run.

## Canonical research sequence and FINAL gate

1. Validate the corrected canonical 900-prompt dataset and provenance.
2. Construct retrieval indexes from TRAIN responses only.
3. Complete DEV candidate-recall and ORACLE CEILING analysis.
4. Run broader live Phi quality/compression attribution with A/B0/B1/C/CF/D; redirect if representation, H emission, continuation state, EOS, serving, or another subsystem dominates.
5. Only if predictor/codebook quality is confirmed as a major bottleneck and B1 is a viable serving target, train Predictor V2 architectures on TRAIN and compare/select a shortlist on DEV. The P-VANILLA versus P-LORA label-source diagnostic changes only the target-continuation source; defer scaling LoRA-target labels if B1 is not viable.
6. Evaluate at most the top two DEV candidates in a small live end-to-end integration run on DEV or a separately frozen integration subset.
7. Combine offline DEV and live DEV evidence to freeze the candidate generator, architecture, checkpoint, and configuration.
8. Evaluate FINAL once, only after the previous freezes and live integration gate pass, with explicit `--allow-final-eval` and a durable one-time claim. FINAL must never select or tune a system.
9. Run larger live end-to-end validation with the frozen system.

The machine-readable workflow state uses `candidate_generator_frozen`, `offline_architecture_shortlist_complete`, `live_integration_gate_passed`, `architecture_frozen`, and `final_evaluated`. FINAL evaluation requires the first four states to pass, the explicit CLI flag, and the one-time claim. `final_evaluated` records completion; it is not a selection prerequisite.

### Predictor target-source diagnostic

If B1 passes the quality and serving viability gate, compare P-VANILLA and P-LORA with identical TRAIN prompt IDs, predictor architecture, initialization seed, features, optimizer, training steps, candidate pool, K, phrase lengths, and ranking objective. Change only whether phrase targets come from matched Vanilla or H-disabled B1 continuations. If verified LoRA-target TRAIN continuations do not exist, start with a nested 100- or 200-prompt TRAIN subset and evaluate both predictors on the same Vanilla DEV and B1 DEV outputs. Do not train on DEV or FINAL responses. Scale to all 630 TRAIN prompts only if B1 remains viable, the subset shows a material opportunity gain on B1 outputs, and the compute budget supports it. If B1 is not viable, record LoRA-target predictor scaling as deferred.
