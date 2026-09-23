# Qwen3-8B + vLLM Production Validation Plan (Sequencing Superseded)

**Plan date:** Tuesday, 2026-09-22  
**Target milestone:** Friday, 2026-09-25  
**Status:** Historical planning snapshot, retained for Qwen/vLLM scope and validation details. Its near-term schedule and phase ordering are superseded by the [canonical predictive-hypertoken roadmap](../experiments/RESEARCH_ROADMAP.md). Do not use this document to reorder the current Phi phases or to start Qwen work before the Phi stability gate. This document does not claim that its planned integrations or benchmarks have been completed.

This document preserves the original Qwen/vLLM validation scope. For current priorities and exact sequencing, follow the canonical roadmap. Historical Phi/Zip2Zip measurements remain recorded in [`RESEARCH_LOG.md`](../RESEARCH_LOG.md); they are not silently rewritten.

## 1. Business question

Does Tokens create incremental value on a realistic optimized serving stack—not merely beat an unoptimized model? Compare Qwen3-8B served with vLLM, including a compatible EAGLE3 configuration, against the same stack with Tokens. Value requires preserved task quality and lower real serving cost or latency, not just fewer nominal positions or decode steps.

The intended commercial shape is an installable runtime/plugin plus a small per-model adapter beside a customer's unchanged base model and existing serving engine. Replacing the server or fully retraining the customer's model is not the goal.

## 2. Scientific question

Measure separately whether Tokens can:

1. Reduce input/prefill work while preserving the model's task behavior.
2. Reduce actual target-model decode work with predictive hypertokens while preserving continuation quality and normal EOS termination.
3. Add value on top of EAGLE3, and whether the two methods can coexist without losing either method's benefit.

Measure continuation after a base-token phrase versus its hypertoken representation: KL divergence, top-1 agreement, top-5 overlap, correct-next-token probability, hidden-state similarity where useful, and multi-token continuation agreement. Distinguish position/token reduction from wall-clock, throughput, and GPU-seconds savings. A shortened, truncated, or broken answer is not a compute win.

## 3. Corrected Phi evidence and current confidence

The existing Phi/Zip2Zip results are historical evidence, not yet corrected reference results. The reported Phase 3 and K-sweep rankings must not guide a final K choice until the checkpoint loader and evaluation interface are audited and the small tests are rerun.

Code inspection for this roadmap found that the joint trainer saves separate nested checkpoint entries (`lora_state_dict`, `input_encoder_state_dict`, `output_encoder_state_dict`), while the Phase 3, K-sweep, Phase 6, and Phase 7 runners currently remap and pass the outer checkpoint mapping to `load_state_dict(..., strict=False)`. This may leave the trained submodules unloaded without making the failure obvious. Audit every affected consumer, load each sub-state explicitly into its corresponding module, and assert expected tensor/hash matches. Also assert Phi base weights remain unchanged. Until those checks pass, preserve but label downstream results provisional.

Do not present historical K-sweep numbers, Oracle-guided POC results, the old “best K” recommendation, or permissive quality scores as corrected findings. The original 60-prompt set remains historical validation evidence; it has already informed policy choices and is not a fresh holdout. See the audit entry in [`RESEARCH_LOG.md`](../RESEARCH_LOG.md) and the detailed history in [`PREDICTIVE_HYPERTOKEN_STUDY.md`](../PREDICTIVE_HYPERTOKEN_STUDY.md).

## 4. Known Phi evaluation issues and required fixes

| Issue | Required audit or correction |
|---|---|
| Nested Step-100 checkpoint may not load into POC runners | Load LoRA, input encoder, and output encoder sub-states explicitly. Assert loaded tensor values/hashes and unchanged base-weight hashes. Audit Phase 3, K-sweep, Phase 6, and Phase 7 consumers. |
| Training uses compressed prompts while a live comparison may use raw prompts | On the fixed 12-prompt subset, compare raw and correctly compressed prompts with the same checkpoint, codebook, prompts, and generation settings. Report quality, emissions, decode reduction, continuation KL, TTFT, and end-to-end latency. Keep train and serving representation aligned. |
| Predictive training targets omit EOS | Change the data contract to compressed response plus EOS; assert EOS is present in labels and evaluate stopping behavior. Do not launch a large Phi retrain just for this correction. Carry the corrected contract into Qwen. |
| MBPP interface may hide the required function name | Use a task-valid MBPP prompt/interface that gives the model the required function signature/name; rerun the small code result and actual tests. Do not grade against a hidden identifier. |
| Alpaca pass check is too permissive | Replace non-empty/word-count/repetition-only scoring as the primary quality measure with relevance, instruction adherence, completeness, topic drift, repetition, truncation, and EOS checks. Use deterministic task checks where practical and manual spot audits otherwise. |
| Repeated tuning on the original 60 prompts | Keep train, policy-tuning validation, and a fresh held-out slice separate. Freeze settings before accessing the fresh holdout; do not tune on it. |
| Seeded phrases are eligible at every decode position | Test a cheap contextual emission gate as a bounded experiment; do not make it a prerequisite for the basic prefill path or bake it into the architecture without evidence. |
| One hypertoken produces one model/KV state, not the states of its base-token expansion | Preserve continuation equivalence as a first-class quality gate; token expansion alone does not prove equivalent model behavior. |
| Many predicted slots may be unused | Historical validation reported roughly 76.9% dead slots. Treat this as provisional until the loader and evaluation are corrected, then measure again; do not assume predictor selection is the only bottleneck. |
| Quality-Aware Oracle is heuristic; OracleV2/ExactOracle prefilter candidates | Call these ranking/teacher heuristics, not perfect quality or unrestricted global oracles. Do not spend this week trying to perfect OracleV2. |

## 5. Why Qwen3-8B

Qwen3-8B is the week's fixed transfer target: it moves evaluation beyond the Phi research vehicle to a separate model family at a meaningful serving scale, without requiring full-model fine-tuning. The choice is a validation scope, not a claim that Qwen is already compatible, faster, or representative of every customer model. Pin the exact model revision and tokenizer before comparing results.

## 6. Why vLLM

vLLM is the first-class v1 deployment target because the product should extend an existing serving stack rather than ask a provider to replace it. Test against the actual serving path and record the exact supported integration surface; local `generate()` results are not production-stack validation. Verify prompt-embedding and extension/plugin APIs from current official documentation before implementation.

## 7. Why EAGLE3 is a required baseline

EAGLE3 is required here as the optimized decode baseline for the product question: does Tokens add value beyond a strong speculative-decoding configuration? This is a requirement for this validation, not an assertion that every deployment uses EAGLE3. Pin the speculator revision and prove compatibility with the selected Qwen/vLLM versions before reporting the combined condition.

## 8. Product architecture under test

```text
customer's Qwen / other supported model (base weights unchanged)
  + existing vLLM serving engine
  + existing quantization, KV cache, batching, and optionally EAGLE3
  + Tokens predictor, codebook/hypermodules, and runtime/plugin
```

The adapter may be calibrated per model family, but the desired path avoids a full 8B fine-tune. Loading/unloading Tokens should leave a usable vanilla path. Retain existing serving optimizations wherever technically compatible; document conflicts rather than assuming additivity.

## 9. Compatibility requirements

**Documentation check, 2026-09-22 (runtime untested):** vLLM's [prompt-embedding input guide](https://docs.vllm.ai/en/v0.30.0/features/prompt_embeds/) documents offline `(sequence_length, hidden_size)` inputs and online Completions/Chat Completions through `--enable-prompt-embeds`. Completions requires embeddings of the full chat-templated prompt; Chat Completions applies the chat template around embedded content. The [vLLM-Project Speculators serving guide](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/serve_vllm/) documents `Qwen/Qwen3-8B` with `RedHatAI/Qwen3-8B-speculator.eagle3`, `method: eagle3`, and three speculative tokens. Name that draft artifact explicitly in condition B; the guide alone does not establish a working pinned model/draft/vLLM/GPU combination. The [vLLM plugin guide](https://docs.vllm.ai/en/v0.30.0/design/plugin_system/) describes extension surfaces, but no general in-engine Tokens prefill callback has been verified. An external embedding-input smoke test is the first bounded C experiment. D and an in-server sidecar remain gated on a real compatibility test. No Qwen/vLLM runtime measurement has been made by this documentation check.

Before baselines, verify with current official vLLM and model documentation and record links, dates, versions, and exact syntax for:

- Qwen3-8B serving support and required model settings.
- Prompt-embedding input support and any shape/position/adapter constraints.
- Supported out-of-tree model, plugin, or runtime-extension hooks.
- Current EAGLE3 integration syntax and compatibility with the selected Qwen3-8B revision.
- Benchmark tooling, metrics, and supported quantization/cache/batching combinations.
- Kaggle environment, GPU availability, quota, and package/CUDA compatibility.

Do not infer compatibility from a related model, an old API, or a successful Transformers-only test. Freeze exact model, tokenizer, vLLM, CUDA/PyTorch, and speculator revisions in every run. For predictive decode, separately validate request-specific dynamic IDs, hyperembedding/logit handling, positions/RoPE spans, KV-cache behavior, output expansion, EOS, and continuation before any performance claim.

## 10. Benchmark matrix

| ID | Configuration | Priority |
|---|---|---|
| A | Qwen3-8B, vanilla vLLM | Required |
| B | Qwen3-8B + compatible EAGLE3 | Required |
| C | Qwen3-8B + Tokens input/prefill compression | Strong target |
| D | Qwen3-8B + EAGLE3 + Tokens input/prefill compression | Strong target |
| E | Qwen3-8B + Tokens predictive decode | Major architecture target; stretch for this week |
| F | Qwen3-8B + EAGLE3 + Tokens predictive decode | Stretch; report only if actually implemented and validated |

The primary comparison is A vs. C for prefill and B vs. D for additive prefill value; E vs. B for predictive decode, and F only when real coexistence has been tested. Never infer F from separate EAGLE and Tokens runs.

## 11. Metrics and workload

Use a fixed workload with short/long prompt crossed with short/long output. Warm up before measurement. Start at concurrency 1, then 2, 4, and 8; increase only if the instance supports it. Save per-request raw JSON, configuration, and logs.

- **Input:** base prompt tokens, effective positions, input MICRO reduction, predictor/codebook/hyperembedding time, and TTFT.
- **Decode:** base-equivalent output tokens, actual target-model passes, predictive hypertokens emitted, decode-step reduction, EAGLE proposals/acceptances, and TPOT/inter-token latency.
- **System:** end-to-end latency, tokens/sec, requests/sec, GPU utilization, peak VRAM, GPU-seconds/request, and concurrency scaling.
- **Quality:** task correctness and actual code tests; a standard MBPP-compatible interface; GSM8K correctness; instruction relevance/adherence; repetition; truncation; EOS termination; and continuation validity.

Report workload mix, warm-up, repetitions, aggregation, and confidence intervals where practical. Do not use “not empty” as a quality check, claim success from position reduction alone, or count broken/truncated output as savings.

## 12. Tuesday–Friday milestones

### Asynchronous execution and commit-linked evidence

Run long compatibility, training-smoke, and benchmark jobs asynchronously so independent work can proceed in parallel. Pin each job to an immutable checkout and record its exact tested commit SHA, model/checkpoint and environment versions, run ID, command/config, logs, raw results, and completion status. Continue independent tasks in another checkout while it runs; do not edit the pinned checkout. While a Phi run is pending, parallelizable work includes official vLLM/Qwen/EAGLE3 API review, workload and cache-key preparation, baseline-cache auditing, free-compute readiness checks, and static harness/evaluator correction. Serialize timing workloads that would contend for the same physical GPU. Wait only when the next decision (such as choosing K, promoting a candidate, or escalating a benchmark tier) depends on that result. Record the result under the tested SHA in the scoreboard/research log; put the tested SHA and the separate result-record commit in the report rather than amending the commit that was tested. The detailed Phi regression tiers are in [`PHI_CONTINUOUS_REGRESSION_BENCHMARK.md`](PHI_CONTINUOUS_REGRESSION_BENCHMARK.md).

### Tuesday, September 22 — short Phi audit and freeze

1. Audit and explicitly fix checkpoint loading across Phase 3/K-sweep and the affected Phase 6/7 runners; confirm expected trained tensors differ from upstream initialization, match the Step-100 checkpoint hashes/values, and frozen Phi base hashes remain identical.
2. Rerun only fixed K = 4, 8, 16, 24, 32 on the fixed 12 prompts.
3. Run the matched raw-vs-compressed prompt A/B on those prompts.
4. Correct EOS targets and stopping checks, MBPP interface, and instruction-quality review; do not scale-retrain Phi.
5. Freeze the corrected small-Phi conclusions and settings. Record corrected best K, quality, micro decode reduction, raw-vs-compressed prompt result, and remaining representation/continuation and code-domain limitations. Create a fresh validation slice, but do not tune against it.
6. Stop Phi policy optimization and move to Qwen.

### Wednesday, September 23 — official compatibility check and baselines

1. Verify the current official vLLM/Qwen/prompt-embedding/EAGLE3/plugin documentation and pin revisions.
2. Establish A and B with the fixed workload, concurrency schedule, warm-up, raw per-request JSON, and the full metric set.
3. Confirm quality and runtime stability before spending the remaining compute on integration.

### Thursday, September 24 — two distinct Tokens tracks

**Track 1: input/prefill.** Prompt → prompt-only predictor → codebook → exact segmentation → hyperembedding synthesis → vLLM prompt embeddings → ordinary Qwen decode. Ensure training and production use the same prompt representation. Implement/measure C and D if the official interfaces and baseline permit.

**Track 2: predictive decode.** Treat as a separate, higher-risk port. Before any Qwen3-8B training, gate on forward/backward, gradient flow, checkpoint save/reload, frozen base hashes, compressed-prompt use, EOS labels, at least one emitted predictive hypertoken, and coherent continuation. Validate request-specific dynamic IDs, input hyperembeddings, dynamic output vectors, position IDs/RoPE span handling, KV-cache correctness, output expansion, a small PEFT/LoRA calibration, and a quality-aware predictor. Use a meaningful, balanced training set across code, reasoning, and instruction/general tasks; roughly 100 examples are not enough to judge the architecture. A smaller Qwen3 variant is acceptable for mechanics. Do not assume Phi modules transfer.

Also test a cheap contextual emission gate (for example, first-token top-N/probability or a hyper-logit/constituent consistency threshold). This is a first-class experiment but is not a blocker for the basic prefill path. Use validation only to choose thresholds; report quality, emissions, decode savings, and added latency. Keep its overhead negligible compared with a transformer pass and do not bake it into the architecture without evidence.

### Friday, September 25 — production validation and decision

Run every condition that has passed its validity gates. A and B are required; C and D are strong targets; E is stretch; F is future/stretch and must not be faked. Answer the commercial questions in Section 16 and publish the results report described in Section 15, including conditions not completed and exact blockers.

## 13. Compute and resource limits

Use free compute only; prefer a Kaggle dual-Tesla-T4 runtime if currently available and authorized. Verify actual device count, memory, quota, and session limits immediately before dispatch. Two GPUs are separate memory devices, not a pooled 30 GB allocation; use only code that explicitly supports the selected multi-GPU arrangement. No paid cloud or API. Keep logs, checkpoints, progress counters, a watchdog, and an ETA for long jobs: investigate roughly 10 minutes without progress and stop/debug a run stalled around 20 minutes.

## 14. Stop conditions and priority if time slips

Stop rather than overclaim if checkpoint hashes do not match, quality falls as emissions rise, continuation fails, the serving integration is invalid, or the job is making no progress. Do not spend money, train on holdout, full-finetune Qwen3-8B, or claim speedup from nominal compression or untested EAGLE compatibility.

If time is short, prioritize in this order:

1. Correct Phi loader and small K sweep.
2. Correct evaluation interfaces/metrics and matched prompt A/B.
3. Qwen3-8B vanilla vLLM baseline.
4. Qwen3-8B + EAGLE3 baseline.
5. Tokens prefill, then EAGLE3 + Tokens prefill.
6. Predictive Qwen decode.
7. Contextual gate and joint EAGLE3 + predictive decode.
8. Further Oracle/predictor optimization.

## 15. Friday deliverables

Create `docs/QWEN3_VLLM_RESULTS.md` after real runs; do not create a blank results file in advance. Include hardware, software/model/tokenizer/vLLM/speculator revisions, benchmark method, prompt/output buckets, concurrency, raw timing data, quality, input/decode reduction, EAGLE acceptance, GPU-seconds/request, confidence intervals where practical, limitations, compatibility blockers, and next recommendation. Link raw JSON and logs. Update `RESEARCH_LOG.md` and this study with corrected Phase 7/K results only after the corrected runs and validity checks. Preserve historical Phi figures and identify which corrected runs supersede them.

Conclude with exactly one evidence-supported category:

A. **Additive production signal — proceed toward productization.**
B. **Promising but integration-limited — solve specific blocker.**
C. **Only beats unoptimized baseline — product thesis weakened.**
D. **No meaningful benefit — reconsider architecture.**

## 16. Friday commercial questions

1. Does Tokens reduce actual Qwen3-8B work in vLLM, and does that yield lower latency or GPU-seconds/request?
2. Is quality preserved across code, reasoning, and instruction workloads?
3. Does input compression add value on top of EAGLE3?
4. Does predictive decode beat EAGLE3 in any workload, and can the two coexist?
5. If coexistence is not demonstrated, what exact interface or technical conflict blocks it, does coexistence appear feasible, do EAGLE3 and Tokens compete for the same gain, and can Tokens still be additive on prefill?
6. How does value change with prompt/output length, concurrency, and domain? Is there a clear first workload?
7. Can a provider install this as a plugin/runtime plus adapter without replacing its engine? How much calibration is needed per model family, and is the artifact commercially small?

## 17. Known unknowns and post-Friday roadmap

Unknown until verified: current vLLM prompt-embedding and plugin surfaces; exact Qwen3-8B/EAGLE3 compatibility; resource fit on free hardware; whether prefill embeddings can preserve task quality; whether Tokens and EAGLE3 compose; whether predictive decode mechanics transfer from Phi; and whether nominal compute reductions translate to serving economics.

After Friday, use the evidence to choose a bounded next step: resolve a named integration blocker, validate the strongest additive prefill path on a fresh holdout, test continuation-consistency losses only with an ablation if needed, or reconsider the architecture if measured production value is absent. Learn continuation safety from base-vs-hypertoken probes over time; a future teacher should estimate expected step savings multiplied by empirically predicted continuation safety. Do not call the current handcrafted teacher a perfect oracle. Keep train, tuning validation, and final holdout isolated throughout.
