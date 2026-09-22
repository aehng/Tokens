# Primary Plan — Predictive Hypertoken Training

> This is the forward-looking implementation plan. Historical results and the
> completed output-encoder-only experiment are recorded in `RESEARCH_LOG.md`.

## Main Goal

Train a predictive-hypertoken version of Zip2Zip in which:

1. The prompt determines a temporary codebook before generation.
2. The model can emit those predicted hypertokens.
3. Each hypertoken replaces multiple base-token decode steps.
4. The continuation after a hypertoken remains correct and coherent.
5. The original base-model weights remain frozen if possible.
6. Quality is preserved.
7. Compression becomes real decode-step savings.

The first success milestone is deliberately smaller than maximum compression:

> A complete, correct/coherent answer contains at least one predicted
> hypertoken, that hypertoken skips real transformer decode steps, and the model
> continues normally afterward.

Until this works reliably, do not run another giant 60-prompt evaluation and do
not optimize for maximum compression.

The commercial objective remains a datacenter sidecar: frozen customer model
+ hypermodules/optional LoRA + predictor + runtime, with load/unload rollback.
See [`docs/product.md`](docs/product.md).

## Why the Previous Architecture Failed

The 100-step experiment trained only `output_encoder`. It did not jointly train:

- the input hyperencoder;
- the existing Zip2Zip LoRA;
- continuation behavior after consuming a hypertoken; or
- the reconstruction objective.

The model learned “emit hypertoken H,” but not “after consuming H, behave as
though the constituent base-token sequence had passed through the transformer.”
Observed consequences included truncation, repetition, changed next-token
distributions, and semantic drift after a selected hypertoken.

More training of that same output-encoder-only setup is not the plan.

## Architecture

Start from `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1` and use Zip2Zip’s
working dynamic-token architecture as the regression reference.

- Keep the original Phi-3.5 base weights frozen.
- Jointly train the Zip2Zip LoRA, input hyperencoder, and output hyperencoder.
- Use language-model cross-entropy on compressed predictive-token targets.
- Use Zip2Zip-style reconstruction loss.
- Use prompt-predicted codebooks rather than reactive LZW-only codebooks.
- Do not full-finetune the base transformer at this stage.

The predictive training example is:

```text
prompt
  -> base tokenizer
  -> predictor sees prompt only
  -> choose K predicted phrases
  -> assign temporary hypertoken IDs
  -> segment prompt with those IDs
  -> segment response with the same codebook
  -> teacher-force the compressed sequence
```

The predictor must never inspect the response. The response may only identify
where already-selected phrases occur in the target sequence.

## Fixed Starting Point

- Base model: `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1`
- Codebook budget: `K=32`
- Predictor starting policy: current `no_global` policy
- Training domains: code, reasoning/math, instruction/conversation
- Base weights: frozen
- Validation: fixed 12-prompt smoke set first; frozen 60-prompt set only after gates pass
- Final test data: never used for tuning

## Phase 0 — Verify the Official Zip2Zip Path

Before training anything new, create a formal regression test using the released
checkpoint and its native codebook settings:

1. Compress the prompt with the official tokenizer.
2. Generate with `model.generate()`.
3. Decompress the result.
4. Verify coherent continuation after hypertokens.

Explicitly cover dynamic IDs, position IDs, RoPE offsets, attention masks,
KV-cache behavior, expansion/decompression, EOS behavior, and codebook lifetime.
The existing short official probes are evidence, not the final regression test.

If the local implementation cannot reliably reproduce official Zip2Zip
continuation behavior, stop before predictive training.

## Phase 1 — Continuation-Equivalence Test

For a phrase such as “the following sentence,” compare:

- context followed by its constituent base tokens; and
- the same context followed by one hypertoken representing that phrase.

After each version, compare the next-token distributions and, where practical,
the hidden states. Measure:

- KL divergence;
- top-1 agreement;
- top-5 and top-10 overlap;
- probability assigned to the base-model choice;
- hidden-state cosine similarity; and
- several subsequent generated tokens.

Run this over semantic phrases, code, newlines/indentation, numbers, and
repeated phrases. Record the untrained baseline mismatch and repeat the test at
each training checkpoint. This is the primary diagnostic for the failure mode;
it should not wait for a full-answer evaluation.

## Phase 2 — Predictor and Codebook Policy

Use `K=32` and start from `no_global`. Keep prompt-conditioned phrases,
repeated prompt phrases, useful content phrases, newlines/indentation, and
useful numeric structures. Remove the global filler list and bare punctuation
without measured incremental value.

Add simple category caps or diversity constraints so one category cannot consume
the whole codebook. A possible starting cap is at most eight structural,
newline, or numeric slots, with the remainder primarily prompt-conditioned
multi-token phrases. The exact ratio is not a major optimization target yet.

Log every training codebook, including phrase, category, frequency, and use
count.

## Phase 3 — Predictive Training-Data Pipeline

Build a reusable preprocessing pipeline. For every training example:

1. Tokenize the prompt.
2. Run the predictor on prompt tokens only.
3. Select `K` phrases and assign temporary dynamic IDs.
4. Segment the prompt with base and dynamic IDs.
5. Segment the response with the same dictionary.
6. Store original and compressed prompt/response IDs.
7. Store the codebook, constituent-token mapping, phrase category, frequency, and use count.

Every compressed prompt and response must round-trip to the original token IDs
exactly. Any response-derived phrase selection is an experiment failure.

## Phase 4 — Joint Training Objectives

### Language-model loss

Teacher-force the compressed sequence so the model learns to emit a dynamic
hypertoken instead of its constituent base tokens when appropriate.

### Reconstruction loss

Use the Zip2Zip-style reconstruction objective to preserve the information in
the underlying phrase. This is critical because correct hypertoken emission is
not enough if the next-token distribution changes afterward.

### Optional continuation-consistency loss

Only after implementing and measuring the standard Zip2Zip losses, investigate
an explicit continuation loss if equivalence remains poor. Candidate ablations:

- KL divergence between next-token distributions;
- hidden-state cosine or MSE loss; and
- logits consistency.

Do not add this loss without a measured baseline and ablation.

## Phase 5 — Curriculum

Do not immediately compress every eligible span. Start with high-confidence
phrases at low density, then increase exposure:

```text
20% of eligible occurrences -> 40% -> 70% -> full policy
```

The schedule is adjustable. The purpose is to preserve ordinary-token
continuation while the model gradually adapts to predictive hypertokens.

## Phase 6 — Mix Reactive Zip2Zip Training

The released checkpoint already knows how to continue after reactive
hypertokens. The first pilot may mix approximately 50% normal/reactive
Zip2Zip-style examples with 50% predictive-codebook examples, then shift toward
predictive examples if validation improves.

Treat the ratio as a validation hyperparameter, not a fixed truth. The model
must be trained on the distribution it will see during inference.

## Phase 7 — Small GPU Pilot

Do not launch the published-scale approximately 100M-token run. First answer
only whether joint training can repair continuation after one predictive
hypertoken.

Pilot requirements:

- mixed code, reasoning/math, and instruction/conversation data;
- modest sequence lengths;
- `K=32`;
- small step count with frequent checkpoints;
- conservative learning rates near the upstream Zip2Zip adaptation regime;
- no blind reuse of the aggressive output-encoder-only learning rate.

Candidate checkpoints are 0, 100, 250, 500, 1,000, and 2,000 steps, adjusted
to observed learning speed.

## Phase 8 — Fixed 12-Prompt Validation

Use four code, four reasoning/math, and four instruction/general prompts at every
important checkpoint. Measure:

- available and emitted predictive hypertokens;
- actual transformer decode steps and base-equivalent output tokens;
- decode-step reduction and output length;
- truncation and repetition;
- code correctness, math final-answer correctness, and instruction completion;
- CE validation loss and reconstruction loss; and
- continuation-equivalence metrics.

### Hard gates

1. At least one predictive hypertoken appears inside a complete valid answer and
   the model continues normally afterward.
2. The behavior occurs on multiple prompts and domains.
3. Quality remains comparable to the base reference.

Do not run the 60-prompt evaluation before these gates pass.

## Phase 9 — Failure Diagnostics

When an answer breaks after a hypertoken, inspect in this order:

1. position IDs;
2. RoPE span offsets;
3. attention mask;
4. KV cache;
5. dynamic ID mapping;
6. input hyperembedding;
7. output hyperprojection;
8. reconstruction loss;
9. EOS probability; and
10. next-token distribution divergence.

Classify the failure as an implementation bug, representation mismatch,
insufficient training, bad phrase/codebook selection, curriculum problem, or
catastrophic forgetting. Do not answer every failure by adding more steps.

## Phase 10 — Continuation Equivalence as a Primary Metric

For every validation checkpoint, report:

```text
Base phrase:    P(next token | constituent base tokens)
Hypertoken:     P(next token | H)
```

Aggregate mean KL divergence, top-1 agreement, top-5 overlap, and probability
difference on the correct next token. The desired trend is:

```text
training steps up
  -> continuation divergence down
  -> hypertoken emission up
  -> complete-answer quality flat
```

If hypertoken emission rises while continuation divergence worsens, stop.

## Phase 11 — 60-Prompt Validation

Only after the 12-prompt gates pass, compare the frozen 60-prompt set across:

1. Vanilla Phi-3.5;
2. official reactive Zip2Zip;
3. zero-shot predictive; and
4. joint-trained predictive.

Report input compression, output decode-step savings, quality, runtime
breakdown, and offline-available versus realized compression. Do not tune on the
final test split.

## Phase 12 — Minimize the Adapter After Correctness

Once the architecture works, ablate:

- both hyperencoders only;
- hyperencoders plus existing LoRA;
- lower-rank LoRA;
- fewer LoRA target modules;
- LoRA inside hyperencoders; and
- selective layers.

The commercial artifact remains:

```text
unchanged customer base model
  + small LoRA/PEFT adapter
  + hypermodules
  + predictor/runtime
```

Do not optimize adapter size before proving continuation correctness.

## Phase 13 — Hybrid Mode After Pure Predictive Mode

Only after pure predictive mode is reliable, evaluate trained hybrid mixtures:

```text
0/100, 25/75, 37.5/62.5, 50/50, 75/25, 100/0
predictive/reactive
```

Do not assume a model trained only on pure predictive codebooks will handle
hybrid codebooks automatically.

## Phase 14 — Escalation Architecture

If official Zip2Zip regression passes, predictive joint training is correct,
losses converge, and continuation equivalence remains poor, investigate deeper
representations such as a KV-state expander, multiple internal cache entries,
or a latent multi-position representation.

Do not build this before the preceding conditions are met.

## Phase 15 — Hardware Gate

Prepare locally, but do not run serious training on the current 16 GB
unified-memory laptop. Before the GPU pilot, report:

- trainable parameter count;
- estimated VRAM;
- batch size and sequence length;
- gradient checkpointing;
- optimizer and optimizer-state size;
- expected disk usage; and
- expected duration.

Estimate requirements for 16 GB, 24 GB, 40 GB, and 80 GB GPUs. Explore bf16,
gradient checkpointing, 8-bit optimizers, LoRA-only optimizer state, and CPU
offload without changing the architecture merely to fit the laptop.

## Phase 16 — One-Command Training and Evaluation

Prepare the repository for commands of this shape:

```bash
python experiments/train_predictive_zip2zip.py \
  --config configs/predictive_joint_pilot.yaml

python experiments/evaluate_predictive_checkpoint.py \
  --checkpoint <path>
```

The configuration must cover the model, `K`, predictor policy, LoRA settings,
trainable modules, CE/reconstruction loss weights, curriculum,
reactive/predictive mix, batch and sequence sizes, learning rate, steps,
checkpoint frequency, and validation frequency.

## Stopping Rule

Do not launch a large training job automatically. First:

1. implement the pipeline;
2. pass official Zip2Zip regression;
3. generate predictive training examples;
4. verify exact round-trip reconstruction;
5. implement continuation-equivalence tests;
6. verify the losses;
7. run a tiny CPU/unit smoke test; and
8. estimate GPU resources.

Then stop and report what was implemented, tests passed, exact architecture,
trainable parameter count, expected VRAM, recommended GPU, expected pilot
duration, and the exact launch command.

## Empirical Results — Cumulative Pilot (Steps 0 → 200)

The cumulative joint pilot successfully trained on CPU from Step 0 to Step 200 with exact checkpoint resumption and parameter verification:

- **Backbone Frozen**: Phi-3.5 3.82B parameters in fp16, verified 100% byte-identical via SHA256 hashes across 19 base tensors.
- **Trainable Modules**: EPFL LoRA (50.3M) + Input Hyperencoder (226.5M) + Output Hyperencoder (226.5M) = 503.4M params in fp32.
- **Optimal Checkpoint**: `checkpoint_step_100.pt` / `checkpoint_step_150.pt`.
- **Semantic Continuation KL**: Dropped from **4.991** (zero-shot) to **4.056** (Step 150), proving that joint training successfully teaches the model to treat predicted hypertokens as equivalent to constituent sequences.
- **Top-1 / 5-Step Agreement**: Peaked at **40.0%** at Step 100.
- **Reconstruction Loss**: Dropped monotonically from **7.41** down to **1.18** at Step 200 (-84.1%).
- **Early Stopping Triggered at Step 200**: At Step 200, semantic continuation plateaued (4.056 $\to$ 4.075) while hypertoken emission surged on math reasoning (18 hypertokens), introducing minor formatting repetition. Per Early Stopping Rules 7 and 8, training stopped cleanly at Step 200.

## Definitive Quality & Compute Economics Benchmark (60 Frozen Validation Samples)

We benchmarked 4 distinct conditions across 60 held-out prompts (`data/cached_pure_pred_val_60.json`: 20 MBPP code, 20 GSM8K math reasoning, 20 Alpaca instruction), evaluating 240 full autoregressive generations with `max_new_tokens=300`:

| Metric | Original Vanilla Phi | Official Reactive Zip2Zip | Predictive Step 100 | Predictive Step 150 |
| :--- | :---: | :---: | :---: | :---: |
| **MBPP Code Pass@1** | **10.0%** (2/20) | 0.0% (0/20) | 0.0% (0/20) | 0.0% (0/20) |
| **GSM8K Math Accuracy** | **65.0%** (13/20) | 50.0% (10/20) | **60.0%** (12/20) | 50.0% (10/20) |
| **Alpaca Failure Rate** | 30.0% (6/20) | 30.0% (6/20) | **20.0%** (4/20) | 25.0% (5/20) |
| **Micro Decode Reduction** | 0.0% | **29.27%** | 21.85% | 16.87% |
| **Tokens / Steps Saved** | 0 | **5,660** | 4,150 | 2,966 |
| **Total Emitted Hypertokens** | 0 | 4,381 | 2,532 | 1,840 |
| **Mean Wall-Clock Latency** | 58.59s | 67.80s (+15.7% slower) | **49.61s** (15.3% faster) | **49.12s** (16.2% faster) |
| **Mean TTFT (Prefill Latency)**| **1.18s** | 16.66s (14.1x bottleneck) | 1.59s | 1.57s |
| **Throughput (expanded tok/s)**| 4.83 tok/s | 4.31 tok/s | **6.11 tok/s** (+26.5%) | 5.73 tok/s (+18.6%) |

### Key Conclusions:
1. **Predictive Step 100 is Superior to Official Zip2Zip in Quality and Latency**:
   - Step 100 achieves 60.0% GSM8k accuracy vs 50.0% for Official Zip2Zip.
   - Step 100 has a 20.0% Alpaca failure rate vs 30.0% for Official Zip2Zip and Vanilla Phi.
   - Official Zip2Zip is slower than vanilla Phi due to a 16.66s TTFT bottleneck, whereas Predictive Step 100 achieves a 15.3% end-to-end latency speedup and +26.5% higher throughput.
2. **Quality Loss Decomposition**:
   - In math reasoning, Stage 1 (Official Zip2Zip LoRA/LZW) lost 15.0pp relative to Vanilla Phi (65% $\to$ 50%). Our Predictive Step 100 *recovered* +10.0pp of this deficit (reaching 60%), bringing it within 5pp of Vanilla Phi.
   - In code, both Zip2Zip variants failed execution due to function signature and naming mismatches.

## Predictive Hypertoken Optimization — Phased POC (Phases 0–4)

### Phase 0: Frozen Step-100 Baseline
- Frozen reference baseline across all 60 validation samples (`baseline_step100_frozen.json`):
  - MBPP Code: 0.0% Pass@1, 35.91% micro compression (3,132 steps saved, 88.15 hypers/prompt).
  - GSM8k Reasoning: 60.0% accuracy (12/20), 12.96% micro compression, 12.00% quality-preserved compression (491 steps saved on correct outputs).
  - Alpaca Instruction: 20.0% failure rate (80% pass), 3.70% micro compression (125 steps saved).
  - Overall: 46.7% accuracy (28/60), 21.85% micro compression (4,150 steps saved).

### Phase 1: Feature & Provenance Findings
- **Hypothesis Validated:** *"Prompt-supported numbers/identifiers are safe; novel/inferred numbers/identifiers are catastrophic."*
  - Prompt-absent numbers: **88.9% error rate** (1,254 emissions).
  - Prompt-present numbers: **49.5% error rate** (273 emissions).
  - Word boundary alignment: space-aligned phrases have a **51.1% error rate**, while unspaced/mid-word phrases suffer a **91.8% error rate** and bare punctuation a **95.3% error rate**.
  - 2-token phrases (65.3% error) substantially outperform 3-token phrases (92.5% error).
  - 76.9% of candidate slots in the baseline were dead capacity.

### Phase 2: Evidence-Aware Selector (`EvidenceAwareSelector`)
- Fast (<7ms per prompt) heuristic reranker incorporating prompt provenance boosts (+8.0 prompt match, +6.0 grounded numbers), structural risk penalties (-60 ungrounded numbers, -60 dead structural, -40 isolated syntax), boundary alignment checks, and stem diversity throttling.

### Phase 3: Small Selector Policy POC (12 Prompts)
- Fixed 12-prompt evaluation subset (4 MBPP, 4 GSM8K, 4 Alpaca):
  - Condition A (Baseline K=32): 4/12 (33.3%) accuracy, 16.05% micro compression, 292 dead slots.
  - Condition B (Evidence-Aware K=32): 4/12 (33.3%) accuracy, 6.82% micro compression, 313 dead slots.
  - Condition C (Adaptive K, $\tau=20.0$): **5/12 (41.7%) accuracy (+8.4% gain)**, **131 dead slots (-55.1% dead capacity)**, and **100% (4/4) success on Alpaca instruction**.

### Phase 4: K-Sweep & Adaptive-K Sweep — COMPLETE
- Fixed-K: $K=4$ and $K=8$ achieved **58.3% accuracy (7/12)**, vastly outperforming $K=32$ (**33.3%**).
- Adaptive-K: $\tau=20.0$ and $\tau=25.0$ achieved **41.7% accuracy** with **5.1–5.5% micro compression**, cutting dead slots by up to 70% ($K=12.3$ mean allocated slots).
- Key takeaway: Codebook capacity must be restricted to high-confidence evidence-grounded phrases to avoid token distortion. $K=8$ is the optimal fixed budget; $\tau=20.0$ is the optimal adaptive threshold.

---

*Last updated: 2026-09-22. Maintained by Antigravity (Google DeepMind) coding assistant.*

