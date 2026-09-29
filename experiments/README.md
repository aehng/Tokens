# experiments/ — Script Directory

> See `../docs/product.md` for the commercial goal.
> See `../RESEARCH_LOG.md` for historical results and current status.
> See [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md) for the one canonical current research sequence.
> See `../PREDICTIVE_HYPERTOKEN_STUDY.md` for study history and implementation context.
> The Qwen/vLLM plan at [`../docs/QWEN3_VLLM_PRODUCTION_VALIDATION.md`](../docs/QWEN3_VLLM_PRODUCTION_VALIDATION.md) is a later-stage planning snapshot, not the immediate roadmap.

The current research direction and phase gates are defined only by
[`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md). The immediate Predictor V2 step
is a CPU/offline DEV comparison of the Phi-only baseline, a prompt-cleaned
external response sourcebook, and their hybrid at 256/512/1024 candidates.
The broader live Phi failure-attribution gate follows that evidence and must
confirm a predictor/codebook bottleneck before any architecture training.

The active quality-evaluation definitions, safety limits, score semantics,
generation-health fields, and cache/version rules are documented in
[QUALITY_BENCHMARK_METHODOLOGY.md](QUALITY_BENCHMARK_METHODOLOGY.md).

The historical pilot configuration and validation notes below remain references
for the predictive training pipeline. They do not override the canonical
roadmap or authorize a large training run.

For Phi quality regressions, follow the [tiered 3-way benchmark policy](../docs/PHI_CONTINUOUS_REGRESSION_BENCHMARK.md). Run meaningful-change benchmarks asynchronously from an immutable checkout pinned to the exact tested commit; continue independent work while they run, and wait only at tier-promotion or other result-dependent gates. Store each result keyed by the full tested commit SHA.

## Existing and Historical Scripts

### Training

| Script | Purpose | Status |
|---|---|---|
| `train_pure_predictive_calibration.py` | Historical output-encoder-only calibration | ✅ 100 steps complete; superseded |
| `train_predictive_zip2zip.py` | Joint predictive pilot (LoRA + input/output encoders) with exact resume | ✅ Implemented; Steps 0–200 completed |

### Evaluation and Verification

| Script | Purpose | Status |
|---|---|---|
| `tests/test_official_zip2zip_regression.py` | Official reactive Zip2Zip regression suite | ✅ 5/5 tests passed |
| `tests/test_predictive_pipeline_roundtrip.py` | Predictor pipeline causality & round-trip suite | ✅ 5/5 tests passed |
| `tests/test_reconstruction_loss.py` | Positional query reconstruction loss verification | ✅ 2/2 tests passed |
| `tests/test_curriculum_ranking.py` | Deterministic ranked curriculum verification | ✅ 4/4 tests passed |
| `tests/test_joint_training_smoke.py` | Base weight frozen hash & backward flow check | ✅ 2/2 tests passed |
| `experiments/test_continuation_equivalence.py` | Base-span vs hypertoken next-token continuation comparison | ✅ Fully operational (overall + semantic) |
| `experiments/run_smoke_generation_check.py` | 3-prompt (code, reasoning, instruction) real-generation test | ✅ Fully operational |
| `compute_full_test_micro_compression.py` | Offline compression audit | ✅ Complete |
| `verify_prompt_guarantees.py` | Causality, round-trip, and pre-prefill checks | ✅ Complete |

The standard training and evaluation commands are:

```bash
# Cumulative training with resume support
python experiments/train_predictive_zip2zip.py \
  --resume-from experiments/checkpoints/predictive_joint_pilot/checkpoint_step_150.pt \
  --target-steps 200 \
  --checkpoint-interval 50

# Continuation equivalence evaluation
python experiments/test_continuation_equivalence.py \
  --checkpoint experiments/checkpoints/predictive_joint_pilot/checkpoint_step_150.pt \
  --output experiments/checkpoints/predictive_joint_pilot/continuation_step_150.json

# Real generation smoke check
python experiments/run_smoke_generation_check.py \
  --checkpoint experiments/checkpoints/predictive_joint_pilot/checkpoint_step_150.pt \
  --output experiments/checkpoints/predictive_joint_pilot/smoke_step_150.json
```

## Planned Pilot Configuration

The configuration must specify:

- base model and codebook budget (`K=32` initially);
- predictor policy and category caps;
- trainable modules and LoRA targets;
- CE and reconstruction-loss weights;
- curriculum density;
- reactive/predictive training mix;
- batch size and sequence length;
- learning rate and optimizer;
- checkpoint and validation frequency; and
- hardware/precision settings.

The original base transformer must remain frozen. The first pilot should use
small mixed-domain data, conservative learning rates, frequent checkpoints, and
candidate checkpoints around steps 0, 100, 250, 500, 1,000, and 2,000.

## Required Validation Gates

The fixed smoke set contains four code, four reasoning/math, and four
instruction/general prompts. Every checkpoint must report:

- predictive hypertokens available and emitted;
- actual decode steps and base-equivalent output tokens;
- decode-step reduction;
- truncation, repetition, and output length;
- domain-appropriate correctness;
- CE and reconstruction losses; and
- continuation-equivalence metrics: KL, top-k agreement, and correct-next-token probability.

The first hard gate is one predicted hypertoken inside a complete valid answer,
with real skipped transformer steps and normal continuation afterward. The
second requires this on multiple prompts/domains. The third requires quality to
remain comparable to the base reference.

## Historical Artifacts

| File | Contents |
|---|---|
| `checkpoints/cached_predictor.pkl` | 23 MB prompt phrase predictor |
| `checkpoints/pure_pred_k32_step50.pt` | Historical output encoder + optimizer state |
| `checkpoints/pure_pred_k32_step100.pt` | Historical output encoder + optimizer state |
| `checkpoints/pure_pred_k32_stageA_training_log.json` | Historical per-step loss, gradient, and timing log |
| `true_hypertoken_decode_results.json` | Exploratory six-prompt live predictive results |

The old 100-step run increased hypertoken emission on a tiny probe, but the
answer could truncate or drift after a selected hypertoken. Those artifacts are
not evidence that the predictive architecture is ready for scale-up.

## Diagnostic and Architecture References

- `src/zip2zip/model.py` — model wrapper and generation lifecycle
- `src/zip2zip/codebook.py` — official dynamic codebook path
- `src/zip2zip/static_codebook.py` — seeded predictive codebook path
- `src/zip2zip/nn/embedding.py` — input hyperembedding
- `src/zip2zip/nn/linear.py` — output hyperprojection
- `src/evaluation/offline_segmenter.py` — exact DP segmentation
- `scratch/` and `experiments/scratch/` — exploratory diagnostics, not final gates
