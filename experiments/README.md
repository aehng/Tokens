# experiments/ — Script Directory

> See `../docs/product.md` for the commercial goal.
> See `../RESEARCH_LOG.md` for historical results and current status.
> See `../PREDICTIVE_HYPERTOKEN_STUDY.md` for the primary implementation plan.

The current research direction is joint predictive-hypertoken training. The
first gate is continuation correctness after a predicted hypertoken, not a
large compression percentage or a giant validation sweep.

## Current Execution Order

Do not launch a large training job automatically. The required order is:

1. Formalize the official reactive Zip2Zip regression.
2. Implement continuation-equivalence measurement.
3. Build and verify the prompt-only predictive training-data pipeline.
4. Implement joint input/output hyperencoder + LoRA training with CE and
   reconstruction losses.
5. Run a tiny CPU/unit smoke test and estimate GPU resources.
6. Run the small GPU pilot with fixed 12-prompt validation.
7. Run the frozen 60-prompt evaluation only after the 12-prompt gates pass.

## Existing and Historical Scripts

### Training

| Script | Purpose | Status |
|---|---|---|
| `train_pure_predictive_calibration.py` | Historical output-encoder-only calibration | ✅ 100 steps complete; superseded |
| `train_predictive_zip2zip.py` | Joint predictive pilot | ⏳ To be implemented |

Do not resume the old output-encoder-only run. Its checkpoint and training log
are retained as historical evidence of the continuation failure.

### Evaluation and Verification

| Script | Purpose | Status |
|---|---|---|
| `test_official_checkpoint.py` | Official reactive Zip2Zip regression | ⏳ Must be formalized and passed first |
| `evaluate_continuation_equivalence.py` | Base-span vs hypertoken next-token comparison | ⏳ To be implemented |
| `prepare_predictive_training_data.py` | Prompt-only codebook and exact round-trip pipeline | ⏳ To be implemented or consolidated |
| `evaluate_predictive_checkpoint.py` | Fixed 12-prompt checkpoint evaluation | ⏳ To be implemented |
| `eval_60prompt_validation.py` | Frozen 60-prompt comparison | ⏸ Blocked until 12-prompt gates pass |
| `compute_full_test_micro_compression.py` | Offline compression audit | ✅ Complete; not the current acceptance gate |
| `verify_prompt_guarantees.py` | Causality, round-trip, and pre-prefill checks | ✅ Complete |
| `true_hypertoken_decode.py` | Exploratory live predictive decode | ⚠ Diagnostic only; not an acceptance evaluation |

The intended commands are:

```bash
python experiments/train_predictive_zip2zip.py \
  --config configs/predictive_joint_pilot.yaml

python experiments/evaluate_predictive_checkpoint.py \
  --checkpoint <path>
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
