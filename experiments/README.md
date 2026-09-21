# experiments/ — Script Directory

> See `../docs/product.md` for the commercial goal (datacenter plug-and-play, frozen customer models).
> See `../RESEARCH_LOG.md` for full project context and history.
> See `../PREDICTIVE_HYPERTOKEN_STUDY.md` for the experimental design.

This directory contains all experiment scripts for the Predictive Hypertoken Calibration Study.

---

## Scripts

### Training

| Script | Purpose | Status |
|---|---|---|
| `train_pure_predictive_calibration.py` | Train `output_encoder` (Level 1 adaptation) | ✅ 100 steps complete |

**Run training** (resume from step 100):
```bash
python experiments/train_pure_predictive_calibration.py --max-steps 500 --resume-from experiments/checkpoints/pure_pred_k32_step100.pt
```

---

### Evaluation

| Script | Purpose | Status |
|---|---|---|
| `eval_60prompt_validation.py` | 5-condition × 60-prompt validation sweep | ❌ Not yet run |
| `compute_full_test_micro_compression.py` | Offline compression audit (7,512 samples) | ✅ Complete |
| `verify_prompt_guarantees.py` | Causality + round-trip verification | ✅ Complete |

**Run the main evaluation sweep**:
```bash
python experiments/eval_60prompt_validation.py
```
Results are saved incrementally to `experiments/checkpoints/eval_60prompt_results.json`.
Safe to interrupt and resume — already-completed (prompt, condition) pairs are skipped.

---

### Diagnostics (scratch/)

| Script | Purpose |
|---|---|
| `scratch/phase0_audit.py` | Full parameter inventory + hash verification of base model |
| `scratch/inspect_lora.py` | LoRA structure inspection (rank, targets, requires_grad) |

---

## Checkpoints

| File | Size | Contents |
|---|---|---|
| `checkpoints/cached_predictor.pkl` | 23 MB | Pre-trained phrase predictor |
| `checkpoints/pure_pred_k32_step50.pt` | 2.718 GB | output_encoder + AdamW optimizer at step 50 |
| `checkpoints/pure_pred_k32_step100.pt` | 2.718 GB | output_encoder + AdamW optimizer at step 100 |
| `checkpoints/pure_pred_k32_stageA_training_log.json` | ~20 KB | Full per-step training log |
| `checkpoints/eval_60prompt_results.json` | *(not yet created)* | 60-prompt validation sweep results |

### Checkpoint Format

```
{
    'step': int,
    'output_encoder_state_dict': ...,   # 226.5M fp32 params = ~906 MB
    'optimizer_state_dict': ...,        # AdamW m+v = ~1.812 GB  
    'loss': float,
}
```

**Why 2.718 GB?** The checkpoint stores only the `output_encoder` (our trained module) + AdamW optimizer state. The 3.8B base model is NOT in the checkpoint.

---

## Training Log Summary

From `checkpoints/pure_pred_k32_stageA_training_log.json`:

| Checkpoint | 3-Probe GSM8k Hypers | Decode Step Savings |
|---|---|---|
| Step 0 (zero-shot) | 1 | 1.54% |
| Step 50 | 2 | 3.03% |
| Step 100 | 3 | 4.48% |

Loss range: 1.3–4.2 (noisy, no plateau at step 100 → likely undertrained).
Average step time: ~27.65 s/step on CPU.

---

## Key Findings So Far

1. **EPFL LoRA already present**: The `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1` checkpoint contains a rank-32 LoRA across all 32 layers (50.33M params, frozen at load time). This is EPFL's original adaptation, not added by us.

2. **Base model confirmed unchanged**: MD5 hash verification shows Phi-3.5 base weights and EPFL LoRA matrices are identical before and after our 100-step training run.

3. **Only `output_encoder` was trained**: Our 100-step run is correctly Level 1 on the adaptation ladder (least invasive).

4. **Hypertoken emission is increasing**: 1 → 2 → 3 hypertokens on the 3-sample probe across steps 0, 50, 100. But this needs verification on the full 60-prompt set.

5. **Loss has not converged**: Training appears undertrained at 100 steps. Model likely benefits from 200–500 steps.
