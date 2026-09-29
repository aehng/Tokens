# Predictor V2 Training and Evaluation Status

Canonical dataset: **validated** (630 TRAIN / 135 DEV / 135 FINAL)
Predictor/codebook bottleneck: **not yet assessed**

| Stage | Status | Artifact |
|---|---|---|
| Dev Sourcebook Comparison | measured | `experiments/results/predictor_v2_sourcebook_dev.json` |
| Candidate Recall Oracle | measured | `experiments/results/predictor_v2_candidate_recall.json` |
| Broader Live Phi Attribution | not-yet-run | `experiments/results/predictor_v2_quality_attribution_gate.json` |
| Candidate Plan | not-yet-run | `experiments/results/predictor_v2_candidate_plan.json` |
| Train Dev Architecture Bakeoff | not-yet-run | `experiments/results/predictor_v2_architecture_bakeoff.json` |
| Offline Architecture Shortlist | not-yet-run | `experiments/results/predictor_v2_architecture_shortlist.json` |
| Integration Subset | full-dev | `None` |
| Live Integration Gate | not-yet-run | `experiments/results/predictor_v2_live_integration_gate.json` |
| Candidate Generator Freeze | not-yet-run | `docs/predictor_v2_candidate_generator_freeze.json` |
| Architecture Freeze | not-yet-run | `docs/predictor_v2_architecture_freeze.json` |
| Final Evaluation | not-yet-run | `experiments/results/predictor_v2_final_result.json` |

## Offline sourcebook comparison

| Candidate source | Pool | K=32 capture interval | p50/p90/p99 retrieval (ms) | Code | Reasoning | Instruction |
|---|---:|---:|---:|---:|---:|---:|
| external_only | 1024 | 0.301–0.560 | 6581.883/11503.539/14795.442 | 0.310–0.684 | 0.274–0.427 | 0.321–0.590 |
| external_only | 256 | 0.234–0.419 | 6708.029/11710.783/15458.054 | 0.242–0.499 | 0.204–0.318 | 0.258–0.458 |
| external_only | 512 | 0.271–0.494 | 6517.569/11297.512/14742.993 | 0.280–0.590 | 0.240–0.374 | 0.295–0.538 |
| hybrid | 1024 | 0.321–0.597 | 6586.851/11510.864/14804.577 | 0.319–0.706 | 0.323–0.503 | 0.321–0.591 |
| hybrid | 256 | 0.256–0.456 | 6712.265/11719.048/15466.992 | 0.255–0.521 | 0.255–0.396 | 0.259–0.459 |
| hybrid | 512 | 0.292–0.533 | 6523.744/11304.825/14752.509 | 0.292–0.617 | 0.289–0.451 | 0.295–0.542 |
| phi_only | 1024 | 0.406–0.732 | 4.213/6.341/9.917 | 0.366–0.731 | 0.564–0.941 | 0.264–0.450 |
| phi_only | 256 | 0.359–0.626 | 4.038/6.650/9.287 | 0.324–0.632 | 0.494–0.779 | 0.241–0.410 |
| phi_only | 512 | 0.400–0.717 | 4.283/6.340/9.281 | 0.364–0.725 | 0.550–0.907 | 0.263–0.448 |

## Machine-readable gates

```json
{
  "architecture_frozen": false,
  "candidate_generator_frozen": false,
  "final_evaluated": false,
  "live_integration_gate_passed": false,
  "offline_architecture_shortlist_complete": false
}
```

## Pending steps

- Broader live Vanilla/Predictive Phi attribution has not been recorded.
- No provisional candidate plan has been selected from DEV evidence.
- TRAIN-only Predictor V2 training and DEV comparison have not been run.
- A one- or two-candidate DEV architecture shortlist has not been frozen.
- Small live end-to-end validation of the shortlisted candidates has not passed.
- Candidate generator cannot freeze until the shortlisted system passes live integration.
- Architecture cannot freeze until the live integration gate passes.
- FINAL remains unavailable until the candidate, shortlist, live, and architecture gates pass.

FINAL is excluded from candidate and architecture selection and remains a one-time evaluation behind `--allow-final-eval`.
