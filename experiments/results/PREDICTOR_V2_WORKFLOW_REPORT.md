# Predictor V2 Training and Evaluation Status

Canonical dataset: **validated** (630 TRAIN / 135 DEV / 135 FINAL)
Predictor/codebook bottleneck confirmed: **False**

| Stage | Status | Artifact |
|---|---|---|
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
