# Predictor V2 Training and Evaluation Architecture

This is the downstream harness for the corrected canonical Vanilla Phi continuation artifact. It does not modify how Vanilla prompts are formatted, how generation terminates, or the active EOS diagnostic. It uses the strategy names already implemented by `ConfigurableCandidateGenerator`: `baseline`, `expanded_associations`, `suffix_conditioned`, and `sparse_lexical`.

## Workflow

```text
Correct canonical 900 Vanilla dataset + independent ID/split inventory
    ↓ validate schema, revisions, generation config, provenance, and hashes
TRAIN / DEV / FINAL views
    ↓ TRAIN only
TRAIN association index with source/config/code provenance
    ↓ DEV only
Candidate recall + oracle analysis across strategies and pool sizes
    ↓ broader live Vanilla vs Predictive Phi baseline and failure attribution
Redirect if another subsystem dominates; continue only if Predictor/codebook quality is a major bottleneck
    ↓ TRAIN fitting + DEV comparison
Predictor V2 architecture bakeoff, then a one- or two-candidate DEV shortlist
    ↓ small live end-to-end integration check on all DEV or an exactly frozen DEV subset
Combine offline DEV and live DEV evidence; freeze candidate generator and architecture/config/checkpoint
    ↓ explicit --allow-final-eval, all upstream gates bound, one claim only
FINAL evaluation once; never use FINAL for selection
    ↓
Larger live end-to-end validation with the frozen system
```

The contract validator reads the JSONL for schema and integrity checks. Training uses TRAIN; candidate and architecture selection use DEV; the broad live attribution uses matched live DEV Vanilla/Predictive Phi runs. The Predictor V2 live integration check tests actual H emission, continuation health, task quality, termination behavior, and decode-step savings for the shortlisted models. A candidate occurrence or capture in offline text is not evidence that decode will emit H tokens or preserve a healthy live trajectory.

The FINAL record view requires a permit bound to the attribution gate, provisional candidate plan, post-live candidate freeze, DEV shortlist, passed live integration gate, architecture freeze, and canonical dataset. No candidate or architecture selection code computes FINAL metrics.

## Hypothesis and redirect gate

Predictor V2 is a leading hypothesis, not an assumed root cause. Before architecture training, record broader live Phi baseline and failure-attribution evidence covering candidate generation/ranking, codebook, H emission, representation, continuation state, EOS/termination, serving, and other observed failures. The evidence must bind to DEV and compare matched Vanilla and Predictive Phi conditions.

If attribution identifies candidate generation, ranking, predictor, or codebook quality as a major bottleneck, the gate passes and the TRAIN/DEV bakeoff may proceed. If H emission, representation, continuation state, EOS, serving, or another subsystem dominates, the gate records `redirected` and names the destination; the Predictor V2 trainer rejects that gate. Do not infer a pass from offline oracle capture alone.

## Canonical input contract

The canonical JSONL has one JSON object per prompt with these required fields:

| Field | Requirement |
|---|---|
| `prompt_id` | Unique, non-empty string in the independently prepared prompt inventory. |
| `domain` | Member of the manifest's allowed domain list. |
| `split` | `TRAIN`, `DEV`, or `FINAL` (lowercase spellings are normalized). |
| `task_prompt_text` | Original task prompt, non-empty. |
| `rendered_prompt_text` | Exact rendered chat prompt passed to the pinned tokenizer, non-empty. |
| `continuation_text` | Vanilla continuation text. |
| `continuation_token_ids` | Non-empty integer IDs within the pinned tokenizer vocabulary. |
| `generated_token_count` | Exactly the length of `continuation_token_ids`. |
| `termination_reason` | Non-empty string. Stop/EOS reasons require a configured stop token; cap reasons must match the configured cap. |
| `termination_token_id` | Null or a valid vocabulary ID; when present it must be the final generated ID. |
| `model_id`, `model_revision` | Same values on every row and equal to the manifest. |
| `tokenizer_id`, `tokenizer_revision` | Same values on every row and equal to the manifest. |
| `generation_config` | Same object on every row and equal to the manifest; `max_new_tokens` is read dynamically. |
| `generation_metadata` | Per-row provenance object. |

The companion manifest uses schema `predictor_v2_canonical_manifest_v1`. It binds the raw JSONL SHA-256, the expected prompt IDs and split assignment, allowed domains, model/tokenizer revisions, tokenizer vocabulary size, generation config, source provenance, and self/provenance hashes. The expected ID list and split assignment must come from the prompt inventory prepared independently of the generated continuations. Do not create them by copying IDs from the artifact being checked.

Create that sidecar after the corrected JSONL and independent inventory/provenance files are ready:

```powershell
python experiments/make_predictor_v2_manifest.py --dataset data/canonical_phi_continuations.jsonl --split-inventory <independent-prompt-split-inventory.json> --provenance <canonical-generation-provenance.json> --tokenizer-vocab-size <pinned-tokenizer-vocab-size> --out data/canonical_phi_continuations.manifest.json
python experiments/validate_predictor_v2_dataset.py --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json
```

The inventory JSON contains `expected_prompt_ids`, `split_ids` with exactly `TRAIN`, `DEV`, and `FINAL`, and `allowed_domains`. The provenance JSON should identify the upstream generation run and its source/code hashes. The manifest command takes revisions/config from the first record; the validator then rejects any row that differs.

Each saved TRAIN index also records hashes for the index builder and its canonical-data/retrieval dependencies. Experiment manifests include the repository `HEAD`, runtime/hardware versions, and a SHA-256 map plus aggregate hash for the Predictor V2 source files, so a run remains identifiable when its worktree has uncommitted edits.

## Run sequence

1. Validate the corrected canonical 900-prompt Vanilla continuation dataset and its independent prompt inventory/provenance using the commands above. Do not use the unrelated EOS diagnostic or generation scratch output as the canonical source. All later artifacts bind to the validated dataset and split hashes.

2. Build and save the TRAIN-only association index:

```powershell
python experiments/rebuild_train_association_index.py --continuations data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --out-pkl experiments/checkpoints/train_only_association_index.pkl --out-summary docs/train_only_association_index_summary.json
```

3. Benchmark the full DEV set across all implemented strategies and pool sizes:

```powershell
python experiments/benchmark_canonical_candidate_recall.py --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --index experiments/checkpoints/train_only_association_index.pkl --pool-sizes 256 512 1024 2048 --out-json experiments/results/predictor_v2_candidate_recall.json --out-md experiments/results/PREDICTOR_V2_CANDIDATE_RECALL.md
```

The benchmark checkpoints each strategy/pool-size result to `<out-json>.resume.json`, including dataset, index, run configuration, and source hashes. After interruption, repeat with `--resume`; mismatched inputs or code are rejected. A plumbing-only `--max-dev-prompts N` run is marked `DEV_SMOKE` and cannot be used for a candidate plan. The benchmark records oracle bounds, candidate capture intervals, domain metrics, measured DEV latency, index size, a labeled shallow memory estimate, and missed-opportunity categories. Offline occurrence/capture is only a retrieval ceiling; it does not establish live H emission, continuation health, preserved task quality, or realized decode-step savings.

4. Complete the broader live Phi baseline and failure attribution on matched DEV prompts. Cover candidate generation/ranking, codebook, H emission, representation, continuation state, EOS/termination, serving, and other observed failure categories. Record an evidence object with `scope: "DEV"`, the canonical dataset and DEV hashes, matched prompt count, domains, both `Vanilla` and `Predictive Phi` conditions, and counts for every category. Record the primary bottleneck and rationale:

```powershell
python experiments/record_predictor_v2_quality_attribution.py --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --evidence experiments/results/phi_quality_attribution_evidence.json --primary-bottleneck <observed-category> --rationale "<evidence-based conclusion>" --out experiments/results/predictor_v2_quality_attribution_gate.json
```

The recognized primary bottleneck values are `candidate_generation`, `candidate_ranking`, `predictor`, `codebook`, `h_emission`, `representation`, `continuation_state`, `eos`, `serving`, and `other`. Only a major candidate/predictor/codebook bottleneck passes the Predictor V2 training gate. A redirect records where to investigate next; it does not start Predictor V2 training.

5. If the attribution gate passes, review full DEV candidate recall and oracle evidence and make an explicit provisional candidate plan. The plan is intentionally not a freeze, because live integration may still disqualify it:

```powershell
python experiments/create_predictor_v2_candidate_plan.py --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --benchmark experiments/results/predictor_v2_candidate_recall.json --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --strategy <reviewed-strategy> --pool-size <reviewed-pool-size> --rationale "<DEV evidence and tradeoff>" --out experiments/results/predictor_v2_candidate_plan.json
```

Valid strategies are `baseline`, `expanded_associations`, `suffix_conditioned`, and `sparse_lexical`; pool sizes are 256, 512, 1024, or 2048.

6. Train architectures on TRAIN and compare on DEV. Ridge uses seed 42; each neural architecture defaults to seeds 42, 43, and 44. The runner builds candidate/label objects only for TRAIN and DEV and can resume completed seed runs only when hashes/configuration match:

```powershell
python experiments/train_predictor_v2_canonical_bakeoff.py --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --index experiments/checkpoints/train_only_association_index.pkl --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --candidate-plan experiments/results/predictor_v2_candidate_plan.json --output-dir experiments/checkpoints/predictor_v2_canonical_bakeoff --out-json experiments/results/predictor_v2_architecture_bakeoff.json --epochs 12 --learning-rate 0.001 --seeds 42 43 44
```

The result records parameter/checkpoint size, training time, inference latency, CPU RSS, CUDA peak allocation when available, DEV aggregate/domain metrics, oracle intervals, and each seed. It does not select a winner. Resume with the same command plus `--resume` after interruption.

7. Review the full multiseed DEV results against Ridge, domain behavior, and quality/latency/memory. Explicitly shortlist one or two architecture/seed pairs, with rationale. The shortlist binds each checkpoint and bakeoff hash:

```powershell
python experiments/freeze_predictor_v2_architecture_shortlist.py --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --candidate-plan experiments/results/predictor_v2_candidate_plan.json --bakeoff experiments/results/predictor_v2_architecture_bakeoff.json --selection-json experiments/results/predictor_v2_shortlist_selection.json --rationale "<DEV comparison and shortlist rationale>" --out experiments/results/predictor_v2_architecture_shortlist.json
```

Each shortlist choice must exist in the DEV bakeoff with an intact checkpoint. FINAL is not opened or used in shortlist selection.

8. Run a small live end-to-end integration evaluation of every shortlisted candidate. Use all DEV prompts, or first freeze the exact prompt IDs of a DEV-only integration subset:

```powershell
python experiments/freeze_predictor_v2_integration_subset.py --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --prompt-ids-json experiments/results/predictor_v2_integration_prompt_ids.json --out experiments/results/predictor_v2_integration_subset.json
```

For each candidate, retain a result artifact and record prompt coverage, domain coverage, checkpoint and plan hashes, observed H emissions, continuation failures, EOS/truncation/repetition, task quality against matched Vanilla, and actual decode steps against the Vanilla baseline. A separate review object must pass `h_emission`, `continuation_state`, `task_quality`, `termination_health`, and `decode_step_savings`, each with evidence notes. Numeric thresholds are an evidence-based human decision, not universal constants embedded in this harness. Then validate and record the live gate:

```powershell
python experiments/record_predictor_v2_live_integration_gate.py --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --candidate-plan experiments/results/predictor_v2_candidate_plan.json --shortlist experiments/results/predictor_v2_architecture_shortlist.json --integration-subset experiments/results/predictor_v2_integration_subset.json --evidence experiments/results/predictor_v2_live_integration_evidence.json --review experiments/results/predictor_v2_live_integration_review.json --out experiments/results/predictor_v2_live_integration_gate.json
```

When using full DEV, omit `--integration-subset` and declare `partition: "DEV"`; when using the frozen subset, declare `partition: "FROZEN_INTEGRATION_SUBSET"`. The exact prompt IDs must match. The gate rejects changed evidence hashes, a missing shortlisted candidate, or failed/missing review checks.

9. Only after the live integration gate passes, create the final candidate-generator freeze. Then freeze the architecture/checkpoint using a review JSON with `multiseed_robustness`, `latency_quality_pareto`, `domain_regression`, and `ridge_baseline`; each must contain `{"passed": true, "notes": "..."}`. The selected architecture and seed must be in the live-tested shortlist and pass live review:

```powershell
python experiments/freeze_predictor_v2_candidates.py --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --candidate-plan experiments/results/predictor_v2_candidate_plan.json --shortlist experiments/results/predictor_v2_architecture_shortlist.json --live-integration-gate experiments/results/predictor_v2_live_integration_gate.json --integration-subset experiments/results/predictor_v2_integration_subset.json --rationale "<combined offline DEV and live DEV evidence>" --out docs/predictor_v2_candidate_generator_freeze.json
python experiments/freeze_predictor_v2_architecture.py --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --bakeoff experiments/results/predictor_v2_architecture_bakeoff.json --candidate-freeze docs/predictor_v2_candidate_generator_freeze.json --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --candidate-plan experiments/results/predictor_v2_candidate_plan.json --shortlist experiments/results/predictor_v2_architecture_shortlist.json --live-integration-gate experiments/results/predictor_v2_live_integration_gate.json --integration-subset experiments/results/predictor_v2_integration_subset.json --architecture <reviewed-model-name> --seed <reviewed-seed> --checkpoint <matching-checkpoint-path> --review-json <DEV-review.json> --rationale "<combined offline DEV and live DEV evidence>" --out docs/predictor_v2_architecture_freeze.json
```

10. FINAL is only available after the candidate generator and architecture freezes and the live integration gate pass. The explicit `--allow-final-eval` flag is mandatory. The claim file is created exclusively before FINAL records are opened; an existing claim or result makes another run fail closed. Treat a claim without a result as a consumed attempt requiring investigation, not permission to tune and retry. FINAL is evaluation-only and cannot change the candidate generator, shortlist, architecture, checkpoint, or configuration:

```powershell
python experiments/evaluate_predictor_v2_final_once.py --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --index experiments/checkpoints/train_only_association_index.pkl --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --candidate-plan experiments/results/predictor_v2_candidate_plan.json --candidate-freeze docs/predictor_v2_candidate_generator_freeze.json --shortlist experiments/results/predictor_v2_architecture_shortlist.json --live-integration-gate experiments/results/predictor_v2_live_integration_gate.json --integration-subset experiments/results/predictor_v2_integration_subset.json --architecture-freeze docs/predictor_v2_architecture_freeze.json --claim experiments/results/predictor_v2_final_eval.claim.json --out experiments/results/predictor_v2_final_result.json --allow-final-eval
```

11. Larger live end-to-end validation follows with the frozen system. It does not reopen architecture selection based on FINAL. Generate the evidence report at any stage:

```powershell
python experiments/report_predictor_v2_workflow.py --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --out-json experiments/results/predictor_v2_workflow_report.json --out-md experiments/results/PREDICTOR_V2_WORKFLOW_REPORT.md
```

The report exposes machine-readable `candidate_generator_frozen`, `offline_architecture_shortlist_complete`, `live_integration_gate_passed`, `architecture_frozen`, and `final_evaluated` states, plus an attribution redirect when Predictor V2 is not the leading cause. Existing legacy bakeoff entry points that accessed `frozen_test_records`/FINAL before architecture freeze now stop with a pointer to this workflow; historical result artifacts are retained but are never inputs to these gates.

## Current gate state

No quality-attribution conclusion, strategy, pool size, architecture, or final result is asserted by this repository change. No real training, live integration run, or FINAL evaluation is run. Full measurements require the corrected canonical 900-prompt artifact, an independent ID/split inventory and provenance manifest, matched broader live Phi attribution, and later evidence-backed DEV decisions.
