# Predictor V2 Training and Evaluation Architecture

This is the downstream harness for the corrected canonical Vanilla Phi continuation artifact. It does not modify how Vanilla prompts are formatted, how generation terminates, or the active EOS diagnostic. The old Phi-only association system remains the provisional baseline. The primary general phrase source is a separate, prompt-conditioned public response corpus; the Phi TRAIN split supplies model-specific retrieval/calibration evidence only. The existing `ConfigurableCandidateGenerator` strategies (`baseline`, `expanded_associations`, `suffix_conditioned`, and `sparse_lexical`) remain available as Phi-only baselines.

The generation-run-pinned independent inventory and the raw records both resolve to **630 TRAIN / 135 DEV / 135 FINAL** (210/45/45 per domain). The separately stated expectation of 540/180/180 is not supported by that inventory and was not applied. The canonical raw artifact is SHA-256 `3a8f59791f6fd06479b2b2869d09d71b57bd1e05b897740b70f18b1941f22ee6`.

## Workflow

```text
Correct canonical 900 Vanilla dataset + independent ID/split inventory
    ↓ validate schema, revisions, generation config, provenance, and hashes
TRAIN / DEV / FINAL views
    ↓ TRAIN only for Phi supervision/indexing; external corpus separately prompt-decontaminated against all 900 task prompts
Pinned Phi TRAIN association index + model-independent prompt/response sourcebook, mined with the pinned Phi tokenizer
    ↓ DEV only
Phi-only vs external-only vs hybrid candidate/oracle analysis at 256/512/1024
    ↓ broader live Vanilla vs Predictive Phi baseline and failure attribution
Redirect if another subsystem dominates; continue only if Predictor/codebook quality is a major bottleneck
    ↓ only if live attribution confirms the predictor/codebook bottleneck
TRAIN fitting + DEV comparison, then a one- or two-candidate DEV shortlist
    ↓ small live end-to-end integration check on all DEV or an exactly frozen DEV subset
Combine offline DEV and live DEV evidence; freeze candidate generator and architecture/config/checkpoint
    ↓ explicit --allow-final-eval, all upstream gates bound, one claim only
FINAL evaluation once; never use FINAL for selection
    ↓
Larger live end-to-end validation with the frozen system
```

The contract validator reads the JSONL for schema and integrity checks. The existing Phi-only association index is built from TRAIN continuations only. The external sourcebook uses a separately pinned public train split and scans every user turn against all 900 canonical task prompts before sampling; the filter reads no canonical DEV or FINAL response. The sourcebook is mined with the pinned Phi tokenizer, while the underlying public prompt/response cache stays outside Git for later tokenizer retargeting. Candidate and architecture selection use DEV. The broad live attribution uses matched live DEV Vanilla/Predictive Phi runs after the offline sourcebook comparison. The Predictor V2 live integration check tests actual H emission, continuation health, task quality, termination behavior, and decode-step savings for the shortlisted models. A candidate occurrence or capture in offline text is not evidence that decode will emit H tokens or preserve a healthy live trajectory.

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
| `provenance` | Per-row object containing `model_id`, `model_revision`, `tokenizer_id`, `tokenizer_revision`, and `generation_config`. Shared canonical fields are checked against the summary provenance bound into the manifest. The loader also accepts the flat spellings used by synthetic fixtures. |
| Row audit fields | The generation artifact retains cap flags (`hit_1024_cap`, `reached_1024`), termination metadata, generation time, and GPU memory fields. The loader carries these flags into its normalized metadata view without rewriting the source JSONL. |

The companion manifest uses schema `predictor_v2_canonical_manifest_v1`. It binds the raw JSONL SHA-256, the expected prompt IDs and split assignment, allowed domains, model/tokenizer revisions, tokenizer vocabulary size, generation config, source provenance, and self/provenance hashes. The expected ID list and split assignment must come from the prompt inventory prepared independently of the generated continuations. Do not create them by copying IDs from the artifact being checked.

Create that sidecar after the corrected JSONL and independent inventory/provenance files are ready:

```powershell
python -m experiments.make_predictor_v2_manifest --dataset data/canonical_phi_continuations.jsonl --split-inventory data/predictor_v2_canonical_split_inventory.json --provenance data/predictor_v2_canonical_generation_provenance.json --tokenizer-vocab-size 32011 --out data/canonical_phi_continuations.manifest.json
python -m experiments.validate_predictor_v2_dataset --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json
```

The committed canonical sidecars are `data/predictor_v2_canonical_split_inventory.json` and `data/predictor_v2_canonical_generation_provenance.json`. They bind the independently pinned source inventory, generation selection, summary, audit, stage-0 audit, and immutable JSONL SHA. The manifest command reads revisions/config from the first record's nested `provenance`; the validator then checks every row. The canonical JSONL itself remains an external, byte-for-byte copy of the audited scratch source and is not required in Git.

Each saved TRAIN index also records hashes for the index builder and its canonical-data/retrieval dependencies. Experiment manifests include the repository `HEAD`, runtime/hardware versions, and a SHA-256 map plus aggregate hash for the Predictor V2 source files, so a run remains identifiable when its worktree has uncommitted edits.

## External sourcebook and run sequence

1. Validate the corrected canonical 900-prompt Vanilla continuation dataset and its independent prompt inventory/provenance using the commands above. Do not use the unrelated EOS diagnostic or generation scratch output as the canonical source. All later artifacts bind to the validated dataset and split hashes.

2. Build and save the existing Phi-only TRAIN association index. This is the comparator, not the general-language sourcebook:

```powershell
python -m experiments.rebuild_train_association_index --continuations data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --out-pkl experiments/checkpoints/train_only_association_index.pkl --out-summary docs/train_only_association_index_summary.json
```

3. Select the first external corpus from [`PREDICTOR_V2_EXTERNAL_SOURCE_SELECTION.json`](PREDICTOR_V2_EXTERNAL_SOURCE_SELECTION.json). It uses the pinned `HuggingFaceH4/ultrachat_200k` `train_sft` revision and a deterministic sample of up to 50,000 clean rows. The selector streams the full source split, checks all user turns against all 900 canonical prompts, and records exact/near-duplicate counts. It does not inspect canonical continuation text. Public response overlap with held-out continuations is not tested because DEV/FINAL responses are forbidden filter inputs:

```powershell
python -m experiments.select_predictor_v2_external_sourcebook --canonical data/canonical_phi_continuations.jsonl --dataset HuggingFaceH4/ultrachat_200k --revision 8049631c405ae6576f93f445c6b8166f76f5505a --split train_sft --sample-size 50000 --working-db scratch/predictor_v2_external_sourcebook/selection.sqlite --out-jsonl scratch/predictor_v2_external_sourcebook/clean_sample.jsonl --report scratch/predictor_v2_external_sourcebook/decontamination_report.json
python -m experiments.build_predictor_v2_external_sourcebook --sample scratch/predictor_v2_external_sourcebook/clean_sample.jsonl --decontamination-report scratch/predictor_v2_external_sourcebook/decontamination_report.json --database scratch/predictor_v2_external_sourcebook/sourcebook.sqlite --manifest scratch/predictor_v2_external_sourcebook/sourcebook.manifest.json
```

The pinned scan covered 207,865 `train_sft` rows: 3 malformed rows were skipped, 207,862 were prompt-clean, and 50,000 were selected deterministically. No exact or near-duplicate prompt was found against the 900 canonical prompts. The result and sample hash are recorded in [`experiments/results/predictor_v2_sourcebook_decontamination.json`](../experiments/results/predictor_v2_sourcebook_decontamination.json). This does not measure overlap between external responses and held-out canonical completions.

The builder creates a chunked/resumable SQLite sourcebook, aggregates unique 2–4 Phi-token phrase counts, rejects special/whitespace/punctuation/malformed spans, and hashes the source, tokenizer, decontamination report, configuration, and database. Keep the selected public response cache, SQLite database, Hugging Face cache, and any temporary artifacts outside Git.

4. Compare Phi-only, external-only, and hybrid retrieval on all of DEV at pool sizes 256/512/1024. The current best Phi-only comparator is `expanded_associations`; 2048 is diagnostic only. The report preserves oracle intervals, per-domain capture, retrieval p50/p90/p99, pool/index size, and missed-opportunity categories:

```powershell
python -m experiments.benchmark_predictor_v2_sourcebook --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --train-index experiments/checkpoints/train_only_association_index.pkl --sourcebook scratch/predictor_v2_external_sourcebook/sourcebook.sqlite --sourcebook-manifest scratch/predictor_v2_external_sourcebook/sourcebook.manifest.json --pool-sizes 256 512 1024 --time-limit-seconds 1.0 --out-json experiments/results/predictor_v2_sourcebook_dev.json --out-md experiments/results/PREDICTOR_V2_SOURCEBOOK_DEV.md
```

For an interrupted phrase build, repeat the same builder command: completed example IDs are skipped and tokenizer/source/configuration mismatches fail closed. The external benchmark is DEV-only; its output is a candidate-opportunity screen, not a live integration result.

5. Benchmark the existing Phi-only candidate strategies on the full DEV set when a separate strategy comparison is useful:

```powershell
python -m experiments.benchmark_canonical_candidate_recall --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --index experiments/checkpoints/train_only_association_index.pkl --pool-sizes 256 512 1024 2048 --time-limit-seconds 1.0 --out-json experiments/results/predictor_v2_candidate_recall.json --out-md experiments/results/PREDICTOR_V2_CANDIDATE_RECALL.md
```

The benchmark checkpoints each strategy/pool-size result to `<out-json>.resume.json`, including dataset, index, run configuration, and source hashes. After interruption, repeat with `--resume`; mismatched inputs or code are rejected. A plumbing-only `--max-dev-prompts N` run is marked `DEV_SMOKE` and cannot be used for a candidate plan. The benchmark records oracle bounds, candidate capture intervals, domain metrics, p50/p90/p99 prompt-time latency, startup/loading cost, index and pool memory, candidate rank/filter/occurrence health, and missed-opportunity examples. Whitespace-only, punctuation-only, and trailing-space/tab phrases are filtered generically. Offline occurrence/capture is only a retrieval ceiling; it does not establish live H emission, continuation health, preserved task quality, or realized decode-step savings.

6. Complete the broader live Phi baseline and failure attribution on matched DEV prompts using the A–D conditions in [the live attribution protocol](PREDICTOR_V2_LIVE_ATTRIBUTION_PLAN.md): Vanilla, predictive with H disabled, occurrence-oracle codebook, continuation-safe oracle codebook, and the current Phi-only TRAIN retrieval/codebook path. The completed sourcebook DEV screen does not support an external/hybrid live arm; add one only if later full DEV evidence changes that decision. Cover candidate generation/ranking, codebook, H emission, representation, continuation state, EOS/termination, serving, and other observed failure categories. Record an evidence object with `scope: "DEV"`, the canonical dataset and DEV hashes, matched prompt count, domains, all live conditions, and counts for every category. Record the primary bottleneck and rationale:

```powershell
python -m experiments.record_predictor_v2_quality_attribution --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --evidence experiments/results/phi_quality_attribution_evidence.json --primary-bottleneck <observed-category> --rationale "<evidence-based conclusion>" --out experiments/results/predictor_v2_quality_attribution_gate.json
```

The recognized primary bottleneck values are `candidate_generation`, `candidate_ranking`, `predictor`, `codebook`, `h_emission`, `representation`, `continuation_state`, `eos`, `serving`, and `other`. Only a major candidate/predictor/codebook bottleneck passes the Predictor V2 training gate. A redirect records where to investigate next; it does not start Predictor V2 training.

7. If the attribution gate passes, review full DEV candidate recall and oracle evidence and make an explicit provisional candidate plan. The plan is intentionally not a freeze, because live integration may still disqualify it:

```powershell
python -m experiments.create_predictor_v2_candidate_plan --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --benchmark experiments/results/predictor_v2_candidate_recall.json --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --strategy <reviewed-strategy> --pool-size <reviewed-pool-size> --rationale "<DEV evidence and tradeoff>" --out experiments/results/predictor_v2_candidate_plan.json
```

Valid strategies are `baseline`, `expanded_associations`, `suffix_conditioned`, `sparse_lexical`, `external_sourcebook`, and `hybrid_sourcebook`; pool sizes are 256, 512, 1024, or 2048. For the four Phi-only strategies, pass `experiments/results/predictor_v2_candidate_recall.json` as `--benchmark`. For `external_sourcebook` or `hybrid_sourcebook`, use `experiments/results/predictor_v2_sourcebook_dev.json`, which records the external/hybrid candidate configuration and sourcebook database hash.

8. Train architectures on TRAIN and compare on DEV. Ridge uses seed 42; each neural architecture defaults to seeds 42, 43, and 44. The runner builds candidate/label objects only for TRAIN and DEV and can resume completed seed runs only when hashes/configuration match. If the candidate plan selects `external_sourcebook` or `hybrid_sourcebook`, pass `--sourcebook` with the matching sourcebook database; its hash must match the DEV candidate plan:

```powershell
python -m experiments.train_predictor_v2_canonical_bakeoff --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --index experiments/checkpoints/train_only_association_index.pkl --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --candidate-plan experiments/results/predictor_v2_candidate_plan.json --sourcebook scratch/predictor_v2_external_sourcebook/sourcebook.sqlite --output-dir experiments/checkpoints/predictor_v2_canonical_bakeoff --out-json experiments/results/predictor_v2_architecture_bakeoff.json --epochs 12 --learning-rate 0.001 --seeds 42 43 44
```

The result records parameter/checkpoint size, training time, inference latency, CPU RSS, CUDA peak allocation when available, DEV aggregate/domain metrics, oracle intervals, and each seed. It does not select a winner. Resume with the same command plus `--resume` after interruption.

9. Review the full multiseed DEV results against Ridge, domain behavior, and quality/latency/memory. Explicitly shortlist one or two architecture/seed pairs, with rationale. The shortlist binds each checkpoint and bakeoff hash:

```powershell
python -m experiments.freeze_predictor_v2_architecture_shortlist --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --candidate-plan experiments/results/predictor_v2_candidate_plan.json --bakeoff experiments/results/predictor_v2_architecture_bakeoff.json --selection-json experiments/results/predictor_v2_shortlist_selection.json --rationale "<DEV comparison and shortlist rationale>" --out experiments/results/predictor_v2_architecture_shortlist.json
```

Each shortlist choice must exist in the DEV bakeoff with an intact checkpoint. FINAL is not opened or used in shortlist selection.

10. Run a small live end-to-end integration evaluation of every shortlisted candidate. Use all DEV prompts, or first freeze the exact prompt IDs of a DEV-only integration subset:

```powershell
python -m experiments.freeze_predictor_v2_integration_subset --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --prompt-ids-json experiments/results/predictor_v2_integration_prompt_ids.json --out experiments/results/predictor_v2_integration_subset.json
```

For each candidate, retain a result artifact and record prompt coverage, domain coverage, checkpoint and plan hashes, observed H emissions, continuation failures, EOS/truncation/repetition, task quality against matched Vanilla, and actual decode steps against the Vanilla baseline. A separate review object must pass `h_emission`, `continuation_state`, `task_quality`, `termination_health`, and `decode_step_savings`, each with evidence notes. Numeric thresholds are an evidence-based human decision, not universal constants embedded in this harness. Then validate and record the live gate:

```powershell
python -m experiments.record_predictor_v2_live_integration_gate --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --candidate-plan experiments/results/predictor_v2_candidate_plan.json --shortlist experiments/results/predictor_v2_architecture_shortlist.json --integration-subset experiments/results/predictor_v2_integration_subset.json --evidence experiments/results/predictor_v2_live_integration_evidence.json --review experiments/results/predictor_v2_live_integration_review.json --out experiments/results/predictor_v2_live_integration_gate.json
```

When using full DEV, omit `--integration-subset` and declare `partition: "DEV"`; when using the frozen subset, declare `partition: "FROZEN_INTEGRATION_SUBSET"`. The exact prompt IDs must match. The gate rejects changed evidence hashes, a missing shortlisted candidate, or failed/missing review checks.

11. Only after the live integration gate passes, create the final candidate-generator freeze. Then freeze the architecture/checkpoint using a review JSON with `multiseed_robustness`, `latency_quality_pareto`, `domain_regression`, and `ridge_baseline`; each must contain `{"passed": true, "notes": "..."}`. The selected architecture and seed must be in the live-tested shortlist and pass live review:

```powershell
python -m experiments.freeze_predictor_v2_candidates --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --candidate-plan experiments/results/predictor_v2_candidate_plan.json --shortlist experiments/results/predictor_v2_architecture_shortlist.json --live-integration-gate experiments/results/predictor_v2_live_integration_gate.json --integration-subset experiments/results/predictor_v2_integration_subset.json --rationale "<combined offline DEV and live DEV evidence>" --out docs/predictor_v2_candidate_generator_freeze.json
python -m experiments.freeze_predictor_v2_architecture --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --bakeoff experiments/results/predictor_v2_architecture_bakeoff.json --candidate-freeze docs/predictor_v2_candidate_generator_freeze.json --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --candidate-plan experiments/results/predictor_v2_candidate_plan.json --shortlist experiments/results/predictor_v2_architecture_shortlist.json --live-integration-gate experiments/results/predictor_v2_live_integration_gate.json --integration-subset experiments/results/predictor_v2_integration_subset.json --architecture <reviewed-model-name> --seed <reviewed-seed> --checkpoint <matching-checkpoint-path> --review-json <DEV-review.json> --rationale "<combined offline DEV and live DEV evidence>" --out docs/predictor_v2_architecture_freeze.json
```

12. FINAL is only available after the candidate generator and architecture freezes and the live integration gate pass. The explicit `--allow-final-eval` flag is mandatory. The claim file is created exclusively before FINAL records are opened; an existing claim or result makes another run fail closed. Treat a claim without a result as a consumed attempt requiring investigation, not permission to tune and retry. If the frozen candidate uses external/hybrid retrieval, pass the sourcebook whose hash is bound into its freeze; a mismatch fails before the one-time claim. FINAL is evaluation-only and cannot change the candidate generator, shortlist, architecture, checkpoint, or configuration:

```powershell
python -m experiments.evaluate_predictor_v2_final_once --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --index experiments/checkpoints/train_only_association_index.pkl --quality-attribution-gate experiments/results/predictor_v2_quality_attribution_gate.json --candidate-plan experiments/results/predictor_v2_candidate_plan.json --candidate-freeze docs/predictor_v2_candidate_generator_freeze.json --shortlist experiments/results/predictor_v2_architecture_shortlist.json --live-integration-gate experiments/results/predictor_v2_live_integration_gate.json --integration-subset experiments/results/predictor_v2_integration_subset.json --architecture-freeze docs/predictor_v2_architecture_freeze.json --sourcebook scratch/predictor_v2_external_sourcebook/sourcebook.sqlite --claim experiments/results/predictor_v2_final_eval.claim.json --out experiments/results/predictor_v2_final_result.json --allow-final-eval
```

13. Larger live end-to-end validation follows with the frozen system. It does not reopen architecture selection based on FINAL. Generate the evidence report at any stage:

```powershell
python -m experiments.report_predictor_v2_workflow --dataset data/canonical_phi_continuations.jsonl --manifest data/canonical_phi_continuations.manifest.json --out-json experiments/results/predictor_v2_workflow_report.json --out-md experiments/results/PREDICTOR_V2_WORKFLOW_REPORT.md
```

The report exposes machine-readable `candidate_generator_frozen`, `offline_architecture_shortlist_complete`, `live_integration_gate_passed`, `architecture_frozen`, and `final_evaluated` states, plus an attribution redirect when Predictor V2 is not the leading cause. Existing legacy bakeoff entry points that accessed `frozen_test_records`/FINAL before architecture freeze now stop with a pointer to this workflow; historical result artifacts are retained but are never inputs to these gates.

## Current gate state

The corrected canonical dataset and TRAIN-only index are validated. The complete 135-prompt DEV candidate/oracle benchmark is recorded in `experiments/results/predictor_v2_candidate_recall.json` and `experiments/results/PREDICTOR_V2_CANDIDATE_RECALL.md`. The full DEV comparison in `experiments/results/predictor_v2_sourcebook_dev.json` and `experiments/results/PREDICTOR_V2_SOURCEBOOK_DEV.md` found no aggregate external or hybrid lead: at pool 1024, bounded capture was 0.406–0.732 for Phi-only, 0.301–0.560 external-only, and 0.321–0.597 hybrid. These are oracle bounds, not confidence intervals. External/hybrid have higher Instruction lower bounds (0.321 versus 0.264 for Phi-only), but the ranges overlap; Phi-only is stronger in Code and especially Reasoning. External/hybrid retrieval latency was much higher (p50 about 6.6 seconds versus about 4 milliseconds for Phi-only in this CPU benchmark). Keep `expanded_associations` at pool 1024 as the provisional comparator, not a frozen generator; the current evidence does not justify external/hybrid live testing. See `docs/PREDICTOR_V2_CORRECTED_CANONICAL_RESULTS.md` for the historical Phi-only baseline analysis.

The current workflow report records the broader live attribution as not run. No Predictor V2 architecture training, live integration evaluation, architecture freeze, or FINAL evaluation has run. Predictor V2 remains a leading hypothesis, not a confirmed root cause; broader live attribution is the next gate and can redirect work to representation, H emission, continuation state, EOS, serving, or another subsystem. Architecture and generator choices still require combined offline DEV and live DEV evidence. FINAL remains unavailable until live integration passes and both generator and architecture freezes are recorded.
