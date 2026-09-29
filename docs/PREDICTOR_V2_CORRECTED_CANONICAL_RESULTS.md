# Corrected Canonical Predictor V2 DEV Results

## Scope and data

This is an offline retrieval and oracle analysis of the corrected canonical Vanilla Phi continuations. It covers all **135 DEV prompts** (45 each in Code, Reasoning, and Instruction), bound to dataset SHA-256 `3a8f59791f6fd06479b2b2869d09d71b57bd1e05b897740b70f18b1941f22ee6` and DEV split SHA-256 `ef52826e58e0a3a0924e8856b781d177253f0960e897f73f42770f7f5e939e67`. The independent inventory resolves the dataset as 630 TRAIN / 135 DEV / 135 FINAL; the unsupported 540/180/180 expectation was not applied. FINAL metrics and text were not used.

The DEV sweep evaluates `baseline`, `expanded_associations`, `suffix_conditioned`, and `sparse_lexical` at pool sizes 256, 512, 1024, and 2048 for K=8/16/32. Full machine-readable results and the generated comparison tables are in `experiments/results/predictor_v2_candidate_recall.json` and `experiments/results/PREDICTOR_V2_CANDIDATE_RECALL.md`.

## Bounded offline comparison

Each capture figure is a lower–upper interval derived from bounded global-occurrence and candidate-pool oracle solves. It describes the hindsight compression opportunity available in the generated candidate pool. It is not observed predictor recall, H emission, continuation quality, or live decode-step savings.

| Strategy / pool | K=8 capture | K=16 capture | K=32 capture | CPU pool-generation p50 / p90 / p99 (ms) |
|---|---:|---:|---:|---:|
| `expanded_associations` / 1024 | 0.298–1.000 | 0.376–1.000 | **0.406–0.732** | 4.084 / 6.473 / 9.423 |
| `sparse_lexical` / 1024 | 0.298–1.000 | 0.371–1.000 | 0.401–0.726 | 8.716 / 10.454 / 13.443 |
| `baseline` / 512 | 0.292–1.000 | 0.359–0.949 | 0.377–0.672 | 3.101 / 8.664 / 18.764 |
| `suffix_conditioned` / 1024 | 0.295–1.000 | 0.362–0.940 | 0.379–0.671 | 2.506 / 3.675 / 4.894 |

`expanded_associations` / 1024 is the provisional offline lead. At K=32 it has the highest capture interval among these settings, with 0.406–0.732 versus 0.401–0.726 for `sparse_lexical` / 1024. Raising its pool limit from 1024 to 2048 did not improve the K=32 interval (both 0.406–0.732) or candidate-pool oracle bound materially, so 2048 is not justified by this run. The 1024 setting is a candidate for later live comparison, not a frozen generator or a gate pass. The broader live Phi attribution has not run and remains able to redirect the work.

The provisional lead is uneven across domains at K=32: Code 0.366–0.731, Reasoning 0.564–0.941, and Instruction 0.264–0.449. Instruction has the weakest lower-bound capture and needs attention in both offline and live evidence. Across all strategies and pool sizes, the global oracle was exact for only 39 of 405 solves; for `expanded_associations` / 1024, the candidate-pool oracle was exact for 264 of 405. The rest preserve solver bounds, so the wide intervals must not be replaced by point estimates or called exact ceilings.

For `expanded_associations` / 1024, the global-occurrence K=32 solver reports 17,093–29,574 saved steps, while the candidate-pool oracle reports 11,993–12,506. The derived capture interval is 0.406–0.732. These are offline oracle bounds on the matched Vanilla continuations, not live speedup measurements.

## Retrieval and missed-opportunity diagnostics

For the selected K=32 global-oracle incumbents in the 135 DEV prompts, 830 of 4,112 phrase instances occur somewhere in the `expanded_associations` / 1024 pool; 3,282 are absent. Of the useful instances present, 133 rank in positions 1–32, 351 rank 33–128, and 346 rank 129 or lower in the pool. Zero were generated and then lost solely to pool-size truncation. The mean pool has 498 candidates (48.7% of its configured capacity). Across all prompt pools, 83.3% of entries do not occur in the matched continuation; 2,664 entries are dead candidates sourced only from the TRAIN global background bank. This is a clutter diagnostic, not evidence that background candidates caused the remaining gap.

The missed-opportunity categories below are diagnostic counts and may overlap; do not sum them as a partition. “No observed prompt signal” is a heuristic and does not establish that a phrase could not be predicted.

| Category | Count | Example / interpretation |
|---|---:|---|
| A. No observed prompt conditioning | 1,525 | For `alpaca_1`, “in art and painting” has no observed word overlap or implemented generator source. Heuristic only. |
| B. Prompt-related but not generated | 1,556 | For `alpaca_1`, examples include “the RGB (” and “The three primary colors”. No useful incumbent was lost to pool-size truncation. |
| C. Present below rank 32 | 697 | For `alpaca_1024`, “. The” appears at rank 87 and “programming language designed for” at rank 125. |
| D. Removed by generic quality filters | 201 useful instances | 135 punctuation-only, 37 trailing-space/tab, and 29 whitespace-only; e.g. `".\n\n"`. |
| E. TRAIN background-only dead entries | 2,664 | Descriptive pool clutter; not a proven cause of missed opportunity. |
| F. Domain-specific gap | Instruction lowest | Example bound gap: `alpaca_1963`, “the talking dog”, 138 steps between candidate and global lower bounds. Code example: `mbpp_906`, whitespace phrase `"\n   "`, 81 steps. |
| G. Oracle-bound uncertainty | 366 / 405 global solves non-exact; 141 / 405 candidate-pool solves non-exact | Retain intervals; do not infer a proven global optimum from incumbent phrases. |

Other useful `expanded_associations` / 1024 health measurements: useful-present rank percentiles are p50 110.5, p90 295.2, p99 481; rejection attempts across generated pools include 3,828 punctuation-only, 2,558 trailing-space/tab, and 1,798 whitespace-only phrases. The serialized TRAIN index is 3,822,460 bytes. First-invocation CPU preparation to generation-ready was 355.681 ms, including 139.453 ms tokenizer load, 173.125 ms dataset validation/load, 25.713 ms index load, and 14.361 ms candidate-generator initialization. This startup measurement excludes model loading. Candidate-pool generation latency is a CPU-only API measurement and excludes oracle solve time.

## Gate state and next experiment

The current machine-readable workflow report is `experiments/results/predictor_v2_workflow_report.json` with its rendered summary at `experiments/results/PREDICTOR_V2_WORKFLOW_REPORT.md`. Candidate generator freeze, architecture shortlist, live integration pass, architecture freeze, and FINAL evaluation are all false/not run. No architecture training or FINAL evaluation was performed for these results.

Next, complete the broader matched live DEV Vanilla/Predictive Phi failure attribution before Predictor V2 training. Compare Vanilla; the predictive path with H disabled; occurrence-oracle and continuation-safe oracle codebooks; and the current learned predictor. Inspect representation, H emission, continuation state, EOS, serving, task quality, and decode steps by prompt and domain. Redirect if one of those areas dominates. Train Predictor V2 on TRAIN and shortlist architectures on DEV only if candidate, predictor, or codebook quality is confirmed as a major bottleneck. Then run a small live end-to-end DEV integration evaluation of at most two architecture candidates. Freeze the candidate generator and architecture only after combining offline DEV and live DEV evidence. Run FINAL once after all gates; FINAL cannot participate in selection. Larger live validation follows the freeze.
