# Stage 1 Attribution Methodology Pin

This note records what the committed 45-prompt Stage 1 run actually measured, from its source code and serialized records. It is a provenance note, not a reanalysis result.

## Pinned run

- Stage 1 source commit recorded by every raw record: `8af6589ba0711018637c8ce41f8584642258cc49`.
- Current report-only branch tip when inspected: `5e0f328f7502e0e69a6dc50cb941da3db77ea16b`.
- Records: `docs/stage1_attribution_records.jsonl`; 180 records, 45 DEV prompt IDs, four conditions.
- Phi model and tokenizer revision: `microsoft/Phi-3.5-mini-instruct`, revision `2fe192450127e6a83f7441aef6e3ca586c338b77`.
- Predictive wrapper repository revision: `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1`, revision `11c461733a79d2a5de6b814585c3361ca2aacbe7`.
- Joint Step-100 checkpoint: `experiments/checkpoints/predictive_joint_pilot/checkpoint_step_100.pt`, SHA-256 `2c3606c075ac96dff1f607043f58241251d837f2340ae950d3dc309e9820fd44`. B, C, and D records all carry this checkpoint name and hash; the runner calls `load_joint_checkpoint(..., expected_step=100, expected_model_id=...)`.
- Prompt rendering uses `build_canonical_prompt_text`, tokenizes with `add_special_tokens=False`, and runs greedy generation (`do_sample=False`), at most 1024 new tokens, pad ID 32000, EOS IDs `[32007, 32001, 32000]`.
- Kaggle's archived Stage 1 log reports 4,076.95 seconds (67.95 minutes) for the benchmark body. The report markdown separately states 76.8 minutes for the GPU run; this discrepancy is retained rather than treating the two durations as interchangeable. The historical experiment tracker reports about 1.65 of the 5.0 project GPU hours consumed before this follow-up; Kaggle's weekly quota is a different accounting window.

## Actual Stage 1 conditions

- **A_vanilla:** native Transformers Phi model at the pinned revision; no Zip2Zip wrapper, joint checkpoint, or H codebook.
- **B_h_disabled:** Zip2Zip wrapper and the Step-100 joint checkpoint; empty codebook, so H is unavailable.
- **C_oracle (rename in new analysis: ORACLE CODEBOOK LIVE):** same wrapped/checkpointed model as B; codebook built per DEV prompt from `continuation_token_ids` in the canonical dataset record. The source is a Vanilla-generated continuation, not the reference answer. The serialized `selected_h_slots` recover the exact supplied phrases; replaying the recorded phrase-selection rule against that prompt's A token sequence reproduces C's ordered codebook on all 45 prompts.
- **D_real_predictor:** same wrapped/checkpointed model; Phi-only `TrainOnlyAssociationIndex` and `ConfigurableCandidateGenerator` with `EXPANDED_ASSOCIATIONS`, target pool 1024, then the configured ranker's top K phrases. Each D record serializes the exact selected phrase list in `selected_h_slots`; it does not serialize the full retrieval pool or ranker provenance hash.

## C codebook and live behavior

`derive_oracle_codebook_phrases` in `src/zip2zip/predictor_v2/attribution_harness.py` counts 2-, 3-, and 4-token n-grams, excludes IDs `{0, 1, 2, 32000..32010}`, orders by `(count * (length - 1), count, -length)` descending, and takes K=32. The selected phrases map to H IDs starting at 32011. The live `StaticCodebookManager` permits phrase lengths 2 through 4. C did **not** force substitutions: it entered the ordinary greedy `model.generate` loop, and the model could emit a base token or a supplied H ID at each decode step.

The run's **5.39%** is live realized compression, not an offline ceiling. The report sums C expanded base-equivalent tokens (16,712) and model decode calls (15,811), then computes `100 * (1 - 15,811 / 16,712) = 5.39%`. The 704 figure is the count of H emissions; because phrase spans have lengths 2–4, their total expansion savings are 901 tokens. The denominator is C's own expanded output, including its terminal token. The 704 H emissions and 5.39% therefore describe different quantities. `transformer_decode_calls` is `len(gen_ids)` in this harness.

This was **ORACLE CODEBOOK LIVE**, not **ORACLE CEILING**. It supplies hindsight phrases derived from the canonical Vanilla continuation, but lets the trained model decide whether to emit H. The existing records contain the exact supplied C and D phrase lists, enabling offline DP opportunity analysis without regenerating outputs.

## Boundaries of this pin

No FINAL records or metrics were read for this note. Stage 1 did not include B0 (wrapped Vanilla weights) or CF (FORCED ORACLE); its A-versus-B comparison bundles wrapper/architecture changes with the trained checkpoint. Stage 1 also did not measure an ORACLE CEILING on A or B. The D records preserve selected phrases but not the ranker's hash or full candidate pool. These are explicit limitations to resolve or retain as `NOT TESTED` in the causal report.

## Required follow-up gates

The follow-up harness now runs A/B0 exact token equality and deterministic same-prefix logit parity before any B1/C/CF/D attribution. B0/B1 records include the unchanged base-Phi hash and the checkpoint loader's changed tensor names and shapes. CF separately records exact forced-H expansion, semantic positions and cache offsets, and unforced immediate next-token agreement after each H. A failed earlier gate blocks the dependent conditions and leaves causal diagnosis unresolved. These are implementation controls only; the live A/B0, B0/B1, and CF measurements remain **NOT TESTED** until the matched DEV run passes the CPU suite and is executed.

The live runner now defaults to the split-isolated DEV JSONL and rejects mixed-split manifests before loading continuation rows. It also rejects resuming an unmarked output file, so it does not need to inspect unknown-split records to recover prior work.
