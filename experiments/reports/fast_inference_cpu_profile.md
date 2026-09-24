# Predictive Fast Inference CPU Validation

> Directional CPU result; not evidence of GPU speedup.
> No live Phi generation or GPU work was performed.

## Correctness

All synthetic reference checks passed: base_only, H2, H3, mixed_prompt, multiple_H, eos_base_token, decode_did_not_rerun_encoders, shifted_original_tail_embedding, output_logits_top1_top5_and_tail, autoregressive_greedy_ids, prompt_segmentation_and_expansion, prompt_tail_ids_shift_and_roundtrip, H2_H3_semantic_positions, fast_position_has_no_aten_any, reset_and_second_codebook.

The fast table matches legacy embedding/logits on the common base/H ID space, preserves top-1 and top-5, handles shifted output-tail IDs, and restores tail IDs during output expansion.

## CPU operator counts

| Path | embedding | arange | mul | add | where | any | linear | bmm | cat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Embedding legacy | 2 | 2 | 5 | 1 | 0 | 0 | 0 | 0 | 0 |
| Embedding fast | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Output legacy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| Output fast | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| Position legacy | 0 | 0 | 0 | 2 | 2 | 2 | 0 | 0 | 0 |
| Position fast | 0 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 0 |

## Directional CPU median time (ms/op)

| Component | Legacy | Fast |
|---|---:|---:|
| Position | 0.0640 | 0.0325 |
| Embedding | 0.0634 | 0.0037 |
| Output projection | 0.0828 | 0.0503 |

## Effective-table memory (bytes)

- Base input table: 1,050,624
- Effective input table: 1,067,008
- Base output head: 1,058,832
- Effective output head: 1,075,344
- Additional CPU RAM held by tables: 2,142,352

Synthetic CPU profiling does not establish any GPU or T4 speedup.
