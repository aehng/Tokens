# Predictor V2 External Sourcebook DEV Comparison

Prompts: 135 (full DEV) · K=32 · FINAL accessed: no

The external sourcebook contains 50,000 deterministically selected examples from the pinned UltraChat `train_sft` revision, with 1,423,205 unique 2-token, 5,240,176 unique 3-token, and 9,001,244 unique 4-token Phi-token phrases. The SQLite database is 3,589,042,176 bytes (SHA-256 `c28285d556604ed8e2d4b78aac699117b99db288024c9e5c3969d1f534674064`). The DEV oracle cap was one second per solve.

| Candidate source | Pool | Capture interval | p50/p90/p99 retrieval (ms) | Mean pool bytes | Code | Reasoning | Instruction |
|---|---:|---:|---:|---:|---:|---:|---:|
| phi_only | 256 | 0.359–0.626 | 4.038/6.650/9.287 | 128078 | 0.324–0.632 | 0.494–0.779 | 0.241–0.410 |
| phi_only | 512 | 0.400–0.717 | 4.283/6.340/9.281 | 216717 | 0.364–0.725 | 0.550–0.907 | 0.263–0.448 |
| phi_only | 1024 | 0.406–0.732 | 4.213/6.341/9.917 | 251930 | 0.366–0.731 | 0.564–0.941 | 0.264–0.450 |
| external_only | 256 | 0.234–0.419 | 6708.029/11710.783/15458.054 | 126732 | 0.242–0.499 | 0.204–0.318 | 0.258–0.458 |
| external_only | 512 | 0.271–0.494 | 6517.569/11297.512/14742.993 | 253895 | 0.280–0.590 | 0.240–0.374 | 0.295–0.538 |
| external_only | 1024 | 0.301–0.560 | 6581.883/11503.539/14795.442 | 508806 | 0.310–0.684 | 0.274–0.427 | 0.321–0.590 |
| hybrid | 256 | 0.256–0.456 | 6712.265/11719.048/15466.992 | 126766 | 0.255–0.521 | 0.255–0.396 | 0.259–0.459 |
| hybrid | 512 | 0.292–0.533 | 6523.744/11304.825/14752.509 | 253921 | 0.292–0.617 | 0.289–0.451 | 0.295–0.542 |
| hybrid | 1024 | 0.321–0.597 | 6586.851/11510.864/14804.577 | 508830 | 0.319–0.706 | 0.323–0.503 | 0.321–0.591 |

The `phi_only` row is the full-DEV rerun of the provisional `expanded_associations` baseline. Capture values are oracle bounds, not confidence intervals. At pool 1024, external/hybrid lower-bound capture is higher for Instruction (0.321 vs 0.264), but the bounds overlap and aggregate capture is lower; Phi-only has stronger Code and Reasoning bounds. External/hybrid CPU retrieval p50 is about 6.6 seconds, versus about 4 milliseconds for Phi-only. These data do not justify carrying the external sourcebook into live testing. Prompt decontamination never reads canonical continuations; external-response overlap with held-out completions is not measured. Offline capture is not evidence of live task-quality parity, H-token emission, continuation health, or decode-step savings.
