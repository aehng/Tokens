# Predictor V2 Contextual Continuation-Safety Probe Study

**Total Probes Evaluated**: 120 stratified contextual pairs
**Accuracy**: 96.7% | **Precision**: 100.0% | **Recall**: 91.8% | **F1**: 0.957
**False-Safe Rate (FPR - Hazards Admitted)**: 0.00%
**False-Unsafe Rate (FNR - Lost Opportunity)**: 8.16%

## 1. Overall Confusion Matrix

| | Empirically Safe (Actual) | Empirically Unsafe / Hazard (Actual) |
|---|---|---|
| **Heuristic Safe (Predicted)** | True Safe: **45** | False Safe (Hazard!): **0** |
| **Heuristic Unsafe (Predicted)** | False Unsafe (Lost Opp): **4** | True Unsafe (Blocked): **71** |

## 2. Category-Specific Breakdown

| Phrase Category | Probes | Actual Safe | Heuristic Safe | Mean Score | False Safe (FP) | False Unsafe (FN) | Defense Effectiveness |
|---|---|---|---|---|---|---|---|
| `grammatical_glue` | 24 | 24 | 24 | 0.74 | 0 | 0 | **BALANCED** |
| `grounded_entity` | 24 | 24 | 21 | 0.86 | 0 | 3 | **BALANCED** |
| `syntax_hazard` | 24 | 0 | 0 | 0.19 | 0 | 0 | **100% BLOCKED** |
| `trailing_whitespace_hazard` | 24 | 0 | 0 | 0.08 | 0 | 0 | **100% BLOCKED** |
| `ungrounded_numeric` | 24 | 1 | 0 | 0.08 | 0 | 1 | **BALANCED** |

## 3. Domain Breakdown (Code, Reasoning, Instruction)

| Domain | Probes | Empirical Safe | Heuristic Safe | Accuracy | False Safe | False Unsafe |
|---|---|---|---|---|---|---|
| `code` | 54 | 16 | 13 | 94.4% | 0 | 3 |
| `instruction` | 31 | 17 | 16 | 96.8% | 0 | 1 |
| `reasoning` | 35 | 16 | 16 | 100.0% | 0 | 0 |

## 4. Key Empirical Safety Insights

1. **Trailing Whitespace Filtering**: Trailing whitespace is a 100% fatal boundary hazard in BPE models because appending a trailing-space hypertoken prevents greedy next-token merging. Heuristic defense correctly assigns severe penalty, blocking trailing whitespace with 0% false-safe admissions.
2. **Numeric Hallucination Barrier**: Ungrounded numerical phrases (e.g. random multipliers, non-existent ports, fabricated constants) are systematically detected by prompt-overlap checking. Only prompt-grounded numbers pass the threshold.
3. **Syntax Hazard Defense**: Mismatched brackets and isolated syntax fragments are filtered out before emission.
4. **Grammatical Glue Stability**: Common grammatical glue phrases exhibit near 100% empirical compatibility across instruction and reasoning contexts.
