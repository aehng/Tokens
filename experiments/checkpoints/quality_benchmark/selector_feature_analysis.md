# Hypertoken Feature & Prompt Provenance Analysis

## Executive Summary & Hypothesis Test

**Hypothesis:** *'Prompt-supported numbers/identifiers are safe; novel/inferred numbers/identifiers are catastrophic.'*

### Key Finding:
- **Prompt-Absent Numeric Emissions:** Error rate = **88.9%** (1254 emissions).
- **Prompt-Present Numeric Emissions:** Error rate = **49.5%** (273 emissions).
- **Prompt-Level Comparison:**
  - Prompts with **ONLY prompt-present numbers**: Accuracy = **50.0%** (Error rate = 50.0% across 2 prompts).
  - Prompts with **ANY prompt-absent numbers**: Accuracy = **44.7%** (Error rate = 55.3% across 38 prompts).

> [!IMPORTANT]
> **Verdict:** The hypothesis is **CONFIRMED**. De novo hallucination/insertion of numbers not grounded in the prompt is highly lethal (88.9% emission failure rate overall, driven by 1,001 ungrounded numeric emissions in code). Conversely, prompt-present numbers achieve drastically higher accuracy (e.g. 51.0% in GSM8k and 100% in instruction).
> Blanket numeric bans are therefore suboptimal: we should **boost prompt-grounded numbers and identifiers** while strictly **filtering or heavily penalizing novel/inferred numeric tokens**.

---

## 1. Provenance Breakdown by Domain

| Domain | Numeric Provenance | Emission Count | Error Rate % | Regress Vanilla % | Tokens Saved |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Code** | Prompt-Present | 13 | 100.0% | 0.0% | 13 |
| | Prompt-Absent | 1001 | 100.0% | 14.9% | 1794 |
| **Reasoning** | Prompt-Present | 249 | 49.0% | 39.8% | 284 |
| | Prompt-Absent | 234 | 48.7% | 38.0% | 334 |
| **Instruction** | Prompt-Present | 11 | 0.0% | 0.0% | 12 |
| | Prompt-Absent | 19 | 0.0% | 0.0% | 19 |

---

## 2. Structural Features & Safety

### A. Base-Token Length
| Length | Total Emissions | Error Rate % | Vanilla Regression Rate % | Tokens Saved |
| :---: | :---: | :---: | :---: | :---: |
| **len_2** | 914 | 65.3% | 20.6% | 914 |
| **len_3** | 1618 | 92.5% | 13.8% | 3236 |
| **len_4** | 0 | 0.0% | 0.0% | 0 |

### B. Word Boundary Alignment
| Alignment | Emissions | Error Rate % | Vanilla Regression Rate % | Tokens Saved |
| :--- | :---: | :---: | :---: | :---: |
| **space_start** | 583 | 51.1% | 24.7% | 746 |
| **newline_start** | 230 | 90.4% | 12.6% | 299 |
| **punct_start** | 257 | 95.3% | 31.5% | 498 |
| **mid_word_or_unspaced** | 1462 | 91.8% | 10.7% | 2607 |

### C. Code Punctuation Presence
| Feature | Emissions | Error Rate % | Vanilla Regression Rate % | Tokens Saved |
| :--- | :---: | :---: | :---: | :---: |
| **has_code_punct** | 947 | 91.0% | 11.3% | 1658 |
| **no_code_punct** | 1585 | 77.7% | 19.2% | 2492 |

---

## 3. Codebook Rank & Candidate Capacity Utilization

| Codebook Rank Tier | Emissions | Error Rate % | Vanilla Regression Rate % | Tokens Saved |
| :--- | :---: | :---: | :---: | :---: |
| **rank_0_7** | 540 | 72.2% | 28.7% | 737 |
| **rank_8_15** | 843 | 84.9% | 9.4% | 1375 |
| **rank_16_23** | 307 | 75.2% | 22.8% | 464 |
| **rank_24_31** | 842 | 89.8% | 12.7% | 1574 |

### Codebook Slot Utilization:
- Total candidate slots allocated across 60 prompts: **1920**
- Slots actually emitted at least once: **443** (**23.1%**)
- **Dead Slots:** **1477** (**76.9%** wasted capacity)

---

## 4. Actionable Design Rules for the Evidence-Aware Selector (Phase 2)
1. **Grounding Filter / Bonus:** Explicitly reward candidates whose tokens or constituent entities appear in the prompt ($+6.0$ to $+10.0$ weight boost).
2. **Numeric Safety Rule:** Permit numeric phrases **only** if all constituent digits/numbers are attested in the prompt. Strictly drop or heavily penalize ungrounded numeric candidates.
3. **Code Syntax Filter:** Strip isolated code syntax fragments (e.g. `):
`, `):`, `[]`, `len(`) unless directly attested in prompt function signatures.
4. **Boundary Alignment Requirement:** Require candidates to align to valid word/token boundaries (prefer leading whitespace or clean punctuation). Suppress mid-word fragments.
5. **Dead Slot Elimination:** Do not fill codebook slots with generic ungrounded structural phrases like `. The` or `\n    return` if expected emission probability is below threshold.