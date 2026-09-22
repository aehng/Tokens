# Offline Predictor Benchmark: Legacy vs Evidence-Aware vs Oracle-Guided

## Executive Summary

In Phase 5, we trained the **Oracle-Guided Predictor-Ranker** using supervision from the Quality-Aware Oracle on the training split (`data/train.jsonl`, 2,779 samples). The model optimizes for **Quality-Aware Value** ($\text{StepsSaved} \times \text{SafetyPrior}$) rather than raw token co-occurrence frequency.

We evaluated all three predictor architectures strictly **OFFLINE** on the 60 held-out validation prompts without live inference generation.

### Comparison Results

| Metric | Legacy CappedPolicy | EvidenceAwareSelector | OracleGuidedPredictor (NEW) | Improvement vs Legacy | Improvement vs Evidence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Codebook Precision** | 28.44% | 20.68% | **27.14%** | **+-1.3pp** | **+6.5pp** |
| **Dead Slot Rate** (lower is better) | 71.56% | 79.32% | **72.86%** | **--1.3pp** | **-6.5pp** |
| **Dangerous Slots** (lower is better) | 1154 (60.1%) | 397 (20.68%) | **510 (26.56%)** | **-644 hazardous slots** | **--113 hazardous slots** |
| **Tokens Saved (DP Offline)** | 1153 (15.6%) | 762 (10.31%) | **959 (12.98%)** | **+-194 tokens** | **+197 tokens** |
| **Quality-Weighted Value** | 586.17 | 648.1 | **983.94** | **+397.8** | **+335.8** |
| **Mean Latency (ms)** | 7.62 ms | 8.93 ms | **8.77 ms** | Fast (< 10 ms requirement) | Fast (< 10 ms requirement) |

---

## 1. Domain Breakdown

### Code (MBPP)
- **Legacy Capped**: Precision = 40.47%, Saved = 512 (13.91%), Dangerous Slots = 464
- **Evidence-Aware**: Precision = 8.28%, Saved = 112 (3.04%), Dangerous Slots = 68
- **Oracle-Guided (NEW)**: Precision = **26.56%**, Saved = **326 (8.86%)**, Dangerous Slots = **79**

### Reasoning (GSM8K)
- **Legacy Capped**: Precision = 37.97%, Saved = 588 (21.51%), Dangerous Slots = 424
- **Evidence-Aware**: Precision = 41.72%, Saved = 571 (20.89%), Dangerous Slots = 271
- **Oracle-Guided (NEW)**: Precision = **47.5%**, Saved = **584 (21.36%)**, Dangerous Slots = **372**

### Instruction (Alpaca)
- **Legacy Capped**: Precision = 6.88%, Saved = 53 (5.44%), Dangerous Slots = 266
- **Evidence-Aware**: Precision = 12.03%, Saved = 79 (8.11%), Dangerous Slots = 58
- **Oracle-Guided (NEW)**: Precision = **7.34%**, Saved = **49 (5.03%)**, Dangerous Slots = **59**

---

## 2. Decision Gate Verdict

> [!IMPORTANT]
> **Decision Gate Analysis:**
> Precision: 27.14% vs 20.68% (Evidence-Aware)
> Quality-Weighted Value: 983.94 vs 648.1
> Dangerous Slots: 510 vs 397
