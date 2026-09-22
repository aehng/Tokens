# Quality & Compute Economics Benchmark Report

Definitive held-out benchmark evaluating Original Phi, Official Reactive Zip2Zip, and Our Predictive Zip2Zip (Steps 100 & 150) across 60 frozen validation samples (20 MBPP code, 20 GSM8K reasoning, 20 Alpaca instruction).

---

## 1. Executive Summary

| Metric | Original Phi | Official Zip2Zip | Predictive Step 100 | Predictive Step 150 |
| :--- | :---: | :---: | :---: | :---: |
| **MBPP Code Pass@1** | **10.0%** (2/20) | 0.0% (0/20) | 0.0% (0/20) | 0.0% (0/20) |
| **GSM8K Accuracy** | **65.0%** (13/20) | 50.0% (10/20) | 60.0% (12/20) | 50.0% (10/20) |
| **Alpaca Failure Rate** | **30.0%** (6/20) | 30.0% (6/20) | 20.0% (4/20) | 25.0% (5/20) |
| **Micro Decode Reduction** | 0.0% | 29.27% | 21.85% | 16.87% |
| **Macro Decode Reduction** | 0.0% | 23.53% | 13.94% | 10.13% |
| **Total Hypertokens Emitted** | 0 | 4381 | 2532 | 1840 |
| **Mean Hypers / Output** | 0.00 | 73.02 | 42.20 | 30.67 |
| **Mean Wall Time / Req** | 58.59s | 67.80s | 49.61s | 49.12s |
| **Mean TTFT (Prefill)** | 1.18s | 16.66s | 1.59s | 1.57s |
| **Mean Throughput (tok/s)**| 4.83 | 4.31 | 6.11 | 5.73 |

---

## 2. Two-Stage Quality Loss Decomposition

We separate quality degradation into:
1. **Stage 1 (Baseline Zip2Zip LoRA / LZW Penalty)**: $\text{Original Phi} \to \text{Official Zip2Zip}$
2. **Stage 2 (Predictive Adaptation Delta)**: $\text{Official Zip2Zip} \to \text{Predictive Model}$

| Task Domain | Metric | Vanilla Phi | Official Zip2Zip | Stage 1 $\Delta$ (Official Loss) | Pred Step 100 | Stage 2 $\Delta$ (Step 100) | Pred Step 150 | Stage 2 $\Delta$ (Step 150) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MBPP Code** | Pass@1 | 10.0% | 0.0% | -10.0pp | 0.0% | +0.0pp | 0.0% | +0.0pp |
| **GSM8K Math** | Accuracy | 65.0% | 50.0% | -15.0pp | 60.0% | +10.0pp | 50.0% | +0.0pp |
| **Alpaca** | Failure Rate | 30.0% | 30.0% | +0.0pp | 20.0% | -10.0pp | 25.0% | -5.0pp |

---

## 3. Compute Cost & Economics Analysis

| Compute Metric | Original Phi | Official Zip2Zip | Predictive Step 100 | Predictive Step 150 |
| :--- | :---: | :---: | :---: | :---: |
| **Total Wall Time (60 prompts)** | 3515.3s | 4067.8s | 2976.9s | 2947.0s |
| **Median Latency / Prompt** | 57.21s | 81.17s | 58.58s | 58.50s |
| **Total Transformer Decode Steps** | 16684 | 13677 | 14841 | 14616 |
| **Total Expanded Output Tokens** | 16684 | 19337 | 18991 | 17582 |
| **Net Decode Steps Saved** | 0 | 5660 | 4150 | 2966 |
| **Micro Decode Step Reduction** | 0.0% | 29.27% | 21.85% | 16.87% |
| **Truncation Count (Hit Cap)** | 43 | 34 | 44 | 41 |
| **Severe Repetition Count** | 6 | 6 | 4 | 5 |

---

## 4. Paired Transition Contingency Tables

Comparing prompt-by-prompt transitions relative to Original Vanilla Phi:

### Code Transitions (vs Original Phi)
| Candidate Condition | Both Correct | Orig Correct, Cand Failed (Regression) | Orig Failed, Cand Correct (Recovery) | Both Failed |
| :--- | :---: | :---: | :---: | :---: |
| **official_zip2zip** | 0 | 2 | 0 | 18 |
| **predictive_step_100** | 0 | 2 | 0 | 18 |
| **predictive_step_150** | 0 | 2 | 0 | 18 |

### Reasoning Transitions (vs Original Phi)
| Candidate Condition | Both Correct | Orig Correct, Cand Failed (Regression) | Orig Failed, Cand Correct (Recovery) | Both Failed |
| :--- | :---: | :---: | :---: | :---: |
| **official_zip2zip** | 9 | 4 | 1 | 6 |
| **predictive_step_100** | 7 | 6 | 5 | 2 |
| **predictive_step_150** | 7 | 6 | 3 | 4 |

---

## 5. Quality-Conditional Compression

Examines whether compression occurs on problems the model actually answers correctly vs problems where it fails or drifts:

| Condition | Domain | Preserved Decode Reduction % | Regressed Decode Reduction % | Quality-Preserved Tokens Saved |
| :--- | :--- | :---: | :---: | :---: |
| **official_zip2zip** | code | 0.00% | 23.08% | 0 / 0 |
| **official_zip2zip** | reasoning | 26.45% | 25.56% | 1079 / 4079 |
| **predictive_step_100** | code | 0.00% | 31.27% | 0 / 0 |
| **predictive_step_100** | reasoning | 12.00% | 15.53% | 491 / 4091 |
| **predictive_step_150** | code | 0.00% | 24.34% | 0 / 0 |
| **predictive_step_150** | reasoning | 8.98% | 11.81% | 296 / 3296 |

---

## 6. Failure Mode Review (11 Regressed Prompts)

Representative failure cases where Original Vanilla Phi succeeded but Zip2Zip variants regressed:

### Sample `mbpp_740` (code)
- **Prompt**: Write a Python function to solve the following problem:
Write a function to convert the given tuple to a key-value dictionary using adjacent elements....
- **Failed Models**: official_zip2zip, predictive_step_100, predictive_step_150
- **Original Output Snippet**: `"\nInput:\n(1, 'apple', 3.14, 'banana')\n\nOutput:\n{1: 'apple', 3.14: 'banana'}\n\nNote:\n- The keys should be the first element"`
- **Official Zip2Zip Snippet**: `'\nInput: (1, 2, 3, 4, 5)\n\nOutput: {1: 2, 2: 3, 3: 4, 4: 5}\n\nExample:\n\n```python\ndef convert_to_dict(tup):\n    return dict'`
- **Pred Step 100 Snippet**: `'\nInput: (1,2,3,4,5)\nOutput: {1:2, 2:3, 3:4, 4:5}\n\ndef convert_to_dict(tup):\n    result = {}\n    for i in range(len(tup)-'` (Hypers emitted: 58)
- **Pred Step 150 Snippet**: `'\nInput: (1,2,3,4,5)\nOutput: {1,:2, 2:3, 3:4, 4:5}\n\ndef tuple_to_dict(tup):\n    d = {}\n    for i in range(len(tup)-1):\n  '` (Hypers emitted: 40)

### Sample `mbpp_969` (code)
- **Prompt**: Write a Python function to solve the following problem:
Write a function to join the tuples if they have similar initial elements.
...
- **Failed Models**: official_zip2zip, predictive_step_100, predictive_step_150
- **Original Output Snippet**: `"\nInput:\n[(1, 'a'), (2, 'b'), (1, 'c'), (3, 'd'), (2, 'e')]\n\nOutput:\n[(1, 'a', 'c'), (2, 'b', 'e'), (3, 'd')]\n\nHere's a P"`
- **Official Zip2Zip Snippet**: `'\nExample:\n\nInput:\n[(1, 2), (3, 4), (1, 5), (6, 7), (1, 8), (9, 10)]\n\nOutput:\n[(1, 2), (3, 4), (1, 5), (6, 7), (1, 8), (9'`
- **Pred Step 100 Snippet**: `"\nInput:\n[(1, 'a'), (1, 'b'), (2, 'c'), (2, 'd'), (3, 'e'), (4, 'f')]\nOutput: [(1, 'ab'), (2, 'cd'), (3, 'e'), (4, 'f')]\n"` (Hypers emitted: 97)
- **Pred Step 150 Snippet**: `'\nInput:\n[(1,2),(1,3),(2,4),(2,5),(3,6),(4,7),(5,8),(6,9),(7,10),(8,11),(9,12),(10,11),(11,12)]\nOutput:\n[(1, 2), (1, 3), '` (Hypers emitted: 83)

### Sample `gsm_3022` (reasoning)
- **Prompt**: Solve the following math problem step by step:
Tommy's home is worth 25% more than he bought it for. He sells it and buys a new house that costs $500,...
- **Failed Models**: official_zip2zip, predictive_step_100, predictive_step_150
- **Original Output Snippet**: `'\n\n\nTo solve this problem, we need to work backwards from the information given. Here are the steps:\n\n1. Tommy bought a n'`
- **Official Zip2Zip Snippet**: `"\nLet's denote the price Tommy bought his first house for as P.\n\nAccording to the information given, Tommy's home is now "`
- **Pred Step 100 Snippet**: `'Tommy bought his first house for $375,000.\nTommy bought his house for $375,0000.\nHe sold it for 25% more than he bought '` (Hypers emitted: 36)
- **Pred Step 150 Snippet**: `'If Tommy bought his new house for $500,000, then he paid 25% of that amount in down payment.\nSo, he paid 0.25 * $500,000'` (Hypers emitted: 28)

### Sample `gsm_6442` (reasoning)
- **Prompt**: Solve the following math problem step by step:
Hot dog buns come in packages of 8. For the school picnic, Mr. Gates bought 30 packages of hot dog buns...
- **Failed Models**: predictive_step_150
- **Original Output Snippet**: `'\n# Answer\nFirst, we need to find out the total number of hot dog buns Mr. Gates bought.\n\nHe bought 30 packages of hot do'`
- **Pred Step 150 Snippet**: `'Mr. Gates bought 30 packages of hot dog buns.\nEach package has 8 hot dog buns.\n30 packages x 8 hot dog buns per package '` (Hypers emitted: 28)

### Sample `gsm_6613` (reasoning)
- **Prompt**: Solve the following math problem step by step:
Mr. Curtis has 325 chickens on his farm where 28 are roosters and the rest are hens. Twenty hens do not...
- **Failed Models**: predictive_step_100
- **Original Output Snippet**: `'\n# Answer\nTo find out how many egg-laying hens Mr. Curtis has on his farm, we need to follow these steps:\n\n1. Calculate '`
- **Pred Step 100 Snippet**: `'Mr. Curtis has 325 - 28 = <<325-28=297>>297 chickens on his farm.\nThe number of egg-laying hens is 297 - 20 = <<297-20=2'` (Hypers emitted: 16)

### Sample `gsm_1655` (reasoning)
- **Prompt**: Solve the following math problem step by step:
Abe owns a restaurant. Every month he spends a third of his budget on food, a quarter of his budget on ...
- **Failed Models**: predictive_step_150
- **Original Output Snippet**: `'\nStep 1: Calculate the amount spent on food.\nAbe spends a third of his budget on food, so we need to find one-third of $'`
- **Pred Step 150 Snippet**: `'Abe spends a third of his budget on food, which is $3000 / 3 = <<3000/3=100>>1000.\nHe also spends a quarter of his budge'` (Hypers emitted: 40)

