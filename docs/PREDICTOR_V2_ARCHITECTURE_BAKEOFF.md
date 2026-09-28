# Predictor V2 Architecture Bake-Off: Framework, Oracle Hierarchy, and Empirical Benchmark

**Date:** 2026-09-28  
**Branch:** `codex/predictor-v2-architecture-bakeoff`  
**Starting Commit:** `97cecfba5147346e79a46361ce7a58c899e9e4ac`  
**Status:** In Progress / Experimental Specification and Benchmark

---

## 1. Executive Summary & Problem Formulation

In earlier phases of the Tokens project, integration correctness of predictive hypertokens with Microsoft Phi-3.5-mini-instruct and vLLM 0.30.0 was proven (Phases 1–10 passed on Kaggle kernels v18 and v19 on Tesla T4). With serving and runtime mechanics verified, the dominant research priority is **task quality and quality-preserved decode acceleration**.

While predictor/codebook quality is the leading hypothesis for historical quality regressions, quality is an end-to-end property of the entire pipeline. We must not assume every issue stems from the predictor, nor should we select an ad-hoc architecture without controlled empirical evidence.

This framework answers six foundational questions:
1. **Global Opportunity:** How much useful hypertoken opportunity actually exists in the base model's own continuation?
2. **Candidate Generation Loss:** How much opportunity is lost because the prompt-only candidate generator misses useful phrases?
3. **Ranker Loss:** How much is lost because the ranker fails to prioritize useful candidates into the top-$K$ codebook?
4. **Continuation Safety Loss:** How much is lost because candidate hypertokens alter continuation dynamics (logits, hidden states, EOS)?
5. **Live Emission Loss:** How much remaining opportunity is actually realized during live autoregressive decode?
6. **Architecture Selection:** Which lightweight predictor architecture yields the Pareto-optimal tradeoff between decode-step reduction and prefill latency (TTFT impact)?

---

## 2. Audit of Current Predictor and Oracles

To establish rigorous science, we audit the existing predictor and oracles, documenting their capabilities and theoretical limitations.

### 2.1 Current Predictor (`OracleGuidedPredictor`)
- **Candidate Sources:**
  1. *Prompt n-grams:* All $n$-grams of length 2–4 appearing in prompt tokens, seeded with base weight 8.0.
  2. *Token Association Index:* Co-occurrence lookups $\text{token\_associations}[\text{token\_id}]$ built from training corpus.
  3. *Static Background Bank:* Top 32 global corpus phrases weighted by $0.2 \times \text{bg\_weight}$.
- **Feature Vector (21 handcrafted features):**
  - `log_raw_weight`: $\log(1 + \text{raw\_weight})$
  - `exact_in_prompt`: Binary flag for exact prompt substring match
  - `prompt_count`: Frequency count in prompt
  - `word_overlap_ratio`: Constituent word overlap with prompt
  - `phrase_len_2`, `phrase_len_3`, `phrase_len_4`: One-hot phrase length
  - `char_len`: Character length of decoded phrase
  - `leading_space`: Binary flag (starts with space or `\n`)
  - `trailing_space`: Binary flag (ends with space or `\t`)
  - `mid_word_start`: Binary flag (starts alphanumeric without leading space)
  - `is_numeric`: Contains digits
  - `numeric_grounded`: All digits present in prompt text
  - `numeric_ungrounded`: Contains digits absent from prompt
  - `is_function_name`: Matches `def foo` or `foo(`
  - `function_grounded`: Function identifier present in prompt
  - `syntax_hazard`: Unbalanced brackets or in `CODE_SYNTAX_FRAGMENTS`
  - `is_grammatical_glue`: Present in `GRAMMATICAL_GLUE` set
  - `domain_code`, `domain_reasoning`, `domain_instruction`: One-hot domain indicators
- **Hard Filters (Pre-ranking rejection):**
  - `is_bare_punctuation`: Stripped punctuation strings
  - `trailing_space`: Rejects strings ending in space or tab
  - `numeric_ungrounded`: Rejects numbers not grounded in prompt
  - `ungrounded_function`: Rejects `def func` when `func` is absent from prompt
- **Scorer & Model:**
  - Ridge regression ($\lambda = 1.0$) with closed-form solution: $\theta = (X^T X + \lambda I)^{-1} X^T y$.
- **Target Variable:**
  - Scalar target: $\text{target\_value} = \text{marginal\_steps\_saved} \times \text{safety\_prior}$ from the Quality-Aware Oracle. Non-oracle candidates received target 0.0.
- **Supervision Source & Historical Limitation:**
  - Supervised on `data/train.jsonl` (2,779 samples).
  - **CRITICAL LIMITATION:** The supervision labels were derived from **human/dataset reference responses** (`train.jsonl`), **NOT** the frozen base model's actual autoregressive continuation. Human reference solutions frequently use different variable names, phrasing, and syntax than Phi-3.5-mini-instruct, causing label mismatch.

### 2.2 Current "Quality-Aware Oracle" (`experiments/build_quality_aware_oracle.py`)
- **Safety Prior Formulation:**
  - Starts with neutral base score $0.80$, scaled by multiplicative heuristic rules:
    - Trailing whitespace hazard: $\times 0.20$
    - Leading boundary aligned: $\times 1.05$; Mid-word hazard: $\times 0.70$
    - Exact prompt match: $\times 1.25$; Constituent words $\ge 67\%$: $\times 1.10$; Novel all words: $\times 0.85$
    - Grounded numeric: $\times 0.90$; Ungrounded numeric: $\times 0.10$
    - Grounded function: $\times 1.15$; Ungrounded function: $\times 0.15$
    - Unbalanced brackets: $\times 0.30$; Isolated syntax: $\times 0.40$
    - High-frequency grammatical glue: $\times 1.30$
    - Clean structural indentation: $\times 1.15$
    - Clamped to $[0.01, 1.0]$.
- **Continuation Probing:**
  - **NO contextual continuation probes.** It loaded summary statistics from an old run (`continuation_step_100.json`) purely for informational logging. Every score is a deterministic rule on phrase text and prompt tokens.
- **Data Source:**
  - Evaluated on dataset reference text (`ground_truth_response` or `response`), not base model continuation.
- **Candidate Filtering & Optimization:**
  - Filters `safety_prior >= 0.50` and `not has_trailing_space`.
  - Truncates candidates to top 40 by isolated value ($\text{isolated\_saved} \times \text{safety\_prior}$).
  - Optimization: Accelerated lazy greedy forward selection using dynamic programming segmentation (`segment_tokens_dp`).
- **Classification:** Heuristic teacher / safety baseline. Not a true oracle.

### 2.3 `OracleV2` (`src/evaluation/oracle_v2.py`)
- **Candidate Extraction:** Rolling $n$-grams (lengths 2–3) from response tokens.
- **Candidate Pre-filtering:** Filters to $\text{count} \ge 2$ or $\text{len} \ge 3$, sorted by $\text{count} \times (\text{len} - 1)$, truncated to `candidate_limit` (default 80, in study 40).
- **Optimization:** Beam search (beam width 2 to 4) over marginal DP savings.
- **Exactness:** Near-optimal heuristic. Severely constrained by candidate limit and greedy beam search.

### 2.4 `ExactOracle` (`src/evaluation/oracle_v2.py`)
- **Mathematical Exactness:** Exhaustively evaluates all $\binom{M}{s}$ subsets for $s \le K$.
- **Pre-truncation:** Truncates candidates to `candidate_pool_limit = 20` based on unconstrained upper bound before combinatorial evaluation.
- **Exactness Limitation:** Exact **only** within that pre-filtered 20-candidate pool. It cannot evaluate candidates ranked 21+, and cannot construct a codebook of size $K > 20$. It is **not** an unrestricted global oracle.

---

## 3. The New Three-Tier Oracle Hierarchy

To replace heuristic conflation, we establish three distinct oracle concepts:

```
[ Canonical Vanilla Continuation Tokens ]
                   │
                   ▼
       ┌───────────────────────┐
       │       ORACLE A        │  Global Occurrence Oracle
       │  (Unrestricted Upper  │  Input: All occurring 2-4 grams in continuation
       │     Bound Ceiling)    │  Method: Exact DP segmentation + Dominance Pruned BnB
       └───────────┬───────────┘
                   │  Loss: Candidate Generation Deficit
                   ▼
       ┌───────────────────────┐
       │       ORACLE B        │  Fixed Candidate-Pool Oracle
       │   (Candidate Ceiling  │  Input: Shared prompt-only candidate pool (256-512)
       │    over Shared Pool)  │  Method: Exact / Lazy Greedy DP subset selection
       └───────────┬───────────┘
                   │  Loss: Ranker Inefficiency
                   ▼
       ┌───────────────────────┐
       │  PREDICTOR RANKERS    │  Lightweight Predictor V2 Candidates
       │   (Prompt-Only K=32)  │  Ridge, Pooled MLP, CNN, GRU, 1-Layer Transformer
       └───────────┬───────────┘
                   │  Loss: Continuation Incompatibility / Safety Hazard
                   ▼
       ┌───────────────────────┐
       │       ORACLE C        │  Empirical Continuation-Safe Oracle
       │    (Quality-Aware     │  Input: Contextual H vs Base continuation probes
       │     Ground Truth)     │  Metrics: KL, Top-1 agreement, EOS delta
       └───────────────────────┘
```

### 3.1 Oracle A: Global Occurrence Oracle
- **Input:** Canonical frozen Vanilla Phi continuation token stream.
- **Candidate Pool:** Every valid occurring $n$-gram of length 2–4 in the continuation. No prompt-only filtering, no safety filtering.
- **Optimization:**
  - Evaluates exact realized decode-step reduction using DP token tiling (`segment_tokens_dp`).
  - Solves $\max_{S \subseteq \mathcal{C}_{\text{all}}, |S| \le K} (N - \text{len}(\text{DP}(S)))$.
  - Exact branch-and-bound with lossless dominance pruning (removing sub-phrases strictly covered by super-phrases with equal or fewer occurrences) and lazy greedy bounds.
  - Where search space permits, mathematically exact; otherwise tight provable upper/lower bounds with optimality gap recorded.
- **Evaluated at:** $K \in \{8, 16, 32\}$.
- **Meaning:** The absolute physical ceiling of decode steps that could be saved if the future token stream were known perfectly.

### 3.2 Oracle B: Fixed Candidate-Pool Oracle
- **Input:** The exact prompt-only candidate pool ($\approx 256\text{--}512$ candidates) generated for that prompt, without seeing the response.
- **Method:** Evaluates which subset of $\le K$ candidates from the fixed pool maximizes realized DP savings on the Vanilla continuation.
- **Candidate Generation Capture:**
  $$\text{Capture}_{\text{cand}} = \frac{\text{StepsSaved}(\text{Oracle B})}{\text{StepsSaved}(\text{Oracle A})}$$
- **Meaning:** Quantifies the exact fraction of theoretical savings lost solely due to candidate generation recall.

### 3.3 Oracle C: Empirical Continuation-Safe Oracle
- **Input:** Specific candidate phrases occurring in specific context positions in Vanilla generations.
- **Measurement:** Compares autoregressive continuation from base path vs hypertoken path:
  - Base Path: $\text{Prompt} \to t_1, t_2, \dots, t_n \to \text{Continuation}$
  - H Path: $\text{Prompt} \to H \to \text{Continuation}$
- **Measured Metrics:**
  - Next-token KL divergence $D_{\text{KL}}(P_{\text{base}} \parallel P_H)$
  - Top-1 token agreement
  - Top-$k$ token overlap
  - 3–5 token continuation match
  - EOS probability shift
- **Contextuality:** Safety is contextual to $(s_{\text{id}}, \text{phrase}, \text{context\_pos})$.
- **Baseline Comparison:** The existing rule-based `heuristic_safety_prior` is benchmarked directly against empirical continuation probes to determine false-safe and false-unsafe rates.

---

## 4. Vanilla Continuation Data Contract & Split Discipline

### 4.1 Data Contract
The predictor's task at inference is: *Given prompt $P$, predict which multi-token phrases the frozen base model will emit in its autoregressive continuation.*
- **Model:** `microsoft/Phi-3.5-mini-instruct` (3.8B parameters)
- **Base Revision:** `2fe192450127e6a83f7441aef6e3ca586c338b77`
- **Tokenizer:** PreTrainedTokenizerFast from Microsoft Phi-3.5-mini-instruct
- **Decoding Contract:** Greedy decoding (`do_sample=False`, `temperature=0.0`), `max_new_tokens=300`, `pad_token_id=eos_token_id`.
- **Prompt Formatter:** Canonical task prompt formatting (`build_mbpp_prompt` for code with test assertions visible, raw instruction prompt for GSM8K and Alpaca).
- **Labels:** Derived strictly from base model generations, never from human reference answers.

### 4.2 Split Discipline
- Historical validation prompts have been tuned on in prior research. We explicitly partition the benchmark into three non-overlapping splits with preserved domain balance:
  - **TRAIN (60%):** 36 prompts (12 code, 12 reasoning, 12 instruction)
  - **DEV (20%):** 12 prompts (4 code, 4 reasoning, 4 instruction) — used for hyperparameter tuning and model selection
  - **FROZEN TEST (20%):** 12 prompts (4 code, 4 reasoning, 4 instruction) — strictly held out, evaluated only once
- Split manifests are deterministically hashed via SHA-256.

---

## 5. Candidate Pool & Feature Extraction Contract

Every architecture receives the exact same candidate pool per prompt:
1. **Prompt n-grams:** Length 2, 3, 4 extracted from prompt token IDs.
2. **Token Associations:** Lookups from the precomputed corpus co-occurrence index.
3. **Background Phrases:** Top 32 domain-general background phrases.
4. **Candidate Size:** 256 to 512 unique candidate phrases per prompt.

Each candidate record contains:
- `tokens`: Tuple of token IDs
- `text`: Decoded UTF-8 string
- `length`: 2, 3, or 4
- `features`: 21-dimensional normalized feature vector
- `labels`:
  - `occurs_in_vanilla`: Boolean
  - `occurrence_count`: Non-negative integer
  - `first_occurrence_index`: Token index of first occurrence (-1 if absent)
  - `first_occurrence_bucket`: Categorical index (0: [0, 31], 1: [32, 127], 2: [128, 255], 3: [256+], 4: Never)
  - `isolated_steps_saved`: $(\text{length} - 1) \times \text{count}$
  - `marginal_dp_saved`: Realized marginal savings in Oracle B forward selection

---

## 6. Predictor V2 Architectures Under Evaluation

All architectures implement the `PredictorScorer` interface:
$$\text{score}(P, \mathcal{C}, X) \to (\hat{p}_{\text{occur}}, \hat{c}_{\text{count}}, \hat{h}_{\text{horizon}}, \hat{p}_{\text{safe}})$$

Primary Ranking Utility:
$$\text{Utility}(H) = \hat{p}_{\text{occur}} \times \hat{c}_{\text{count}} \times (\text{len}(H) - 1)$$

### Architecture Specifications:

| Model ID | Architecture | Context Representation | Candidate Representation | Interaction / Heads | Target Parameters |
|---|---|---|---|---|---|
| **A** | **Retrained Ridge Baseline** | Handcrafted 21 features | Features only | Linear closed-form $\hat{y} = X\theta + b$ | 22 params |
| **B** | **Pooled Embedding + MLP** | Mean, Max, Recency-weighted pool of token embeddings (dim=96) | Mean pool of candidate token embeddings (dim=96) | Concat $[u, v, u \odot v, \langle u, v \rangle, X_{\text{hand}}] \to$ 2-layer MLP (192 $\to$ 96 $\to$ heads) | ~150k params |
| **C** | **CNN + Suffix Model** | 1D Conv (kernels 2, 3, 5, filters=32 each) + Suffix pool (last 32 tokens) | 1D Conv over candidate tokens | Concat $[h_{\text{conv}}, h_{\text{suffix}}, h_{\text{cand}}, X_{\text{hand}}] \to$ MLP | ~250k params |
| **D** | **Small GRU + MLP** | 1-Layer causal GRU (hidden=128), final hidden state | Candidate token embedding pool | Concat $[h_{\text{gru}}, h_{\text{cand}}, X_{\text{hand}}] \to$ MLP | ~300k params |
| **E** | **1-Layer Transformer** | 1-Layer Transformer Encoder (dim=128, 4 heads, FFN=256) | Learned query pooling over candidate | Cross-attention / Concat $[z_{\text{prompt}}, z_{\text{cand}}, X_{\text{hand}}] \to$ MLP | ~450k params |

---

## 7. Pre-Registered Hypotheses

- **H0:** Handcrafted Ridge regression remains competitive with neural rankers because prompt co-occurrence and token-length priors dominate occurrence probability.
- **H1:** Simple pooled semantic prompt context captures $>85\%$ of the gain achievable by sequence models at $<20\%$ of the compute cost.
- **H2:** Local order and suffix conditioning (CNN / Suffix) improve occurrence prediction for code and reasoning by prioritizing the prompt's final request sentences.
- **H3:** Recurrent sequential context (GRU) provides minimal incremental benefit over CNN/suffix pooling while increasing CPU TTFT latency.
- **H4:** Global self-attention (Transformer) does not justify its $O(N^2)$ prefill compute overhead for candidate ranking.
- **H5:** Candidate generation recall, rather than ranking, is the primary bottleneck in capturing the Global Occurrence Oracle ceiling.
- **H6:** The historical rule-based `heuristic_safety_prior` has a false-safe rate $>25\%$ when evaluated against empirical continuation KL probes.
- **H7:** Safe-oracle codebooks preserve quality where unconstrained codebooks fail, confirming that continuation safety is an independent axis from occurrence frequency.

---

## 8. Live Attribution Experiment Protocol (Phase 18)

Prepared for execution following offline selection (subject to GPU resource approval):
- **Arm A:** Vanilla Phi-3.5-mini-instruct (Control)
- **Arm B:** Predictive model with hypertokens disabled ($K=0$)
- **Arm C1:** Predictive model + Global Occurrence Oracle Codebook (Diagnostic future knowledge)
- **Arm C2:** Predictive model + Empirically Safety-Filtered Oracle Codebook
- **Arm D:** Predictive model + Legacy `OracleGuidedPredictor`
- **Arm E:** Predictive model + Top-1 Predictor V2 Architecture
- **Arm F:** Predictive model + Top-2 Predictor V2 Architecture

---
