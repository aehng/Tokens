"""Oracle Compression Experiment: Theoretical Ceiling Analysis.

Evaluates how much token positions can be saved when an oracle knows the prompt and response
ahead of time and seeds the optimal 2-token and 3-token hypertokens into a request-specific codebook.
Tests across budgets: 64, 128, 256, 512, 1024.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple
from dataclasses import dataclass, asdict

import torch
from transformers import AutoTokenizer

from zip2zip.segmenter import DynamicSegmenter


@dataclass
class OracleMetrics:
    budget: int
    base_prompt_tokens: int
    base_response_tokens: int
    base_total_tokens: int
    hyper_prompt_tokens: int
    hyper_response_tokens: int
    hyper_total_tokens: int
    total_compression_pct: float
    prompt_compression_pct: float
    response_compression_pct: float
    avg_tokens_per_hypertoken: float
    codebook_utilization_rate: float
    active_hypertokens_count: int


def extract_candidate_ngrams(
    token_ids: Sequence[int],
    disabled_ids: Set[int],
    min_len: int = 2,
    max_len: int = 3,
) -> Counter[Tuple[int, ...]]:
    """Extract all valid n-grams that do not contain special/disabled tokens."""
    counts: Counter[Tuple[int, ...]] = Counter()
    n = len(token_ids)
    for length in range(min_len, max_len + 1):
        for i in range(n - length + 1):
            gram = tuple(token_ids[i : i + length])
            if not any(t in disabled_ids for t in gram):
                counts[gram] += 1
    return counts


def select_oracle_codebook(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    budget: int,
    disabled_ids: Set[int],
    initial_vocab_size: int = 32064,
) -> Dict[Tuple[int, ...], int]:
    """Greedily select the top-K highest saving n-grams from the prompt and response.

    Savings metric: count * (length - 1).
    """
    full_seq = list(prompt_ids) + list(response_ids)
    counts = extract_candidate_ngrams(full_seq, disabled_ids, min_len=2, max_len=3)

    # Score candidates: priority given to tokens appearing in the response (output steps)
    response_counts = extract_candidate_ngrams(response_ids, disabled_ids, min_len=2, max_len=3)

    def score_gram(gram: Tuple[int, ...]) -> float:
        total_c = counts[gram]
        resp_c = response_counts.get(gram, 0)
        # We value response tokens slightly more because reducing generation steps is the primary goal
        savings = (resp_c * 1.5 + (total_c - resp_c)) * (len(gram) - 1)
        return savings

    # Rank candidate grams
    ranked = sorted(
        [g for g, c in counts.items() if c >= 2 or (len(g) == 3 and c >= 1)],
        key=score_gram,
        reverse=True,
    )

    selected_grams = ranked[:budget]
    codebook: Dict[Tuple[int, ...], int] = {}
    for idx, gram in enumerate(selected_grams):
        codebook[gram] = initial_vocab_size + idx
    return codebook


def evaluate_sample_oracle(
    prompt_ids: List[int],
    response_ids: List[int],
    budget: int,
    disabled_ids: Set[int],
    initial_vocab_size: int = 32064,
) -> OracleMetrics:
    """Run oracle evaluation for a single sample at a given budget."""
    codebook = select_oracle_codebook(
        prompt_ids, response_ids, budget, disabled_ids, initial_vocab_size
    )

    segmenter = DynamicSegmenter(
        subtokens_to_hyper=codebook,
        disabled_ids=disabled_ids,
        max_subtokens=3,
    )

    seg_prompt = segmenter.segment(prompt_ids)
    seg_response = segmenter.segment(response_ids)

    base_p_len = len(prompt_ids)
    base_r_len = len(response_ids)
    base_tot = base_p_len + base_r_len

    hyp_p_len = len(seg_prompt)
    hyp_r_len = len(seg_response)
    hyp_tot = hyp_p_len + hyp_r_len

    tot_comp = (1.0 - hyp_tot / base_tot) * 100.0 if base_tot > 0 else 0.0
    p_comp = (1.0 - hyp_p_len / base_p_len) * 100.0 if base_p_len > 0 else 0.0
    r_comp = (1.0 - hyp_r_len / base_r_len) * 100.0 if base_r_len > 0 else 0.0

    # Count hypertoken usage
    used_hypertokens: Set[int] = set()
    total_subtokens_represented = 0
    total_hyper_occurrences = 0

    rev_codebook = {v: k for k, v in codebook.items()}
    for tok in seg_prompt + seg_response:
        if tok in rev_codebook:
            used_hypertokens.add(tok)
            total_subtokens_represented += len(rev_codebook[tok])
            total_hyper_occurrences += 1

    avg_tokens = (
        total_subtokens_represented / total_hyper_occurrences
        if total_hyper_occurrences > 0
        else 0.0
    )
    utilization = len(used_hypertokens) / budget if budget > 0 else 0.0

    return OracleMetrics(
        budget=budget,
        base_prompt_tokens=base_p_len,
        base_response_tokens=base_r_len,
        base_total_tokens=base_tot,
        hyper_prompt_tokens=hyp_p_len,
        hyper_response_tokens=hyp_r_len,
        hyper_total_tokens=hyp_tot,
        total_compression_pct=tot_comp,
        prompt_compression_pct=p_comp,
        response_compression_pct=r_comp,
        avg_tokens_per_hypertoken=avg_tokens,
        codebook_utilization_rate=utilization * 100.0,
        active_hypertokens_count=len(used_hypertokens),
    )


def run_oracle_experiment(
    samples: List[Tuple[str, str, str]],  # (domain, prompt, response)
    tokenizer_name: str = "microsoft/Phi-3.5-mini-instruct",
    budgets: Sequence[int] = (64, 128, 256, 512, 1024),
) -> Dict[str, Dict[int, OracleMetrics]]:
    """Run oracle experiment across domains and budgets."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    disabled_ids = set(tokenizer.all_special_ids)
    if tokenizer.pad_token_id is not None:
        disabled_ids.add(tokenizer.pad_token_id)
    initial_vocab_size = 32064

    # Group samples by domain
    domains: Dict[str, List[Tuple[str, str]]] = {}
    for dom, p, r in samples:
        domains.setdefault(dom, []).append((p, r))
    domains["overall"] = [(p, r) for _, p, r in samples]

    results: Dict[str, Dict[int, OracleMetrics]] = {}

    for dom_name, pair_list in domains.items():
        results[dom_name] = {}
        for budget in budgets:
            agg_base_p = 0
            agg_base_r = 0
            agg_hyp_p = 0
            agg_hyp_r = 0
            agg_util = 0.0
            agg_avg_tokens = 0.0
            agg_active = 0.0

            for prompt, response in pair_list:
                p_ids = tokenizer.encode(prompt, add_special_tokens=False)
                r_ids = tokenizer.encode(response, add_special_tokens=False)
                if len(p_ids) == 0 or len(r_ids) == 0:
                    continue

                m = evaluate_sample_oracle(
                    p_ids, r_ids, budget, disabled_ids, initial_vocab_size
                )
                agg_base_p += m.base_prompt_tokens
                agg_base_r += m.base_response_tokens
                agg_hyp_p += m.hyper_prompt_tokens
                agg_hyp_r += m.hyper_response_tokens
                agg_util += m.codebook_utilization_rate
                agg_avg_tokens += m.avg_tokens_per_hypertoken
                agg_active += m.active_hypertokens_count

            num_s = len(pair_list)
            base_tot = agg_base_p + agg_base_r
            hyp_tot = agg_hyp_p + agg_hyp_r

            tot_comp = (1.0 - hyp_tot / base_tot) * 100.0 if base_tot > 0 else 0.0
            p_comp = (1.0 - agg_hyp_p / agg_base_p) * 100.0 if agg_base_p > 0 else 0.0
            r_comp = (1.0 - agg_hyp_r / agg_base_r) * 100.0 if agg_base_r > 0 else 0.0

            results[dom_name][budget] = OracleMetrics(
                budget=budget,
                base_prompt_tokens=agg_base_p,
                base_response_tokens=agg_base_r,
                base_total_tokens=base_tot,
                hyper_prompt_tokens=agg_hyp_p,
                hyper_response_tokens=agg_hyp_r,
                hyper_total_tokens=hyp_tot,
                total_compression_pct=tot_comp,
                prompt_compression_pct=p_comp,
                response_compression_pct=r_comp,
                avg_tokens_per_hypertoken=agg_avg_tokens / num_s,
                codebook_utilization_rate=agg_util / num_s,
                active_hypertokens_count=int(agg_active / num_s),
            )

    return results


def get_curated_eval_samples() -> List[Tuple[str, str, str]]:
    """Return a curated benchmark dataset covering Code, Math/Reasoning, and Instruction Following."""
    samples: List[Tuple[str, str, str]] = [
        # Domain 1: Code Generation
        (
            "code",
            "Write a Python function to compute the longest palindromic substring using dynamic programming.",
            """def longest_palindromic_substring(s: str) -> str:
    n = len(s)
    if n <= 1:
        return s
    dp = [[False] * n for _ in range(n)]
    start = 0
    max_len = 1
    for i in range(n):
        dp[i][i] = True
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                if length > max_len:
                    start = i
                    max_len = length
    return s[start : start + max_len]
""",
        ),
        (
            "code",
            "Implement an LRU Cache in Python using an OrderedDict with capacity eviction.",
            """from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
""",
        ),
        (
            "code",
            "Implement a PyTorch module for Multi-Head Self Attention from scratch.",
            """import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(context)
""",
        ),
        # Domain 2: Math and Step-by-Step Reasoning
        (
            "reasoning",
            "Solve the following word problem step by step: Sarah has 4 boxes with 12 apples each. She gives 15 apples to her neighbors and makes 3 pies with 8 apples each. How many apples are left?",
            """Let us solve the problem step by step.
Step 1: Calculate the total number of apples Sarah starts with.
Sarah has 4 boxes, and each box contains 12 apples.
Total initial apples = 4 * 12 = 48 apples.

Step 2: Calculate the number of apples given to neighbors.
Sarah gives away 15 apples.
Apples remaining after giving to neighbors = 48 - 15 = 33 apples.

Step 3: Calculate the number of apples used for baking pies.
Sarah makes 3 pies, and each pie requires 8 apples.
Total apples used for pies = 3 * 8 = 24 apples.

Step 4: Calculate the final remaining apples.
Apples remaining = 33 - 24 = 9 apples.

Therefore, Sarah has 9 apples left.
""",
        ),
        (
            "reasoning",
            "Find all real solutions to the quadratic equation 2x^2 - 7x + 3 = 0. Show all steps.",
            """To solve the quadratic equation:
2x^2 - 7x + 3 = 0

We identify the coefficients of the standard quadratic form ax^2 + bx + c = 0:
a = 2
b = -7
c = 3

Step 1: Compute the discriminant D.
D = b^2 - 4ac
D = (-7)^2 - 4 * (2) * (3)
D = 49 - 24 = 25

Since the discriminant D > 0, there are two distinct real solutions.

Step 2: Apply the quadratic formula.
x = (-b +/- sqrt(D)) / (2a)
x = (-(-7) +/- sqrt(25)) / (2 * 2)
x = (7 +/- 5) / 4

Step 3: Evaluate the two roots.
First root:
x_1 = (7 + 5) / 4 = 12 / 4 = 3

Second root:
x_2 = (7 - 5) / 4 = 2 / 4 = 1/2 = 0.5

Therefore, the solutions are x = 3 and x = 0.5.
""",
        ),
        # Domain 3: Instruction Following / QA
        (
            "instruction",
            "Explain the concept of Tokenization in Natural Language Processing and compare Byte-Pair Encoding with WordPiece.",
            """Tokenization is a fundamental preprocessing step in Natural Language Processing (NLP) that breaks raw text strings into discrete units called tokens. These tokens serve as the basic vocabulary elements input into transformer language models.

1. What is Tokenization?
Raw text consists of continuous streams of characters. Tokenization segments characters into meaningful units, mapping each token to a unique integer ID in a fixed vocabulary. Common approaches include word-level, character-level, and subword tokenization.

2. Byte-Pair Encoding (BPE):
- Origin: Originally a data compression algorithm.
- Mechanism: BPE begins with a base vocabulary of individual characters or bytes. It iteratively finds the most frequent adjacent pair of tokens across the corpus and merges them into a new single token.
- Vocabulary Construction: The process continues for a preset number of merge operations until the desired vocabulary size is achieved.
- Usage: Widely used in GPT models (GPT-2, GPT-4), RoBERTa, and LLaMA.

3. WordPiece:
- Mechanism: Similar to BPE, WordPiece starts with a character vocabulary and iteratively merges pairs. However, instead of simply picking the highest frequency pair, WordPiece selects the pair that maximizes the likelihood of the training data according to a language model.
- Representation: Uses special prefix symbols (such as '##') to indicate subword tokens that are continuations of a word rather than word beginnings.
- Usage: Used in BERT, DistilBERT, and Electra.

In summary, both algorithms prevent Out-Of-Vocabulary (OOV) errors by falling back to characters, but BPE optimizes for raw occurrence frequency whereas WordPiece optimizes for training data likelihood.
""",
        ),
        (
            "instruction",
            "Summarize the key advantages and potential trade-offs of microservices architecture compared to a monolithic architecture.",
            """Microservices architecture decomposes a software application into a suite of small, independent, and loosely coupled services, each running in its own process and communicating via lightweight protocols (such as HTTP REST or gRPC).

Key Advantages:
1. Independent Deployment: Teams can update, fix, and deploy individual services without redeploying the entire application stack.
2. Technology Diversity: Different services can use different programming languages, databases, and frameworks best suited for their specific domain.
3. Fault Isolation: A crash or memory leak in one service (e.g., notification service) does not necessarily bring down critical user flows (e.g., checkout).
4. Granular Scalability: High-load components can be scaled horizontally on dedicated clusters without paying to scale idle components.

Key Trade-offs and Challenges:
1. Operational Complexity: Requires robust orchestration (Kubernetes, Docker), service discovery, API gateways, and distributed logging/tracing.
2. Network Latency: Interservice network calls introduce latency overhead compared to in-memory function calls in a monolith.
3. Data Consistency: Managing distributed transactions across multiple microservices often necessitates eventual consistency rather than ACID transactions.
4. Testing Difficulty: End-to-end integration testing becomes significantly harder with dozens of independently evolving services.
""",
        ),
    ]
    return samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="experiments/oracle_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    samples = get_curated_eval_samples()
    budgets = [64, 128, 256, 512, 1024]

    print("Running Oracle Ceiling Experiment across 3 domains (Code, Reasoning, Instruction)...")
    results = run_oracle_experiment(samples, budgets=budgets)

    # Print formatted markdown table
    print("\n" + "=" * 80)
    print("ORACLE THEORETICAL CEILING RESULTS")
    print("=" * 80)
    print(f"{'Domain':<14} | {'Budget':<6} | {'Base Tokens':<11} | {'Hyper Tokens':<12} | {'Total Comp %':<12} | {'Resp Comp %':<11} | {'Util %':<8} | {'Avg L':<5}")
    print("-" * 80)

    serializable = {}
    for domain, budget_data in results.items():
        serializable[domain] = {}
        for b, m in budget_data.items():
            serializable[domain][b] = asdict(m)
            print(
                f"{domain:<14} | {b:<6} | {m.base_total_tokens:<11} | {m.hyper_total_tokens:<12} | "
                f"{m.total_compression_pct:>10.2f}% | {m.response_compression_pct:>9.2f}% | "
                f"{m.codebook_utilization_rate:>6.1f}% | {m.avg_tokens_per_hypertoken:>4.2f}"
            )
        print("-" * 80)

    with open(args.output, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved full results to {args.output}")
