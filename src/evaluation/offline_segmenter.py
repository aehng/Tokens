"""
Optimal Dynamic Programming (DP) Offline Token Segmenter for Request-Specific Codebooks.

Computes the mathematically exact minimal token representation of a sequence of base tokens
given a codebook of 2-3 base-token hypertokens.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple


def segment_tokens_dp(
    tokens: List[int],
    codebook_phrases: Set[Tuple[int, ...]],
) -> Tuple[int, List[Tuple[int, ...]], Dict]:
    """
    Finds the optimal non-overlapping tiling of `tokens` using Base tokens (length 1)
    and `codebook_phrases` (lengths 2 or 3) that minimizes the total emitted token count.

    Returns:
        (compressed_length, emitted_tiles, stats)
    """
    n = len(tokens)
    if n == 0:
        return 0, [], {"compressed_tokens": 0, "base_tokens": 0, "hypertoken_emissions": 0, "unique_hypertokens_used": 0}

    # dp[i] = min tokens to cover prefix tokens[:i]
    dp = [0] * (n + 1)
    backptr = [1] * (n + 1)

    for i in range(1, n + 1):
        # 1. Base token (step = 1)
        best_cost = dp[i - 1] + 1
        best_step = 1

        # 2. Length-2 hypertoken
        if i >= 2:
            t2 = (tokens[i - 2], tokens[i - 1])
            if t2 in codebook_phrases:
                c2 = dp[i - 2] + 1
                if c2 <= best_cost:  # prefer longer step if cost is equal
                    best_cost = c2
                    best_step = 2

        # 3. Length-3 hypertoken
        if i >= 3:
            t3 = (tokens[i - 3], tokens[i - 2], tokens[i - 1])
            if t3 in codebook_phrases:
                c3 = dp[i - 3] + 1
                if c3 <= best_cost:  # prefer longer step
                    best_cost = c3
                    best_step = 3

        dp[i] = best_cost
        backptr[i] = best_step

    # Backtrack to reconstruct tiles
    tiles = []
    curr = n
    hypertoken_emissions = 0
    used_hypertokens = set()
    total_hyper_subtokens = 0

    while curr > 0:
        step = backptr[curr]
        tile = tuple(tokens[curr - step : curr])
        tiles.append(tile)
        if step > 1:
            hypertoken_emissions += 1
            used_hypertokens.add(tile)
            total_hyper_subtokens += step
        curr -= step

    tiles.reverse()

    stats = {
        "base_tokens": n,
        "compressed_tokens": dp[n],
        "tokens_saved": n - dp[n],
        "compression_ratio": dp[n] / n if n > 0 else 1.0,
        "compression_pct": (1.0 - dp[n] / n) * 100.0 if n > 0 else 0.0,
        "hypertoken_emissions": hypertoken_emissions,
        "unique_hypertokens_used": len(used_hypertokens),
        "codebook_utilization": len(used_hypertokens) / len(codebook_phrases) if len(codebook_phrases) > 0 else 0.0,
        "avg_subtokens_per_hypertoken": total_hyper_subtokens / hypertoken_emissions if hypertoken_emissions > 0 else 0.0,
    }

    return dp[n], tiles, stats


def compute_oracle_codebook(
    tokens: List[int],
    k: int = 32,
    min_length: int = 2,
    max_length: int = 3,
) -> Set[Tuple[int, ...]]:
    """
    Computes the optimal Oracle codebook of size K that maximizes tokens saved on `tokens`.
    Greedily selects top K non-overlapping frequent n-grams.
    """
    from collections import Counter

    n = len(tokens)
    if n < min_length:
        return set()

    # Count candidate n-grams
    counts = Counter()
    for l in range(min_length, max_length + 1):
        for i in range(n - l + 1):
            ngram = tuple(tokens[i : i + l])
            counts[ngram] += 1

    # Weight by potential savings: count * (l - 1)
    weighted = [
        (ngram, cnt * (len(ngram) - 1))
        for ngram, cnt in counts.items()
        if cnt >= 1
    ]
    weighted.sort(key=lambda x: x[1], reverse=True)

    oracle_set = set()
    for ngram, _ in weighted[:k]:
        oracle_set.add(ngram)

    return oracle_set
