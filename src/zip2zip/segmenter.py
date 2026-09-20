from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import torch


class DynamicSegmenter:
    """Optimal dynamic programming token segmenter for seeded hypertoken dictionaries.

    Takes a sequence of base token IDs and replaces matching 2-3 base-token subsequences
    with seeded hypertokens, strictly minimizing the total resulting token count.
    Never merges across special-token or disabled-token boundaries.
    """

    def __init__(
        self,
        subtokens_to_hyper: Dict[Tuple[int, ...], int],
        disabled_ids: Optional[Set[int]] = None,
        max_subtokens: int = 3,
    ) -> None:
        """
        Args:
            subtokens_to_hyper: Dictionary mapping tuple of base token IDs -> hypertoken ID.
            disabled_ids: Set of token IDs (e.g. special/control tokens) that must never be merged.
            max_subtokens: Maximum length of a hypertoken sub-sequence (typically 3).
        """
        self.subtokens_to_hyper = subtokens_to_hyper
        self.disabled_ids = set(disabled_ids) if disabled_ids else set()
        self.max_subtokens = max_subtokens

    def segment(self, token_ids: Sequence[int]) -> List[int]:
        """Segment a sequence of base tokens using DP to minimize total token count.

        Args:
            token_ids: List of base token IDs.

        Returns:
            Optimally compressed list of tokens containing base tokens and hypertokens.
        """
        n = len(token_ids)
        if n == 0:
            return []

        # dp[i] = min token count for token_ids[:i]
        # parent[i] = (prev_index, token_emitted)
        dp = [float("inf")] * (n + 1)
        parent: List[Tuple[int, int]] = [(0, 0)] * (n + 1)
        dp[0] = 0

        for i in range(n):
            current_cost = dp[i]
            if current_cost == float("inf"):
                continue

            # Option 1: emit single base token
            t_curr = token_ids[i]
            if current_cost + 1 < dp[i + 1]:
                dp[i + 1] = current_cost + 1
                parent[i + 1] = (i, t_curr)

            # Option 2: emit a matching hypertoken of length L in [2..max_subtokens]
            # Check for disabled/special tokens
            if t_curr in self.disabled_ids:
                continue

            for length in range(2, min(self.max_subtokens + 1, n - i + 1)):
                subseq = tuple(token_ids[i : i + length])
                # Check if any token in candidate span is disabled
                if any(t in self.disabled_ids for t in subseq):
                    # Cannot cross or include disabled token
                    break

                if subseq in self.subtokens_to_hyper:
                    hyper_id = self.subtokens_to_hyper[subseq]
                    next_i = i + length
                    # Cost is 1 (replacing 'length' tokens with 1 hypertoken)
                    if current_cost + 1 < dp[next_i]:
                        dp[next_i] = current_cost + 1
                        parent[next_i] = (i, hyper_id)

        # Backtrack to reconstruct the optimal token sequence
        result: List[int] = []
        curr = n
        while curr > 0:
            prev_i, tok = parent[curr]
            result.append(tok)
            curr = prev_i

        result.reverse()
        return result

    def segment_batch(
        self, batch_ids: Sequence[Sequence[int]]
    ) -> List[List[int]]:
        """Segment a batch of token ID sequences."""
        return [self.segment(seq) for seq in batch_ids]


def segment_tokens_with_dictionary(
    token_ids: Sequence[int],
    subtokens_to_hyper: Dict[Tuple[int, ...], int],
    disabled_ids: Optional[Set[int]] = None,
    max_subtokens: int = 3,
) -> List[int]:
    """Helper function to segment a single sequence of base tokens."""
    segmenter = DynamicSegmenter(
        subtokens_to_hyper=subtokens_to_hyper,
        disabled_ids=disabled_ids,
        max_subtokens=max_subtokens,
    )
    return segmenter.segment(token_ids)
