"""Contextual Hypertoken Emission Gate.

A lightweight LogitsProcessor that restricts hypertoken competition during autoregressive
decoding to contexts where a hypertoken's first constituent base token is plausible
under the current base model logits (e.g. top-N, probability, or logit gap).

This prevents boundary/syntactic mismatches, runaway formatting loops, and out-of-context
emissions without requiring architecture changes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import torch
from transformers import LogitsProcessor

from zip2zip.static_codebook import StaticCodebookManager


class ContextualEmissionGate(LogitsProcessor):
    """Restricts hypertoken logits based on the plausibility of their first base token."""

    def __init__(
        self,
        static_mgr: StaticCodebookManager,
        top_n: Optional[int] = 32,
        min_prob: Optional[float] = None,
        logit_gap: Optional[float] = None,
        enabled: bool = True,
    ) -> None:
        """Initialize the emission gate.

        Args:
            static_mgr: StaticCodebookManager holding the active seeded codebook.
            top_n: If set, only allow hypertoken H=(t_1, t_2, ...) if t_1 is in top_n base tokens.
            min_prob: If set, only allow H if P(t_1) >= min_prob in the base token distribution.
            logit_gap: If set, only allow H if logit(t_1) >= max(base_logits) - logit_gap.
            enabled: Master switch. If False, processor acts as a no-op passthrough.
        """
        self.static_mgr = static_mgr
        self.top_n = top_n
        self.min_prob = min_prob
        self.logit_gap = logit_gap
        self.enabled = enabled

        self.initial_vocab_size = static_mgr.initial_vocab_size
        self.max_codebook_size = static_mgr.max_codebook_size

        # Build mapping from hypertoken slot index (0 to num_seeded-1) -> first base token ID
        self.hyper_first_tokens: List[Tuple[int, int]] = []
        for hyper_id, subtokens in sorted(static_mgr.hyper_to_subtokens.items()):
            if subtokens:
                self.hyper_first_tokens.append((hyper_id, subtokens[0]))

        # Telemetry
        self.positions_evaluated = 0
        self.total_candidates_considered = 0
        self.total_candidates_gated_out = 0
        self.total_candidates_permitted = 0
        self.first_token_ranks: List[int] = []

    def reset_telemetry(self) -> None:
        self.positions_evaluated = 0
        self.total_candidates_considered = 0
        self.total_candidates_gated_out = 0
        self.total_candidates_permitted = 0
        self.first_token_ranks.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return cumulative gating telemetry statistics."""
        mean_rank = (
            float(sum(self.first_token_ranks) / len(self.first_token_ranks))
            if self.first_token_ranks
            else 0.0
        )
        return {
            "enabled": self.enabled,
            "top_n": self.top_n,
            "min_prob": self.min_prob,
            "logit_gap": self.logit_gap,
            "positions_evaluated": self.positions_evaluated,
            "total_candidates_considered": self.total_candidates_considered,
            "total_candidates_gated_out": self.total_candidates_gated_out,
            "total_candidates_permitted": self.total_candidates_permitted,
            "gate_rate_pct": (
                round(
                    100.0
                    * self.total_candidates_gated_out
                    / max(self.total_candidates_considered, 1),
                    2,
                )
            ),
            "mean_first_token_rank": round(mean_rank, 2),
        }

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if not self.enabled or not self.hyper_first_tokens:
            return scores

        batch_size = scores.size(0)
        base_logits = scores[:, : self.initial_vocab_size]

        # Fast precomputation of base distribution criteria
        # 1. Top-N indices per batch item
        top_n_mask = None
        if self.top_n is not None and self.top_n > 0:
            k_val = min(self.top_n, self.initial_vocab_size)
            _, top_indices = torch.topk(base_logits, k=k_val, dim=-1)
            # Create a boolean tensor of shape (batch_size, initial_vocab_size)
            top_n_mask = torch.zeros_like(base_logits, dtype=torch.bool)
            top_n_mask.scatter_(1, top_indices, True)

        # 2. Probability threshold
        prob_dist = None
        if self.min_prob is not None and self.min_prob > 0.0:
            prob_dist = torch.softmax(base_logits, dim=-1)

        # 3. Logit gap threshold
        max_base_logits = None
        if self.logit_gap is not None:
            max_base_logits, _ = torch.max(base_logits, dim=-1, keepdim=True)

        self.positions_evaluated += batch_size

        for hyper_id, first_tok in self.hyper_first_tokens:
            self.total_candidates_considered += batch_size
            for b in range(batch_size):
                allowed = True
                if top_n_mask is not None and not top_n_mask[b, first_tok].item():
                    allowed = False
                elif prob_dist is not None and prob_dist[b, first_tok].item() < self.min_prob:
                    allowed = False
                elif max_base_logits is not None and (
                    base_logits[b, first_tok].item() < (max_base_logits[b].item() - self.logit_gap)
                ):
                    allowed = False

                if not allowed:
                    scores[b, hyper_id] = float("-inf")
                    self.total_candidates_gated_out += 1
                else:
                    self.total_candidates_permitted += 1

        return scores
