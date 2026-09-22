"""Predictive Training Data Pipeline with Lossless Round-Trip Guarantee.

Implements Phase 3:
1. Takes prompt and response.
2. Selects K=32 candidate phrases using PROMPT TOKENS ONLY (strictly causal).
3. Maps phrases to temporary hypertoken IDs (starting at initial_vocab_size = 32011).
4. Segments prompt using base + dynamic IDs.
5. Segments response using the SAME dictionary.
6. Verifies 100% exact lossless round-trip token reconstruction:
   decode_sequence(compressed_prompt) == original_prompt_ids
   decode_sequence(compressed_response) == original_response_ids
7. Prepares training tensors (input_ids, labels with prompt masked to -100, codebook_tensor).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple, Any
import torch
from transformers import PreTrainedTokenizerBase

from zip2zip.predictor_policy import CappedPredictorPolicy
from src.evaluation.offline_segmenter import segment_tokens_dp


class PredictivePipeline:
    """Preprocesses training examples into compressed predictive sequences with lossless guarantees."""

    def __init__(
        self,
        predictor_policy: CappedPredictorPolicy,
        tokenizer: PreTrainedTokenizerBase,
        initial_vocab_size: int = 32011,
        max_codebook_size: int = 32,
        max_subtokens: int = 4,
        pad_token_id: int = 32000,
    ) -> None:
        self.policy = predictor_policy
        self.tokenizer = tokenizer
        self.initial_vocab_size = initial_vocab_size
        self.max_codebook_size = max_codebook_size
        self.max_subtokens = max_subtokens
        self.pad_token_id = pad_token_id

    def decode_sequence(
        self, sequence: Sequence[int], codebook_dict: Dict[Tuple[int, ...], int]
    ) -> List[int]:
        """Expand all hypertokens in a sequence back into base tokens."""
        hyper_to_subtokens = {v: list(k) for k, v in codebook_dict.items()}
        expanded: List[int] = []
        for tid in sequence:
            if tid in hyper_to_subtokens:
                expanded.extend(hyper_to_subtokens[tid])
            else:
                expanded.append(tid)
        return expanded

    def process_sample(
        self,
        prompt_text: str,
        response_text: str,
        domain: str = "general",
        curriculum_density: float = 1.0,
    ) -> Dict[str, Any]:
        """Process one (prompt, response) pair into a compressed training sample."""
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)

        # 1. Select codebook using PROMPT ONLY
        codebook_dict, policy_meta = self.policy.select_codebook(prompt_ids)
        ordered_phrases = list(codebook_dict.keys())

        # Curriculum density support: if curriculum_density < 1.0, take top slice of deterministic ranked list
        if curriculum_density < 1.0 and ordered_phrases:
            import math
            keep_count = max(1, int(math.ceil(len(ordered_phrases) * curriculum_density)))
            ordered_phrases = ordered_phrases[:keep_count]
            codebook_dict = {p: self.initial_vocab_size + i for i, p in enumerate(ordered_phrases)}
        phrases_set = set(ordered_phrases)

        # 2. Segment prompt with codebook
        p_comp_len, p_tiles, _ = segment_tokens_dp(prompt_ids, phrases_set)
        compressed_prompt: List[int] = [
            t[0] if len(t) == 1 else codebook_dict[tuple(t)]
            for t in p_tiles
        ]

        # 3. Segment response with the SAME codebook
        r_comp_len, r_tiles, _ = segment_tokens_dp(response_ids, phrases_set)
        compressed_response: List[int] = [
            t[0] if len(t) == 1 else codebook_dict[tuple(t)]
            for t in r_tiles
        ]

        # 4. Lossless Round-Trip Verification (Hard Assertion)
        recon_prompt = self.decode_sequence(compressed_prompt, codebook_dict)
        recon_response = self.decode_sequence(compressed_response, codebook_dict)

        if recon_prompt != prompt_ids:
            raise ValueError(
                f"Prompt round-trip mismatch! Original len {len(prompt_ids)}, recon len {len(recon_prompt)}"
            )
        if recon_response != response_ids:
            raise ValueError(
                f"Response round-trip mismatch! Original len {len(response_ids)}, recon len {len(recon_response)}"
            )

        # 5. Construct training tokens and labels
        input_ids = compressed_prompt + compressed_response
        # Mask prompt tokens with -100 so loss is computed only on response tokens
        labels = [-100] * len(compressed_prompt) + compressed_response

        # 6. Build codebook tensor for encoder: (max_codebook_size, max_subtokens)
        codebook_tensor = torch.full(
            (self.max_codebook_size, self.max_subtokens),
            self.pad_token_id,
            dtype=torch.long,
        )
        for gram, hid in codebook_dict.items():
            slot_idx = hid - self.initial_vocab_size
            if slot_idx < self.max_codebook_size:
                codebook_tensor[slot_idx, : len(gram)] = torch.tensor(gram, dtype=torch.long)

        # 7. Compute hypertoken spans
        spans = torch.ones(self.max_codebook_size, dtype=torch.long)
        for gram, hid in codebook_dict.items():
            slot_idx = hid - self.initial_vocab_size
            if slot_idx < self.max_codebook_size:
                spans[slot_idx] = len(gram)

        # Response statistics
        resp_hypers_used = sum(1 for t in r_tiles if len(t) > 1 and tuple(t) in phrases_set)

        return {
            "domain": domain,
            "original_prompt_ids": prompt_ids,
            "original_response_ids": response_ids,
            "compressed_prompt_ids": compressed_prompt,
            "compressed_response_ids": compressed_response,
            "input_ids": input_ids,
            "labels": labels,
            "codebook_dict": {str(k): v for k, v in codebook_dict.items()},
            "codebook_tuples": list(codebook_dict.keys()),
            "codebook_tensor": codebook_tensor,
            "spans": spans,
            "prompt_tokens_saved": len(prompt_ids) - len(compressed_prompt),
            "response_tokens_saved": len(response_ids) - len(compressed_response),
            "response_hypers_used": resp_hypers_used,
            "policy_meta": policy_meta,
        }
