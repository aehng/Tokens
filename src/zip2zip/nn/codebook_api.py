"""Structural codebook-manager API used by HyperEmbedding / HyperLinear.

The predictive/static path duck-types this protocol. The legacy LZW
``CodebookManager`` also satisfies it, but importing that class pulls in
``zip2zip-compression``. Keep this module free of that dependency.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, runtime_checkable

import torch


@runtime_checkable
class HyperCodebookManager(Protocol):
    initial_vocab_size: int
    max_codebook_size: int
    fast_inference_ready: bool
    effective_embedding_weight_cache: Optional[torch.Tensor]
    effective_linear_weight_cache: Optional[torch.Tensor]
    effective_linear_bias_cache: Optional[torch.Tensor]

    def get_hyper_embedding_weights(
        self,
        ids: torch.LongTensor,
        base_weight: torch.Tensor,
        encoder_fn: Any,
    ) -> torch.Tensor: ...

    def get_hyper_linear_weights(
        self, base_weight: torch.Tensor, encoder_fn: Any
    ) -> torch.Tensor: ...

    def reset(self) -> None: ...

    def prepare_input_ids(
        self,
        ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.LongTensor: ...

    def init_codebooks_and_hyper_weight_cache(
        self, batch_size: int, codebooks: Optional[List[Any]] = None
    ) -> None: ...
