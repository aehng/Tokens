"""Out-of-tree vLLM proof for predictive hypertokens.

The public contract in this package is pure: logical ids, spans, and
request ownership. The vLLM model is loaded only by the plugin entry point.
"""

from tokens_vllm.contract import (
    BASE_VOCAB_SIZE,
    CODEBOOK_SIZE,
    INITIAL_VOCAB_SIZE,
    LOGICAL_VOCAB_SIZE,
    MAX_POSITION_EMBEDDINGS,
    MAX_SUBTOKENS,
    PredictiveCodebook,
    expand_logical_id,
    insert_hypertoken_logits,
    logical_kind,
    physical_embed_id,
    reconstruct_semantic_offset,
    semantic_positions,
    token_request_indices,
    validate_codebook,
)

__all__ = [
    "BASE_VOCAB_SIZE",
    "CODEBOOK_SIZE",
    "INITIAL_VOCAB_SIZE",
    "LOGICAL_VOCAB_SIZE",
    "MAX_POSITION_EMBEDDINGS",
    "MAX_SUBTOKENS",
    "PredictiveCodebook",
    "expand_logical_id",
    "insert_hypertoken_logits",
    "logical_kind",
    "physical_embed_id",
    "reconstruct_semantic_offset",
    "semantic_positions",
    "token_request_indices",
    "validate_codebook",
]
