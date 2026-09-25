"""Predictive Phi-3.5 causal LM for vLLM 0.30.0 Model Runner V2.

Physical embedding and LM-head rows stay at 32064. The Hugging Face config
vocab size stays at 32096 so the sampler sees the logical vocabulary.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
import torch.nn as nn

from tokens_vllm.contract import (
    BASE_VOCAB_SIZE,
    CODEBOOK_SIZE,
    LOGICAL_VOCAB_SIZE,
    validate_position_mode,
)
from tokens_vllm.remap import insert_h_logits, remap_logical_ids
from tokens_vllm.warmup import mask_inactive_h_logits

_H_ENABLED_ENV = "TOKENS_PREDICTIVE_H_ENABLED"


def _h_enabled_from_env() -> bool:
    return os.environ.get(_H_ENABLED_ENV, "0") == "1"


class PredictivePhi3Model(nn.Module):
    """Placeholder replaced by the vLLM Llama subclass at import time.

    The real class is built in ``_build_classes`` so importing this module
    without vLLM fails only when the plugin loads.
    """


class PredictivePhi3ForCausalLM(nn.Module):
    """Placeholder. See ``_build_classes``."""


def _build_classes() -> None:
    global PredictivePhi3Model, PredictivePhi3ForCausalLM
    from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel
    from vllm.model_executor.models.phi3 import Phi3ForCausalLM

    class _PredictivePhi3Model(LlamaModel):
        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            physical, is_h, slots = remap_logical_ids(input_ids)
            embeds = self.embed_tokens(physical)
            state = getattr(self, "predictive_state", None)
            if state is None or not getattr(self, "h_enabled", False):
                return embeds
            req = state.token_req_indices[: input_ids.shape[0]]
            active = state.h_active[req]
            h_vec = state.h_input[req, slots]
            use_h = is_h & active
            return torch.where(use_h.unsqueeze(-1), h_vec.to(dtype=embeds.dtype), embeds)

    class _PredictivePhi3ForCausalLM(Phi3ForCausalLM):
        def __init__(self, *, vllm_config, prefix: str = ""):
            config = vllm_config.model_config.hf_config
            logical = int(getattr(config, "vocab_size", LOGICAL_VOCAB_SIZE))
            if logical != LOGICAL_VOCAB_SIZE:
                raise ValueError(
                    f"logical vocab_size must be {LOGICAL_VOCAB_SIZE}, got {logical}"
                )
            base_vocab = int(
                getattr(config, "predictive_base_vocab_size", BASE_VOCAB_SIZE)
            )
            if base_vocab != BASE_VOCAB_SIZE:
                raise ValueError(
                    f"physical vocab must be {BASE_VOCAB_SIZE}, got {base_vocab}"
                )
            # Llama sizes the embedding and LM head from config.vocab_size.
            # Present the physical width only while those parameters are built,
            # then restore the logical width the sampler already snapshotted.
            config.vocab_size = base_vocab
            try:
                super().__init__(vllm_config=vllm_config, prefix=prefix)
            finally:
                config.vocab_size = logical
            self.logical_vocab_size = logical
            self.physical_vocab_size = base_vocab
            self.position_mode = validate_position_mode(
                getattr(config, "position_mode", "compressed")
            )
            self.model.position_mode = self.position_mode
            self.h_enabled = _h_enabled_from_env()
            self.model.h_enabled = self.h_enabled
            self.predictive_state = None
            self.input_encoder = None
            self.output_encoder = None
            self.pad_token_id = int(getattr(config, "eos_token_id", 32000) or 32000)

        def _init_model(self, vllm_config, prefix: str = "", layer_type=None):
            kwargs = {"vllm_config": vllm_config, "prefix": prefix}
            if layer_type is not None:
                kwargs["layer_type"] = layer_type
            return _PredictivePhi3Model(**kwargs)

        @staticmethod
        def get_model_state_cls():
            from tokens_vllm.state import PredictiveModelState

            return PredictiveModelState

        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.model.embed_input_ids(input_ids)

        def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
            base_logits = self.logits_processor(self.lm_head, hidden_states)
            if base_logits is None:
                return None
            if base_logits.shape[-1] != BASE_VOCAB_SIZE:
                raise RuntimeError(
                    "physical logits width "
                    f"{base_logits.shape[-1]} != {BASE_VOCAB_SIZE}"
                )
            rows = base_logits.shape[0]
            state = self.predictive_state
            if state is None or not self.h_enabled:
                h_logits = torch.full(
                    (rows, CODEBOOK_SIZE),
                    float("-inf"),
                    dtype=base_logits.dtype,
                    device=base_logits.device,
                )
            else:
                req = state.logit_req_indices[:rows]
                h_weight = state.h_output[req].to(dtype=hidden_states.dtype)
                flat = hidden_states.reshape(rows, -1)
                h_logits = torch.bmm(flat.unsqueeze(1), h_weight.transpose(1, 2)).squeeze(1)
                h_logits = h_logits.to(dtype=base_logits.dtype)
                h_logits = mask_inactive_h_logits(h_logits, state.h_active[req])
            return insert_h_logits(base_logits, h_logits)

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            return LlamaForCausalLM.load_weights(self, weights)

    PredictivePhi3Model = _PredictivePhi3Model
    PredictivePhi3ForCausalLM = _PredictivePhi3ForCausalLM


try:
    _build_classes()
except ImportError:
    # Package import stays usable without vLLM. The plugin imports this module
    # only after vLLM is installed.
    pass
