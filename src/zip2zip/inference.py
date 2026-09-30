"""Explicit inference-only model preparation helpers."""

from __future__ import annotations

from torch import nn

from zip2zip.nn.embedding import HyperEmbedding
from zip2zip.nn.linear import HyperLinear


def prepare_model_for_inference(
    model: nn.Module,
    *,
    merge_lora: bool = True,
) -> nn.Module:
    """Set up a Zip2Zip model for inference, optionally merging PEFT adapters.

    This helper is intentionally explicit and inference-only. It never runs
    from model construction or training code.
    """
    if not hasattr(model, "base_model"):
        raise TypeError("prepare_model_for_inference expects a Zip2Zip model")

    base_model = model.base_model
    if merge_lora:
        try:
            from peft import PeftMixedModel, PeftModel
            peft_model_types = (PeftModel, PeftMixedModel)
        except ImportError:
            peft_model_types = ()
    else:
        peft_model_types = ()
    if merge_lora and isinstance(base_model, peft_model_types):
        merge = getattr(base_model, "merge_and_unload", None)
        if merge is None:
            raise TypeError("the active PEFT model does not support merge_and_unload")
        merged_model = merge(safe_merge=True)
        if merged_model is None:
            raise RuntimeError("PEFT merge_and_unload returned no base model")
        model.base_model = merged_model
        manager = getattr(model, "codebook_manager", None)
        clear_caches = getattr(manager, "clear_weight_caches", None)
        if clear_caches is not None:
            # Effective tables prepared before a LoRA merge contain old weights.
            clear_caches()

    model.eval()
    install_hook = getattr(model, "_install_base_position_generation_hook", None)
    if install_hook is not None:
        install_hook()

    base_model = model.base_model
    input_embedding = base_model.get_input_embeddings()
    output_projection = base_model.get_output_embeddings()
    if not isinstance(input_embedding, HyperEmbedding):
        raise RuntimeError("HyperEmbedding did not survive inference preparation")
    if not isinstance(output_projection, HyperLinear):
        raise RuntimeError("HyperLinear did not survive inference preparation")

    manager = getattr(model, "codebook_manager", None)
    if input_embedding.codebook_manager is not manager:
        raise RuntimeError("input HyperEmbedding is attached to a stale codebook manager")
    if output_projection.codebook_manager is not manager:
        raise RuntimeError("output HyperLinear is attached to a stale codebook manager")
    return model
