"""Tokens / Zip2Zip package.

Legacy LZW helpers (``CodebookManager``, ``Zip2ZipTokenizer``) are imported
lazily so the predictive/static stack can load without ``zip2zip-compression``.
"""

from .static_codebook import StaticCodebookManager
from .emission_gate import ContextualEmissionGate
from .segmenter import DynamicSegmenter, segment_tokens_with_dictionary
from .model import Zip2ZipModel
from .inference import prepare_model_for_inference
from .config import Zip2ZipConfig, CompressionConfig
from .nn.encoders.config import (
    EncoderType,
    AttentionEncoderConfig,
    TransformerEncoderConfig,
)

_LAZY_EXPORTS = {
    "CodebookManager": (".codebook", "CodebookManager"),
    "Zip2ZipTokenizer": (".tokenizer", "Zip2ZipTokenizer"),
}

__all__ = [
    "Zip2ZipModel",
    "prepare_model_for_inference",
    "Zip2ZipTokenizer",
    "Zip2ZipConfig",
    "CompressionConfig",
    "EncoderType",
    "AttentionEncoderConfig",
    "TransformerEncoderConfig",
    "CodebookManager",
    "StaticCodebookManager",
    "ContextualEmissionGate",
    "DynamicSegmenter",
    "segment_tokens_with_dictionary",
]


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


