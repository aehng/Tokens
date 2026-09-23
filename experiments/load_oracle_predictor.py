"""Load the trusted oracle-guided predictor artifact across script entrypoints."""

from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path
from typing import Any, BinaryIO


def _install_transformers_compat_shims() -> None:
    """Install sys.modules shims for transformers modules that were renamed.

    The oracle_guided_predictor.pkl was pickled against an older version of
    transformers that had ``transformers.tokenization_utils_tokenizers``.
    Modern transformers (>=4.44) removed that submodule; its contents (e.g.
    ``AddedToken``) now live in ``transformers.tokenization_utils_base``.
    We pre-register the old name so Python's pickle resolver can find it.
    """
    old_name = "transformers.tokenization_utils_tokenizers"
    if old_name not in sys.modules:
        try:
            importlib.import_module(old_name)
        except ModuleNotFoundError:
            new_mod = importlib.import_module("transformers.tokenization_utils_base")
            sys.modules[old_name] = new_mod


class _OraclePredictorUnpickler(pickle.Unpickler):
    """Resolve predictor artifacts saved while the training file was __main__."""

    def find_class(self, module: str, name: str) -> Any:
        # Resolve old __main__ path (predictor saved from training script).
        if module == "__main__" and name == "OracleGuidedPredictor":
            from experiments.train_oracle_guided_predictor import OracleGuidedPredictor
            return OracleGuidedPredictor
        # Resolve moved transformers submodule before delegating to pickle.
        if module == "transformers.tokenization_utils_tokenizers":
            module = "transformers.tokenization_utils_base"
        return super().find_class(module, name)


def load_oracle_predictor(source: str | Path | BinaryIO) -> Any:
    """Load a trusted local pickle, including the old __main__ class path.

    Pickle can execute code during deserialization. Only use this for the
    project's locally maintained predictor artifacts, never user uploads.
    """
    _install_transformers_compat_shims()
    if hasattr(source, "read"):
        return _OraclePredictorUnpickler(source).load()
    with Path(source).open("rb") as predictor_file:
        return _OraclePredictorUnpickler(predictor_file).load()
