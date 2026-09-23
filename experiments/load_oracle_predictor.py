"""Load the trusted oracle-guided predictor artifact across script entrypoints."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, BinaryIO


class _OraclePredictorUnpickler(pickle.Unpickler):
    """Resolve predictor artifacts saved while the training file was __main__."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "OracleGuidedPredictor":
            from experiments.train_oracle_guided_predictor import OracleGuidedPredictor

            return OracleGuidedPredictor
        return super().find_class(module, name)


def load_oracle_predictor(source: str | Path | BinaryIO) -> Any:
    """Load a trusted local pickle, including the old __main__ class path.

    Pickle can execute code during deserialization. Only use this for the
    project's locally maintained predictor artifacts, never user uploads.
    """
    if hasattr(source, "read"):
        return _OraclePredictorUnpickler(source).load()
    with Path(source).open("rb") as predictor_file:
        return _OraclePredictorUnpickler(predictor_file).load()
