"""Strict mount and artifact checks for the approved Kaggle Step-100 dataset.

This module deliberately imports no model or CUDA dependencies so Kaggle
launchers can validate their inputs before importing PyTorch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence


STEP100_DATASET_SLUG = "elikearl/tokens-step100-gpu-smoke"
STEP100_DATASET_VERSION = 1
STEP100_DATASET_MOUNTS = (
    Path("/kaggle/input/tokens-step100-gpu-smoke"),
    Path("/kaggle/input/datasets/elikearl/tokens-step100-gpu-smoke"),
)

STEP100_CHECKPOINT_NAME = "checkpoint_step_100.pt"
STEP100_CHECKPOINT_SHA256 = "2c3606c075ac96dff1f607043f58241251d837f2340ae950d3dc309e9820fd44"
STEP100_CHECKPOINT_SIZE_BYTES = 6_041_080_070
STEP100_PREDICTOR_NAME = "oracle_guided_predictor.pkl"
STEP100_PREDICTOR_SHA256 = "5ea21e53f119e7416e719b835536b4e5ce3b7fab1a989040f3b3052dbe4908f7"
STEP100_PREDICTOR_SIZE_BYTES = 24_512_819


def resolve_approved_step100_mount(
    approved_mounts: Sequence[Path] = STEP100_DATASET_MOUNTS,
    *,
    is_dir: Callable[[Path], bool] | None = None,
) -> Path:
    """Return the one known mount that exists; reject absent or ambiguous inputs."""
    directory_exists = is_dir or Path.is_dir
    present = [
        Path(mount) for mount in approved_mounts if directory_exists(Path(mount))
    ]
    if not present:
        choices = ", ".join(str(Path(mount)) for mount in approved_mounts)
        raise FileNotFoundError(
            f"approved Step-100 dataset mount not found; checked only: {choices}"
        )
    if len(present) != 1:
        choices = ", ".join(str(mount) for mount in present)
        raise RuntimeError(
            f"ambiguous Step-100 dataset mounts; expected exactly one, found: {choices}"
        )
    return present[0]


def validate_step100_artifact_manifest(
    manifest: Mapping[str, object],
    *,
    dataset_slug: str = STEP100_DATASET_SLUG,
) -> None:
    """Require the exact private Step-100 artifact and canonical provenance."""
    expected_top_level = {
        "schema": "tokens_kaggle_private_artifacts_v1",
        "repo": "aehng/Tokens",
        "private_dataset_id": dataset_slug,
        "dataset_visibility": "private",
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "phi_revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
        "zip2zip_id": "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
        "zip2zip_revision": "11c461733a79d2a5de6b814585c3361ca2aacbe7",
        "predictive_condition": "predictive_step_100_compressed_prompt",
        "prompt_representation": "predictive_codebook_dp_segmented",
        "emission_gate_top_n": None,
    }
    for key, expected in expected_top_level.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Step-100 artifact manifest {key} mismatch: "
                f"expected {expected!r}, got {manifest.get(key)!r}"
            )

    expected_artifacts = {
        STEP100_CHECKPOINT_NAME: {
            "sha256": STEP100_CHECKPOINT_SHA256,
            "size_bytes": STEP100_CHECKPOINT_SIZE_BYTES,
        },
        STEP100_PREDICTOR_NAME: {
            "sha256": STEP100_PREDICTOR_SHA256,
            "size_bytes": STEP100_PREDICTOR_SIZE_BYTES,
        },
    }
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("Step-100 artifact manifest has no per-file provenance")

    for name, expected in expected_artifacts.items():
        top_level_section = (
            "checkpoint" if name == STEP100_CHECKPOINT_NAME else "predictor"
        )
        for section_name in ("files", top_level_section):
            section = files if section_name == "files" else manifest.get(section_name)
            artifact = (
                section.get(name)
                if section_name == "files" and isinstance(section, Mapping)
                else section
            )
            if not isinstance(artifact, Mapping):
                raise ValueError(f"Step-100 artifact manifest is missing {section_name}.{name}")
            for field, expected_value in expected.items():
                if artifact.get(field) != expected_value:
                    raise ValueError(
                        f"Step-100 artifact {name} {field} mismatch in {section_name}: "
                        f"expected {expected_value!r}, got {artifact.get(field)!r}"
                    )


def sha256_file(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Hash a file incrementally without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_step100_manifest(mount: Path) -> Mapping[str, object]:
    """Load the mounted manifest and enforce its identity before model imports."""
    manifest_path = Path(mount) / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Step-100 artifact manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    if not isinstance(manifest, Mapping):
        raise ValueError("Step-100 artifact manifest must be a JSON object")
    validate_step100_artifact_manifest(manifest)
    return manifest
