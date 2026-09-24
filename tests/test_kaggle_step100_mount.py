import hashlib
import json
from pathlib import Path

import pytest

from experiments.kaggle_step100_mount import (
    STEP100_CHECKPOINT_NAME,
    STEP100_CHECKPOINT_SHA256,
    STEP100_CHECKPOINT_SIZE_BYTES,
    STEP100_DATASET_MOUNTS,
    STEP100_DATASET_SLUG,
    STEP100_PREDICTOR_NAME,
    STEP100_PREDICTOR_SHA256,
    STEP100_PREDICTOR_SIZE_BYTES,
    load_and_validate_step100_manifest,
    resolve_approved_step100_mount,
    sha256_file,
    validate_step100_artifact_manifest,
)


def _manifest():
    checkpoint = {
        "sha256": STEP100_CHECKPOINT_SHA256,
        "size_bytes": STEP100_CHECKPOINT_SIZE_BYTES,
    }
    predictor = {
        "sha256": STEP100_PREDICTOR_SHA256,
        "size_bytes": STEP100_PREDICTOR_SIZE_BYTES,
    }
    return {
        "schema": "tokens_kaggle_private_artifacts_v1",
        "repo": "aehng/Tokens",
        "private_dataset_id": STEP100_DATASET_SLUG,
        "dataset_visibility": "private",
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "phi_revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
        "zip2zip_id": "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
        "zip2zip_revision": "11c461733a79d2a5de6b814585c3361ca2aacbe7",
        "predictive_condition": "predictive_step_100_compressed_prompt",
        "prompt_representation": "predictive_codebook_dp_segmented",
        "emission_gate_top_n": None,
        "checkpoint": checkpoint,
        "predictor": predictor,
        "files": {
            STEP100_CHECKPOINT_NAME: checkpoint,
            STEP100_PREDICTOR_NAME: predictor,
        },
    }


def test_mount_resolver_selects_the_known_nested_kaggle_path(tmp_path):
    direct, nested = (tmp_path / "direct", tmp_path / "nested")
    nested.mkdir()

    selected = resolve_approved_step100_mount(
        [direct, nested], is_dir=lambda path: path.is_dir()
    )

    assert selected == nested


def test_mount_resolver_fails_if_no_approved_mount_exists(tmp_path):
    candidates = [tmp_path / "direct", tmp_path / "nested"]

    with pytest.raises(FileNotFoundError, match="approved Step-100 dataset mount not found"):
        resolve_approved_step100_mount(candidates)


def test_mount_resolver_fails_if_two_approved_mounts_exist(tmp_path):
    candidates = [tmp_path / "direct", tmp_path / "nested"]
    for candidate in candidates:
        candidate.mkdir()

    with pytest.raises(RuntimeError, match="ambiguous Step-100 dataset mounts"):
        resolve_approved_step100_mount(candidates)


def test_default_approved_mounts_include_confirmed_kaggle_nested_location():
    assert STEP100_DATASET_MOUNTS[1] == Path(
        "/kaggle/input/datasets/elikearl/tokens-step100-gpu-smoke"
    )


def test_step100_manifest_matches_canonical_checkpoint_predictor_and_revisions():
    validate_step100_artifact_manifest(_manifest())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("private_dataset_id", "elikearl/other-dataset"),
        ("phi_revision", "unreviewed-revision"),
        ("predictive_condition", "step_150"),
    ],
)
def test_step100_manifest_rejects_wrong_dataset_or_model_provenance(field, value):
    manifest = _manifest()
    manifest[field] = value

    with pytest.raises(ValueError, match=field):
        validate_step100_artifact_manifest(manifest)


def test_step100_manifest_rejects_noncanonical_checkpoint_hash():
    manifest = _manifest()
    manifest["checkpoint"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="checkpoint_step_100.pt sha256 mismatch"):
        validate_step100_artifact_manifest(manifest)


def test_load_manifest_reads_only_from_selected_mount_and_checks_identity(tmp_path):
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps(_manifest()), encoding="utf-8"
    )

    loaded = load_and_validate_step100_manifest(tmp_path)

    assert loaded["private_dataset_id"] == STEP100_DATASET_SLUG


def test_sha256_file_streams_file_contents(tmp_path):
    sample = tmp_path / "small-artifact.bin"
    sample.write_bytes(b"reviewed artifact bytes")

    assert sha256_file(sample, chunk_size=5) == hashlib.sha256(
        b"reviewed artifact bytes"
    ).hexdigest()
