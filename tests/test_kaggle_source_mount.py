"""CPU checks for locating the proof source dataset under a Kaggle input root."""

from pathlib import Path

import pytest

from experiments.kaggle.bootstrap_kaggle import (
    APPROVED_SOURCE_SHA,
    SourceDatasetError,
    expand_approved_source_sha,
    find_source_dataset,
)

SHA = "dae9bd73f5c3fc4af2633195f3ba6dcebabfe59e"
OTHER = "85722889bb41e30719baf333a00ff6088bc0ec88"


def _mount(root: Path, name: Path | str, sha: str, *, archive: bool = True) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SOURCE_SHA.txt").write_text(sha + "\n", encoding="utf-8")
    if archive:
        (directory / "source.tar.gz").write_bytes(b"proof-archive")
    return directory


def test_approved_sha_placeholder_expands_to_the_commit():
    assert APPROVED_SOURCE_SHA == "$Format:%H$"
    source = (
        'note "$Format:%H$"\n'
        'APPROVED_SOURCE_SHA = "$Format:%H$"\n'
        '_SHA_PLACEHOLDER = \'APPROVED_SOURCE_SHA = "$Format:%H$"\'\n'
    )
    expanded = expand_approved_source_sha(source, SHA)
    assert f'APPROVED_SOURCE_SHA = "{SHA}"\n' in expanded
    assert expanded.count(f'APPROVED_SOURCE_SHA = "{SHA}"') == 1
    assert '_SHA_PLACEHOLDER = \'APPROVED_SOURCE_SHA = "$Format:%H$"\'' in expanded
    bootstrap = Path("experiments/kaggle/bootstrap_kaggle.py").read_text(encoding="utf-8")
    packed = expand_approved_source_sha(bootstrap, SHA)
    assignments = [
        line.strip()
        for line in packed.splitlines()
        if line.strip().startswith("APPROVED_SOURCE_SHA = ")
    ]
    assert assignments == [f'APPROVED_SOURCE_SHA = "{SHA}"']
    with pytest.raises(ValueError):
        expand_approved_source_sha(source, "not-a-sha")


def test_matching_source_dataset_is_selected(tmp_path: Path):
    kaggle = tmp_path / "kaggle" / "input"
    _mount(kaggle, "tokens-step100-gpu-smoke", OTHER)
    chosen = _mount(kaggle, "tokens-vllm-predictive-source", SHA)
    (kaggle / "tokens-step100-gpu-smoke" / "checkpoint_step_100.pt").write_bytes(b"weights")
    found = find_source_dataset(kaggle, SHA)
    assert found == chosen.resolve()
    assert (found / "source.tar.gz").is_file()


def test_wrong_sha_is_rejected(tmp_path: Path):
    kaggle = tmp_path / "kaggle" / "input"
    _mount(kaggle, "tokens-vllm-predictive-source", OTHER)
    with pytest.raises(SourceDatasetError, match="exactly one"):
        find_source_dataset(kaggle, SHA)


def test_multiple_matching_source_datasets_are_rejected(tmp_path: Path):
    kaggle = tmp_path / "kaggle" / "input"
    _mount(kaggle, "tokens-vllm-predictive-source", SHA)
    _mount(kaggle, Path("datasets") / "elikearl" / "tokens-vllm-predictive-source", SHA)
    with pytest.raises(SourceDatasetError, match="exactly one"):
        find_source_dataset(kaggle, SHA)


def test_missing_archive_is_rejected(tmp_path: Path):
    kaggle = tmp_path / "kaggle" / "input"
    _mount(kaggle, "tokens-vllm-predictive-source", SHA, archive=False)
    with pytest.raises(SourceDatasetError, match="source.tar.gz"):
        find_source_dataset(kaggle, SHA)
