"""Extracted-archive dry-run for the attribution Kaggle source bundle."""

from __future__ import annotations

from pathlib import Path

from experiments.pack_attribution_source import (
    REQUIRED_SYMBOLS,
    archive_relative,
    dry_run_extracted_package,
    extract_source_archive,
    iter_source_files,
    write_source_archive,
)


def test_attribution_source_archive_contains_runner_and_harness_symbols(tmp_path: Path):
    files = {archive_relative(path) for path in iter_source_files()}
    assert "experiments/run_phi_attribution_benchmark.py" in files
    assert "src/zip2zip/predictor_v2/attribution_harness.py" in files
    assert "src/zip2zip/nn/codebook_api.py" in files
    archive = tmp_path / "attribution_source.tar.gz"
    report = write_source_archive(archive)
    assert archive.is_file()
    assert report["sha256"]
    extracted = extract_source_archive(archive, tmp_path / "extracted")
    dry = dry_run_extracted_package(extracted)
    assert dry["status"] == "PASS"
    assert dry["stdout"] == "DRY_RUN_OK"
    assert dry["runner_help_status"] == "PASS"
    assert "--conditions" in dry["runner_help_excerpt"]
    assert "--logit-only" in dry["runner_help_excerpt"]
    harness = (extracted / "src/zip2zip/predictor_v2/attribution_harness.py").read_text(encoding="utf-8")
    assert "def select_stratified_dev_prompts" in harness
    assert "STRATIFIED_DEV12_PROMPT_IDS" in harness
    required = {symbol for _, symbol in REQUIRED_SYMBOLS}
    assert "select_stratified_dev_prompts" in required
