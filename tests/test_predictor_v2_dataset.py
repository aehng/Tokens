"""Tests for Predictor V2 Dataset and Manifest."""

import pytest
from src.zip2zip.predictor_v2.dataset import create_deterministic_splits, BakeoffDataset
from src.zip2zip.predictor_v2.vanilla_labels import load_canonical_vanilla_records, compute_dataset_manifest_hash


def test_dataset_manifest_and_splits():
    recs = load_canonical_vanilla_records()
    assert len(recs) == 60
    h1 = compute_dataset_manifest_hash(recs)
    h2 = compute_dataset_manifest_hash(recs)
    assert h1 == h2

    manifest = create_deterministic_splits(recs, seed=42)
    assert len(manifest.train_ids) == 36
    assert len(manifest.dev_ids) == 12
    assert len(manifest.frozen_test_ids) == 12

    dataset = BakeoffDataset(recs, manifest)
    dataset.verify_split_integrity()
