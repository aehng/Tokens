"""Deterministic Dataset Partitioning & Split Discipline for Predictor V2.

Guarantees:
- Zero prompt overlap between TRAIN, DEV, and FROZEN TEST.
- Exact domain stratification (code, reasoning, instruction balanced across splits).
- Deterministic SHA-256 fingerprinting of split manifests.
- Frozen test split is isolated and never accessed during model development or early stopping.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.zip2zip.predictor_v2.vanilla_labels import VanillaContinuationRecord

DEFAULT_SPLIT_MANIFEST_PATH = "data/predictor_v2_split_manifest.json"


@dataclass
class SplitManifest:
    schema: str
    git_base: str
    seed: int
    train_ids: List[str]
    dev_ids: List[str]
    frozen_test_ids: List[str]
    domain_counts: Dict[str, Dict[str, int]]
    manifest_hash: str

    def verify_zero_leakage(self) -> None:
        s_train = set(self.train_ids)
        s_dev = set(self.dev_ids)
        s_test = set(self.frozen_test_ids)
        assert len(s_train & s_dev) == 0, f"Leakage: train and dev overlap on {s_train & s_dev}"
        assert len(s_train & s_test) == 0, f"Leakage: train and test overlap on {s_train & s_test}"
        assert len(s_dev & s_test) == 0, f"Leakage: dev and test overlap on {s_dev & s_test}"


def create_deterministic_splits(
    records: Sequence[VanillaContinuationRecord],
    train_ratio: float = 0.60,
    dev_ratio: float = 0.20,
    seed: int = 42,
    git_base: str = "97cecfba5147346e79a46361ce7a58c899e9e4ac",
) -> SplitManifest:
    """Partitions records into TRAIN, DEV, and FROZEN TEST with stratified domain balance."""
    by_domain: Dict[str, List[VanillaContinuationRecord]] = {}
    for r in records:
        by_domain.setdefault(r.domain, []).append(r)

    rng = random.Random(seed)
    train_ids: List[str] = []
    dev_ids: List[str] = []
    frozen_test_ids: List[str] = []
    domain_counts: Dict[str, Dict[str, int]] = {}

    for dom in sorted(by_domain.keys()):
        dom_records = sorted(by_domain[dom], key=lambda x: x.prompt_id)
        rng.shuffle(dom_records)
        n = len(dom_records)
        n_train = int(round(n * train_ratio))
        n_dev = int(round(n * dev_ratio))
        n_test = n - n_train - n_dev

        dom_train = [r.prompt_id for r in dom_records[:n_train]]
        dom_dev = [r.prompt_id for r in dom_records[n_train : n_train + n_dev]]
        dom_test = [r.prompt_id for r in dom_records[n_train + n_dev :]]

        train_ids.extend(dom_train)
        dev_ids.extend(dom_dev)
        frozen_test_ids.extend(dom_test)

        domain_counts[dom] = {
            "train": len(dom_train),
            "dev": len(dom_dev),
            "test": len(dom_test),
            "total": n,
        }

    train_ids.sort()
    dev_ids.sort()
    frozen_test_ids.sort()

    manifest_str = f"seed:{seed}|train:{','.join(train_ids)}|dev:{','.join(dev_ids)}|test:{','.join(frozen_test_ids)}"
    manifest_hash = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()

    manifest = SplitManifest(
        schema="predictor_v2_split_manifest_v1",
        git_base=git_base,
        seed=seed,
        train_ids=train_ids,
        dev_ids=dev_ids,
        frozen_test_ids=frozen_test_ids,
        domain_counts=domain_counts,
        manifest_hash=manifest_hash,
    )
    manifest.verify_zero_leakage()
    return manifest


def save_split_manifest(manifest: SplitManifest, path: str = DEFAULT_SPLIT_MANIFEST_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)


def load_split_manifest(path: str = DEFAULT_SPLIT_MANIFEST_PATH) -> SplitManifest:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    manifest = SplitManifest(**d)
    manifest.verify_zero_leakage()
    return manifest


class BakeoffDataset:
    """Container for partitioned Vanilla continuation records."""

    def __init__(
        self,
        records: Sequence[VanillaContinuationRecord],
        manifest: SplitManifest,
    ):
        self.manifest = manifest
        self.manifest.verify_zero_leakage()
        
        record_map = {r.prompt_id: r for r in records}
        self.train_records = [record_map[pid] for pid in manifest.train_ids if pid in record_map]
        self.dev_records = [record_map[pid] for pid in manifest.dev_ids if pid in record_map]
        self.frozen_test_records = [record_map[pid] for pid in manifest.frozen_test_ids if pid in record_map]

    def verify_split_integrity(self) -> None:
        train_set = {r.prompt_id for r in self.train_records}
        dev_set = {r.prompt_id for r in self.dev_records}
        test_set = {r.prompt_id for r in self.frozen_test_records}
        assert len(train_set & dev_set) == 0
        assert len(train_set & test_set) == 0
        assert len(dev_set & test_set) == 0
        assert len(train_set) + len(dev_set) + len(test_set) == len(self.train_records) + len(self.dev_records) + len(self.frozen_test_records)
