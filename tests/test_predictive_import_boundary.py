"""Prove the predictive/static stack can import without zip2zip-compression."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


BLOCKER = r"""
import sys

class _BlockZip2ZipCompression:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "zip2zip_compression" or fullname.startswith("zip2zip_compression."):
            raise ImportError("zip2zip-compression is intentionally unavailable")
        return None

sys.meta_path.insert(0, _BlockZip2ZipCompression())

import zip2zip
from zip2zip.static_codebook import StaticCodebookManager
from zip2zip.model import Zip2ZipModel
from zip2zip.predictor_v2.ablation_gates import normalize_wrapper_logits
from src.zip2zip.predictor_v2.attribution_harness import select_stratified_dev_prompts
from src.zip2zip.predictor_v2.forced_oracle import force_oracle_substitutions, h_vs_base_prefix_pair

assert "zip2zip_compression" not in sys.modules
mgr = StaticCodebookManager(
    initial_vocab_size=8,
    max_codebook_size=2,
    max_subtokens=4,
    embedding_dim=4,
    pad_token_id=0,
)
mgr.set_seeded_codebook({}, batch_size=1)
assert mgr.num_seeded == 0
assert callable(select_stratified_dev_prompts)
print("IMPORT_BOUNDARY_OK")
"""


def test_predictive_static_stack_imports_without_zip2zip_compression():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src")])
    completed = subprocess.run(
        [sys.executable, "-c", BLOCKER],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "IMPORT_BOUNDARY_OK" in completed.stdout
