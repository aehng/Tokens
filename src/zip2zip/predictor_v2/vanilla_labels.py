"""Canonical Vanilla Continuation Data Contract & Label Extraction for Predictor V2.

This module defines the data contract for base-model continuation labels.
Primary occurrence and horizon labels MUST come from canonical Vanilla model generations,
NOT from human/dataset reference answers.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from transformers import AutoTokenizer

CANONICAL_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
CANONICAL_BASE_REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"
RAW_RESULTS_PATH = "experiments/checkpoints/quality_benchmark/raw_results.jsonl"
VAL_PROMPTS_PATH = "data/cached_pure_pred_val_60.json"


def get_canonical_tokenizer() -> AutoTokenizer:
    """Loads the canonical Phi-3.5 tokenizer from local snapshot or hub."""
    local_hub = os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/snapshots")
    if os.path.exists(local_hub):
        snaps = os.listdir(local_hub)
        if snaps:
            snap_dir = os.path.join(local_hub, snaps[0])
            return AutoTokenizer.from_pretrained(snap_dir)
    return AutoTokenizer.from_pretrained(CANONICAL_MODEL_ID, local_files_only=True)


@dataclass
class VanillaContinuationRecord:
    prompt_id: str
    domain: str
    prompt_text: str
    prompt_token_ids: List[int]
    continuation_text: str
    continuation_token_ids: List[int]
    base_model: str = CANONICAL_MODEL_ID
    base_revision: str = CANONICAL_BASE_REVISION
    tokenizer_name: str = CANONICAL_MODEL_ID
    generation_config: Dict[str, Any] = None
    dataset_source: str = "raw_results.jsonl:original_phi"

    def __post_init__(self):
        if self.generation_config is None:
            self.generation_config = {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": 300,
                "pad_token_id": 32000,
                "eos_token_id": 32000,
            }

    def compute_hash(self) -> str:
        s = f"{self.prompt_id}|{self.domain}|{self.prompt_text}|{self.continuation_text}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_canonical_vanilla_records(
    raw_results_path: str = RAW_RESULTS_PATH,
    val_prompts_path: str = VAL_PROMPTS_PATH,
) -> List[VanillaContinuationRecord]:
    """Loads and validates the 60 canonical Vanilla Phi generations from raw_results.jsonl.
    
    Verifies provenance and tokenization against microsoft/Phi-3.5-mini-instruct.
    """
    if not os.path.exists(raw_results_path):
        raise FileNotFoundError(f"Canonical raw results not found at {raw_results_path}")

    # Load prompt definitions
    prompt_meta: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(val_prompts_path):
        with open(val_prompts_path, "r", encoding="utf-8") as f:
            for s in json.load(f):
                prompt_meta[s["id"]] = s

    tokenizer = get_canonical_tokenizer()
    records: List[VanillaContinuationRecord] = []

    with open(raw_results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("condition") != "original_phi":
                continue

            pid = r["prompt_id"]
            domain = r["domain"]
            
            # Use canonical prompt text
            if pid in prompt_meta:
                meta = prompt_meta[pid]
                prompt_text = meta["prompt"]
            else:
                prompt_text = r.get("prompt_text", "")

            continuation_text = r.get("output_text", "")
            
            p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            c_ids = tokenizer.encode(continuation_text, add_special_tokens=False)

            rec = VanillaContinuationRecord(
                prompt_id=pid,
                domain=domain,
                prompt_text=prompt_text,
                prompt_token_ids=p_ids,
                continuation_text=continuation_text,
                continuation_token_ids=c_ids,
            )
            records.append(rec)

    if len(records) != 60:
        raise ValueError(f"Expected exactly 60 original_phi records, found {len(records)}")

    # Sort deterministically by prompt_id
    records.sort(key=lambda x: x.prompt_id)
    return records


def compute_dataset_manifest_hash(records: Sequence[VanillaContinuationRecord]) -> str:
    """Computes a SHA-256 fingerprint over all records."""
    hasher = hashlib.sha256()
    for r in sorted(records, key=lambda x: x.prompt_id):
        hasher.update(r.compute_hash().encode("utf-8"))
    return hasher.hexdigest()
