"""Multi-domain Dataset Loader with strict Zero-Leakage Train/Val/Test Split.

Loads datasets across 3 key domains:
1. Code: MBPP (Python programming problems and solutions)
2. Reasoning: GSM8K (Multi-step mathematical reasoning)
3. Instruction/QA: Stanford Alpaca (Instruction following)

Performs deterministic 80% Train / 10% Val / 10% Test split.
Saves data/train.jsonl, data/val.jsonl, data/test.jsonl.
Strictly ensures test data is isolated and never accessed during phrase bank construction.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
import requests


@dataclass
class DatasetSample:
    domain: str
    id: str
    prompt: str
    response: str


def fetch_mbpp_samples() -> List[DatasetSample]:
    """Fetch Python code generation samples from MBPP."""
    url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    samples: List[DatasetSample] = []
    for line in resp.text.strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)
        task_id = str(data.get("task_id", len(samples)))
        prompt_text = data.get("text", "")
        test_setup = "\n".join(data.get("test_list", []))
        code = data.get("code", "")

        full_prompt = f"Write a Python function to solve the following problem:\n{prompt_text}\n"
        full_response = f"{code}\n\n# Tests\n{test_setup}"

        samples.append(
            DatasetSample(
                domain="code",
                id=f"mbpp_{task_id}",
                prompt=full_prompt,
                response=full_response,
            )
        )
    return samples


def fetch_gsm8k_samples() -> List[DatasetSample]:
    """Fetch mathematical reasoning samples from GSM8K."""
    # Fetch from both train and test to have sufficient pool
    urls = [
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    ]
    samples: List[DatasetSample] = []
    for url in urls:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        for line in resp.text.strip().split("\n"):
            if not line:
                continue
            data = json.loads(line)
            q = data.get("question", "")
            a = data.get("answer", "")
            samples.append(
                DatasetSample(
                    domain="reasoning",
                    id=f"gsm_{len(samples)}",
                    prompt=f"Solve the following math problem step by step:\n{q}\n",
                    response=a,
                )
            )
    return samples


def fetch_alpaca_samples(limit: int = 2000) -> List[DatasetSample]:
    """Fetch general instruction following samples from Stanford Alpaca."""
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    resp = requests.get(url, timeout=45)
    resp.raise_for_status()
    raw = json.loads(resp.text)

    # Use first `limit` items for balanced domain representation
    samples: List[DatasetSample] = []
    for idx, item in enumerate(raw[:limit]):
        inst = item.get("instruction", "")
        inp = item.get("input", "")
        out = item.get("output", "")

        if inp:
            prompt = f"Instruction: {inst}\nInput: {inp}\nAnswer:"
        else:
            prompt = f"Instruction: {inst}\nAnswer:"

        samples.append(
            DatasetSample(
                domain="instruction",
                id=f"alpaca_{idx}",
                prompt=prompt,
                response=out,
            )
        )
    return samples


def create_splits(
    data_dir: str = "data",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[DatasetSample], List[DatasetSample], List[DatasetSample]]:
    """Build and save stratified train/val/test splits."""
    os.makedirs(data_dir, exist_ok=True)

    print("Fetching datasets from authoritative sources...")
    mbpp = fetch_mbpp_samples()
    print(f"Loaded {len(mbpp)} code samples (MBPP).")

    gsm = fetch_gsm8k_samples()
    print(f"Loaded {len(gsm)} reasoning samples (GSM8K).")

    alpaca = fetch_alpaca_samples(limit=2000)
    print(f"Loaded {len(alpaca)} instruction samples (Alpaca).")

    rng = random.Random(seed)

    train_all: List[DatasetSample] = []
    val_all: List[DatasetSample] = []
    test_all: List[DatasetSample] = []

    # Stratified split per domain
    for domain_name, sample_list in [("code", mbpp), ("reasoning", gsm), ("instruction", alpaca)]:
        rng.shuffle(sample_list)
        n = len(sample_list)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        d_train = sample_list[:n_train]
        d_val = sample_list[n_train : n_train + n_val]
        d_test = sample_list[n_train + n_val :]

        train_all.extend(d_train)
        val_all.extend(d_val)
        test_all.extend(d_test)
        print(f"Domain {domain_name:<12}: {len(d_train):>5} train, {len(d_val):>4} val, {len(d_test):>4} test")

    # Shuffle combined sets
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    # Save to jsonl
    for name, s_list in [("train", train_all), ("val", val_all), ("test", test_all)]:
        path = os.path.join(data_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for s in s_list:
                f.write(json.dumps(asdict(s)) + "\n")
        print(f"Saved {len(s_list)} samples to {path}")

    return train_all, val_all, test_all


def load_split(split: str = "test", data_dir: str = "data") -> List[DatasetSample]:
    """Load an existing split from disk."""
    path = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split {path} not found. Run dataset_loader.py first.")
    samples: List[DatasetSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                samples.append(DatasetSample(**d))
    return samples


if __name__ == "__main__":
    train, val, test = create_splits()
    print(f"\nFinal Split Summary:")
    print(f"  TRAIN: {len(train)} (80%)")
    print(f"  VAL:   {len(val)} (10%)")
    print(f"  TEST:  {len(test)} (10%)")
    print("Zero-leakage dataset preparation complete.")
