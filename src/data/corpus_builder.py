"""
Corpus builder for large-scale offline tokenization and vocabulary benchmark.
Ingests:
- OpenAssistant OASST1 (Apache 2.0)
- AllenAI WildChat (ODC-By)
- CodeAlpaca 20k (CC-BY 4.0)
- GSM8K (MIT)
- MBPP (CC-BY 4.0)

Features:
- Prompt-level exact deduplication (SHA-256)
- Conversation/thread-level strictly isolated 80/10/10 splitting
- Pre-tokenization with microsoft/Phi-3.5-mini-instruct (cached token IDs)
- Full metadata tracking
"""

import gzip
import hashlib
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import fastparquet
import numpy as np
from fastparquet import core
from transformers import AutoTokenizer


def compute_thread_split(thread_id: str) -> str:
    """Deterministically assign a thread/conversation to train (80%), val (10%), or test (10%)."""
    h = int(hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 80:
        return "train"
    elif h < 90:
        return "val"
    else:
        return "test"


def load_oasst1(max_pairs: Optional[int] = None) -> List[Dict]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="OpenAssistant/oasst1", filename="2023-04-12_oasst_ready.messages.jsonl.gz", repo_type="dataset")

    msgs = {}
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            msgs[m["message_id"]] = m

    pairs = []
    for m_id, m in msgs.items():
        if m.get("role") == "assistant" and m.get("lang") == "en":
            p_id = m.get("parent_id")
            if p_id in msgs:
                p_msg = msgs[p_id]
                prompt_text = p_msg.get("text", "").strip()
                resp_text = m.get("text", "").strip()
                if len(prompt_text) > 10 and len(resp_text) > 10:
                    pairs.append({
                        "id": f"oasst1_{m_id}",
                        "source_dataset": "oasst1",
                        "domain": "conversation",
                        "thread_id": m["message_tree_id"],
                        "prompt": prompt_text,
                        "response": resp_text,
                    })
                    if max_pairs and len(pairs) >= max_pairs:
                        break
    return pairs


def load_wildchat(max_pairs: Optional[int] = None) -> List[Dict]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="allenai/WildChat", filename="data/train-00000-of-00006.parquet", repo_type="dataset")

    pf = fastparquet.ParquetFile(p)
    pairs = []

    with open(p, "rb") as f:
        for rg_idx, rg in enumerate(pf.row_groups):
            df_meta = pf[rg_idx].to_pandas(columns=["conversation_id", "language", "model"])
            
            col_content = next(c for c in rg.columns if c.meta_data.path_in_schema == ["conversation", "list", "element", "content"])
            assign_content = np.empty(col_content.meta_data.num_values, dtype="O")
            core.read_col(col_content, pf.schema, f, assign=assign_content)

            col_role = next(c for c in rg.columns if c.meta_data.path_in_schema == ["conversation", "list", "element", "role"])
            assign_role = np.empty(col_role.meta_data.num_values, dtype="O")
            core.read_col(col_role, pf.schema, f, assign=assign_role)

            for i in range(len(df_meta)):
                if df_meta["language"].iloc[i] != "English":
                    continue
                conv_id = df_meta["conversation_id"].iloc[i]
                contents = assign_content[i]
                roles = assign_role[i]

                if contents is not None and roles is not None and len(contents) >= 2:
                    user_msg, asst_msg = None, None
                    for c_text, r_name in zip(contents, roles):
                        if r_name == "user" and user_msg is None:
                            user_msg = c_text
                        elif r_name == "assistant" and user_msg is not None:
                            asst_msg = c_text
                            break
                    if user_msg and asst_msg:
                        u_str = str(user_msg).strip()
                        a_str = str(asst_msg).strip()
                        if len(u_str) > 10 and len(a_str) > 10:
                            pairs.append({
                                "id": f"wildchat_{conv_id}_{len(pairs)}",
                                "source_dataset": "wildchat",
                                "domain": "conversation",
                                "thread_id": conv_id,
                                "prompt": u_str,
                                "response": a_str,
                            })
                            if max_pairs and len(pairs) >= max_pairs:
                                return pairs
    return pairs


def load_codealpaca(max_pairs: Optional[int] = None) -> List[Dict]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="sahil2801/CodeAlpaca-20k", filename="code_alpaca_20k.json", repo_type="dataset")

    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pairs = []
    for idx, item in enumerate(raw):
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()
        prompt = f"{inst}\n{inp}".strip() if inp else inst
        if len(prompt) > 10 and len(out) > 10:
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            pairs.append({
                "id": f"codealpaca_{idx}",
                "source_dataset": "codealpaca",
                "domain": "code",
                "thread_id": f"codealpaca_thread_{prompt_hash}",
                "prompt": prompt,
                "response": out,
            })
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def load_gsm8k(max_pairs: Optional[int] = None) -> List[Dict]:
    from huggingface_hub import hf_hub_download
    p_train = hf_hub_download(repo_id="openai/gsm8k", filename="main/train-00000-of-00001.parquet", repo_type="dataset")
    p_test = hf_hub_download(repo_id="openai/gsm8k", filename="main/test-00000-of-00001.parquet", repo_type="dataset")

    pairs = []
    idx = 0
    for p in [p_train, p_test]:
        if not os.path.exists(p):
            continue
        pf = fastparquet.ParquetFile(p)
        df = pf.to_pandas()
        for _, row in df.iterrows():
            q = str(row["question"]).strip()
            a = str(row["answer"]).strip()
            if len(q) > 10 and len(a) > 10:
                q_hash = hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]
                pairs.append({
                    "id": f"gsm8k_{idx}",
                    "source_dataset": "gsm8k",
                    "domain": "reasoning",
                    "thread_id": f"gsm8k_thread_{q_hash}",
                    "prompt": q,
                    "response": a,
                })
                idx += 1
                if max_pairs and len(pairs) >= max_pairs:
                    return pairs
    return pairs


def load_mbpp(max_pairs: Optional[int] = None) -> List[Dict]:
    """Load MBPP examples from existing processed files."""
    pairs = []
    idx = 0
    for fname in ["data/train.jsonl", "data/val.jsonl", "data/test.jsonl"]:
        if not os.path.exists(fname):
            continue
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("domain") == "code" or d.get("dataset") == "code":
                    prompt = d.get("prompt", "").strip()
                    resp = d.get("response", "").strip()
                    if len(prompt) > 10 and len(resp) > 10:
                        p_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
                        pairs.append({
                            "id": f"mbpp_{idx}",
                            "source_dataset": "mbpp",
                            "domain": "code",
                            "thread_id": f"mbpp_thread_{p_hash}",
                            "prompt": prompt,
                            "response": resp,
                        })
                        idx += 1
                        if max_pairs and len(pairs) >= max_pairs:
                            return pairs
    return pairs


def build_corpus(
    targets: Dict[str, int],
    tokenizer_name: str = "microsoft/Phi-3.5-mini-instruct",
    output_dir: str = "data",
    prefix: str = "large_corpus",
) -> Dict[str, int]:
    """
    Build deduplicated, pre-tokenized, thread-isolated train/val/test splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading tokenizer {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print("Collecting raw pairs across datasets...")
    raw_samples: List[Dict] = []

    if targets.get("oasst1", 0) > 0:
        oasst = load_oasst1(targets["oasst1"])
        print(f"Loaded {len(oasst)} OASST1 pairs.")
        raw_samples.extend(oasst)

    if targets.get("wildchat", 0) > 0:
        wild = load_wildchat(targets["wildchat"])
        print(f"Loaded {len(wild)} WildChat pairs.")
        raw_samples.extend(wild)

    if targets.get("codealpaca", 0) > 0:
        code_alp = load_codealpaca(targets["codealpaca"])
        print(f"Loaded {len(code_alp)} CodeAlpaca pairs.")
        raw_samples.extend(code_alp)

    if targets.get("gsm8k", 0) > 0:
        gsm = load_gsm8k(targets["gsm8k"])
        print(f"Loaded {len(gsm)} GSM8K pairs.")
        raw_samples.extend(gsm)

    if targets.get("mbpp", 0) > 0:
        mbpp = load_mbpp(targets["mbpp"])
        print(f"Loaded {len(mbpp)} MBPP pairs.")
        raw_samples.extend(mbpp)

    print(f"Total raw pairs gathered: {len(raw_samples)}")

    # Deduplication by prompt text hash
    seen_prompts = set()
    deduped_samples = []
    for s in raw_samples:
        p_key = hashlib.sha256(s["prompt"].strip().lower().encode("utf-8")).hexdigest()
        if p_key in seen_prompts:
            continue
        seen_prompts.add(p_key)
        deduped_samples.append(s)

    print(f"Deduplicated pairs: {len(deduped_samples)} (dropped {len(raw_samples) - len(deduped_samples)} duplicates)")

    # Pre-tokenize
    print("Tokenizing prompts and responses...")
    prompts = [s["prompt"] for s in deduped_samples]
    responses = [s["response"] for s in deduped_samples]

    prompt_enc = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    resp_enc = tokenizer(responses, add_special_tokens=False)["input_ids"]

    splits = {"train": [], "val": [], "test": []}

    for i, s in enumerate(deduped_samples):
        p_ids = prompt_enc[i]
        r_ids = resp_enc[i]
        if len(p_ids) == 0 or len(r_ids) == 0:
            continue

        split_assignment = compute_thread_split(s["thread_id"])

        record = {
            "id": s["id"],
            "source_dataset": s["source_dataset"],
            "domain": s["domain"],
            "thread_id": s["thread_id"],
            "prompt": s["prompt"],
            "response": s["response"],
            "prompt_token_ids": p_ids,
            "response_token_ids": r_ids,
            "prompt_length": len(p_ids),
            "response_length": len(r_ids),
        }
        splits[split_assignment].append(record)

    # Save to disk
    counts = {}
    for split_name, items in splits.items():
        out_file = os.path.join(output_dir, f"{prefix}_{split_name}.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        counts[split_name] = len(items)
        print(f"Saved {len(items)} samples to {out_file}")

    return counts


if __name__ == "__main__":
    # Stage 1 targets: ~5,000 total pairs
    stage1_targets = {
        "oasst1": 1500,
        "wildchat": 1500,
        "codealpaca": 1200,
        "gsm8k": 800,
        "mbpp": 300,
    }
    print("Building Stage 1 Corpus (~5,000 samples)...")
    c = build_corpus(stage1_targets, prefix="corpus_stage1")
    print("Stage 1 build complete:", c)
