"""Canonical Microsoft Phi-3.5-mini-instruct Continuation Generation Script.

Generates actual frozen Vanilla Phi continuations for the scaled 900-prompt dataset.
Can run on GPU (Kaggle T4 or local CUDA) or CPU (smoke test).
Features:
- Incremental streaming to JSONL (100% restartable)
- Deterministic greedy decoding (temperature=0.0, do_sample=False, max_new_tokens=300)
- Canonical prompt formatting (build_mbpp_prompt for code with assertions; raw prompt for reasoning and instruction)
- Comprehensive provenance logging (model ID, revision, git SHA, torch/transformers versions, timestamps)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# Compatibility patch for transformers >= 4.48 / 5.x with Phi-3.5 trust_remote_code
if not hasattr(DynamicCache, "from_legacy_cache"):
    @classmethod
    def from_legacy_cache(cls, past_key_values=None):
        if isinstance(past_key_values, DynamicCache):
            return past_key_values
        cache = cls()
        if past_key_values is None:
            return cache
        for layer_idx, (key_states, value_states) in enumerate(past_key_values):
            cache.update(key_states, value_states, layer_idx)
        return cache
    DynamicCache.from_legacy_cache = from_legacy_cache

if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, *args, **kwargs: self.get_seq_length()

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from experiments.mbpp_prompt import build_mbpp_prompt

CANONICAL_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
CANONICAL_BASE_REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"
DEFAULT_MANIFEST_PATH = "docs/predictor_v2_scaled_dataset_manifest.json"
DEFAULT_OUTPUT_PATH = "data/canonical_phi_continuations.jsonl"
DEFAULT_MAX_NEW_TOKENS = 300


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Generate Canonical Phi-3.5 Continuations")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH, help="Path to scaled dataset manifest")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSONL path")
    parser.add_argument("--model-id", default=CANONICAL_MODEL_ID, help="HF model ID or path")
    parser.add_argument("--revision", default=CANONICAL_BASE_REVISION, help="Model revision commit")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device", default=None, help="Device (cuda, cpu, xpu, auto)")
    parser.add_argument("--max-prompts", type=int, default=None, help="Optional limit for smoke testing")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("CANONICAL PHI-3.5 CONTINUATION GENERATOR")
    print(f"Device: {device} | Model: {args.model_id} (rev: {args.revision[:8]})")
    print(f"Manifest: {args.manifest}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # 1. Load manifest
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    prompts_dict = manifest["prompts"]
    print(f"Loaded {len(prompts_dict)} prompts from manifest.")

    # 2. Check existing progress (restartability)
    completed_ids: Set[str] = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    completed_ids.add(rec["prompt_id"])
                except Exception:
                    pass
    print(f"Found {len(completed_ids)} already completed prompts in {args.output}.")

    # Filter pending prompts
    pending = [p for pid, p in prompts_dict.items() if pid not in completed_ids]
    if args.max_prompts is not None:
        pending = pending[: args.max_prompts]
    print(f"Pending prompts to generate: {len(pending)}")

    if not pending:
        print("All prompts already generated! Exiting cleanly.")
        return

    # 3. Load Tokenizer & Model
    print(f"Loading tokenizer from {args.model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    except Exception:
        # Fallback to local cache if offline
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)

    print(f"Loading model on {device} (torch_dtype={torch.float16 if device == 'cuda' else torch.float32})...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.revision,
            torch_dtype=dtype,
            trust_remote_code=False,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            local_files_only=True,
            trust_remote_code=False,
        )
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Provenance bundle
    provenance = {
        "model_id": args.model_id,
        "model_revision": args.revision,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "transformers_version": sys.modules.get("transformers", {}).__version__ if "transformers" in sys.modules else "unknown",
        "git_sha": get_git_sha(),
        "chat_template": True,
        "generation_config": {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": 32000,
            "eos_token_id": [32007, 32001, 32000],
        },
    }

    # 4. Generate and stream
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    out_fh = open(args.output, "a", encoding="utf-8")

    total_tokens_generated = 0
    t_start_all = time.perf_counter()

    for idx, item in enumerate(pending, 1):
        pid = item["prompt_id"]
        dom = item["domain"]
        split = item["split"]

        # Canonical prompt formatting
        if dom == "code":
            task_text = build_mbpp_prompt(item)
        else:
            task_text = item["prompt_text"]

        messages = [{"role": "user", "content": task_text}]
        rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        p_tokens = tokenizer.encode(rendered_prompt, add_special_tokens=False)
        p_len = len(p_tokens)
        input_ids = torch.tensor([p_tokens], dtype=torch.long, device=device)

        attention_mask = torch.ones_like(input_ids)
        t0 = time.perf_counter()
        with torch.no_grad():
            output_seq = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=32000,
                eos_token_id=[32007, 32001, 32000],
            )
        gen_time_ms = (time.perf_counter() - t0) * 1000.0

        gen_tokens = output_seq[0, p_len:].tolist()
        num_gen = len(gen_tokens)
        total_tokens_generated += num_gen

        last_tok = gen_tokens[-1] if gen_tokens else None
        eos_reached = last_tok in (32007, 32001, 32000)
        termination_reason = "eos" if eos_reached else "max_new_tokens"

        continuation_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        raw_continuation_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)

        rec = {
            "prompt_id": pid,
            "domain": dom,
            "split": split,
            "task_prompt_text": task_text,
            "rendered_prompt_text": rendered_prompt,
            "prompt_text": rendered_prompt,
            "prompt_token_ids": p_tokens,
            "continuation_text": continuation_text,
            "raw_continuation_text": raw_continuation_text,
            "continuation_token_ids": gen_tokens,
            "generated_token_count": num_gen,
            "num_continuation_tokens": num_gen,
            "termination_reason": termination_reason,
            "termination_token_id": last_tok,
            "hit_max_new_tokens": (termination_reason == "max_new_tokens"),
            "eos_reached": eos_reached,
            "generation_time_ms": round(gen_time_ms, 2),
            "provenance": provenance,
        }


        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()

        if idx % 10 == 0 or idx == len(pending):
            elapsed = time.perf_counter() - t_start_all
            tok_per_sec = total_tokens_generated / max(0.1, elapsed)
            print(f"[{idx}/{len(pending)}] Generated {num_gen} tokens for {pid} ({dom}) in {gen_time_ms:.1f}ms | Overall: {tok_per_sec:.1f} tok/s")

    out_fh.close()
    print(f"\nGeneration complete! Saved {len(pending)} records to {args.output}")


if __name__ == "__main__":
    main()
