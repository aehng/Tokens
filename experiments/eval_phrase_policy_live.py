"""Exact-match ceiling on the frozen model's own continuation.

Generates a short greedy answer with hypertokens masked, then asks how many
tokens each phrase policy could have verified. The text is unchanged.
"""
import json
import os
import pickle
import sys
from collections import defaultdict

import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from experiments.phrase_policy import POLICIES, select_phrases
from src.evaluation.offline_segmenter import segment_tokens_dp
from zip2zip import Zip2ZipModel

INITIAL = 32011
MAX_NEW = 48
PER_DOMAIN = 2
MODEL = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"


class MaskHyper(LogitsProcessor):
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def __call__(self, input_ids, scores):
        if scores.shape[-1] > self.vocab_size:
            scores = scores.clone()
            scores[..., self.vocab_size :] = float("-inf")
        return scores


def savings(token_ids, phrases):
    if not token_ids:
        return 0, 0
    comp, _, _ = segment_tokens_dp(list(token_ids), set(phrases))
    return len(token_ids), len(token_ids) - comp


def main():
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    with open("experiments/checkpoints/cached_predictor.pkl", "rb") as f:
        predictor = pickle.load(f)
    records = json.load(open("data/cached_pure_pred_val_60.json", encoding="utf-8"))
    chosen = []
    seen = defaultdict(int)
    for rec in records:
        if seen[rec["domain"]] < PER_DOMAIN:
            chosen.append(rec)
            seen[rec["domain"]] += 1

    print("Loading frozen model...", flush=True)
    model = Zip2ZipModel.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).eval()
    mask = LogitsProcessorList([MaskHyper(INITIAL)])

    totals = {p: {"base": 0, "saved": 0} for p in POLICIES}
    by_domain = {p: defaultdict(lambda: {"base": 0, "saved": 0}) for p in POLICIES}

    for rec in chosen:
        prompt_ids = rec["prompt_token_ids"]
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        with torch.no_grad():
            out = model.base_model.generate(
                input_ids=ids,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                logits_processor=mask,
            )
        gen = out[0, len(prompt_ids) :].tolist()
        text = tokenizer.decode(gen, skip_special_tokens=True).replace("\n", " / ")
        print("\n" + rec["id"], rec["domain"], flush=True)
        print(text[:220], flush=True)
        for policy in POLICIES:
            phrases = select_phrases(
                predictor, tokenizer, prompt_ids, policy, budget=32, domain=rec["domain"]
            )
            base, saved = savings(gen, phrases)
            totals[policy]["base"] += base
            totals[policy]["saved"] += saved
            by_domain[policy][rec["domain"]]["base"] += base
            by_domain[policy][rec["domain"]]["saved"] += saved
            pct = 100.0 * saved / base if base else 0.0
            print(f"  {policy:<28} {pct:5.1f}%  ({saved} tokens)", flush=True)

    print("\nLIVE EXACT-MATCH CEILING", flush=True)
    for policy in POLICIES:
        b = totals[policy]
        pct = 100.0 * b["saved"] / b["base"] if b["base"] else 0.0
        print(f"{policy:<28} {pct:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
