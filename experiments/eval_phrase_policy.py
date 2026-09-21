"""Compare phrase policies on the 60 held-out prompts.

Compression is the exact-match ceiling: a verifier may accept a predicted
phrase only where the reference tokens already contain it. Gold responses
estimate the opportunity. Pass --live to measure the same ceiling on the
frozen model's own greedy continuation.
"""
import json
import os
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from transformers import AutoTokenizer

from experiments.phrase_policy import POLICIES, kind_counts, select_phrases
from src.evaluation.offline_segmenter import segment_tokens_dp

VAL_PATH = "data/cached_pure_pred_val_60.json"
PREDICTOR_PATH = "experiments/checkpoints/cached_predictor.pkl"
OUT_PATH = "experiments/phrase_policy_results.json"


def compress(token_ids, phrases):
    if not token_ids:
        return 0, 0
    comp_len, _, _ = segment_tokens_dp(list(token_ids), set(phrases))
    return len(token_ids), len(token_ids) - comp_len


def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    with open(PREDICTOR_PATH, "rb") as f:
        predictor = pickle.load(f)
    records = json.load(open(VAL_PATH, encoding="utf-8"))

    totals = {p: defaultdict(lambda: {"base": 0, "saved": 0, "n": 0, "kinds": defaultdict(int)}) for p in POLICIES}

    for rec in records:
        prompt_ids = rec["prompt_token_ids"]
        response_ids = tokenizer.encode(rec["ground_truth_response"], add_special_tokens=False)
        domain = rec["domain"]
        for policy in POLICIES:
            phrases = select_phrases(
                predictor, tokenizer, prompt_ids, policy, budget=32, domain=domain
            )
            base, saved = compress(response_ids, phrases)
            bucket = totals[policy][domain]
            bucket["base"] += base
            bucket["saved"] += saved
            bucket["n"] += 1
            for kind, count in kind_counts(tokenizer, phrases).items():
                bucket["kinds"][kind] += count
            all_bucket = totals[policy]["all"]
            all_bucket["base"] += base
            all_bucket["saved"] += saved
            all_bucket["n"] += 1
            for kind, count in kind_counts(tokenizer, phrases).items():
                all_bucket["kinds"][kind] += count

    report = {}
    print(f"{'policy':<28} {'all':>8} {'code':>8} {'instr':>8} {'reason':>8}  kinds/prompt")
    for policy in POLICIES:
        report[policy] = {}
        cells = []
        for domain in ("all", "code", "instruction", "reasoning"):
            b = totals[policy][domain]
            pct = 100.0 * b["saved"] / b["base"] if b["base"] else 0.0
            kinds = {k: round(v / b["n"], 2) for k, v in b["kinds"].items()} if b["n"] else {}
            report[policy][domain] = {
                "micro_compression_pct": pct,
                "base_tokens": b["base"],
                "saved_tokens": b["saved"],
                "kinds_per_prompt": kinds,
            }
            cells.append(f"{pct:7.2f}%")
        kinds = report[policy]["all"]["kinds_per_prompt"]
        kind_s = " ".join(f"{k}={v:.1f}" for k, v in sorted(kinds.items()))
        print(f"{policy:<28} {cells[0]} {cells[1]} {cells[2]} {cells[3]}  {kind_s}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
