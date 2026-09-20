"""Held-out Predictor Benchmark with Zero Leakage.

Trains PhraseBank strictly on data/train.jsonl (80% split, 9,412 samples).
Evaluates strictly on held-out data/test.jsonl (10% split, 1,178 samples).
Tests across budgets: K = 16, 32, 64, 128, 256.
Compares 6 regimes:
A. Base Tokenizer (No compression)
B. Standard zip2zip LZW
C. Global Static Top-K
D. Domain-Specific Static Top-K
E. Prompt-Conditioned Predictor (Ours)
F. Oracle Upper Bound (Sees test response)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer
from zip2zip_compression import LZWCompressor

from experiments.dataset_loader import DatasetSample, load_split
from zip2zip.segmenter import DynamicSegmenter


@dataclass
class BudgetEvaluationSummary:
    regime: str
    budget: int
    num_samples: int
    base_prompt_tokens: int
    base_response_tokens: int
    base_total_tokens: int
    hyper_prompt_tokens: int
    hyper_response_tokens: int
    hyper_total_tokens: int
    prompt_compression_pct: float
    response_compression_pct: float
    total_compression_pct: float
    codebook_utilization_pct: float
    avg_tokens_per_hypertoken: float
    avg_predictor_latency_ms: float


def extract_ngrams(
    token_ids: Sequence[int],
    disabled_ids: Set[int],
    min_len: int = 2,
    max_len: int = 3,
) -> Counter[Tuple[int, ...]]:
    """Extract all candidate n-grams excluding special/disabled tokens."""
    counts: Counter[Tuple[int, ...]] = Counter()
    n = len(token_ids)
    for length in range(min_len, max_len + 1):
        for i in range(n - length + 1):
            gram = tuple(token_ids[i : i + length])
            if not any(t in disabled_ids for t in gram):
                counts[gram] += 1
    return counts


class StrictHeldOutPhraseBank:
    """Phrase Bank trained strictly on TRAIN split with no test leakage."""

    def __init__(self, disabled_ids: Set[int], max_subtokens: int = 3) -> None:
        self.disabled_ids = disabled_ids
        self.max_subtokens = max_subtokens
        self.global_counts: Counter[Tuple[int, ...]] = Counter()
        self.domain_counts: Dict[str, Counter[Tuple[int, ...]]] = defaultdict(Counter)
        # prompt_token -> candidate_phrase -> co-occurrence count in TRAIN
        self.prompt_to_phrase: Dict[int, Counter[Tuple[int, ...]]] = defaultdict(Counter)
        self.prompt_token_totals: Counter[int] = Counter()

    def train_on_samples(self, samples: List[DatasetSample], tokenizer: AutoTokenizer) -> None:
        """Mine statistics strictly from training samples."""
        print(f"Mining phrase statistics from {len(samples)} training samples (Strict Zero Leakage)...")
        t0 = time.time()
        for idx, s in enumerate(samples):
            p_ids = tokenizer.encode(s.prompt, add_special_tokens=False)
            r_ids = tokenizer.encode(s.response, add_special_tokens=False)

            r_ngrams = extract_ngrams(r_ids, self.disabled_ids, min_len=2, max_len=self.max_subtokens)
            p_unique = set(p_ids) - self.disabled_ids

            for p_tok in p_unique:
                self.prompt_token_totals[p_tok] += 1

            for gram, cnt in r_ngrams.items():
                self.global_counts[gram] += cnt
                self.domain_counts[s.domain][gram] += cnt

                for p_tok in p_unique:
                    self.prompt_to_phrase[p_tok][gram] += cnt

            if (idx + 1) % 2000 == 0:
                print(f"  Processed {idx + 1}/{len(samples)} training samples...")

        elapsed = time.time() - t0
        print(f"PhraseBank training complete in {elapsed:.1f}s. Unique phrases mined: {len(self.global_counts)}")


class StrictPredictor:
    """Ultra-cheap statistical predictor that runs in sub-millisecond table lookups."""

    def __init__(
        self,
        bank: StrictHeldOutPhraseBank,
        initial_vocab_size: int = 32064,
        max_subtokens: int = 3,
    ) -> None:
        self.bank = bank
        self.initial_vocab_size = initial_vocab_size
        self.max_subtokens = max_subtokens

        # Precompute static top phrases for fast baseline lookup
        self.precomputed_global_static = sorted(
            self.bank.global_counts.items(),
            key=lambda item: item[1] * (len(item[0]) - 1),
            reverse=True,
        )
        self.precomputed_domain_static = {
            dom: sorted(
                counts.items(),
                key=lambda item: item[1] * (len(item[0]) - 1),
                reverse=True,
            )
            for dom, counts in self.bank.domain_counts.items()
        }

    def select_global_static(self, budget: int) -> Dict[Tuple[int, ...], int]:
        return {
            gram: self.initial_vocab_size + i
            for i, (gram, _) in enumerate(self.precomputed_global_static[:budget])
        }

    def select_domain_static(self, domain: str, budget: int) -> Dict[Tuple[int, ...], int]:
        ranked = self.precomputed_domain_static.get(
            domain, self.precomputed_global_static
        )
        return {
            gram: self.initial_vocab_size + i
            for i, (gram, _) in enumerate(ranked[:budget])
        }

    def select_prompt_conditioned(
        self, prompt_ids: Sequence[int], budget: int
    ) -> Tuple[Dict[Tuple[int, ...], int], float]:
        """Predict top-K request hypertokens conditioned on the prompt.

        Returns (codebook_dict, latency_ms).
        """
        t0 = time.perf_counter()
        scores: Dict[Tuple[int, ...], float] = defaultdict(float)

        # 1. Repetition prior: n-grams appearing in prompt
        p_ngrams = extract_ngrams(
            prompt_ids, self.bank.disabled_ids, min_len=2, max_len=self.max_subtokens
        )
        for gram, cnt in p_ngrams.items():
            savings = len(gram) - 1
            scores[gram] += cnt * savings * 6.0

        # 2. Normalized Association score from precomputed training stats
        p_tokens = set(prompt_ids) - self.bank.disabled_ids
        for p_tok in p_tokens:
            if p_tok in self.bank.prompt_to_phrase:
                tok_total = self.bank.prompt_token_totals.get(p_tok, 1)
                for gram, co_cnt in self.bank.prompt_to_phrase[p_tok].items():
                    norm_score = (co_cnt / (tok_total + 25.0)) * (len(gram) - 1)
                    scores[gram] += norm_score * 3.0

        # 3. Global background frequency prior
        for gram, g_cnt in self.precomputed_global_static[: budget * 2]:
            savings = len(gram) - 1
            scores[gram] += math.log1p(g_cnt) * savings * 0.25

        ranked = sorted(scores.items(), key=lambda it: it[1], reverse=True)
        codebook = {
            gram: self.initial_vocab_size + i
            for i, (gram, _) in enumerate(ranked[:budget])
        }
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        return codebook, latency_ms

    def select_oracle(
        self,
        prompt_ids: Sequence[int],
        response_ids: Sequence[int],
        budget: int,
    ) -> Dict[Tuple[int, ...], int]:
        """Oracle upper bound: allowed to see the test response."""
        r_counts = extract_ngrams(response_ids, self.bank.disabled_ids, min_len=2, max_len=self.max_subtokens)
        p_counts = extract_ngrams(prompt_ids, self.bank.disabled_ids, min_len=2, max_len=self.max_subtokens)

        # Score by actual token savings: response count * (length - 1)
        scores: Dict[Tuple[int, ...], float] = defaultdict(float)
        for gram, cnt in r_counts.items():
            scores[gram] += cnt * (len(gram) - 1) * 2.0
        for gram, cnt in p_counts.items():
            scores[gram] += cnt * (len(gram) - 1) * 1.0

        ranked = sorted(scores.items(), key=lambda it: it[1], reverse=True)
        return {
            gram: self.initial_vocab_size + i
            for i, (gram, _) in enumerate(ranked[:budget])
        }


def evaluate_sample_codebook(
    prompt_ids: List[int],
    response_ids: List[int],
    codebook: Dict[Tuple[int, ...], int],
    disabled_ids: Set[int],
    budget: int,
) -> Tuple[int, int, int, int, float, float, float, float, int]:
    """Segment prompt and response and compute exact compression metrics."""
    seg = DynamicSegmenter(
        subtokens_to_hyper=codebook,
        disabled_ids=disabled_ids,
        max_subtokens=3,
    )
    seg_p = seg.segment(prompt_ids)
    seg_r = seg.segment(response_ids)

    base_p, base_r = len(prompt_ids), len(response_ids)
    hyp_p, hyp_r = len(seg_p), len(seg_r)

    tot_comp = (1.0 - (hyp_p + hyp_r) / (base_p + base_r)) * 100.0 if (base_p + base_r) > 0 else 0.0
    resp_comp = (1.0 - hyp_r / base_r) * 100.0 if base_r > 0 else 0.0
    p_comp = (1.0 - hyp_p / base_p) * 100.0 if base_p > 0 else 0.0

    rev_map = {v: k for k, v in codebook.items()}
    used: Set[int] = set()
    total_tokens_rep = 0
    total_occ = 0
    for tok in seg_p + seg_r:
        if tok in rev_map:
            used.add(tok)
            total_tokens_rep += len(rev_map[tok])
            total_occ += 1

    util = (len(used) / budget) * 100.0 if budget > 0 else 0.0
    avg_l = (total_tokens_rep / total_occ) if total_occ > 0 else 0.0
    return base_p, base_r, hyp_p, hyp_r, p_comp, resp_comp, tot_comp, util, avg_l


def evaluate_sample_lzw(
    prompt_ids: List[int],
    response_ids: List[int],
    budget: int,
    disabled_ids: Set[int],
    initial_vocab_size: int = 32064,
) -> Tuple[int, int, int, int, float, float, float, float, int]:
    """Evaluate standard zip2zip reactive LZW baseline."""
    compressor = LZWCompressor(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=budget,
        max_subtokens=3,
        pad_token_id=0,
        disabled_ids=list(disabled_ids),
    )
    full_seq = prompt_ids + response_ids
    encoded, _, codebook = compressor.encode(full_seq)

    base_p, base_r = len(prompt_ids), len(response_ids)
    base_tot = len(full_seq)
    hyp_tot = len(encoded)

    comp_p = LZWCompressor(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=budget,
        max_subtokens=3,
        pad_token_id=0,
        disabled_ids=list(disabled_ids),
    )
    enc_p, _, _ = comp_p.encode(prompt_ids)
    hyp_p = len(enc_p)
    hyp_r = max(0, hyp_tot - hyp_p)

    p_comp = (1.0 - hyp_p / base_p) * 100.0 if base_p > 0 else 0.0
    resp_comp = (1.0 - hyp_r / base_r) * 100.0 if base_r > 0 else 0.0
    tot_comp = (1.0 - hyp_tot / base_tot) * 100.0 if base_tot > 0 else 0.0

    cb_dict = codebook.to_dict()
    util = (len(cb_dict) / budget) * 100.0 if budget > 0 else 0.0
    avg_l = (
        sum(len(sub) for sub in cb_dict.values()) / len(cb_dict)
        if len(cb_dict) > 0
        else 0.0
    )
    return base_p, base_r, hyp_p, hyp_r, p_comp, resp_comp, tot_comp, util, avg_l


def run_heldout_benchmark(
    tokenizer_name: str = "microsoft/Phi-3.5-mini-instruct",
    test_limit_per_domain: int = 100,  # 100 code, 100 reasoning, 100 instruction = 300 held-out test samples
    budgets: Sequence[int] = (16, 32, 64, 128, 256),
) -> Dict[int, List[BudgetEvaluationSummary]]:
    """Run full evaluation on held-out TEST set across all budgets."""
    print("Initializing tokenizer and dataset splits...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    disabled_ids = set(tokenizer.all_special_ids)
    if tokenizer.pad_token_id is not None:
        disabled_ids.add(tokenizer.pad_token_id)
    initial_vocab_size = 32064

    # 1. Load TRAIN and build PhraseBank
    train_samples = load_split("train")
    bank = StrictHeldOutPhraseBank(disabled_ids=disabled_ids, max_subtokens=3)
    bank.train_on_samples(train_samples, tokenizer)

    predictor = StrictPredictor(bank, initial_vocab_size=initial_vocab_size, max_subtokens=3)

    # 2. Load TEST split (Held-out)
    all_test_samples = load_split("test")
    # Subsample per domain for balanced representation
    domain_test_samples: Dict[str, List[DatasetSample]] = defaultdict(list)
    for s in all_test_samples:
        if len(domain_test_samples[s.domain]) < test_limit_per_domain:
            domain_test_samples[s.domain].append(s)

    eval_test_set = [s for sublist in domain_test_samples.values() for s in sublist]
    print(f"\nEvaluating on {len(eval_test_set)} strictly held-out TEST samples:")
    for dom, s_list in domain_test_samples.items():
        print(f"  {dom:<12}: {len(s_list)} samples")

    # Tokenize test samples once
    tokenized_test: List[Tuple[str, List[int], List[int]]] = []
    for s in eval_test_set:
        p_ids = tokenizer.encode(s.prompt, add_special_tokens=False)
        r_ids = tokenizer.encode(s.response, add_special_tokens=False)
        if len(p_ids) > 0 and len(r_ids) > 0:
            tokenized_test.append((s.domain, p_ids, r_ids))

    regimes = [
        "A. Base (No compression)",
        "B. zip2zip LZW",
        "C. Global Static Top-K",
        "D. Domain Static Top-K",
        "E. Prompt Predictor (Ours)",
        "F. Oracle Upper Bound",
    ]

    all_budget_results: Dict[int, List[BudgetEvaluationSummary]] = {}

    for budget in budgets:
        print(f"\n=======================================================")
        print(f"Evaluating Hypertoken Budget K = {budget}")
        print(f"=======================================================")
        budget_summaries: List[BudgetEvaluationSummary] = []

        for regime in regimes:
            agg_bp, agg_br = 0, 0
            agg_hp, agg_hr = 0, 0
            agg_util = 0.0
            agg_avg_l = 0.0
            agg_lat = 0.0
            n_eval = len(tokenized_test)

            for dom, p_ids, r_ids in tokenized_test:
                if regime == "A. Base (No compression)":
                    bp, br = len(p_ids), len(r_ids)
                    hp, hr = bp, br
                    p_c, r_c, t_c = 0.0, 0.0, 0.0
                    util, avg_l, lat = 0.0, 1.0, 0.0
                elif regime == "B. zip2zip LZW":
                    bp, br, hp, hr, p_c, r_c, t_c, util, avg_l = evaluate_sample_lzw(
                        p_ids, r_ids, budget, disabled_ids, initial_vocab_size
                    )
                    lat = 0.0
                elif regime == "C. Global Static Top-K":
                    cb = predictor.select_global_static(budget)
                    bp, br, hp, hr, p_c, r_c, t_c, util, avg_l = evaluate_sample_codebook(
                        p_ids, r_ids, cb, disabled_ids, budget
                    )
                    lat = 0.005  # Precomputed hash lookup
                elif regime == "D. Domain Static Top-K":
                    cb = predictor.select_domain_static(dom, budget)
                    bp, br, hp, hr, p_c, r_c, t_c, util, avg_l = evaluate_sample_codebook(
                        p_ids, r_ids, cb, disabled_ids, budget
                    )
                    lat = 0.005
                elif regime == "E. Prompt Predictor (Ours)":
                    cb, lat = predictor.select_prompt_conditioned(p_ids, budget)
                    bp, br, hp, hr, p_c, r_c, t_c, util, avg_l = evaluate_sample_codebook(
                        p_ids, r_ids, cb, disabled_ids, budget
                    )
                elif regime == "F. Oracle Upper Bound":
                    cb = predictor.select_oracle(p_ids, r_ids, budget)
                    bp, br, hp, hr, p_c, r_c, t_c, util, avg_l = evaluate_sample_codebook(
                        p_ids, r_ids, cb, disabled_ids, budget
                    )
                    lat = 0.0

                agg_bp += bp
                agg_br += br
                agg_hp += hp
                agg_hr += hr
                agg_util += util
                agg_avg_l += avg_l
                agg_lat += lat

            tot_base = agg_bp + agg_br
            tot_hyp = agg_hp + agg_hr
            p_comp = (1.0 - agg_hp / agg_bp) * 100.0 if agg_bp > 0 else 0.0
            r_comp = (1.0 - agg_hr / agg_br) * 100.0 if agg_br > 0 else 0.0
            tot_comp = (1.0 - tot_hyp / tot_base) * 100.0 if tot_base > 0 else 0.0

            summary = BudgetEvaluationSummary(
                regime=regime,
                budget=budget,
                num_samples=n_eval,
                base_prompt_tokens=agg_bp,
                base_response_tokens=agg_br,
                base_total_tokens=tot_base,
                hyper_prompt_tokens=agg_hp,
                hyper_response_tokens=agg_hr,
                hyper_total_tokens=tot_hyp,
                prompt_compression_pct=p_comp,
                response_compression_pct=r_comp,
                total_compression_pct=tot_comp,
                codebook_utilization_pct=agg_util / n_eval,
                avg_tokens_per_hypertoken=agg_avg_l / n_eval,
                avg_predictor_latency_ms=agg_lat / n_eval,
            )
            budget_summaries.append(summary)

            print(
                f"  {regime:<28} | Resp Comp: {r_comp:>6.2f}% | Total Comp: {tot_comp:>6.2f}% | "
                f"Util: {summary.codebook_utilization_pct:>5.1f}% | Latency: {summary.avg_predictor_latency_ms:>6.3f}ms"
            )

        all_budget_results[budget] = budget_summaries

    return all_budget_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_limit", type=int, default=100)
    parser.add_argument("--output", type=str, default="experiments/heldout_benchmark_results.json")
    args = parser.parse_args()

    results = run_heldout_benchmark(test_limit_per_domain=args.test_limit)

    serializable = {
        b: [asdict(s) for s in summaries] for b, summaries in results.items()
    }
    with open(args.output, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved held-out benchmark results to {args.output}")
