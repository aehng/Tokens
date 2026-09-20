"""Predictive Dynamic Token Vocabulary: Non-Neural Predictor and Baseline Comparisons.

Implements and evaluates:
1. Base Tokenizer (no hypertokens)
2. Global Static Top-K Most Frequent Phrases
3. Domain-Specific Top-K Phrases
4. Prompt-Conditioned Predictive Vocabulary Selector
5. Standard zip2zip LZW Compressor Baseline
6. Oracle Theoretical Upper Bound
"""

from __future__ import annotations

import sys
import os
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer
from zip2zip_compression import LZWCompressor
from zip2zip.segmenter import DynamicSegmenter
from experiments.oracle_compression import (
    extract_candidate_ngrams,
    evaluate_sample_oracle,
    get_curated_eval_samples,
)


@dataclass
class RegimeResult:
    name: str
    budget: int
    base_prompt_tokens: int
    base_response_tokens: int
    base_total_tokens: int
    hyper_prompt_tokens: int
    hyper_response_tokens: int
    hyper_total_tokens: int
    total_compression_pct: float
    response_compression_pct: float
    utilization_rate_pct: float
    avg_tokens_per_hypertoken: float
    active_hypertokens_count: int


class PhraseBank:
    """Mined repository of 2-3 token phrases and prompt-phrase association statistics."""

    def __init__(self, disabled_ids: Set[int], max_subtokens: int = 3) -> None:
        self.disabled_ids = disabled_ids
        self.max_subtokens = max_subtokens
        self.global_counts: Counter[Tuple[int, ...]] = Counter()
        self.domain_counts: Dict[str, Counter[Tuple[int, ...]]] = defaultdict(Counter)
        # prompt_token_id -> candidate_phrase -> co-occurrence count
        self.prompt_to_phrase_cooccurrence: Dict[int, Counter[Tuple[int, ...]]] = defaultdict(Counter)

    def train_on_corpus(self, corpus: List[Tuple[str, List[int], List[int]]]) -> None:
        """Mine n-grams and associations from (domain, prompt_ids, response_ids)."""
        for domain, p_ids, r_ids in corpus:
            full_seq = p_ids + r_ids
            # Extract response n-grams (target vocabulary)
            r_ngrams = extract_candidate_ngrams(r_ids, self.disabled_ids, min_len=2, max_len=self.max_subtokens)
            for gram, cnt in r_ngrams.items():
                self.global_counts[gram] += cnt
                self.domain_counts[domain][gram] += cnt

                # Track association with prompt tokens
                unique_prompt_tokens = set(p_ids) - self.disabled_ids
                for p_tok in unique_prompt_tokens:
                    self.prompt_to_phrase_cooccurrence[p_tok][gram] += cnt

            # Also include high-frequency prompt n-grams that may recur in generation
            p_ngrams = extract_candidate_ngrams(p_ids, self.disabled_ids, min_len=2, max_len=self.max_subtokens)
            for gram, cnt in p_ngrams.items():
                self.global_counts[gram] += cnt
                self.domain_counts[domain][gram] += cnt


class PredictiveVocabularyOptimizer:
    """Predicts request-specific hypertokens before autoregressive generation starts."""

    def __init__(
        self,
        phrase_bank: PhraseBank,
        initial_vocab_size: int = 32064,
        max_subtokens: int = 3,
    ) -> None:
        self.bank = phrase_bank
        self.initial_vocab_size = initial_vocab_size
        self.max_subtokens = max_subtokens

    def select_global_static(self, budget: int) -> Dict[Tuple[int, ...], int]:
        """Select top-K globally most frequent phrases."""
        ranked = sorted(
            self.bank.global_counts.items(),
            key=lambda item: item[1] * (len(item[0]) - 1),
            reverse=True,
        )
        codebook: Dict[Tuple[int, ...], int] = {}
        for idx, (gram, _) in enumerate(ranked[:budget]):
            codebook[gram] = self.initial_vocab_size + idx
        return codebook

    def select_domain_specific(self, domain: str, budget: int) -> Dict[Tuple[int, ...], int]:
        """Select top-K domain-specific phrases."""
        dom_counts = self.bank.domain_counts.get(domain, self.bank.global_counts)
        ranked = sorted(
            dom_counts.items(),
            key=lambda item: item[1] * (len(item[0]) - 1),
            reverse=True,
        )
        codebook: Dict[Tuple[int, ...], int] = {}
        for idx, (gram, _) in enumerate(ranked[:budget]):
            codebook[gram] = self.initial_vocab_size + idx
        return codebook

    def select_prompt_conditioned(
        self, prompt_ids: List[int], budget: int
    ) -> Dict[Tuple[int, ...], int]:
        """Predict request-specific hypertokens conditioned on the prompt text.

        Combines:
        1. Repetition prior: 2-3 token phrases found directly inside the prompt
           (entity names, function signatures, variables, keywords).
        2. Co-occurrence prior: Phrases correlated with prompt tokens in training data.
        3. Global frequency prior: Universal grammatical/syntactic structures.
        """
        candidate_scores: Dict[Tuple[int, ...], float] = defaultdict(float)

        # 1. Phrases appearing in the prompt itself (high likelihood of repetition in output)
        prompt_ngrams = extract_candidate_ngrams(
            prompt_ids, self.bank.disabled_ids, min_len=2, max_len=self.max_subtokens
        )
        for gram, cnt in prompt_ngrams.items():
            savings = len(gram) - 1
            # High boost for phrases present in the prompt
            candidate_scores[gram] += cnt * savings * 8.0

        # 2. Co-occurrence with prompt tokens
        unique_prompt_tokens = set(prompt_ids) - self.bank.disabled_ids
        for p_tok in unique_prompt_tokens:
            if p_tok in self.bank.prompt_to_phrase_cooccurrence:
                for gram, co_cnt in self.bank.prompt_to_phrase_cooccurrence[p_tok].items():
                    savings = len(gram) - 1
                    candidate_scores[gram] += co_cnt * savings * 2.0

        # 3. Global background frequency
        for gram, g_cnt in self.bank.global_counts.most_common(budget * 2):
            savings = len(gram) - 1
            candidate_scores[gram] += math.log1p(g_cnt) * savings * 0.5

        # Rank candidates
        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        codebook: Dict[Tuple[int, ...], int] = {}
        for idx, (gram, _) in enumerate(ranked[:budget]):
            codebook[gram] = self.initial_vocab_size + idx
        return codebook


def evaluate_seeded_codebook(
    prompt_ids: List[int],
    response_ids: List[int],
    codebook: Dict[Tuple[int, ...], int],
    disabled_ids: Set[int],
    budget: int,
) -> Tuple[int, int, int, int, float, float, float, float, int]:
    """Segment prompt and response with a seeded codebook and compute metrics."""
    segmenter = DynamicSegmenter(
        subtokens_to_hyper=codebook,
        disabled_ids=disabled_ids,
        max_subtokens=3,
    )
    seg_p = segmenter.segment(prompt_ids)
    seg_r = segmenter.segment(response_ids)

    base_p = len(prompt_ids)
    base_r = len(response_ids)
    hyp_p = len(seg_p)
    hyp_r = len(seg_r)

    tot_comp = (1.0 - (hyp_p + hyp_r) / (base_p + base_r)) * 100.0
    resp_comp = (1.0 - hyp_r / base_r) * 100.0

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
    return base_p, base_r, hyp_p, hyp_r, tot_comp, resp_comp, util, avg_l, len(used)


def evaluate_lzw(
    prompt_ids: List[int],
    response_ids: List[int],
    budget: int,
    disabled_ids: Set[int],
    initial_vocab_size: int = 32064,
) -> Tuple[int, int, int, int, float, float, float, float, int]:
    """Evaluate standard zip2zip reactive LZW compressor on the same prompt & response."""
    compressor = LZWCompressor(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=budget,
        max_subtokens=3,
        pad_token_id=0,
        disabled_ids=list(disabled_ids),
    )
    # LZW processes the concatenated prompt + response stream
    full_seq = prompt_ids + response_ids
    encoded, _, codebook = compressor.encode(full_seq)

    # Re-decode to inspect the prompt/response boundary compression
    p_len = len(prompt_ids)
    # In LZW encode, boundary is not cleanly decoupled, but we can measure the compressed output
    base_tot = len(full_seq)
    hyp_tot = len(encoded)
    tot_comp = (1.0 - hyp_tot / base_tot) * 100.0

    # For response estimation in LZW: encode prompt first, then response
    comp_p = LZWCompressor(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=budget,
        max_subtokens=3,
        pad_token_id=0,
        disabled_ids=list(disabled_ids),
    )
    enc_p, _, _ = comp_p.encode(prompt_ids)
    hyp_p = len(enc_p)
    # Remaining tokens belong to response
    hyp_r = max(0, hyp_tot - hyp_p)
    resp_comp = (1.0 - hyp_r / len(response_ids)) * 100.0

    codebook_dict = codebook.to_dict()
    util = (len(codebook_dict) / budget) * 100.0 if budget > 0 else 0.0
    avg_l = (
        sum(len(sub) for sub in codebook_dict.values()) / len(codebook_dict)
        if len(codebook_dict) > 0
        else 0.0
    )
    return len(prompt_ids), len(response_ids), hyp_p, hyp_r, tot_comp, resp_comp, util, avg_l, len(codebook_dict)


def run_predictor_experiment(
    budget: int = 256,
    tokenizer_name: str = "microsoft/Phi-3.5-mini-instruct",
) -> List[RegimeResult]:
    """Compare all 6 regimes on the curated benchmark dataset."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    disabled_ids = set(tokenizer.all_special_ids)
    if tokenizer.pad_token_id is not None:
        disabled_ids.add(tokenizer.pad_token_id)
    initial_vocab_size = 32064

    raw_samples = get_curated_eval_samples()
    tokenized_corpus = []
    for dom, p, r in raw_samples:
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        r_ids = tokenizer.encode(r, add_special_tokens=False)
        tokenized_corpus.append((dom, p_ids, r_ids))

    # Build and populate phrase bank
    bank = PhraseBank(disabled_ids=disabled_ids, max_subtokens=3)
    bank.train_on_corpus(tokenized_corpus)

    optimizer = PredictiveVocabularyOptimizer(
        phrase_bank=bank,
        initial_vocab_size=initial_vocab_size,
        max_subtokens=3,
    )

    regimes = [
        "1. Base Tokenizer",
        "2. Global Static Top-K",
        "3. Domain-Specific Top-K",
        "4. Prompt-Conditioned Predictor",
        "5. zip2zip LZW Baseline",
        "6. Oracle Upper Bound",
    ]

    results: List[RegimeResult] = []

    for regime in regimes:
        agg_bp, agg_br = 0, 0
        agg_hp, agg_hr = 0, 0
        agg_util = 0.0
        agg_avg_l = 0.0
        agg_active = 0.0
        n_samples = len(tokenized_corpus)

        for dom, p_ids, r_ids in tokenized_corpus:
            if regime == "1. Base Tokenizer":
                bp, br = len(p_ids), len(r_ids)
                hp, hr = bp, br
                util, avg_l, act = 0.0, 1.0, 0
            elif regime == "2. Global Static Top-K":
                cb = optimizer.select_global_static(budget)
                bp, br, hp, hr, _, _, util, avg_l, act = evaluate_seeded_codebook(
                    p_ids, r_ids, cb, disabled_ids, budget
                )
            elif regime == "3. Domain-Specific Top-K":
                cb = optimizer.select_domain_specific(dom, budget)
                bp, br, hp, hr, _, _, util, avg_l, act = evaluate_seeded_codebook(
                    p_ids, r_ids, cb, disabled_ids, budget
                )
            elif regime == "4. Prompt-Conditioned Predictor":
                cb = optimizer.select_prompt_conditioned(p_ids, budget)
                bp, br, hp, hr, _, _, util, avg_l, act = evaluate_seeded_codebook(
                    p_ids, r_ids, cb, disabled_ids, budget
                )
            elif regime == "5. zip2zip LZW Baseline":
                bp, br, hp, hr, _, _, util, avg_l, act = evaluate_lzw(
                    p_ids, r_ids, budget, disabled_ids, initial_vocab_size
                )
            elif regime == "6. Oracle Upper Bound":
                m = evaluate_sample_oracle(p_ids, r_ids, budget, disabled_ids, initial_vocab_size)
                bp, br = m.base_prompt_tokens, m.base_response_tokens
                hp, hr = m.hyper_prompt_tokens, m.hyper_response_tokens
                util, avg_l, act = m.codebook_utilization_rate, m.avg_tokens_per_hypertoken, m.active_hypertokens_count

            agg_bp += bp
            agg_br += br
            agg_hp += hp
            agg_hr += hr
            agg_util += util
            agg_avg_l += avg_l
            agg_active += act

        tot_base = agg_bp + agg_br
        tot_hyp = agg_hp + agg_hr
        tot_comp = (1.0 - tot_hyp / tot_base) * 100.0 if tot_base > 0 else 0.0
        resp_comp = (1.0 - agg_hr / agg_br) * 100.0 if agg_br > 0 else 0.0

        results.append(
            RegimeResult(
                name=regime,
                budget=budget,
                base_prompt_tokens=agg_bp,
                base_response_tokens=agg_br,
                base_total_tokens=tot_base,
                hyper_prompt_tokens=agg_hp,
                hyper_response_tokens=agg_hr,
                hyper_total_tokens=tot_hyp,
                total_compression_pct=tot_comp,
                response_compression_pct=resp_comp,
                utilization_rate_pct=agg_util / n_samples,
                avg_tokens_per_hypertoken=agg_avg_l / n_samples,
                active_hypertokens_count=int(agg_active / n_samples),
            )
        )

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--output", type=str, default="experiments/predictor_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Running Non-Neural Predictor Comparison at Hypertoken Budget = {args.budget}...")
    res = run_predictor_experiment(budget=args.budget)

    print("\n" + "=" * 90)
    print(f"PREDICTIVE VOCABULARY COMPARISON (Budget = {args.budget} Hypertokens)")
    print("=" * 90)
    print(f"{'Regime':<32} | {'Base Resp':<9} | {'Hyp Resp':<8} | {'Resp Comp %':<11} | {'Total Comp %':<12} | {'Util %':<7}")
    print("-" * 90)

    for r in res:
        print(
            f"{r.name:<32} | {r.base_response_tokens:<9} | {r.hyper_response_tokens:<8} | "
            f"{r.response_compression_pct:>10.2f}% | {r.total_compression_pct:>11.2f}% | "
            f"{r.utilization_rate_pct:>6.1f}%"
        )
    print("-" * 90)

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in res], f, indent=2)
    print(f"\nSaved results to {args.output}")
