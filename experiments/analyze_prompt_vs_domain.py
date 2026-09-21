"""Detailed comparative analysis of Prompt Predictor vs Domain Static on Validation Data.

Answers the core research question:
Does conditioning on the individual prompt actually improve OUTPUT compression
beyond simply knowing the domain?

Measures for K=32 and K=64 across Code, Reasoning, and Instruction:
1. Response & Total Compression % for:
   - Global Static
   - Domain Static
   - Prompt Predictor
   - Oracle
2. Hypertoken Vocabulary Overlap (Jaccard & shared count)
3. Unique Phrases introduced by Prompt Predictor (not in Domain Static)
4. Response Hit Rate: % of unique prompt phrases that appear in the unseen response
5. Marginal Token Savings: Net tokens saved in the response attributable specifically
   to the prompt-conditioned additions
"""

from __future__ import annotations

import os
import sys
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from transformers import AutoTokenizer
from experiments.dataset_loader import load_split, DatasetSample
from experiments.optimized_predictor import OptimizedPhraseIndex, FastPredictor
from experiments.heldout_predictor_benchmark import evaluate_sample_codebook, extract_ngrams


def run_prompt_vs_domain_analysis(budgets=(32, 64)):
    print("Loading tokenizer and OptimizedPhraseIndex...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    disabled_ids = set(tokenizer.all_special_ids)
    if tokenizer.pad_token_id is not None:
        disabled_ids.add(tokenizer.pad_token_id)

    with open("experiments/optimized_phrase_index.pkl", "rb") as f:
        opt_index: OptimizedPhraseIndex = pickle.load(f)

    predictor = FastPredictor(opt_index, initial_vocab_size=32011)

    val_samples = load_split("val")
    samples_by_domain: Dict[str, List[DatasetSample]] = defaultdict(list)
    for s in val_samples:
        samples_by_domain[s.domain].append(s)

    # Use 50 samples per domain from validation set (150 total)
    eval_domains = ["code", "reasoning", "instruction"]
    eval_samples = {dom: samples_by_domain[dom][:50] for dom in eval_domains}

    print(f"\nAnalyzing {sum(len(v) for v in eval_samples.values())} validation samples across domains: {eval_domains}")

    results_by_budget = {}

    for K in budgets:
        print(f"\n" + "=" * 80)
        print(f"BUDGET K = {K}")
        print("=" * 80)

        domain_summaries = {}

        for dom in eval_domains:
            samples = eval_samples[dom]

            # Metric accumulators
            comp_global = []
            comp_domain = []
            comp_prompt = []
            comp_oracle = []

            overlaps = []
            num_uniques = []
            unique_hit_rates = []
            net_marginal_savings = []

            domain_static_cb = predictor.select_domain_static(dom, budget=K)
            domain_phrases_set = set(domain_static_cb.keys())
            global_static_cb = predictor.select_global_static(budget=K)

            for s in samples:
                p_ids = tokenizer.encode(s.prompt, add_special_tokens=False)
                r_ids = tokenizer.encode(s.response, add_special_tokens=False)

                # 1. Global Static
                *_, r_g, _, _, _ = evaluate_sample_codebook(p_ids, r_ids, global_static_cb, disabled_ids, K)
                comp_global.append(r_g)

                # 2. Domain Static
                *_, r_d, _, _, _ = evaluate_sample_codebook(p_ids, r_ids, domain_static_cb, disabled_ids, K)
                comp_domain.append(r_d)

                # 3. Prompt Predictor
                prompt_cb, _ = predictor.select_prompt_conditioned(p_ids, budget=K)
                *_, r_p, _, _, _ = evaluate_sample_codebook(p_ids, r_ids, prompt_cb, disabled_ids, K)
                comp_prompt.append(r_p)

                # 4. Oracle
                r_ngrams = extract_ngrams(r_ids, disabled_ids, min_len=2, max_len=3)
                oracle_ranked = sorted(r_ngrams.items(), key=lambda x: x[1] * (len(x[0]) - 1), reverse=True)
                oracle_cb = {gram: 32011 + i for i, (gram, _) in enumerate(oracle_ranked[:K])}
                *_, r_o, _, _, _ = evaluate_sample_codebook(p_ids, r_ids, oracle_cb, disabled_ids, K)
                comp_oracle.append(r_o)

                # Vocabulary Overlap & Marginal Analysis
                prompt_phrases_set = set(prompt_cb.keys())
                shared = prompt_phrases_set & domain_phrases_set
                unique_to_prompt = prompt_phrases_set - domain_phrases_set
                overlap_pct = (len(shared) / K) * 100.0
                overlaps.append(overlap_pct)
                num_uniques.append(len(unique_to_prompt))

                # Extract response n-grams to check hits
                response_ngrams = set(r_ngrams.keys())
                if unique_to_prompt:
                    hits = sum(1 for phr in unique_to_prompt if phr in response_ngrams)
                    hit_rate = (hits / len(unique_to_prompt)) * 100.0
                else:
                    hit_rate = 0.0
                unique_hit_rates.append(hit_rate)

                # Net marginal savings in response tokens:
                # Actual tokens compressed by Prompt Predictor minus tokens compressed by Domain Static
                # len(r_ids) * (r_p - r_d) / 100.0
                tokens_saved_prompt = len(r_ids) * (r_p / 100.0)
                tokens_saved_domain = len(r_ids) * (r_d / 100.0)
                net_marginal_savings.append(tokens_saved_prompt - tokens_saved_domain)

            summary = {
                "global_resp_comp": np.mean(comp_global),
                "domain_resp_comp": np.mean(comp_domain),
                "prompt_resp_comp": np.mean(comp_prompt),
                "oracle_resp_comp": np.mean(comp_oracle),
                "mean_overlap_pct": np.mean(overlaps),
                "mean_unique_phrases": np.mean(num_uniques),
                "mean_unique_hit_rate": np.mean(unique_hit_rates),
                "mean_net_token_diff": np.mean(net_marginal_savings),
            }
            domain_summaries[dom] = summary

        results_by_budget[K] = domain_summaries

        # Print detailed table for this budget
        print(f"\n{'Domain':<14} | {'Global Comp':<12} | {'Domain Comp':<12} | {'Prompt Comp':<12} | {'Oracle Comp':<12} | {'Overlap %':<10} | {'Unique Hits %':<14} | {'Net Token Diff':<14}")
        print("-" * 110)
        for dom in eval_domains:
            s = domain_summaries[dom]
            print(f"{dom.capitalize():<14} | {s['global_resp_comp']:<11.2f}% | {s['domain_resp_comp']:<11.2f}% | {s['prompt_resp_comp']:<11.2f}% | {s['oracle_resp_comp']:<11.2f}% | {s['mean_overlap_pct']:<9.1f}% | {s['mean_unique_hit_rate']:<13.1f}% | {s['mean_net_token_diff']:<+13.2f}")

    return results_by_budget


if __name__ == "__main__":
    run_prompt_vs_domain_analysis()
