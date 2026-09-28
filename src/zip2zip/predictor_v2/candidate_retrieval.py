"""Modular Candidate Retrieval Strategies and Candidate Pool Generator for Predictor V2.

Implements 4 distinct candidate retrieval configurations:
1. BASELINE:
   - Prompt n-grams (lengths 2..4)
   - Standard 1-hop token associations from TRAIN index
   - Global static background bank
2. EXPANDED_ASSOCIATIONS:
   - Prompt n-grams (lengths 2..4)
   - Deep 1-hop token associations (higher top-K cutoff)
   - 2-hop token associations (associations of associated tokens, attenuated)
   - Global static background bank
3. SUFFIX_CONDITIONED:
   - Local suffix n-grams (emphasizing prompt-ending 16 tokens)
   - Suffix token associations (weighted higher for prompt suffix tokens)
   - Prompt-wide n-grams and baseline associations
   - Global static background bank
4. SPARSE_LEXICAL:
   - Prompt n-grams and standard associations
   - BM25 / TF-IDF retrieval against the 630 TRAIN prompts to retrieve
     continuation n-grams from the top-matching training examples.
   - Global static background bank

All configurations support nominal candidate pool sizes:
  N in {256, 512, 1024, 2048}

STRICT LEAKAGE GUARANTEE:
All association tables, BM25 indices, and static banks are built EXCLUSIVELY
from the 630 TRAIN continuations. Zero DEV or FINAL prompts/continuations.
"""

from __future__ import annotations

import heapq
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from transformers import AutoTokenizer

from src.zip2zip.predictor_v2.candidate_pool import (
    CandidateRecord,
    extract_handcrafted_features,
    get_first_occurrence_bucket,
    is_bare_punctuation,
)
from src.zip2zip.predictor_v2.vanilla_labels import VanillaContinuationRecord


class RetrievalStrategy(str, Enum):
    BASELINE = "baseline"
    EXPANDED_ASSOCIATIONS = "expanded_associations"
    SUFFIX_CONDITIONED = "suffix_conditioned"
    SPARSE_LEXICAL = "sparse_lexical"


@dataclass
class TrainOnlyAssociationIndex:
    """Precomputed phrase association index built strictly on TRAIN continuations."""
    # prompt_token -> list of (phrase_tuple, weight)
    token_associations: Dict[int, List[Tuple[Tuple[int, ...], float]]] = field(default_factory=dict)
    # Global background ranked phrases: list of (phrase_tuple, score)
    precomputed_global_static: List[Tuple[Tuple[int, ...], float]] = field(default_factory=list)
    # Domain background ranked phrases: domain -> list of phrase tuples
    precomputed_domain_static: Dict[str, List[Tuple[int, ...]]] = field(default_factory=dict)
    # Disabled token IDs (special tokens, BOS, EOS, padding, etc.)
    disabled_ids: Set[int] = field(default_factory=set)
    # Train prompt BM25 corpus data: list of (prompt_id, set_of_words, continuation_phrases_with_weights)
    train_prompt_lexical_docs: List[Dict[str, Any]] = field(default_factory=list)
    # Metadata
    train_prompt_ids: List[str] = field(default_factory=list)
    total_train_continuations: int = 0
    max_subtokens: int = 4

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> TrainOnlyAssociationIndex:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj


class ConfigurableCandidateGenerator:
    """Configurable Candidate Generator supporting multiple retrieval strategies and pool sizes."""

    def __init__(
        self,
        index: TrainOnlyAssociationIndex,
        tokenizer: AutoTokenizer,
        max_subtokens: int = 4,
    ):
        self.index = index
        self.tokenizer = tokenizer
        self.max_subtokens = max_subtokens
        self.disabled_ids = set(index.disabled_ids)

        # Build reverse index for BM25 lexical lookup if train docs are present
        self.bm25_doc_count = len(index.train_prompt_lexical_docs)
        self.bm25_df: Counter[str] = Counter()
        self.bm25_docs: List[Dict[str, Any]] = []
        if self.bm25_doc_count > 0:
            for doc in index.train_prompt_lexical_docs:
                words = set(re.findall(r"\b\w+\b", doc["prompt_text"].lower()))
                for w in words:
                    self.bm25_df[w] += 1
                self.bm25_docs.append({
                    "prompt_id": doc["prompt_id"],
                    "words": words,
                    "word_list": re.findall(r"\b\w+\b", doc["prompt_text"].lower()),
                    "phrases": doc["continuation_phrases"],
                })

    def generate_candidate_pool(
        self,
        prompt_ids: Sequence[int],
        prompt_text: str,
        domain: str = "general",
        strategy: RetrievalStrategy = RetrievalStrategy.BASELINE,
        target_pool_size: int = 512,
    ) -> Dict[Tuple[int, ...], Dict[str, Any]]:
        """Generates candidate pool of size target_pool_size using the selected strategy."""
        p_ids = [t for t in prompt_ids if t not in self.disabled_ids]
        n = len(p_ids)
        cands: Dict[Tuple[int, ...], Dict[str, Any]] = {}

        def add_candidate(gram: Tuple[int, ...], source: str, weight: float):
            if any(t in self.disabled_ids for t in gram):
                return
            if len(gram) < 2 or len(gram) > self.max_subtokens:
                return
            if gram not in cands:
                cands[gram] = {"sources": set(), "weight": 0.0}
            cands[gram]["sources"].add(source)
            cands[gram]["weight"] += weight

        # 1. Prompt n-grams (universal across all strategies)
        for l in range(2, min(self.max_subtokens + 1, n + 1)):
            for i in range(n - l + 1):
                gram = tuple(p_ids[i : i + l])
                base_w = 8.0
                if strategy == RetrievalStrategy.SUFFIX_CONDITIONED:
                    # Boost n-grams appearing in the prompt suffix (last 16 tokens)
                    if i >= max(0, n - 16):
                        base_w *= 2.5
                add_candidate(gram, "prompt_ngram", base_w)

        # 2. Strategy-specific retrieval
        if strategy == RetrievalStrategy.BASELINE:
            # Standard 1-hop associations (top 24 per token)
            for tok in set(p_ids):
                assoc_list = self.index.token_associations.get(tok, [])
                for gram, w in assoc_list[:24]:
                    add_candidate(gram, "token_association", float(w))

        elif strategy == RetrievalStrategy.EXPANDED_ASSOCIATIONS:
            # Deeper 1-hop associations (top 48 per token)
            one_hop_cands: Dict[Tuple[int, ...], float] = defaultdict(float)
            for tok in set(p_ids):
                assoc_list = self.index.token_associations.get(tok, [])
                for gram, w in assoc_list[:48]:
                    add_candidate(gram, "token_association_1hop", float(w))
                    one_hop_cands[gram] += float(w)

            # 2-hop associations (from constituent tokens of top 1-hop candidates)
            top_1hop = sorted(one_hop_cands.items(), key=lambda x: x[1], reverse=True)[:32]
            for gram_1hop, w1 in top_1hop:
                for sub_tok in gram_1hop:
                    sub_assocs = self.index.token_associations.get(sub_tok, [])
                    for gram_2hop, w2 in sub_assocs[:12]:
                        add_candidate(gram_2hop, "token_association_2hop", float(w1 * w2 * 0.15))

        elif strategy == RetrievalStrategy.SUFFIX_CONDITIONED:
            # Higher weighting for tokens appearing near the prompt suffix
            suffix_tokens = set(p_ids[-16:]) if n > 0 else set()
            for idx_pos, tok in enumerate(p_ids):
                assoc_list = self.index.token_associations.get(tok, [])
                is_suffix = tok in suffix_tokens or idx_pos >= max(0, n - 16)
                weight_mul = 2.0 if is_suffix else 1.0
                limit = 32 if is_suffix else 16
                for gram, w in assoc_list[:limit]:
                    add_candidate(gram, "suffix_association" if is_suffix else "token_association", float(w) * weight_mul)

        elif strategy == RetrievalStrategy.SPARSE_LEXICAL:
            # Baseline 1-hop associations
            for tok in set(p_ids):
                assoc_list = self.index.token_associations.get(tok, [])
                for gram, w in assoc_list[:24]:
                    add_candidate(gram, "token_association", float(w))

            # BM25 sparse lexical retrieval against TRAIN prompt documents
            if self.bm25_doc_count > 0:
                query_words = re.findall(r"\b\w+\b", prompt_text.lower())
                scores: List[Tuple[int, float]] = []
                # BM25 scoring parameters
                k1 = 1.2
                b = 0.75
                avgdl = sum(len(d["word_list"]) for d in self.bm25_docs) / max(1, self.bm25_doc_count)

                for doc_idx, doc in enumerate(self.bm25_docs):
                    doc_len = len(doc["word_list"])
                    doc_words = doc["words"]
                    score = 0.0
                    for qw in query_words:
                        if qw in doc_words:
                            df = self.bm25_df[qw]
                            idf = math.log((self.bm25_doc_count - df + 0.5) / (df + 0.5) + 1.0)
                            # Term freq in doc
                            tf = doc["word_list"].count(qw)
                            numerator = tf * (k1 + 1)
                            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                            score += idf * (numerator / denominator)

                    if score > 0.0:
                        scores.append((doc_idx, score))

                # Top-5 nearest train prompts
                scores.sort(key=lambda x: x[1], reverse=True)
                for doc_idx, sim_score in scores[:5]:
                    doc = self.bm25_docs[doc_idx]
                    norm_sim = min(sim_score, 10.0) / 10.0
                    for ph_tuple, ph_weight in doc["phrases"].items():
                        add_candidate(ph_tuple, "lexical_neighbor", float(ph_weight) * norm_sim * 2.0)

        # 3. Domain & Global background bank
        # Add background items to ensure diversity and fill pool
        num_bg = min(128, target_pool_size // 2)
        for gram, bg_w in self.index.precomputed_global_static[:num_bg]:
            add_candidate(gram, "background_bank", float(bg_w) * 0.25)

        # 4. Filter bare punctuation and trailing whitespace
        filtered: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        for gram, data in cands.items():
            if is_bare_punctuation(gram, self.tokenizer):
                continue
            text = self.tokenizer.decode(list(gram))
            if text.endswith(" ") or text.endswith("\t"):
                continue
            filtered[gram] = data

        # 5. Truncate to target_pool_size by weight
        if len(filtered) > target_pool_size:
            # Deterministic sorting: weight descending, length ascending, tokens tuple ascending
            sorted_items = sorted(
                filtered.items(),
                key=lambda x: (x[1]["weight"], len(x[0]), x[0]),
                reverse=True,
            )
            filtered = dict(sorted_items[:target_pool_size])

        return filtered

    def build_candidate_records(
        self,
        record: VanillaContinuationRecord,
        strategy: RetrievalStrategy = RetrievalStrategy.BASELINE,
        target_pool_size: int = 512,
    ) -> List[CandidateRecord]:
        """Builds candidate pool and populates CandidateRecord objects."""
        cand_dict = self.generate_candidate_pool(
            record.prompt_token_ids,
            record.prompt_text,
            domain=record.domain,
            strategy=strategy,
            target_pool_size=target_pool_size,
        )

        continuation_tokens = record.continuation_token_ids
        n_cont = len(continuation_tokens)

        # Build occurrence index for fast lookup
        cont_counts: Counter[Tuple[int, ...]] = Counter()
        first_idx_map: Dict[Tuple[int, ...], int] = {}

        for l in range(2, min(self.max_subtokens + 1, n_cont + 1)):
            for i in range(n_cont - l + 1):
                g = tuple(continuation_tokens[i : i + l])
                cont_counts[g] += 1
                if g not in first_idx_map:
                    first_idx_map[g] = i

        cand_records: List[CandidateRecord] = []
        for gram, meta in cand_dict.items():
            text = self.tokenizer.decode(list(gram))
            p_len = len(gram)
            raw_w = meta["weight"]
            sources = sorted(list(meta["sources"]))

            count = cont_counts.get(gram, 0)
            occurs = count > 0
            first_idx = first_idx_map.get(gram, -1)
            bucket = get_first_occurrence_bucket(first_idx)
            isolated_saved = (p_len - 1) * count

            feats = extract_handcrafted_features(
                phrase_text=text,
                phrase_tokens=gram,
                raw_weight=raw_w,
                prompt_text=record.prompt_text,
                prompt_tokens=record.prompt_token_ids,
                domain=record.domain,
            )

            cand_records.append(
                CandidateRecord(
                    prompt_id=record.prompt_id,
                    tokens=gram,
                    text=text,
                    length=p_len,
                    sources=sources,
                    raw_association_weight=raw_w,
                    features=feats,
                    occurs_in_vanilla=occurs,
                    occurrence_count=count,
                    first_occurrence_index=first_idx,
                    first_occurrence_bucket=bucket,
                    isolated_steps_saved=isolated_saved,
                )
            )

        return cand_records
