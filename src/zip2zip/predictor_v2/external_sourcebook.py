"""Prompt-cleaned external response sourcebook for Predictor V2.

The benchmark prompt text is the only canonical data consulted by the corpus
filter. Canonical DEV/FINAL continuation text is never an input to this module.
Sourcebook phrase IDs are always produced by a caller-supplied, pinned tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
import time
import unicodedata
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.zip2zip.predictor_v2.candidate_pool import candidate_filter_reason, candidate_filter_reason_from_text


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
CHAR_NGRAM = 3
MINHASH_SEEDS = tuple(range(16))
MINHASH_BANDS = 8
MAX_SOURCE_RESPONSE_TOKENS = 512


class SourcebookError(ValueError):
    """Raised when external sourcebook inputs or provenance are invalid."""


@dataclass(frozen=True)
class CanonicalPrompt:
    prompt_id: str
    domain: str
    text: str


@dataclass(frozen=True)
class ExternalExample:
    example_id: str
    prompt: str
    response: str
    source_domain: str = "unlabeled"
    retrieval_prompt: str | None = None
    filter_prompts: tuple[str, ...] = ()


def normalize_prompt(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("prompt must be text")
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(value.split())


def _word_tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(normalize_prompt(text)))


def _char_ngrams(text: str, n: int = CHAR_NGRAM) -> frozenset[str]:
    value = " " + " ".join(_word_tokens(text)) + " "
    return frozenset(value[i : i + n] for i in range(max(0, len(value) - n + 1)))


def _jaccard(left: frozenset[Any], right: frozenset[Any]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _minhash_signature(words: frozenset[str]) -> tuple[int, ...]:
    if not words:
        return tuple(0xFFFFFFFF for _ in MINHASH_SEEDS)
    encoded = [word.encode("utf-8") for word in words]
    result = []
    for seed in MINHASH_SEEDS:
        prefix = struct.pack(">I", seed)
        result.append(min(zlib.crc32(item, zlib.crc32(prefix)) for item in encoded))
    return tuple(result)


class PromptContaminationIndex:
    """Deterministic exact, lexical, and fuzzy index over all benchmark prompts."""

    def __init__(self, prompts: Sequence[CanonicalPrompt]):
        if len(prompts) != 900:
            raise SourcebookError(f"decontamination requires all 900 canonical prompts; got {len(prompts)}")
        if len({item.prompt_id for item in prompts}) != 900:
            raise SourcebookError("canonical prompt IDs must be unique")
        self.prompts = tuple(prompts)
        self._exact: dict[str, int] = {}
        self._words: list[frozenset[str]] = []
        self._chars: list[frozenset[str]] = []
        self._bands: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        self._word_df: Counter[str] = Counter()
        self._word_postings: dict[str, list[int]] = defaultdict(list)
        for idx, item in enumerate(self.prompts):
            normalized = normalize_prompt(item.text)
            if not normalized:
                raise SourcebookError(f"canonical prompt {item.prompt_id} is empty")
            self._exact[hashlib.sha256(normalized.encode("utf-8")).hexdigest()] = idx
            words = frozenset(_word_tokens(item.text))
            self._words.append(words)
            self._chars.append(_char_ngrams(item.text))
            self._word_df.update(words)
            for word in words:
                self._word_postings[word].append(idx)
            signature = _minhash_signature(words)
            for band in range(MINHASH_BANDS):
                start = band * (len(signature) // MINHASH_BANDS)
                key = (band, signature[start], signature[start + 1])
                self._bands[key].append(idx)

    def match(self, prompt: str) -> dict[str, Any] | None:
        normalized = normalize_prompt(prompt)
        if not normalized:
            return None
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        exact_idx = self._exact.get(digest)
        if exact_idx is not None:
            target = self.prompts[exact_idx]
            return {"method": "exact_normalized_sha256", "canonical_prompt_id": target.prompt_id, "canonical_domain": target.domain, "word_jaccard": 1.0, "char_trigram_jaccard": 1.0}

        words = frozenset(_word_tokens(prompt))
        if len(words) < 3:
            return None
        candidates: set[int] = set()
        signature = _minhash_signature(words)
        for band in range(MINHASH_BANDS):
            start = band * (len(signature) // MINHASH_BANDS)
            candidates.update(self._bands.get((band, signature[start], signature[start + 1]), ()))
        # Rare-word postings make short, reordered task descriptions robust to
        # the MinHash sketch while keeping query work bounded by the 900 rows.
        for word in words:
            if self._word_df[word] <= 8:
                candidates.update(self._word_postings[word])

        chars = _char_ngrams(prompt)
        scored: list[tuple[float, float, int]] = []
        for idx in candidates:
            word_score = _jaccard(words, self._words[idx])
            char_score = _jaccard(chars, self._chars[idx])
            if word_score >= 0.84 or (len(words) >= 6 and char_score >= 0.88):
                scored.append((word_score, char_score, idx))
        if not scored:
            return None
        word_score, char_score, idx = max(scored, key=lambda value: (value[0], value[1], -value[2]))
        target = self.prompts[idx]
        method = "near_duplicate_lexical" if word_score >= 0.84 else "near_duplicate_fuzzy_char_trigram"
        return {"method": method, "canonical_prompt_id": target.prompt_id, "canonical_domain": target.domain, "word_jaccard": round(word_score, 6), "char_trigram_jaccard": round(char_score, 6)}


def load_canonical_prompt_texts(path: str | Path) -> list[CanonicalPrompt]:
    """Read prompt identifiers/domain/text only; never expose answer fields."""
    prompts: list[CanonicalPrompt] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("task_prompt_text") or row.get("prompt_text")
            if not isinstance(prompt, str):
                raise SourcebookError(f"canonical row {line_number} has no task prompt text")
            prompts.append(CanonicalPrompt(str(row.get("prompt_id", "")), str(row.get("domain", "")), prompt))
    if len(prompts) != 900:
        raise SourcebookError(f"expected 900 canonical prompt rows, found {len(prompts)}")
    return prompts


def parse_external_example(row: Mapping[str, Any]) -> ExternalExample | None:
    """Extract a user prompt and final assistant response from chat-like data."""
    messages = row.get("messages")
    prompt = row.get("prompt")
    response: str | None = None
    retrieval_prompt: str | None = None
    filter_prompts: tuple[str, ...] = ()
    if isinstance(messages, list):
        users = [m.get("content") for m in messages if isinstance(m, Mapping) and str(m.get("role", "")).casefold() in {"user", "human"} and isinstance(m.get("content"), str)]
        assistants = [(idx, m.get("content")) for idx, m in enumerate(messages) if isinstance(m, Mapping) and str(m.get("role", "")).casefold() in {"assistant", "gpt"} and isinstance(m.get("content"), str)]
        if users:
            prompt = users[-1]
        if assistants:
            response_idx, response = assistants[-1]
            context = [m for m in messages[:response_idx] if isinstance(m, Mapping) and isinstance(m.get("content"), str)]
            retrieval_prompt = "\n".join(f"{str(m.get('role', 'user'))}: {m['content']}" for m in context)
            filter_prompts = tuple(str(m["content"]) for m in context if str(m.get("role", "")).casefold() in {"user", "human"} and str(m["content"]).strip())
    if not isinstance(prompt, str) or not isinstance(response, str):
        return None
    prompt, response = prompt.strip(), response.strip()
    if not prompt or not response:
        return None
    raw_id = row.get("id", row.get("prompt_id"))
    if raw_id is None:
        raw_id = hashlib.sha256((prompt + "\0" + response).encode("utf-8")).hexdigest()
    if not filter_prompts:
        filter_prompts = (prompt,)
    return ExternalExample(str(raw_id), prompt, response, str(row.get("domain", "unlabeled")), retrieval_prompt or prompt, filter_prompts)


def _phrase_key(phrase: Sequence[int]) -> bytes:
    return struct.pack(">" + "I" * len(phrase), *(int(token) for token in phrase))


def _pack_ids(ids: Sequence[int]) -> bytes:
    return zlib.compress(struct.pack(">" + "I" * len(ids), *(int(item) for item in ids)), level=6)


def _unpack_ids(blob: bytes) -> list[int]:
    raw = zlib.decompress(blob)
    if len(raw) % 4:
        raise SourcebookError("stored token sequence is corrupt")
    return list(struct.unpack(">" + "I" * (len(raw) // 4), raw))


def _response_ngrams(ids: Sequence[int], disabled_ids: set[int], tokenizer: Any, max_tokens: int = MAX_SOURCE_RESPONSE_TOKENS) -> tuple[Counter[tuple[int, ...]], Counter[str]]:
    tokens = ids[:max_tokens]
    observed: Counter[tuple[int, ...]] = Counter()
    rejections: Counter[str] = Counter()
    for length in (2, 3, 4):
        for start in range(len(tokens) - length + 1):
            phrase = tuple(tokens[start : start + length])
            if any(token in disabled_ids for token in phrase):
                rejections["special_token"] += 1
                continue
            observed[phrase] += 1
    phrases: Counter[tuple[int, ...]] = Counter()
    for phrase, count in observed.items():
        text = tokenizer.decode(list(phrase))
        reason = candidate_filter_reason_from_text(text)
        if reason:
            rejections[reason] += count
            continue
        if "\ufffd" in text or any(ord(char) < 32 and char not in "\n\t\r" for char in text):
            rejections["malformed_fragment"] += count
            continue
        phrases[phrase] = count
    return phrases, rejections


class SourcebookBuilder:
    """Chunked, resumable SQLite phrase sourcebook builder."""

    SCHEMA = "predictor_v2_external_sourcebook_v1"

    def __init__(self, path: str | Path, *, tokenizer: Any, tokenizer_id: str, tokenizer_revision: str, config: Mapping[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.tokenizer_id = tokenizer_id
        self.tokenizer_revision = tokenizer_revision
        self.config = {"max_phrase_tokens": 4, "max_response_tokens": MAX_SOURCE_RESPONSE_TOKENS, **dict(config or {})}
        self.config_sha256 = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA temp_store=FILE")
        self._initialize()

    def _initialize(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS examples (
                example_id TEXT PRIMARY KEY, prompt TEXT NOT NULL, response_ids BLOB NOT NULL,
                source_domain TEXT NOT NULL, prompt_length INTEGER NOT NULL, prompt_words TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS example_terms (
                term TEXT NOT NULL, example_id TEXT NOT NULL, tf INTEGER NOT NULL,
                PRIMARY KEY(term, example_id)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS example_terms_by_doc ON example_terms(example_id);
            CREATE TABLE IF NOT EXISTS term_stats (term TEXT PRIMARY KEY, df INTEGER NOT NULL, idf REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS phrase_types (
                phrase_key BLOB PRIMARY KEY, phrase_length INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
        """)
        stored_schema = self._meta("schema")
        if stored_schema is not None and (stored_schema != self.SCHEMA or self._meta("config_sha256") != self.config_sha256 or self._meta("tokenizer_id") != self.tokenizer_id or self._meta("tokenizer_revision") != self.tokenizer_revision):
            raise SourcebookError("resume database provenance does not match the requested sourcebook build")
        for key, value in {"schema": self.SCHEMA, "config_sha256": self.config_sha256, "tokenizer_id": self.tokenizer_id, "tokenizer_revision": self.tokenizer_revision, "config": json.dumps(self.config, sort_keys=True)}.items():
            self.db.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES (?,?)", (key, value))
        self.db.commit()

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def add_chunk(self, examples: Iterable[ExternalExample], *, chunk_size: int = 128) -> int:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        disabled = set(getattr(self.tokenizer, "all_special_ids", ()) or ())
        for key in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
            value = getattr(self.tokenizer, key, None)
            if isinstance(value, int):
                disabled.add(value)
        disabled.update({0, 1, 2, *range(32000, 32011)})
        inserted = 0
        pending = 0
        self.db.execute("BEGIN")
        try:
            for item in examples:
                if not isinstance(item, ExternalExample) or not item.example_id or not item.prompt.strip() or not item.response.strip():
                    continue
                if self.db.execute("SELECT 1 FROM examples WHERE example_id=?", (item.example_id,)).fetchone():
                    continue
                retrieval_prompt = item.retrieval_prompt or item.prompt
                words = TOKEN_RE.findall(normalize_prompt(retrieval_prompt))
                if not words:
                    continue
                ids = self.tokenizer.encode(item.response, add_special_tokens=False)
                if not ids:
                    continue
                ids = [int(token) for token in ids[: int(self.config["max_response_tokens"])]]
                self.db.execute("INSERT INTO examples VALUES (?,?,?,?,?,?)", (item.example_id, retrieval_prompt, _pack_ids(ids), item.source_domain, len(words), json.dumps(words, ensure_ascii=False)))
                self.db.executemany("INSERT INTO example_terms(term,example_id,tf) VALUES (?,?,?)", ((term, item.example_id, count) for term, count in Counter(words).items()))
                phrases, rejections = _response_ngrams(ids, disabled, self.tokenizer, int(self.config["max_response_tokens"]))
                self.db.executemany("INSERT OR IGNORE INTO phrase_types(phrase_key,phrase_length) VALUES (?,?)", ((_phrase_key(phrase), len(phrase)) for phrase in phrases))
                for name, count in rejections.items():
                    self.db.execute("INSERT INTO counters(name,value) VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value", (f"rejected_{name}", count))
                self.db.execute("INSERT INTO counters(name,value) VALUES ('source_examples',1) ON CONFLICT(name) DO UPDATE SET value=value+1")
                self.db.execute("INSERT INTO counters(name,value) VALUES ('phrase_occurrences',?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value", (sum(phrases.values()),))
                self.db.execute("INSERT INTO counters(name,value) VALUES ('phrase_occurrences_2',?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value", (sum(count for phrase, count in phrases.items() if len(phrase) == 2),))
                self.db.execute("INSERT INTO counters(name,value) VALUES ('phrase_occurrences_3',?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value", (sum(count for phrase, count in phrases.items() if len(phrase) == 3),))
                self.db.execute("INSERT INTO counters(name,value) VALUES ('phrase_occurrences_4',?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value", (sum(count for phrase, count in phrases.items() if len(phrase) == 4),))
                inserted += 1
                pending += 1
                if pending >= chunk_size:
                    self.db.commit()
                    self.db.execute("BEGIN")
                    pending = 0
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self._finalize_terms()
        return inserted

    def _finalize_terms(self) -> None:
        count = self.db.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
        self.db.execute("DELETE FROM term_stats")
        self.db.execute("""
            INSERT INTO term_stats(term,df,idf)
            SELECT term, COUNT(*), log(1.0 + (? - COUNT(*) + 0.5) / (COUNT(*) + 0.5))
            FROM example_terms GROUP BY term
        """, (count,))
        self.db.commit()

    def summary(self) -> dict[str, Any]:
        counts = {str(length): int(count) for length, count in self.db.execute("SELECT phrase_length, COUNT(*) FROM phrase_types GROUP BY phrase_length")}
        examples = int(self.db.execute("SELECT COUNT(*) FROM examples").fetchone()[0])
        counters = {name: int(value) for name, value in self.db.execute("SELECT name,value FROM counters")}
        return {
            "schema": self.SCHEMA,
            "source_prompts": examples,
            "unique_phrases_by_phi_token_length": {str(n): counts.get(str(n), 0) for n in (2, 3, 4)},
            "phrase_occurrences_by_phi_token_length": {str(n): counters.get(f"phrase_occurrences_{n}", 0) for n in (2, 3, 4)},
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "config": self.config,
            "config_sha256": self.config_sha256,
            "database_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "counters": counters,
        }

    def close(self) -> None:
        self.db.close()


class ExternalSourcebookCandidateGenerator:
    """BM25 prompt retrieval over a sourcebook; phrases are mined from responses."""

    def __init__(self, database_path: str | Path, tokenizer: Any, *, top_source_examples: int = 48, max_subtokens: int = 4):
        from src.zip2zip.predictor_v2.candidate_retrieval import ConfigurableCandidateGenerator, TrainOnlyAssociationIndex

        self.database_path = Path(database_path)
        self.tokenizer = tokenizer
        self.max_subtokens = max_subtokens
        self.disabled_ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
        self.disabled_ids.update({0, 1, 2, *range(32000, 32011)})
        self.top_source_examples = top_source_examples
        self.db = sqlite3.connect(f"file:{self.database_path.resolve().as_posix()}?mode=ro", uri=True)
        self.example_count = int(self.db.execute("SELECT COUNT(*) FROM examples").fetchone()[0])
        self.term_count = int(self.db.execute("SELECT COUNT(*) FROM term_stats").fetchone()[0])
        self.average_prompt_length = float(self.db.execute("SELECT COALESCE(AVG(prompt_length),1.0) FROM examples").fetchone()[0])
        self.index_bytes = self.database_path.stat().st_size
        self._candidate_records_builder = ConfigurableCandidateGenerator(TrainOnlyAssociationIndex(), tokenizer, max_subtokens)

    def close(self) -> None:
        self.db.close()

    def generate_candidate_pool(self, prompt_ids: Sequence[int], prompt_text: str, domain: str = "general", strategy: Any = "external_sourcebook", target_pool_size: int = 512, diagnostics: dict[str, Any] | None = None) -> dict[tuple[int, ...], dict[str, Any]]:
        started = time.perf_counter()
        query = Counter(TOKEN_RE.findall(normalize_prompt(prompt_text)))
        query_terms = list(query)
        candidates: dict[tuple[int, ...], dict[str, Any]] = {}
        rejections: dict[tuple[int, ...], str] = {}
        if query_terms:
            placeholders = ",".join("?" for _ in query_terms)
            rows = self.db.execute(
                f"""SELECT et.example_id,
                    SUM(ts.idf * et.tf * 2.2 / (et.tf + 1.2 * (0.25 + 0.75 * (e.prompt_length / ?)))) AS score,
                    e.response_ids
                    FROM example_terms et
                    JOIN term_stats ts ON ts.term=et.term
                    JOIN examples e ON e.example_id=et.example_id
                    WHERE et.term IN ({placeholders})
                    GROUP BY et.example_id
                    ORDER BY score DESC, et.example_id ASC LIMIT ?""",
                (max(self.average_prompt_length, 1.0), *query_terms, self.top_source_examples),
            ).fetchall()
            max_score = max((float(row[1]) for row in rows), default=0.0)
            for rank, (example_id, raw_score, token_blob) in enumerate(rows):
                relevance = float(raw_score) / max_score if max_score else 0.0
                ids = _unpack_ids(token_blob)
                phrases, _ = _response_ngrams(ids, self.disabled_ids, self.tokenizer)
                for phrase, count in phrases.most_common(320):
                    reason = candidate_filter_reason(phrase, self.tokenizer)
                    if reason:
                        rejections[phrase] = reason
                        continue
                    payload = candidates.setdefault(phrase, {"sources": set(), "weight": 0.0})
                    payload["sources"].add("external_sourcebook")
                    payload["weight"] += relevance * math.log1p(count) * (1.0 + 0.12 * (len(phrase) - 2)) / (1.0 + rank * 0.025)
        ordered = sorted(candidates.items(), key=lambda item: (item[1]["weight"], -len(item[0]), item[0]), reverse=True)
        pool = dict(ordered[:target_pool_size])
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({
                "generation_latency_ms": (time.perf_counter() - started) * 1000.0,
                "generated_candidate_count": len(candidates),
                "quality_rejections": rejections,
                "rank_by_phrase": {phrase: rank for rank, (phrase, _) in enumerate(ordered, 1)},
                "sources_by_phrase": {phrase: set(meta["sources"]) for phrase, meta in candidates.items()},
                "pool_source_counts": {"external_sourcebook": len(pool)} if pool else {},
                "pool_phrase_length_counts": {str(length): sum(len(phrase) == length for phrase in pool) for length in range(2, self.max_subtokens + 1)},
                "retrieved_source_examples": len(rows) if query_terms else 0,
                "source_index_bytes": self.index_bytes,
                "source_index_examples": self.example_count,
                "source_index_terms": self.term_count,
            })
        return pool

    def build_candidate_records(self, record: Any, strategy: Any = "external_sourcebook", target_pool_size: int = 512, candidate_pool: Mapping[tuple[int, ...], dict[str, Any]] | None = None) -> list[Any]:
        from src.zip2zip.predictor_v2.candidate_retrieval import RetrievalStrategy
        return self._candidate_records_builder.build_candidate_records(record, strategy=RetrievalStrategy.BASELINE, target_pool_size=target_pool_size, candidate_pool=candidate_pool)


def make_hybrid_pool(external: Mapping[tuple[int, ...], Mapping[str, Any]], phi: Mapping[tuple[int, ...], Mapping[str, Any]], target_pool_size: int, *, external_weight: float = 1.0, phi_weight: float = 1.0) -> dict[tuple[int, ...], dict[str, Any]]:
    """Deterministic max-normalized weighted union of external and Phi proposals."""
    merged: dict[tuple[int, ...], dict[str, Any]] = {}
    for source_name, source, scale in (("external_sourcebook", external, external_weight), ("phi_train_retrieval", phi, phi_weight)):
        maximum = max((float(payload.get("weight", 0.0)) for payload in source.values()), default=0.0)
        for phrase, payload in source.items():
            item = merged.setdefault(tuple(phrase), {"sources": set(), "weight": 0.0})
            item["sources"].add(source_name)
            raw_weight = float(payload.get("weight", 0.0))
            item["weight"] += (raw_weight / maximum if maximum > 0 else 0.0) * scale
    ordered = sorted(merged.items(), key=lambda item: (item[1]["weight"], -len(item[0]), item[0]), reverse=True)
    return dict(ordered[:target_pool_size])


def phrase_book_summary(path: str | Path) -> dict[str, Any]:
    with sqlite3.connect(path) as db:
        meta = dict(db.execute("SELECT key,value FROM metadata"))
        counts = {str(length): int(count) for length, count in db.execute("SELECT phrase_length,COUNT(*) FROM phrase_types GROUP BY phrase_length")}
        examples = int(db.execute("SELECT COUNT(*) FROM examples").fetchone()[0])
        counters = dict(db.execute("SELECT name,value FROM counters"))
    return {"schema": meta.get("schema"), "source_prompts": examples, "unique_phrases_by_phi_token_length": {str(n): counts.get(str(n), 0) for n in (2, 3, 4)}, "counters": counters, "tokenizer_id": meta.get("tokenizer_id"), "tokenizer_revision": meta.get("tokenizer_revision"), "database_bytes": Path(path).stat().st_size}
