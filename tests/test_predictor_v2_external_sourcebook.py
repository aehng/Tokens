from __future__ import annotations

import json
import re

import pytest

from src.zip2zip.predictor_v2.candidate_retrieval import ConfigurableCandidateGenerator, TrainOnlyAssociationIndex
from src.zip2zip.predictor_v2.external_sourcebook import (
    CanonicalPrompt,
    ExternalExample,
    ExternalSourcebookCandidateGenerator,
    PromptContaminationIndex,
    SourcebookBuilder,
    SourcebookError,
    _response_ngrams,
    load_canonical_prompt_texts,
    make_hybrid_pool,
    parse_external_example,
)
from src.zip2zip.predictor_v2.vanilla_labels import VanillaContinuationRecord
from src.zip2zip.predictor_v2.canonical_dataset import sha256_file


class FakeTokenizer:
    all_special_ids = [999]
    bos_token_id = 999
    eos_token_id = 999
    pad_token_id = 999
    unk_token_id = 999

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.reverse: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        pieces = re.findall(r"\s*\S+", text)
        output = []
        for piece in pieces:
            if piece not in self.vocab:
                token_id = len(self.vocab) + 10
                self.vocab[piece] = token_id
                self.reverse[token_id] = piece
            output.append(self.vocab[piece])
        return output

    def decode(self, ids: list[int]) -> str:
        return "".join(self.reverse.get(token_id, "") for token_id in ids)


def _canonical_prompts() -> list[CanonicalPrompt]:
    return [CanonicalPrompt(f"prompt-{i}", ("code", "reasoning", "instruction")[i % 3], f"Canonical benchmark request {i} with unique details alpha{i} beta{i} gamma{i}") for i in range(900)]


def test_decontamination_removes_exact_reordered_near_duplicate_and_fuzzy_text():
    prompts = _canonical_prompts()
    index = PromptContaminationIndex(prompts)

    exact = prompts[0].text.upper().replace("  ", " ")
    assert index.match(exact)["method"] == "exact_normalized_sha256"

    target = "Please calculate the total average of these exact values using simple arithmetic"
    reordered = "using simple arithmetic please calculate the total average of these exact values"
    reordered_index = PromptContaminationIndex([CanonicalPrompt(f"p{i}", "reasoning", target if i == 0 else f"unrelated benchmark prompt {i} has distinct content number {i}") for i in range(900)])
    match = reordered_index.match(reordered)
    assert match is not None
    assert match["canonical_prompt_id"] == "p0"
    assert match["method"] == "near_duplicate_lexical"

    lightly_reformatted = target.replace("arithmetic", "arithmetical")
    fuzzy = reordered_index.match(lightly_reformatted)
    assert fuzzy is not None
    assert fuzzy["canonical_prompt_id"] == "p0"


def test_filter_requires_all_900_prompt_texts_and_never_indexes_responses(tmp_path):
    rows = []
    for index in range(900):
        rows.append({"prompt_id": str(index), "domain": "instruction", "task_prompt_text": f"task prompt unique item {index}", "continuation_text": "held out secret phrase omega" if index == 0 else "other response"})
    canonical_path = tmp_path / "canonical.jsonl"
    canonical_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    canonical_prompts = load_canonical_prompt_texts(canonical_path)
    assert len(canonical_prompts) == 900
    index = PromptContaminationIndex(canonical_prompts)
    assert index.match("held out secret phrase omega") is None
    assert not any(hasattr(item, "continuation_text") for item in canonical_prompts)
    with pytest.raises(SourcebookError, match="all 900"):
        PromptContaminationIndex(canonical_prompts[:899])


def test_phi_phrase_extraction_enforces_two_to_four_tokens_and_special_token_filtering():
    tokenizer = FakeTokenizer()
    ids = tokenizer.encode("alpha beta gamma delta", add_special_tokens=False)
    ids.extend([999, *tokenizer.encode("epsilon zeta", add_special_tokens=False)])
    phrases, rejections = _response_ngrams(ids, {999}, tokenizer)
    assert phrases
    assert all(2 <= len(phrase) <= 4 for phrase in phrases)
    assert all(999 not in phrase for phrase in phrases)
    assert rejections["special_token"] > 0


def test_sourcebook_build_is_deterministic_resumable_and_provenance_bound(tmp_path):
    examples = [
        ExternalExample("a", "write a short greeting", "Hello there friend. I can help you today."),
        ExternalExample("b", "explain a small calculation", "The total is easy to calculate by addition."),
    ]
    summaries = []
    paths = []
    for name in ("first", "second"):
        db_path = tmp_path / f"{name}.sqlite"
        tokenizer = FakeTokenizer()
        builder = SourcebookBuilder(db_path, tokenizer=tokenizer, tokenizer_id="fake-phi", tokenizer_revision="rev-1", config={"fixture": "deterministic"})
        assert builder.add_chunk(examples, chunk_size=1) == 2
        before = builder.summary()
        assert builder.add_chunk(examples, chunk_size=1) == 0
        after = builder.summary()
        assert after["unique_phrases_by_phi_token_length"] == before["unique_phrases_by_phi_token_length"]
        summaries.append(after["unique_phrases_by_phi_token_length"])
        paths.append(db_path)
        builder.close()
    assert summaries[0] == summaries[1]
    assert sha256_file(paths[0]) == sha256_file(paths[1])

    with pytest.raises(SourcebookError, match="provenance"):
        SourcebookBuilder(paths[0], tokenizer=FakeTokenizer(), tokenizer_id="fake-phi", tokenizer_revision="rev-2", config={"fixture": "deterministic"})


def test_external_retrieval_is_deterministic_and_builds_existing_candidate_records(tmp_path):
    db_path = tmp_path / "source.sqlite"
    tokenizer = FakeTokenizer()
    builder = SourcebookBuilder(db_path, tokenizer=tokenizer, tokenizer_id="fake-phi", tokenizer_revision="rev-1")
    builder.add_chunk([
        ExternalExample("a", "write a short greeting", "Hello there friend. I can help you today."),
        ExternalExample("b", "write a friendly welcome", "Hello dear friend. Welcome to the group today."),
        ExternalExample("c", "calculate a math result", "The answer is four after simple addition."),
    ])
    builder.close()
    generator = ExternalSourcebookCandidateGenerator(db_path, tokenizer, top_source_examples=3)
    first_diag: dict = {}
    second_diag: dict = {}
    first = generator.generate_candidate_pool([], "write a friendly greeting", target_pool_size=64, diagnostics=first_diag)
    second = generator.generate_candidate_pool([], "write a friendly greeting", target_pool_size=64, diagnostics=second_diag)
    assert first == second
    assert first_diag["rank_by_phrase"] == second_diag["rank_by_phrase"]
    assert all(2 <= len(phrase) <= 4 for phrase in first)
    legacy = VanillaContinuationRecord(prompt_id="dev-1", domain="instruction", prompt_text="write a friendly greeting", prompt_token_ids=[], continuation_text="hello", continuation_token_ids=[])
    candidate_records = generator.build_candidate_records(legacy, candidate_pool=first)
    assert len(candidate_records) == len(first)
    assert all(record.prompt_id == "dev-1" for record in candidate_records)
    generator.close()

    configured = ConfigurableCandidateGenerator(TrainOnlyAssociationIndex(), tokenizer, external_sourcebook_path=str(db_path))
    external_pool = configured.generate_candidate_pool([], "write a friendly greeting", strategy="external_sourcebook", target_pool_size=64)
    hybrid_pool = configured.generate_candidate_pool([], "write a friendly greeting", strategy="hybrid_sourcebook", target_pool_size=64)
    assert external_pool
    assert hybrid_pool
    assert all("external_sourcebook" in item["sources"] for item in external_pool.values())
    assert any("phi_train_retrieval" in item["sources"] for item in hybrid_pool.values()) is False
    configured.close()


def test_hybrid_union_combines_provenance_and_deterministically_ranks():
    external = {
        (1, 2): {"sources": {"external_sourcebook"}, "weight": 3.0},
        (3, 4, 5): {"sources": {"external_sourcebook"}, "weight": 1.0},
    }
    phi = {
        (1, 2): {"sources": {"token_association"}, "weight": 2.0},
        (6, 7): {"sources": {"phi_train"}, "weight": 4.0},
    }
    result = make_hybrid_pool(external, phi, 2)
    assert list(result) == [(1, 2), (6, 7)]
    assert result[(1, 2)]["sources"] == {"external_sourcebook", "phi_train_retrieval"}
    assert result[(1, 2)]["weight"] == 1.5


def test_malformed_external_records_are_skipped():
    assert parse_external_example({"prompt": "missing assistant answer", "messages": [{"role": "user", "content": "question"}]}) is None
    parsed = parse_external_example({"id": "ok", "messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]})
    assert parsed is not None
    assert parsed.example_id == "ok"
    assert parsed.prompt == "question"
    assert parsed.response == "answer"
