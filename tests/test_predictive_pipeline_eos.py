"""Small deterministic checks for EOS-terminated predictive training targets."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zip2zip.predictive_pipeline import PredictivePipeline


class TinyTokenizer:
    eos_token_id = 99

    encodings = {
        "prompt": [1, 2, 1, 2],
        "response": [1, 2, 1],
        "response_with_eos": [1, 2, 99],
        "alternate_response": [2, 1],
    }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(self.encodings[text])


class TinyPolicy:
    def __init__(self):
        self.prompts = []

    def select_codebook(self, prompt_ids):
        self.prompts.append(list(prompt_ids))
        return {(1, 2): 100}, {"selected_from": list(prompt_ids)}


def make_pipeline():
    policy = TinyPolicy()
    return PredictivePipeline(policy, TinyTokenizer(), initial_vocab_size=100), policy


def test_prompt_mask_and_eos_terminated_lossless_response():
    pipeline, _ = make_pipeline()
    sample = pipeline.process_sample("prompt", "response")
    prompt_len = len(sample["compressed_prompt_ids"])

    assert sample["labels"][:prompt_len] == [-100] * prompt_len
    assert sample["labels"][prompt_len:] == sample["compressed_response_ids"]
    assert sample["input_ids"][-1] == sample["labels"][-1] == 99
    assert sample["original_response_ids"] == [1, 2, 1, 99]
    assert sample["compressed_response_ids"] == [100, 1, 99]
    assert pipeline.decode_sequence(
        sample["compressed_response_ids"],
        {(1, 2): 100},
    ) == sample["original_response_ids"]


def test_existing_terminal_eos_is_not_duplicated():
    pipeline, _ = make_pipeline()
    sample = pipeline.process_sample("prompt", "response_with_eos")

    assert sample["compressed_response_ids"][-1] == 99
    assert sample["compressed_response_ids"].count(99) == 1
    assert sample["original_response_ids"] == [1, 2, 99]
    assert pipeline.decode_sequence(sample["compressed_response_ids"], {(1, 2): 100}) == [1, 2, 99]


def test_tokenizer_without_eos_id_fails_explicitly():
    pipeline, _ = make_pipeline()
    pipeline.tokenizer.eos_token_id = None

    try:
        pipeline.process_sample("prompt", "response")
    except ValueError as exc:
        assert "eos_token_id" in str(exc)
    else:
        raise AssertionError("Expected a clear error when the tokenizer has no EOS token ID")


def test_codebook_selection_receives_prompt_only_for_different_responses():
    pipeline, policy = make_pipeline()
    first = pipeline.process_sample("prompt", "response")
    second = pipeline.process_sample("prompt", "alternate_response")

    assert policy.prompts == [[1, 2, 1, 2], [1, 2, 1, 2]]
    assert first["codebook_tuples"] == second["codebook_tuples"] == [(1, 2)]
    assert first["original_response_ids"] != second["original_response_ids"]
    assert pipeline.decode_sequence(second["compressed_response_ids"], {(1, 2): 100}) == second["original_response_ids"]
