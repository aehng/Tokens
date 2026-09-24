"""CPU-only correctness, operator-profile, and directional timing harness.

This script intentionally does not inspect or use CUDA. Its timing results are
not evidence of GPU speedup.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zip2zip import StaticCodebookManager  # noqa: E402
from zip2zip.nn.embedding import HyperEmbedding  # noqa: E402
from zip2zip.nn.linear import HyperLinear  # noqa: E402


VOCAB = 2048
TAIL = 4
CODEBOOK_SIZE = 32
HIDDEN = 128
MAX_SUBTOKENS = 3
PAD_ID = 0
PHRASES = [[2, 3], [4, 5, 6]]
EOS_ID = VOCAB - 1


class MeanEncoder:
    def get_encoder_fn(self):
        def encode(updates, base_weight, pad_token_id):
            valid = updates.ne(pad_token_id)
            vectors = F.embedding(updates, base_weight)
            return (vectors * valid.unsqueeze(-1)).sum(dim=-2) / valid.sum(
                dim=-1, keepdim=True
            ).clamp_min(1)

        return encode


class TinyCausalModel(nn.Module):
    def __init__(self, embedding, output_head):
        super().__init__()
        self.embedding = embedding
        self.output_head = output_head

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output_head


class TinyPredictiveModel(nn.Module):
    def __init__(self, manager, embedding, output_head):
        super().__init__()
        self.base_model = TinyCausalModel(embedding, output_head)
        self.input_encoder = MeanEncoder()
        self.output_encoder = MeanEncoder()
        self.codebook_manager = manager


def make_manager():
    manager = StaticCodebookManager(
        initial_vocab_size=VOCAB,
        max_codebook_size=CODEBOOK_SIZE,
        max_subtokens=MAX_SUBTOKENS,
        embedding_dim=HIDDEN,
        pad_token_id=PAD_ID,
    )
    manager.set_seeded_codebook(PHRASES)
    return manager


def make_model(manager):
    embedding = nn.Embedding(VOCAB + TAIL, HIDDEN, padding_idx=PAD_ID)
    output_head = nn.Linear(HIDDEN, VOCAB + TAIL, bias=True)
    model = TinyPredictiveModel(
        manager,
        HyperEmbedding(
            config=None,
            encoder=MeanEncoder(),
            num_embeddings=VOCAB + TAIL,
            embedding_dim=HIDDEN,
            padding_idx=PAD_ID,
            device=torch.device("cpu"),
            dtype=embedding.weight.dtype,
            initial_vocab_size=VOCAB,
            codebook_manager=manager,
        ),
        HyperLinear(
            config=None,
            encoder=MeanEncoder(),
            in_features=HIDDEN,
            out_features=VOCAB + TAIL,
            bias=True,
            device=torch.device("cpu"),
            dtype=output_head.weight.dtype,
            initial_vocab_size=VOCAB,
            codebook_manager=manager,
        ),
    ).eval()
    with torch.no_grad():
        model.base_model.embedding.weight.copy_(embedding.weight)
        model.base_model.output_head.weight.copy_(output_head.weight)
        model.base_model.output_head.bias.copy_(output_head.bias)
    manager.attach_to_model(model)
    return model


def op_counts(operation):
    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as trace:
        operation()
    return {
        event.key: event.count
        for event in trace.key_averages()
        if event.key
        in {
            "aten::embedding",
            "aten::arange",
            "aten::mul",
            "aten::add",
            "aten::where",
            "aten::any",
            "aten::linear",
            "aten::mm",
            "aten::bmm",
            "aten::cat",
        }
    }


def median_ms(operation, *, iterations=100, repeats=5):
    for _ in range(10):
        operation()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            operation()
        samples.append((time.perf_counter() - started) * 1000 / iterations)
    return round(statistics.median(samples), 4)


def greedy_fixture(embedding, output_head, manager, prompt_ids, steps):
    sequence = prompt_ids.clone()
    generated = []
    for _ in range(steps):
        hidden = embedding(sequence)[:, -1:, :]
        logits = output_head(hidden)[:, -1, :].clone()
        logits[..., VOCAB + CODEBOOK_SIZE :] = float("-inf")
        logits = manager.mask_unused_logits(logits)
        next_id = logits.argmax(dim=-1, keepdim=True)
        generated.append(next_id)
        sequence = torch.cat((sequence, next_id), dim=-1)
    return torch.cat(generated, dim=-1)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(230923)
    ref_manager = make_manager()
    ref_model = make_model(ref_manager)
    fast_manager = make_manager()
    fast_model = make_model(fast_manager)
    with torch.no_grad():
        fast_model.base_model.embedding.weight.copy_(
            ref_model.base_model.embedding.weight
        )
        fast_model.base_model.output_head.weight.copy_(
            ref_model.base_model.output_head.weight
        )
        fast_model.base_model.output_head.bias.copy_(
            ref_model.base_model.output_head.bias
        )

    # Warm the legacy H-vector caches before capturing its reference outputs.
    ref_ids = torch.tensor([[1, VOCAB, 7, VOCAB + 1]])
    ref_model.base_model.embedding(ref_ids)
    hidden = torch.randn(1, 4, HIDDEN)
    legacy_logits = ref_model.base_model.output_head(hidden)

    fast_memory = fast_manager.prepare_inference_tables(fast_model)
    setup_encoder_calls = (
        fast_manager.input_encoder_calls,
        fast_manager.output_encoder_calls,
    )
    fast_embedding = fast_model.base_model.embedding
    fast_head = fast_model.base_model.output_head
    legacy_embedding = ref_model.base_model.embedding
    legacy_head = ref_model.base_model.output_head

    cases = {
        "base_only": torch.tensor([[1, 2, 3, 4]]),
        "H2": torch.tensor([[VOCAB]]),
        "H3": torch.tensor([[VOCAB + 1]]),
        "mixed_prompt": ref_ids,
        "multiple_H": torch.tensor([[VOCAB, VOCAB + 1, VOCAB]]),
        "eos_base_token": torch.tensor([[EOS_ID]]),
    }
    checks = {}
    for name, ids in cases.items():
        old_emb = legacy_embedding(ids)
        new_emb = fast_embedding(ids)
        torch.testing.assert_close(new_emb, old_emb, atol=1e-6, rtol=1e-6)
        checks[name] = "pass"
    assert (
        fast_manager.input_encoder_calls,
        fast_manager.output_encoder_calls,
    ) == setup_encoder_calls
    decode_encoder_calls = (
        fast_manager.input_encoder_calls - setup_encoder_calls[0],
        fast_manager.output_encoder_calls - setup_encoder_calls[1],
    )
    checks["decode_did_not_rerun_encoders"] = "pass"

    tail_ids = torch.tensor([[VOCAB + CODEBOOK_SIZE, VOCAB + CODEBOOK_SIZE + 2]])
    expected_tail_emb = legacy_embedding.weight[
        torch.tensor([[VOCAB, VOCAB + 2]])
    ]
    torch.testing.assert_close(fast_embedding(tail_ids), expected_tail_emb)
    checks["shifted_original_tail_embedding"] = "pass"

    fast_logits = fast_head(hidden)
    torch.testing.assert_close(fast_logits, legacy_logits, atol=1e-5, rtol=1e-6)
    assert torch.equal(fast_logits.argmax(dim=-1), legacy_logits.argmax(dim=-1))
    assert torch.equal(
        torch.topk(fast_logits, 5, dim=-1).indices,
        torch.topk(legacy_logits, 5, dim=-1).indices,
    )
    torch.testing.assert_close(
        fast_logits[..., VOCAB + CODEBOOK_SIZE :],
        F.linear(hidden, legacy_head.weight, legacy_head.bias)[..., VOCAB:],
    )
    checks["output_logits_top1_top5_and_tail"] = "pass"
    legacy_generated = greedy_fixture(
        legacy_embedding, legacy_head, ref_manager, torch.tensor([[1, 2]]), 5
    )
    fast_generated = greedy_fixture(
        fast_embedding, fast_head, fast_manager, torch.tensor([[1, 2]]), 5
    )
    assert torch.equal(fast_generated, legacy_generated)
    checks["autoregressive_greedy_ids"] = "pass"

    expected_expansion = [2, 3, 4, 5, 6, VOCAB, VOCAB + 2]
    assert fast_manager.decode_sequence(
        [VOCAB, VOCAB + 1, VOCAB + CODEBOOK_SIZE, VOCAB + CODEBOOK_SIZE + 2]
    ) == expected_expansion
    assert fast_manager.segment_sequence([2, 3, 4, 5, 6]) == [VOCAB, VOCAB + 1]
    prepared_tail = fast_manager.prepare_input_sequence(
        [VOCAB + 1, VOCAB + 3], compress=False
    )
    assert prepared_tail == [VOCAB + CODEBOOK_SIZE + 1, VOCAB + CODEBOOK_SIZE + 3]
    tail_embedding = fast_embedding(torch.tensor([prepared_tail]))
    expected_tail_embedding = torch.stack(
        [legacy_embedding.weight[VOCAB + 1], legacy_embedding.weight[VOCAB + 3]]
    ).unsqueeze(0)
    assert torch.equal(tail_embedding, expected_tail_embedding)
    assert fast_manager.decode_sequence(prepared_tail) == [VOCAB + 1, VOCAB + 3]
    checks["prompt_segmentation_and_expansion"] = "pass"
    checks["prompt_tail_ids_shift_and_roundtrip"] = "pass"

    ref_manager.reset()
    fast_manager.reset()
    position_ids = torch.tensor([[1, VOCAB, 2, VOCAB + 1]])
    old_positions = ref_manager.prepare_input_ids(position_ids)
    new_positions = fast_manager.prepare_input_ids(position_ids)
    assert torch.equal(new_positions, old_positions)
    assert new_positions.tolist() == [[0, 2, 3, 6]]
    checks["H2_H3_semantic_positions"] = "pass"

    profile_ids = torch.tensor([[1, VOCAB, 2, VOCAB + 1]])
    op_profile = {
        "embedding_legacy": op_counts(lambda: legacy_embedding(ref_ids)),
        "embedding_fast": op_counts(lambda: fast_embedding(ref_ids)),
        "output_legacy": op_counts(lambda: legacy_head(hidden)),
        "output_fast": op_counts(lambda: fast_head(hidden)),
        "position_legacy": op_counts(
            lambda: (setattr(ref_manager, "base_position_offset", None),
                     ref_manager.prepare_input_ids(profile_ids))
        ),
        "position_fast": op_counts(
            lambda: (setattr(fast_manager, "base_position_offset", None),
                     fast_manager.prepare_input_ids(profile_ids))
        ),
    }
    assert op_profile["position_fast"].get("aten::any", 0) == 0
    checks["fast_position_has_no_aten_any"] = "pass"

    # Profile-sized synthetic tensors stay modest and all work is on CPU.
    timing_ids = torch.tensor([[1, VOCAB, 2, VOCAB + 1]])
    timings = {
        "position_legacy_ms": median_ms(
            lambda: (setattr(ref_manager, "base_position_offset", None),
                     ref_manager.prepare_input_ids(timing_ids))
        ),
        "position_fast_ms": median_ms(
            lambda: (setattr(fast_manager, "base_position_offset", None),
                     fast_manager.prepare_input_ids(timing_ids))
        ),
        "embedding_legacy_ms": median_ms(lambda: legacy_embedding(ref_ids)),
        "embedding_fast_ms": median_ms(lambda: fast_embedding(ref_ids)),
        "output_legacy_ms": median_ms(lambda: legacy_head(hidden)),
        "output_fast_ms": median_ms(lambda: fast_head(hidden)),
    }

    # A new request codebook must build new effective rows, not reuse stale ones.
    old_h0 = fast_manager.effective_embedding_weight_cache[VOCAB].clone()
    old_output_h0 = fast_manager.effective_linear_weight_cache[VOCAB].clone()
    fast_manager.set_seeded_codebook([[7, 8], [9, 10, 11]])
    assert not fast_manager.fast_inference_ready
    fast_manager.prepare_inference_tables(fast_model)
    new_h0 = fast_manager.effective_embedding_weight_cache[VOCAB]
    new_output_h0 = fast_manager.effective_linear_weight_cache[VOCAB]
    assert not torch.equal(old_h0, new_h0)
    assert not torch.equal(old_output_h0, new_output_h0)
    checks["reset_and_second_codebook"] = "pass"

    report = {
        "device": "cpu",
        "timing_interpretation": "directional CPU result; not evidence of GPU speedup",
        "no_gpu_work_performed": True,
        "dimensions": {
            "base_vocab_rows": VOCAB + TAIL,
            "initial_vocab_size": VOCAB,
            "codebook_size": CODEBOOK_SIZE,
            "hidden_size": HIDDEN,
        },
        "checks": checks,
        "greedy_generated_ids": fast_generated[0].tolist(),
        "encoder_calls_after_first_setup": {
            "input": setup_encoder_calls[0],
            "output": setup_encoder_calls[1],
            "additional_decode_calls": sum(decode_encoder_calls),
        },
        "memory_bytes": fast_memory,
        "operator_counts": op_profile,
        "median_timing_ms": timings,
        "notes": [
            "Synthetic deterministic encoder/model only; no live Phi generation was run.",
            "The legacy output path preserves tail logits but does not embed shifted tail IDs.",
            "The fast effective table supports shifted tail IDs and reverses that shift on decode.",
            "CPU timing and operator counts do not predict T4 latency.",
        ],
    }
    output_dir = ROOT / "experiments" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fast_inference_cpu_profile.json"
    md_path = output_dir / "fast_inference_cpu_profile.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    print(json.dumps({"checks": checks, "median_timing_ms": timings}, indent=2))


def render_markdown(report):
    ops = report["operator_counts"]
    times = report["median_timing_ms"]
    memory = report["memory_bytes"]
    lines = [
        "# Predictive Fast Inference CPU Validation",
        "",
        "> Directional CPU result; not evidence of GPU speedup.",
        "> No live Phi generation or GPU work was performed.",
        "",
        "## Correctness",
        "",
        "All synthetic reference checks passed: "
        + ", ".join(report["checks"].keys())
        + ".",
        "",
        "The fast table matches legacy embedding/logits on the common base/H ID "
        "space, preserves top-1 and top-5, handles shifted output-tail IDs, and "
        "restores tail IDs during output expansion.",
        "",
        "## CPU operator counts",
        "",
        "| Path | embedding | arange | mul | add | where | any | linear | bmm | cat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [
        ("Embedding legacy", ops["embedding_legacy"]),
        ("Embedding fast", ops["embedding_fast"]),
        ("Output legacy", ops["output_legacy"]),
        ("Output fast", ops["output_fast"]),
        ("Position legacy", ops["position_legacy"]),
        ("Position fast", ops["position_fast"]),
    ]
    for label, counts in rows:
        cols = [
            counts.get("aten::embedding", 0),
            counts.get("aten::arange", 0),
            counts.get("aten::mul", 0),
            counts.get("aten::add", 0),
            counts.get("aten::where", 0),
            counts.get("aten::any", 0),
            counts.get("aten::linear", 0),
            counts.get("aten::bmm", 0),
            counts.get("aten::cat", 0),
        ]
        lines.append("| " + label + " | " + " | ".join(map(str, cols)) + " |")
    lines.extend(
        [
            "",
            "## Directional CPU median time (ms/op)",
            "",
            "| Component | Legacy | Fast |",
            "|---|---:|---:|",
            f"| Position | {times['position_legacy_ms']:.4f} | {times['position_fast_ms']:.4f} |",
            f"| Embedding | {times['embedding_legacy_ms']:.4f} | {times['embedding_fast_ms']:.4f} |",
            f"| Output projection | {times['output_legacy_ms']:.4f} | {times['output_fast_ms']:.4f} |",
            "",
            "## Effective-table memory (bytes)",
            "",
            f"- Base input table: {memory['base_input_embedding_bytes']:,}",
            f"- Effective input table: {memory['effective_input_embedding_bytes']:,}",
            f"- Base output head: {memory['base_output_head_bytes']:,}",
            f"- Effective output head: {memory['effective_output_head_bytes']:,}",
            f"- Additional CPU RAM held by tables: {memory['additional_cpu_ram_bytes']:,}",
            "",
            "Synthetic CPU profiling does not establish any GPU or T4 speedup.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
