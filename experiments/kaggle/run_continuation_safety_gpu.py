"""Run small CUDA continuation-safety probes selected from validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from experiments import run_quality_benchmark as benchmark
from src.zip2zip import StaticCodebookManager


INITIAL_VOCAB = benchmark.INITIAL_VOCAB
SAFE_THRESHOLDS = {
    "max_kl": 0.10,
    "min_top1_agreement": True,
    "min_top5_overlap": 0.60,
    "min_hidden_cosine": 0.98,
    "max_abs_eos_probability_delta": 0.01,
    "require_greedy_continuation_match": True,
}


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, data: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _expand(token_ids: list[int], manager: StaticCodebookManager) -> list[int]:
    expanded: list[int] = []
    for token_id in token_ids:
        if token_id in manager.hyper_to_subtokens:
            expanded.extend(manager.hyper_to_subtokens[token_id])
        else:
            expanded.append(token_id)
    return expanded


def _tensor_distribution(logits: torch.Tensor, eos_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_logits = logits[0, -1, :INITIAL_VOCAB].float()
    log_prob = F.log_softmax(base_logits, dim=-1)
    prob = log_prob.exp()
    topk = torch.topk(prob, 10).indices
    return log_prob, prob, topk, prob[eos_id]


def _run_one(model: Any, tokenizer: Any, probe: dict[str, Any], device: torch.device) -> dict[str, Any]:
    phrase_ids = [int(token) for token in probe["phrase_tokens"]]
    context_ids = [int(token) for token in probe["context_token_ids"]]
    codebook = {tuple(phrase_ids): INITIAL_VOCAB}
    manager = StaticCodebookManager(
        initial_vocab_size=INITIAL_VOCAB,
        max_codebook_size=1,
        max_subtokens=4,
        embedding_dim=model.zip2zip_config.encoder.hidden_size,
        pad_token_id=tokenizer.pad_token_id or 32000,
        disabled_ids=list(model.zip2zip_config.compression.disabled_ids),
    )
    manager.set_seeded_codebook(codebook, batch_size=1, device=device)
    manager.attach_to_model(model)
    try:
        base_ids = context_ids + phrase_ids
        hyper_ids = context_ids + [INITIAL_VOCAB]
        input_a = torch.tensor([base_ids], dtype=torch.long, device=device)
        input_b = torch.tensor([hyper_ids], dtype=torch.long, device=device)
        position_a = torch.arange(len(base_ids), dtype=torch.long, device=device).unsqueeze(0)
        semantic_position_b = (
            list(range(len(context_ids))) + [len(context_ids) + len(phrase_ids) - 1]
        )
        position_b = torch.tensor([semantic_position_b], dtype=torch.long, device=device)

        with torch.inference_mode():
            out_a = model.base_model(
                input_a,
                position_ids=position_a,
                output_hidden_states=True,
                use_cache=False,
            )
            log_prob_a, prob_a, topk_a, eos_prob_a = _tensor_distribution(
                out_a.logits, tokenizer.eos_token_id
            )
            hidden_a = out_a.hidden_states[-1][0, -1].float()

            out_b = model.base_model(
                input_b,
                position_ids=position_b,
                output_hidden_states=True,
                use_cache=False,
            )
            log_prob_b, prob_b, topk_b, eos_prob_b = _tensor_distribution(
                out_b.logits, tokenizer.eos_token_id
            )
            hidden_b = out_b.hidden_states[-1][0, -1].float()

            continuation_a = model.generate(
                input_ids=input_a,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0, len(base_ids) :].tolist()
            continuation_b_raw = model.generate(
                input_ids=input_b,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0, len(hyper_ids) :].tolist()

        continuation_a_expanded = _expand(continuation_a, manager)
        continuation_b_expanded = _expand(continuation_b_raw, manager)
        top1_a = int(prob_a.argmax().item())
        top1_b = int(prob_b.argmax().item())
        kl = F.kl_div(log_prob_b, log_prob_a, log_target=True, reduction="sum").item()
        top5_a = set(int(value) for value in topk_a[:5].tolist())
        top5_b = set(int(value) for value in topk_b[:5].tolist())
        top10_a = set(int(value) for value in topk_a.tolist())
        top10_b = set(int(value) for value in topk_b.tolist())
        cosine = F.cosine_similarity(hidden_a.unsqueeze(0), hidden_b.unsqueeze(0)).item()
        eos_delta = float(eos_prob_b.item() - eos_prob_a.item())
        continuation_match = continuation_a_expanded == continuation_b_expanded
        safe = (
            kl <= SAFE_THRESHOLDS["max_kl"]
            and top1_a == top1_b
            and len(top5_a & top5_b) / 5.0 >= SAFE_THRESHOLDS["min_top5_overlap"]
            and cosine >= SAFE_THRESHOLDS["min_hidden_cosine"]
            and abs(eos_delta) <= SAFE_THRESHOLDS["max_abs_eos_probability_delta"]
            and continuation_match
        )
        return {
            "prompt_id": probe["prompt_id"],
            "domain": probe["domain"],
            "risk_stratum": probe["risk_stratum"],
            "phrase": probe["phrase_text"],
            "phrase_tokens": phrase_ids,
            "phrase_length": len(phrase_ids),
            "in_predictor_top_k": bool(probe["in_predictor_top_k"]),
            "in_quality_aware_candidates": bool(probe["in_quality_aware_candidates"]),
            "reference_occurrences": int(probe["reference_occurrences"]),
            "reference_steps_saved_if_selected": int(probe["reference_steps_saved_if_selected"]),
            "reference_start_token": int(probe["reference_start_token"]),
            "context_token_count": len(context_ids),
            "context_sha256": hashlib.sha256(bytes(str(context_ids), "utf-8")).hexdigest(),
            "context_source": "validation_reference_prefix_retokenized; suffix capped at 256 tokens",
            "kl_base_to_hyper": float(kl),
            "top1_agreement": top1_a == top1_b,
            "top1_base_token_id": top1_a,
            "top1_hyper_token_id": top1_b,
            "top5_overlap": len(top5_a & top5_b) / 5.0,
            "top10_overlap": len(top10_a & top10_b) / 10.0,
            "hidden_cosine": float(cosine),
            "base_eos_probability": float(eos_prob_a.item()),
            "hyper_eos_probability": float(eos_prob_b.item()),
            "eos_probability_delta_hyper_minus_base": eos_delta,
            "continuation_base_token_ids": continuation_a_expanded,
            "continuation_hyper_token_ids": continuation_b_expanded,
            "continuation_base_text": tokenizer.decode(continuation_a_expanded, skip_special_tokens=True),
            "continuation_hyper_text": tokenizer.decode(continuation_b_expanded, skip_special_tokens=True),
            "greedy_continuation_match": continuation_match,
            "screen_safe": safe,
        }
    finally:
        manager.detach_from_model(model)
        model.codebook_manager.reset()


def _summarize(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    output = {}
    metric_names = (
        "kl_base_to_hyper",
        "top5_overlap",
        "top10_overlap",
        "hidden_cosine",
        "eos_probability_delta_hyper_minus_base",
    )
    for label, rows in sorted(groups.items()):
        output[label] = {
            "count": len(rows),
            "screen_safe_count": sum(bool(row["screen_safe"]) for row in rows),
            "top1_agreement_rate": statistics.mean(float(row["top1_agreement"]) for row in rows),
            "greedy_continuation_match_rate": statistics.mean(
                float(row["greedy_continuation_match"]) for row in rows
            ),
        }
        for metric in metric_names:
            output[label][f"mean_{metric}"] = statistics.mean(float(row[metric]) for row in rows)
    return output


def _update_funnel(funnel_path: Path, records: list[dict[str, Any]]) -> None:
    funnel = _read(funnel_path)
    intervals_by_prompt: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for record in records:
        if record["screen_safe"] and record["in_predictor_top_k"]:
            start = int(record["reference_start_token"])
            end = start + int(record["phrase_length"])
            saved = int(record["phrase_length"]) - 1
            intervals_by_prompt[record["prompt_id"]].append((start, end, saved))
    observed_saved = 0
    observed_phrase_count = 0
    for intervals in intervals_by_prompt.values():
        last_end = -1
        for start, end, saved in sorted(intervals, key=lambda item: (item[1], item[0], -item[2])):
            if start >= last_end:
                observed_saved += saved
                observed_phrase_count += 1
                last_end = end
    oracle_saved = int(funnel["stages"]["quality_aware_oracle"]["tokens_saved"])
    funnel["continuation_safety_capture"] = {
        "status": "measured_context_specific_sample_lower_bound",
        "screen_safe_definition": SAFE_THRESHOLDS,
        "probe_count": len(records),
        "screen_safe_probe_count": sum(bool(row["screen_safe"]) for row in records),
        "predictor_top_k_screen_safe_probe_count": sum(
            bool(row["screen_safe"] and row["in_predictor_top_k"]) for row in records
        ),
        "non_overlapping_probed_top_k_reference_occurrences": observed_phrase_count,
        "safe_reference_saved_tokens": observed_saved,
        "safe_capture_pct_of_12_prompt_quality_oracle": round(
            100.0 * observed_saved / max(oracle_saved, 1), 4
        ),
        "interpretation": "Only the exact sampled validation reference occurrence is credited. This is a lower-bound probe count, not a projected full-run continuation-safe compression estimate.",
    }
    _write(funnel_path, funnel)
    markdown_path = funnel_path.with_suffix(".md")
    existing = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else ""
    addendum = [
        "",
        "## Empirical continuation safety (single-T4 validation probe)",
        "",
        f"Probes: {len(records)}; strict screen-safe: {funnel['continuation_safety_capture']['screen_safe_probe_count']}; among predictor Top-K: {funnel['continuation_safety_capture']['predictor_top_k_screen_safe_probe_count']}.",
        f"Only non-overlapping exact sampled reference occurrences passed by the screen are credited: {observed_saved} saved reference steps across {observed_phrase_count} occurrences ({funnel['continuation_safety_capture']['safe_capture_pct_of_12_prompt_quality_oracle']}% of the recomputed 12-prompt quality-aware oracle opportunity).",
        "This is a context-specific sampled lower bound, not a projected full-run estimate.",
        "",
    ]
    marker = "## Empirical continuation safety (single-T4 validation probe)"
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip() + "\n"
    markdown_path.write_text(existing.rstrip() + "\n" + "\n".join(addendum).lstrip("\n"), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Continuation probe requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_name(device) != args.expected_gpu_name and args.expected_gpu_name:
        raise RuntimeError(
            f"Unexpected GPU: expected {args.expected_gpu_name!r}, got {torch.cuda.get_device_name(device)!r}"
        )
    selection = _read(args.probe_selection)
    if selection.get("split") != "validation_only":
        raise RuntimeError("Refusing continuation probes without an explicit validation-only selection")

    bundle = benchmark.load_predictive_model_bundle(
        args.checkpoint,
        device=str(device),
        base_revision=benchmark.DEFAULT_PHI_REVISION,
        model_revision=benchmark.DEFAULT_ZIP2ZIP_REVISION,
        expected_step=100,
    )
    report = bundle["checkpoint_load_report"]
    if report.get("base_hash_status") != "verified":
        raise RuntimeError(f"Checkpoint frozen-backbone hashes were not verified: {report.get('base_hash_status')}")
    for component in ("lora", "input_encoder", "output_encoder"):
        if int(report["components"][component]["changed_tensor_count"]) <= 0:
            raise RuntimeError(f"Checkpoint component was not activated: {component}")
    predictor_sha = benchmark.file_sha256(args.predictor)
    if predictor_sha != args.expected_predictor_sha256:
        raise RuntimeError("Predictor artifact hash mismatch before continuation probes")

    model, tokenizer = bundle["model"], bundle["tokenizer"]
    records = []
    try:
        for probe in selection["probes"]:
            if not 2 <= len(probe["phrase_tokens"]) <= 4:
                raise RuntimeError(f"Invalid probe phrase length: {probe['phrase_text']!r}")
            records.append(_run_one(model, tokenizer, probe, device))
    finally:
        del model
        del tokenizer
        bundle.clear()
        torch.cuda.empty_cache()

    summary = {
        "schema": "continuation_safety_gpu_v1",
        "split": "validation_only",
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(device),
        "tested_commit": args.tested_commit,
        "checkpoint_sha256": benchmark.file_sha256(args.checkpoint),
        "predictor_sha256": predictor_sha,
        "checkpoint_load_report": report,
        "probe_method": {
            "base_path": "context tokens + phrase constituent base tokens; distribution measured after final constituent",
            "hypertoken_path": "same decoded context + one seeded hypertoken; position set to the semantic final-constituent position",
            "continuation": "three-token deterministic Zip2Zip greedy rollout on both paths, expanded to base-token IDs before comparison",
            "context_source": "prompt plus teacher-forced validation reference prefix, retokenized and capped to the preceding 256 token IDs; not an exact captured free-generation cache state",
            "safe_thresholds": SAFE_THRESHOLDS,
            "limitations": "small stratified sample; probe safety is context-specific and does not establish causal or full-trajectory safety",
        },
        "probe_count": len(records),
        "summary": {
            "mean_kl_base_to_hyper": statistics.mean(row["kl_base_to_hyper"] for row in records) if records else None,
            "top1_agreement_rate": statistics.mean(float(row["top1_agreement"]) for row in records) if records else None,
            "mean_top5_overlap": statistics.mean(row["top5_overlap"] for row in records) if records else None,
            "mean_top10_overlap": statistics.mean(row["top10_overlap"] for row in records) if records else None,
            "mean_hidden_cosine": statistics.mean(row["hidden_cosine"] for row in records) if records else None,
            "mean_eos_probability_delta": statistics.mean(row["eos_probability_delta_hyper_minus_base"] for row in records) if records else None,
            "mean_absolute_eos_probability_delta": statistics.mean(abs(row["eos_probability_delta_hyper_minus_base"]) for row in records) if records else None,
            "greedy_continuation_match_rate": statistics.mean(float(row["greedy_continuation_match"]) for row in records) if records else None,
            "screen_safe_count": sum(bool(row["screen_safe"]) for row in records),
        },
        "by_domain": _summarize(records, "domain"),
        "by_risk_stratum": _summarize(records, "risk_stratum"),
        "probes": records,
    }
    output_path = Path(args.output)
    _write(output_path, summary)
    md_path = output_path.with_suffix(".md")
    lines = [
        "# Empirical hypertoken continuation safety (GPU)",
        "",
        f"Device: {summary['gpu_name']} (`cuda:0`); validation probes: {len(records)}.",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Mean KL(base || H) | {summary['summary']['mean_kl_base_to_hyper']:.6f} |",
        f"| Top-1 agreement | {summary['summary']['top1_agreement_rate']:.1%} |",
        f"| Mean top-5 overlap | {summary['summary']['mean_top5_overlap']:.1%} |",
        f"| Mean top-10 overlap | {summary['summary']['mean_top10_overlap']:.1%} |",
        f"| Mean hidden cosine | {summary['summary']['mean_hidden_cosine']:.5f} |",
        f"| Mean EOS probability delta (H − base) | {summary['summary']['mean_eos_probability_delta']:+.6f} |",
        f"| Three-token greedy continuation match | {summary['summary']['greedy_continuation_match_rate']:.1%} |",
        f"| Strict screen-safe probes | {summary['summary']['screen_safe_count']}/{len(records)} |",
        "",
        "## Results by phrase risk stratum",
        "",
        "| Stratum | N | Safe | Mean KL | Top-1 | Top-5 overlap | Hidden cosine | Mean EOS delta | Continuation match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in summary["by_risk_stratum"].items():
        lines.append(
            f"| {label} | {row['count']} | {row['screen_safe_count']} | {row['mean_kl_base_to_hyper']:.5f} | {row['top1_agreement_rate']:.1%} | {row['mean_top5_overlap']:.1%} | {row['mean_hidden_cosine']:.5f} | {row['mean_eos_probability_delta_hyper_minus_base']:+.5f} | {row['greedy_continuation_match_rate']:.1%} |"
        )
    lines.extend([
        "",
        "The safety screen is deliberately strict: KL ≤ 0.10, top-1 agreement, top-5 overlap ≥ 0.60, hidden cosine ≥ 0.98, absolute EOS probability delta ≤ 0.01, and exact three-token greedy continuation match. These thresholds are an exploratory screen, not a calibrated quality guarantee.",
        "",
        "Contexts come from prompt plus validation-reference prefixes, retokenized and truncated to 256 tokens. This probes local context sensitivity but does not reproduce exact free-running generation cache states. See JSON for per-probe provenance and exact results.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _update_funnel(Path(args.funnel_json), records)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictor", default="experiments/checkpoints/oracle_guided_predictor.pkl")
    parser.add_argument("--expected-predictor-sha256", required=True)
    parser.add_argument("--probe-selection", required=True)
    parser.add_argument("--funnel-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tested-commit", required=True)
    parser.add_argument("--expected-gpu-name", default="")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
