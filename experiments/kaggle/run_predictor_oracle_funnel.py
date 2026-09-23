"""Measure a validation-only predictor/oracle funnel on the fixed Tier-1 prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from experiments import run_quality_benchmark as benchmark
from experiments.build_quality_aware_oracle import (
    CODE_SYNTAX_FRAGMENTS,
    GRAMMATICAL_GLUE,
    compute_safety_prior,
    extract_all_candidate_phrases,
)
from experiments.load_oracle_predictor import load_oracle_predictor
from src.evaluation.offline_segmenter import segment_tokens_dp
from src.zip2zip.predictor_policy import CappedPredictorPolicy, classify_phrase, is_bare_punctuation


def _json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _candidate_pool(policy: CappedPredictorPolicy, prompt_ids: list[int]) -> dict[tuple[int, ...], float]:
    scores: dict[tuple[int, ...], float] = defaultdict(float)
    for phrase, count in policy.extract_prompt_ngrams(prompt_ids).items():
        scores[phrase] += count * (len(phrase) - 1) * 8.0
    prompt_tokens = set(prompt_ids) - policy.disabled_ids
    for prompt_token in prompt_tokens:
        for phrase, weight in getattr(policy.index, "token_associations", {}).get(prompt_token, []):
            scores[tuple(phrase)] += float(weight)
    return {
        phrase: score
        for phrase, score in scores.items()
        if not policy.filter_bare_punctuation or not is_bare_punctuation(phrase, policy.tokenizer)
    }


def _quality_aware_codebook(
    prompt_text: str,
    prompt_ids: list[int],
    response_ids: list[int],
    domain: str,
    tokenizer: Any,
) -> tuple[set[tuple[int, ...]], dict[str, Any]]:
    counts = extract_all_candidate_phrases(response_ids, min_len=2, max_len=4)
    ranked: list[tuple[float, tuple[int, ...]]] = []
    safety_by_phrase: dict[tuple[int, ...], dict[str, Any]] = {}
    for phrase, count in counts.items():
        phrase_text = tokenizer.decode(list(phrase))
        safety = compute_safety_prior(phrase_text, list(phrase), prompt_text, prompt_ids, domain)
        safety_by_phrase[phrase] = safety
        if safety["safety_prior"] >= 0.50 and not safety["has_trailing_space"]:
            ranked.append(((len(phrase) - 1) * count * safety["safety_prior"], phrase))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    candidates = [phrase for _, phrase in ranked[:40]]

    selected: set[tuple[int, ...]] = set()
    current_saved = 0
    available = set(candidates)
    for _ in range(32):
        best_phrase = None
        best_gain = 0
        for phrase in sorted(available):
            candidate_saved = segment_tokens_dp(response_ids, selected | {phrase})[2]["tokens_saved"]
            gain = candidate_saved - current_saved
            if gain > best_gain:
                best_gain = gain
                best_phrase = phrase
        if best_phrase is None:
            break
        selected.add(best_phrase)
        available.remove(best_phrase)
        current_saved += best_gain

    _, tiles, dp_stats = segment_tokens_dp(response_ids, selected)
    used_phrases = {tile for tile in tiles if len(tile) > 1}
    return selected, {
        "candidate_count": len(candidates),
        "codebook_size": len(selected),
        "used_phrase_count": len(used_phrases),
        "tokens_saved": dp_stats["tokens_saved"],
        "safety_by_phrase": safety_by_phrase,
    }


def _phrase_features(
    phrase: tuple[int, ...],
    tokenizer: Any,
    prompt_text: str,
    prompt_ids: list[int],
    domain: str,
) -> dict[str, Any]:
    text = tokenizer.decode(list(phrase))
    safety = compute_safety_prior(text, list(phrase), prompt_text, prompt_ids, domain)
    words = re.findall(r"\b\w+\b", text.lower())
    prompt_words = set(re.findall(r"\b\w+\b", prompt_text.lower()))
    if safety["is_exact_in_prompt"]:
        grounding = "exact"
    elif words and any(word in prompt_words for word in words):
        grounding = "partial"
    else:
        grounding = "none"
    digits = bool(re.findall(r"\d+", text))
    prompt_digits = set(re.findall(r"\d+", prompt_text))
    numeric_grounding = all(digit in prompt_digits for digit in re.findall(r"\d+", text))
    identifier_match = re.search(r"\b(?:def|class)\s+([A-Za-z_]\w*)|\b([A-Za-z_]\w*)\s*\(", text)
    identifier = bool(identifier_match)
    category = classify_phrase(phrase, tokenizer)
    stripped = text.strip()
    structural_syntax = category == "structural" or stripped in CODE_SYNTAX_FRAGMENTS
    grammatical_glue = text in GRAMMATICAL_GLUE or (" " + stripped) in GRAMMATICAL_GLUE
    whitespace_boundary = (
        text[:1].isspace()
        or text[-1:].isspace()
        or any(character in text for character in "\n\r\t")
    )
    if digits:
        risk_stratum = "grounded_numeric" if numeric_grounding else "ungrounded_numeric"
    elif identifier and domain == "code":
        risk_stratum = "identifier_function"
    elif structural_syntax:
        risk_stratum = "structural_syntax"
    elif grammatical_glue:
        risk_stratum = "grammatical_glue"
    elif grounding == "exact":
        risk_stratum = "grounded_exact"
    elif grounding == "partial":
        risk_stratum = "grounded_partial"
    else:
        risk_stratum = "ungrounded_content"
    return {
        "phrase_text": text,
        "phrase_tokens": list(phrase),
        "phrase_length": len(phrase),
        "predictor_category": category,
        "prompt_grounding": grounding,
        "has_digits": digits,
        "numeric_grounding": numeric_grounding if digits else None,
        "identifier_or_function": identifier,
        "structural_syntax": structural_syntax,
        "grammatical_glue": grammatical_glue,
        "whitespace_boundary": whitespace_boundary,
        "has_trailing_space": safety["has_trailing_space"],
        "safety_prior": safety["safety_prior"],
        "safety_prior_label": safety["is_safe"],
        "safety_prior_reasons": safety["reasons"],
        "risk_stratum": risk_stratum,
    }


def _first_subsequence(sequence: list[int], phrase: tuple[int, ...]) -> int | None:
    width = len(phrase)
    for index in range(len(sequence) - width + 1):
        if tuple(sequence[index : index + width]) == phrase:
            return index
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    with Path(args.validation_data).open("r", encoding="utf-8") as source:
        all_samples = json.load(source)
    samples = benchmark.select_prompt_subset(
        all_samples,
        args.prompt_ids_file,
        {"code": 4, "reasoning": 4, "instruction": 4},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        benchmark.PHI_MODEL_ID, revision=benchmark.DEFAULT_PHI_REVISION
    )
    raw_predictor = load_oracle_predictor(args.predictor)
    predictor_index = getattr(raw_predictor, "index", raw_predictor)
    policy = CappedPredictorPolicy(predictor_index, tokenizer)

    live_by_id: dict[str, dict[str, Any]] = {}
    with Path(args.live_results).open("r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("condition") == "predictive_step_100_compressed_prompt":
                live_by_id[record["prompt_id"]] = record
    with Path(args.phase5_manifest).open("r", encoding="utf-8") as source:
        phase5_manifest = json.load(source)
    phase5_config = (
        phase5_manifest.get("identity", {})
        .get("conditions", {})
        .get("predictive_step_100_compressed_prompt", {})
    )
    if (
        phase5_config.get("prompt_representation") != "predictive_codebook_dp_segmented"
        or phase5_config.get("emission_gate_top_n") is not None
    ):
        raise RuntimeError("Phase-5 reference is not the committed canonical compressed-prompt no-gate run")

    stages: dict[str, dict[str, Any]] = {
        stage: {"tokens_saved": 0, "base_tokens": 0, "phrase_count": 0, "slots": 0, "domains": {}}
        for stage in ("quality_aware_oracle", "predictor_candidate_pool", "predictor_top_k", "top_k_occurs_in_reference", "live_emission_cpu_phase5")
    }
    domain_totals: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    probes_by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    prompt_details = []
    total_reference_tokens = 0

    for sample in samples:
        prompt_id = sample["id"]
        domain = sample["domain"]
        prompt_text = benchmark._prompt_text(sample)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        response_text = str(sample.get("ground_truth_response", sample.get("response", "")))
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        total_reference_tokens += len(response_ids)

        quality_codebook, quality_stats = _quality_aware_codebook(
            prompt_text, prompt_ids, response_ids, domain, tokenizer
        )
        candidate_pool = _candidate_pool(policy, prompt_ids)
        codebook, _ = policy.select_codebook(prompt_ids)
        expected_codebook_sha = hashlib.sha256(
            repr(sorted(codebook.items())).encode("utf-8")
        ).hexdigest()
        topk_phrases = set(codebook)
        occurring_topk = {
            phrase for phrase in topk_phrases if _first_subsequence(response_ids, phrase) is not None
        }

        phrase_sets = {
            "quality_aware_oracle": quality_codebook,
            "predictor_candidate_pool": set(candidate_pool),
            "predictor_top_k": topk_phrases,
            "top_k_occurs_in_reference": occurring_topk,
        }
        per_prompt_stages = {}
        for stage, phrases in phrase_sets.items():
            _, tiles, stats = segment_tokens_dp(response_ids, phrases)
            used = {tile for tile in tiles if len(tile) > 1}
            per_prompt_stages[stage] = {
                "tokens_saved": stats["tokens_saved"],
                "base_tokens": len(response_ids),
                "phrase_count": len(used),
                "slots": len(phrases),
                "candidate_pool_size": len(phrases),
            }
            summary = stages[stage]
            summary["tokens_saved"] += stats["tokens_saved"]
            summary["base_tokens"] += len(response_ids)
            summary["phrase_count"] += len(used)
            summary["slots"] += len(phrases)
            domain_totals[stage][domain]["tokens_saved"] += stats["tokens_saved"]
            domain_totals[stage][domain]["base_tokens"] += len(response_ids)
            domain_totals[stage][domain]["phrase_count"] += len(used)
            domain_totals[stage][domain]["slots"] += len(phrases)

        qa_candidates = []
        for phrase, safety in quality_stats["safety_by_phrase"].items():
            if safety["safety_prior"] >= 0.50 and not safety["has_trailing_space"]:
                qa_candidates.append(phrase)
        possible_probe_phrases = set(qa_candidates) | topk_phrases
        for phrase in possible_probe_phrases:
            start = _first_subsequence(response_ids, phrase)
            if start is None:
                continue
            features = _phrase_features(phrase, tokenizer, prompt_text, prompt_ids, domain)
            features.update(
                {
                    "prompt_id": prompt_id,
                    "domain": domain,
                    "in_predictor_top_k": phrase in topk_phrases,
                    "in_quality_aware_candidates": phrase in set(qa_candidates),
                    "reference_occurrences": sum(
                        1
                        for index in range(len(response_ids) - len(phrase) + 1)
                        if tuple(response_ids[index : index + len(phrase)]) == phrase
                    ),
                    "reference_steps_saved_if_selected": max(0, (len(phrase) - 1))
                    * sum(
                        1
                        for index in range(len(response_ids) - len(phrase) + 1)
                        if tuple(response_ids[index : index + len(phrase)]) == phrase
                    ),
                    "reference_start_token": start,
                    "candidate_sources": sorted(
                        (["predictor_top_k"] if phrase in topk_phrases else [])
                        + (["quality_aware_candidate"] if phrase in set(qa_candidates) else [])
                    ),
                    "context_token_ids": (prompt_ids + response_ids[:start])[-256:],
                }
            )
            probes_by_stratum[(domain, features["risk_stratum"])].append(features)

        live_record = live_by_id.get(prompt_id)
        if live_record is None:
            raise RuntimeError(f"Phase-5 CPU no-gate output missing for validation prompt {prompt_id}")
        if live_record.get("codebook_sha256") != expected_codebook_sha:
            raise RuntimeError(
                f"Phase-5 live codebook for {prompt_id} does not match the current prompt-only predictor"
            )
        emitted = live_record.get("hypertokens_emitted", []) or []
        live_stage = {
            "tokens_saved": int(live_record.get("tokens_saved", 0)),
            "base_tokens": int(live_record.get("expanded_output_tokens", 0)),
            "phrase_count": len(emitted),
            "slots": int(live_record.get("codebook_size", 0)),
            "quality_pass": bool(
                live_record.get("problem_pass")
                or live_record.get("exact_correct")
                or live_record.get("mechanical_instruction_pass")
            ),
            "hypertokens": emitted,
            "run_id": "3de046a1b858dc7a",
            "record_schema": live_record.get("record_schema"),
            "generated_from_commit": live_record.get("generated_from_commit"),
        }
        stages["live_emission_cpu_phase5"]["tokens_saved"] += live_stage["tokens_saved"]
        stages["live_emission_cpu_phase5"]["base_tokens"] += live_stage["base_tokens"]
        stages["live_emission_cpu_phase5"]["phrase_count"] += live_stage["phrase_count"]
        stages["live_emission_cpu_phase5"]["slots"] += live_stage["slots"]
        domain_totals["live_emission_cpu_phase5"][domain]["tokens_saved"] += live_stage["tokens_saved"]
        domain_totals["live_emission_cpu_phase5"][domain]["base_tokens"] += live_stage["base_tokens"]
        domain_totals["live_emission_cpu_phase5"][domain]["phrase_count"] += live_stage["phrase_count"]
        domain_totals["live_emission_cpu_phase5"][domain]["slots"] += live_stage["slots"]

        prompt_details.append(
            {
                "prompt_id": prompt_id,
                "domain": domain,
                "reference_tokens": len(response_ids),
                "quality_aware_oracle_slots": len(quality_codebook),
                "quality_aware_oracle_candidate_count": quality_stats["candidate_count"],
                "predictor_candidate_pool_size": len(candidate_pool),
                "predictor_top_k_slots": len(codebook),
                "predictor_codebook_sha256": hashlib.sha256(
                    repr(sorted(codebook.items())).encode("utf-8")
                ).hexdigest(),
                "predictor_codebook": [
                    {"phrase": tokenizer.decode(list(phrase)), "tokens": list(phrase), "id": token_id}
                    for phrase, token_id in codebook.items()
                ],
                "stages": per_prompt_stages,
                "live_phase5_cpu": live_stage,
            }
        )

    for stage, by_domain in domain_totals.items():
        stages[stage]["domains"] = {}
        for domain, values in by_domain.items():
            stages[stage]["domains"][domain] = {
                **dict(values),
                "compression_pct": round(
                    100.0 * values["tokens_saved"] / max(values["base_tokens"], 1), 3
                ),
            }
    for stage in ("quality_aware_oracle", "predictor_candidate_pool", "predictor_top_k", "top_k_occurs_in_reference"):
        stages[stage]["compression_pct"] = round(
            100.0 * stages[stage]["tokens_saved"] / max(stages[stage]["base_tokens"], 1), 3
        )
        stages[stage]["cumulative_vs_quality_oracle_pct"] = round(
            100.0 * stages[stage]["tokens_saved"]
            / max(stages["quality_aware_oracle"]["tokens_saved"], 1),
            3,
        )
    stages["live_emission_cpu_phase5"]["compression_pct"] = round(
        100.0 * stages["live_emission_cpu_phase5"]["tokens_saved"]
        / max(stages["live_emission_cpu_phase5"]["base_tokens"], 1),
        3,
    )
    stages["live_emission_cpu_phase5"]["quality_pass_saved_tokens"] = sum(
        int(prompt["live_phase5_cpu"]["tokens_saved"])
        for prompt in prompt_details
        if prompt["live_phase5_cpu"]["quality_pass"]
    )
    stages["live_emission_cpu_phase5"]["quality_pass_saved_pct"] = round(
        100.0
        * stages["live_emission_cpu_phase5"]["quality_pass_saved_tokens"]
        / max(stages["live_emission_cpu_phase5"]["base_tokens"], 1),
        3,
    )

    selected_probes = []
    for stratum in sorted(probes_by_stratum):
        candidates = sorted(
            probes_by_stratum[stratum],
            key=lambda item: (
                not item["in_predictor_top_k"],
                -item["reference_steps_saved_if_selected"],
                tuple(item["phrase_tokens"]),
                item["prompt_id"],
            ),
        )
        if candidates:
            selected_probes.append(candidates[0])
    selected_probes = selected_probes[:30]

    funnel = {
        "schema": "predictor_oracle_funnel_v1",
        "split": "validation_only",
        "prompt_ids": [sample["id"] for sample in samples],
        "prompt_count": len(samples),
        "domain_counts": dict(Counter(sample["domain"] for sample in samples)),
            "prompt_formatter": benchmark.PROMPT_FORMATTER_VERSION,
        "tokenizer_id": benchmark.PHI_MODEL_ID,
        "tokenizer_revision": benchmark.DEFAULT_PHI_REVISION,
        "predictor_policy": {
            "budget": policy.budget,
            "max_structural_slots": policy.max_structural_slots,
            "allow_numeric": policy.allow_numeric,
            "filter_bare_punctuation": policy.filter_bare_punctuation,
        },
        "method": {
            "quality_aware_oracle": "same safety prior threshold >= 0.50, top 40 safety-weighted candidates, greedy exact-DP marginal selection up to 32 slots; deterministic phrase-tuple tie break",
            "predictor_pool": "all prompt n-grams plus token-association phrases before Top-K truncation, with bare punctuation filter",
            "offline_savings": "validation reference response re-segmented with exact dynamic programming; stages 1-4 share the reference-token denominator",
            "live_emission": "historical CPU Phase-5 no-gate free-generation records on the same 12 validation IDs; do not interpret as GPU measurements or as identical output-token denominators",
            "quality_gate": "domain evaluator pass fields; Alpaca pass is mechanical only",
            "continuation_probes": "at most one deterministic reference-context example per domain and risk stratum; selected phrase carries exact reference occurrence and 256-token context suffix",
        },
        "quality_aware_oracle_60_prompt_reference": {
            "micro_compression_pct": 35.05,
            "source": "experiments/checkpoints/quality_benchmark/quality_oracle_analysis.json",
            "note": "historical full validation result; not the denominator for this 12-prompt recalculation",
        },
        "total_reference_tokens": total_reference_tokens,
        "stages": stages,
        "prompt_details": prompt_details,
        "continuation_probe_count": len(selected_probes),
        "continuation_probe_strata": [
            {"domain": probe["domain"], "risk_stratum": probe["risk_stratum"], "prompt_id": probe["prompt_id"], "phrase": probe["phrase_text"]}
            for probe in selected_probes
        ],
        "continuation_safety_capture": {
            "status": "awaiting_gpu_probe",
            "safe_reference_saved_tokens": None,
            "safe_capture_pct_of_quality_oracle": None,
        },
    }

    destination = Path(args.output_dir)
    _json_write(destination / "predictor_oracle_funnel.json", funnel)
    _json_write(destination / "continuation_probe_selection.json", {
        "schema": "continuation_probe_selection_v1",
        "split": "validation_only",
        "probes": selected_probes,
    })
    lines = [
        "# Predictor / oracle funnel (validation only)",
        "",
        f"Prompts: {len(samples)} fixed Tier-1 validation prompts; domains: {funnel['domain_counts']}.",
        "",
        "| Stage | Saved steps | Compression on reference | Slots | Used phrases | Capture vs QA oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage, label in (
        ("quality_aware_oracle", "Quality-aware oracle"),
        ("predictor_candidate_pool", "Predictor candidate pool"),
        ("predictor_top_k", "Predictor Top-K"),
        ("top_k_occurs_in_reference", "Top-K phrases occurring in reference"),
    ):
        row = stages[stage]
        lines.append(
            f"| {label} | {row['tokens_saved']} | {row['compression_pct']}% | {row['slots']} | {row['phrase_count']} | {row['cumulative_vs_quality_oracle_pct']}% |"
        )
    live = stages["live_emission_cpu_phase5"]
    lines.extend([
        f"| Phase-5 CPU live emission (free generation) | {live['tokens_saved']} | {live['compression_pct']}% of generated base-equivalent tokens | {live['slots']} | {live['phrase_count']} emissions | {round(100*live['tokens_saved']/max(stages['quality_aware_oracle']['tokens_saved'],1),3)}%* |",
        "",
        f"*The live-generation row uses different free-running text than the teacher-forced reference rows; its ratio is descriptive, not a matched retention estimate. Historical quality-aware oracle result on all 60 validation prompts was 35.05%; the 12-prompt oracle is recomputed with this report's deterministic tie-break.",
        "",
        f"Quality-pass CPU Phase-5 saved steps: {live['quality_pass_saved_tokens']} ({live['quality_pass_saved_pct']}% of live generated base-equivalent tokens). Alpaca pass is mechanical only.",
        "",
        f"Continuation probe selection: {len(selected_probes)} phrases across {len(probes_by_stratum)} available domain/risk strata; exact GPU metrics are pending in this pre-run package report.",
        "",
        "Domain stage details and phrase-level provenance are in `predictor_oracle_funnel.json`.",
    ])
    (destination / "predictor_oracle_funnel.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return funnel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-data", default="data/cached_pure_pred_val_60.json")
    parser.add_argument("--prompt-ids-file", default="experiments/checkpoints/quality_benchmark/poc_12_prompt_ids.json")
    parser.add_argument("--predictor", default="experiments/checkpoints/oracle_guided_predictor.pkl")
    parser.add_argument("--live-results", required=True)
    parser.add_argument(
        "--phase5-manifest",
        default="experiments/checkpoints/quality_benchmark/tier1_runs/3de046a1b858dc7a/run_manifest.json",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
