from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from src.zip2zip.predictor_v2.candidate_retrieval import (
    ConfigurableCandidateGenerator,
    RetrievalStrategy,
)
from src.zip2zip.predictor_v2.canonical_dataset import (
    CanonicalDatasetError,
    CanonicalDatasetViews,
    FinalAccessError,
    create_canonical_manifest,
    load_canonical_dataset,
    sha256_file,
    write_canonical_manifest,
)
from src.zip2zip.predictor_v2.experiment_protocol import (
    ARCHITECTURE_FREEZE_SCHEMA,
    CANDIDATE_FREEZE_SCHEMA,
    claim_final_evaluation,
    issue_final_access_permit,
    load_architecture_freeze,
    load_architecture_shortlist,
    load_candidate_freeze,
    load_candidate_plan,
    make_architecture_shortlist,
    make_architecture_freeze,
    make_candidate_freeze,
    make_candidate_plan,
    make_live_integration_gate,
    make_quality_attribution_gate,
    make_integration_subset_freeze,
    sha256_json,
    validate_resume_pair,
    write_json_exclusive,
)
from src.zip2zip.predictor_v2.interfaces import MultiTaskPredictions, PredictorScorer
from src.zip2zip.predictor_v2.train_index import build_index_from_views, build_train_only_index


GENERATION_CONFIG = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 17,
    "eos_token_id": [9],
}


class FakeTokenizer:
    all_special_ids = [9]
    bos_token_id = None
    eos_token_id = 9
    pad_token_id = None
    unk_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [10, 11, 12]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class TinyRanker(PredictorScorer):
    def fit(self, train_records, train_candidates, dev_records=None, dev_candidates=None, epochs=1, lr=1e-3):
        assert {record.prompt_id for record in train_records} == set(train_candidates)
        assert {record.prompt_id for record in dev_records} == set(dev_candidates)
        return {"synthetic": True}

    def score_candidates(self, prompt_ids, candidates, domain="general"):
        weights = np.array([max(0.0, item.raw_association_weight) for item in candidates], dtype=np.float32)
        denom = max(float(weights.max()) if len(weights) else 0.0, 1.0)
        probabilities = weights / denom
        return MultiTaskPredictions(
            p_occurs=probabilities,
            expected_count=probabilities,
            horizon_logits=np.zeros((len(candidates), 5), dtype=np.float32),
            p_safe=np.ones(len(candidates), dtype=np.float32),
            ranking_scores=probabilities,
        )

    def get_parameter_count(self):
        return 1

    def get_model_size_bytes(self):
        return 1


def _record(prompt_id: str, split: str, domain: str = "code") -> dict:
    return {
        "prompt_id": prompt_id,
        "domain": domain,
        "split": split,
        "task_prompt_text": f"Task for {prompt_id}",
        "rendered_prompt_text": f"<|user|>{prompt_id}<|assistant|>",
        "continuation_text": "answer",
        "continuation_token_ids": [20, 21, 20, 21, 9],
        "generated_token_count": 5,
        "termination_reason": "eos",
        "termination_token_id": 9,
        "model_id": "example/phi",
        "model_revision": "model-rev-1",
        "tokenizer_id": "example/phi-tokenizer",
        "tokenizer_revision": "tokenizer-rev-1",
        "generation_config": dict(GENERATION_CONFIG),
        "generation_metadata": {"run_id": "synthetic-run", "worker": "cpu-fixture"},
    }


def _write_dataset(tmp_path: Path, records: list[dict], *, manifest_records: list[dict] | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset = tmp_path / "canonical.jsonl"
    dataset.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    inventory = manifest_records if manifest_records is not None else records
    split_ids = {
        split: sorted(record["prompt_id"] for record in inventory if record["split"] == split)
        for split in ("TRAIN", "DEV", "FINAL")
    }
    manifest = create_canonical_manifest(
        dataset,
        expected_prompt_ids=sorted(record["prompt_id"] for record in inventory),
        split_ids=split_ids,
        allowed_domains=["code", "reasoning", "instruction"],
        model_id="example/phi",
        model_revision="model-rev-1",
        tokenizer_id="example/phi-tokenizer",
        tokenizer_revision="tokenizer-rev-1",
        tokenizer_vocab_size=64,
        generation_config=GENERATION_CONFIG,
        provenance={"source_run_id": "synthetic-run", "source_sha256": "fixture"},
    )
    manifest_path = tmp_path / "canonical.manifest.json"
    write_canonical_manifest(manifest_path, manifest)
    return dataset, manifest_path, manifest


def _records() -> list[dict]:
    return [
        _record("tr-1", "TRAIN", "code"),
        _record("tr-2", "TRAIN", "reasoning"),
        _record("dev-1", "DEV", "code"),
        _record("dev-2", "DEV", "instruction"),
        _record("final-1", "FINAL", "reasoning"),
        _record("final-2", "FINAL", "instruction"),
    ]


def _review() -> dict:
    return {
        "multiseed_robustness": {"passed": True, "notes": "Synthetic gate fixture only."},
        "latency_quality_pareto": {"passed": True, "notes": "Synthetic gate fixture only."},
        "domain_regression": {"passed": True, "notes": "Synthetic gate fixture only."},
        "ridge_baseline": {"passed": True, "notes": "Synthetic gate fixture only."},
    }


def test_canonical_loader_validates_dynamic_cap_and_returns_immutable_views(tmp_path):
    dataset, manifest_path, _ = _write_dataset(tmp_path, _records())
    views, manifest = load_canonical_dataset(dataset, manifest_path)

    assert len(views.train) == 2
    assert len(views.dev) == 2
    assert len(views.final_ids) == 2
    train_ids = {record.prompt_id for record in views.train}
    dev_ids = {record.prompt_id for record in views.dev}
    final_ids = set(views.final_ids)
    assert not (train_ids & dev_ids or train_ids & final_ids or dev_ids & final_ids)
    assert manifest["generation_config"]["max_new_tokens"] == 17
    assert len(views.train[0].continuation_token_ids) == 5
    assert isinstance(views.train, tuple)
    with pytest.raises(FrozenInstanceError):
        views.train[0].domain = "instruction"
    with pytest.raises(FinalAccessError):
        views.open_final(None)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.__setitem__(1, dict(rows[0])), "duplicate prompt_id"),
        (
            lambda rows: rows[0]["continuation_token_ids"].__setitem__(0, 64),
            "continuation_token_ids",
        ),
        (lambda rows: rows[1].__setitem__("generation_config", {**GENERATION_CONFIG, "max_new_tokens": 18}), "generation_config differs"),
        (lambda rows: rows[2].__setitem__("split", "TRAIN"), "record splits do not match"),
        (lambda rows: rows[3].__setitem__("model_revision", "other-revision"), "model_revision differs"),
        (lambda rows: rows[3].__setitem__("domain", "unknown-domain"), "invalid domain"),
        (lambda rows: rows[4].__setitem__("termination_token_id", 21), "termination token must be the last"),
    ],
)
def test_malformed_records_fail_clearly(tmp_path, mutate, message):
    rows = _records()
    expected = [dict(row) for row in rows]
    mutate(rows)
    dataset, manifest_path, _ = _write_dataset(tmp_path, rows, manifest_records=expected)
    with pytest.raises(CanonicalDatasetError, match=message):
        load_canonical_dataset(dataset, manifest_path)


def test_missing_expected_prompt_id_fails(tmp_path):
    rows = _records()
    expected = [dict(row) for row in rows]
    dataset, manifest_path, _ = _write_dataset(tmp_path, rows[:-1], manifest_records=expected)
    with pytest.raises(CanonicalDatasetError, match="prompt IDs differ from expected inventory"):
        load_canonical_dataset(dataset, manifest_path)


def test_dataset_and_provenance_hash_mismatches_fail(tmp_path):
    dataset, manifest_path, manifest = _write_dataset(tmp_path, _records())
    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CanonicalDatasetError, match="dataset_sha256"):
        load_canonical_dataset(dataset, manifest_path)

    dataset, manifest_path, manifest = _write_dataset(tmp_path / "provenance", _records())
    manifest["provenance"] = {"source_run_id": "changed"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CanonicalDatasetError, match="provenance_sha256"):
        load_canonical_dataset(dataset, manifest_path)


def test_train_index_rejects_non_train_records_and_ignores_other_split_content(tmp_path):
    records = _records()
    dataset, manifest_path, _ = _write_dataset(tmp_path, records)
    views, _ = load_canonical_dataset(dataset, manifest_path)
    tokenizer = FakeTokenizer()
    index, summary = build_index_from_views(views, tokenizer=tokenizer, config={"min_cooccurrence": 1})
    assert index.train_prompt_ids == ["tr-1", "tr-2"]
    assert [item["prompt_id"] for item in index.train_prompt_lexical_docs] == ["tr-1", "tr-2"]
    assert summary["source_dataset_sha256"] == views.dataset_sha256
    assert summary["code_file_sha256"]
    assert "src/zip2zip/predictor_v2/train_index.py" in summary["code_file_sha256"]

    with pytest.raises(CanonicalDatasetError, match="TRAIN records only"):
        build_train_only_index(
            list(views.train) + list(views.dev),
            tokenizer=tokenizer,
            dataset_sha256=views.dataset_sha256,
            train_split_sha256=views.train_split_sha256,
            source_manifest_sha256=views.manifest_sha256,
        )

    for changed_index, split_name in ((2, "DEV"), (-1, "FINAL")):
        changed = [dict(record) for record in records]
        changed[changed_index]["continuation_text"] = f"changed {split_name} only"
        changed[changed_index]["continuation_token_ids"] = [31, 30, 31, 30, 9]
        dataset2, manifest_path2, _ = _write_dataset(tmp_path / f"changed-{split_name.lower()}", changed)
        views2, _ = load_canonical_dataset(dataset2, manifest_path2)
        index2, _ = build_index_from_views(views2, tokenizer=tokenizer, config={"min_cooccurrence": 1})
        assert views.train_split_sha256 == views2.train_split_sha256
        assert index.token_associations == index2.token_associations
        assert index.precomputed_global_static == index2.precomputed_global_static
        assert index.train_prompt_lexical_docs == index2.train_prompt_lexical_docs


def test_candidate_strategies_are_deterministic_and_support_all_pool_sizes(tmp_path):
    dataset, manifest_path, _ = _write_dataset(tmp_path, _records())
    views, _ = load_canonical_dataset(dataset, manifest_path)
    index, _ = build_index_from_views(views, tokenizer=FakeTokenizer(), config={"min_cooccurrence": 1})
    generator = ConfigurableCandidateGenerator(index, FakeTokenizer())
    prompt_ids = [10, 11, 12]
    for strategy in RetrievalStrategy:
        first = generator.generate_candidate_pool(prompt_ids, "prompt", "code", strategy, 256)
        second = generator.generate_candidate_pool(prompt_ids, "prompt", "code", strategy, 256)
        assert first == second
        for size in (256, 512, 1024, 2048):
            pool = generator.generate_candidate_pool(prompt_ids, "prompt", "code", strategy, size)
            assert len(pool) <= size


def test_candidate_benchmark_emits_bounds_domains_tail_latency_and_memory(tmp_path, monkeypatch):
    import importlib

    benchmark = importlib.import_module("experiments.benchmark_canonical_candidate_recall")
    dataset, manifest_path, _ = _write_dataset(tmp_path, _records())
    views, _ = load_canonical_dataset(dataset, manifest_path)
    index, _ = build_index_from_views(views, tokenizer=FakeTokenizer(), config={"min_cooccurrence": 1})
    index_path = tmp_path / "train-index.pkl"
    index.save(str(index_path))
    monkeypatch.setattr(benchmark, "load_manifest_tokenizer", lambda manifest: FakeTokenizer())
    output_json = tmp_path / "candidate-recall.json"
    output_md = tmp_path / "candidate-recall.md"
    result = benchmark.run_benchmark(
        dataset_path=str(dataset),
        manifest_path=str(manifest_path),
        index_path=str(index_path),
        out_json=str(output_json),
        out_md=str(output_md),
        pool_sizes=(256,),
        time_limit_seconds=2.0,
    )
    assert result["scope"] == "DEV"
    assert result["is_full_dev"] is True
    assert result["run_manifest"]["source_code"]["tree_sha256"]
    assert result["run_manifest"]["source_code"]["files_sha256"]
    assert set(result["strategy_results"]) == {strategy.value for strategy in RetrievalStrategy}
    metrics = result["strategy_results"]["baseline"]["256"]
    assert metrics["oracle_metrics_by_k"]["32"]["global_oracle_steps"].keys() >= {"lower_bound", "upper_bound"}
    assert metrics["dev_domain_breakdown_by_k"]["32"]
    assert "p99_ms" in metrics["candidate_generation_latency"]["overall"]
    assert metrics["memory"]["index_file_bytes"] > 0
    assert output_json.is_file() and output_md.is_file()


def test_candidate_benchmark_resumes_completed_combinations_only_when_hashes_match(tmp_path, monkeypatch):
    import importlib

    benchmark = importlib.import_module("experiments.benchmark_canonical_candidate_recall")
    dataset, manifest_path, _ = _write_dataset(tmp_path, _records())
    views, _ = load_canonical_dataset(dataset, manifest_path)
    index, _ = build_index_from_views(views, tokenizer=FakeTokenizer(), config={"min_cooccurrence": 1})
    index_path = tmp_path / "train-index.pkl"
    index.save(str(index_path))
    monkeypatch.setattr(benchmark, "load_manifest_tokenizer", lambda manifest: FakeTokenizer())
    output_json = tmp_path / "candidate-recall.json"
    output_md = tmp_path / "candidate-recall.md"
    resume_state = tmp_path / "candidate-recall.resume.json"
    save_state = benchmark._save_resume_state
    writes = 0

    def interrupt_after_first_combination(path, state):
        nonlocal writes
        writes += 1
        save_state(path, state)
        if writes == 2:
            raise RuntimeError("synthetic interruption after durable combination checkpoint")

    monkeypatch.setattr(benchmark, "_save_resume_state", interrupt_after_first_combination)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        benchmark.run_benchmark(
            dataset_path=str(dataset),
            manifest_path=str(manifest_path),
            index_path=str(index_path),
            out_json=str(output_json),
            out_md=str(output_md),
            pool_sizes=(256,),
            time_limit_seconds=2.0,
            resume_state_path=str(resume_state),
        )
    assert len(json.loads(resume_state.read_text(encoding="utf-8"))["completed_combinations"]["baseline"]) == 1
    monkeypatch.setattr(benchmark, "_save_resume_state", save_state)
    resumed = benchmark.run_benchmark(
        dataset_path=str(dataset),
        manifest_path=str(manifest_path),
        index_path=str(index_path),
        out_json=str(output_json),
        out_md=str(output_md),
        pool_sizes=(256,),
        time_limit_seconds=2.0,
        resume_state_path=str(resume_state),
        resume=True,
    )
    assert set(resumed["strategy_results"]) == {strategy.value for strategy in RetrievalStrategy}
    assert output_json.is_file() and output_md.is_file()
    saved_json = output_json.read_bytes()
    output_json.unlink()
    recovered = benchmark.run_benchmark(
        dataset_path=str(dataset),
        manifest_path=str(manifest_path),
        index_path=str(index_path),
        out_json=str(output_json),
        out_md=str(output_md),
        pool_sizes=(256,),
        time_limit_seconds=2.0,
        resume_state_path=str(resume_state),
        resume=True,
    )
    assert json.loads(output_json.read_text(encoding="utf-8")) == recovered
    assert output_json.read_bytes() == saved_json

    corrupt_state = tmp_path / "corrupt.resume.json"
    corrupt_payload = json.loads(resume_state.read_text(encoding="utf-8"))
    corrupt_payload["completed_combinations"]["unexpected"] = {}
    corrupt_state.write_text(json.dumps(corrupt_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        benchmark.run_benchmark(
            dataset_path=str(dataset),
            manifest_path=str(manifest_path),
            index_path=str(index_path),
            out_json=str(tmp_path / "corrupt.json"),
            out_md=str(tmp_path / "corrupt.md"),
            pool_sizes=(256,),
            time_limit_seconds=2.0,
            resume_state_path=str(corrupt_state),
            resume=True,
        )
    with pytest.raises(ValueError, match="does not match"):
        benchmark.run_benchmark(
            dataset_path=str(dataset),
            manifest_path=str(manifest_path),
            index_path=str(index_path),
            out_json=str(tmp_path / "mismatch.json"),
            out_md=str(tmp_path / "mismatch.md"),
            pool_sizes=(512,),
            time_limit_seconds=2.0,
            resume_state_path=str(resume_state),
            resume=True,
        )


def _write_attribution_gate(tmp_path, views):
    evidence = tmp_path / "attribution-evidence.json"
    evidence.write_text(json.dumps({
        "scope": "DEV",
        "dataset_sha256": views.dataset_sha256,
        "dev_split_sha256": views.dev_split_sha256,
        "matched_prompt_count": len(views.dev),
        "domains": sorted({record.domain for record in views.dev}),
        "baseline_conditions": ["Vanilla", "Predictive Phi"],
        "failure_attribution_counts": {
            "candidate_generation": 0, "candidate_ranking": 0, "codebook": 1,
            "h_emission": 0, "representation": 0, "continuation_state": 0,
            "eos": 0, "serving": 0, "other": 0,
        },
    }), encoding="utf-8")
    gate_path = tmp_path / "attribution-gate.json"
    make_quality_attribution_gate(
        evidence_path=evidence, primary_bottleneck="codebook", rationale="Synthetic fixture only.",
        views=views, output_path=gate_path,
    )
    return gate_path


def test_quality_attribution_can_redirect_and_blocks_predictor_training(tmp_path):
    dataset, manifest_path, _ = _write_dataset(tmp_path, _records())
    views, _ = load_canonical_dataset(dataset, manifest_path)
    evidence_path = tmp_path / "attribution-evidence.json"
    evidence_path.write_text(json.dumps({
        "scope": "DEV", "dataset_sha256": views.dataset_sha256,
        "dev_split_sha256": views.dev_split_sha256, "matched_prompt_count": len(views.dev),
        "domains": ["code"], "baseline_conditions": ["Vanilla", "Predictive Phi"],
        "failure_attribution_counts": {name: 0 for name in (
            "candidate_generation", "candidate_ranking", "codebook", "h_emission",
            "representation", "continuation_state", "eos", "serving", "other",
        )},
    }), encoding="utf-8")
    gate_path = tmp_path / "redirected-gate.json"
    gate = make_quality_attribution_gate(
        evidence_path=evidence_path, primary_bottleneck="eos", rationale="Synthetic redirect fixture.",
        views=views, output_path=gate_path,
    )
    assert gate["status"] == "redirected"
    assert gate["redirect_to"] == "eos"
    with pytest.raises(CanonicalDatasetError, match="redirect before training"):
        make_candidate_plan(
            benchmark_path=tmp_path / "not-used.json", strategy="baseline", pool_size=256,
            rationale="Should be blocked.", quality_attribution_gate_path=gate_path,
            views=views, output_path=tmp_path / "candidate-plan.json",
        )


def test_integration_subset_rejects_final_and_unknown_prompt_ids(tmp_path):
    dataset, manifest_path, _ = _write_dataset(tmp_path, _records())
    views, _ = load_canonical_dataset(dataset, manifest_path)
    with pytest.raises(CanonicalDatasetError, match="DEV prompt IDs"):
        make_integration_subset_freeze(prompt_ids=[views.final_ids[0]], views=views, output_path=tmp_path / "bad-subset.json")


def test_workflow_report_exposes_revised_gate_state(tmp_path):
    import importlib

    report_module = importlib.import_module("experiments.report_predictor_v2_workflow")
    dataset, manifest_path, _ = _write_dataset(tmp_path, _records())
    absent = str(tmp_path / "not-created.json")
    report, markdown = report_module.build_report(
        dataset_path=str(dataset),
        manifest_path=str(manifest_path),
        candidate_benchmark_path=absent,
        quality_attribution_gate_path=absent,
        candidate_plan_path=absent,
        architecture_shortlist_path=absent,
        integration_subset_path=None,
        live_integration_gate_path=absent,
        candidate_freeze_path=absent,
        architecture_bakeoff_path=absent,
        architecture_freeze_path=absent,
        final_claim_path=absent,
        final_result_path=absent,
    )
    assert report["gate_state"] == {
        "candidate_generator_frozen": False,
        "offline_architecture_shortlist_complete": False,
        "live_integration_gate_passed": False,
        "architecture_frozen": False,
        "final_evaluated": False,
    }
    assert "FINAL is excluded from candidate and architecture selection" in markdown


def test_freeze_gates_bind_evidence_and_final_access_is_one_time(tmp_path):
    dataset, manifest_path, _ = _write_dataset(tmp_path, _records())
    views, _ = load_canonical_dataset(dataset, manifest_path)
    attribution_path = _write_attribution_gate(tmp_path, views)
    benchmark = tmp_path / "candidate-benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "scope": "DEV",
                "is_full_dev": True,
                "dataset_sha256": views.dataset_sha256,
                "train_split_sha256": views.train_split_sha256,
                "dev_split_sha256": views.dev_split_sha256,
                "train_index_sha256": "synthetic-index-hash",
                "train_index_provenance_sha256": "synthetic-index-provenance-hash",
                "strategy_results": {
                    "baseline": {
                        "256": {
                            "candidate_config": {"strategy": "baseline", "pool_size": 256},
                            "oracle_metrics_by_k": {"32": {"capture_interval": {"lower": 0.1, "upper": 0.2}}},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "candidate-plan.json"
    plan = make_candidate_plan(
        benchmark_path=benchmark,
        strategy="baseline",
        pool_size=256,
        rationale="Synthetic freeze gate test.",
        quality_attribution_gate_path=attribution_path,
        views=views,
        output_path=plan_path,
    )
    checkpoint = tmp_path / "ridge.pkl"
    checkpoint.write_bytes(b"synthetic checkpoint")
    bakeoff = tmp_path / "bakeoff.json"
    bakeoff.write_text(
        json.dumps(
            {
                "scope": "DEV",
                "is_full_dev": True,
                "dataset_sha256": views.dataset_sha256,
                "train_split_sha256": views.train_split_sha256,
                "dev_split_sha256": views.dev_split_sha256,
                "candidate_plan_sha256": sha256_file(plan_path),
                "final_accessed": False,
                "training_config": {"epochs": 1},
                "architectures": {
                    "Ridge": {
                        "seed_runs": [
                            {
                                "seed": 42,
                                "checkpoint_path": str(checkpoint.resolve()),
                                "checkpoint_sha256": sha256_file(checkpoint),
                                "dev_k16_by_domain": {"code": 1},
                                "inference_latency_ms": {"p50_ms": 1.0},
                                "dev_k16_realized_steps": 1,
                                "architecture_config": {"name": "Ridge", "training": {"optimizer": "closed_form_ridge"}},
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    shortlist_path = tmp_path / "shortlist.json"
    make_architecture_shortlist(
        bakeoff_path=bakeoff,
        quality_attribution_gate_path=attribution_path,
        candidate_plan_path=plan_path,
        selections=[{"architecture": "Ridge", "seed": 42}],
        rationale="Synthetic shortlist only.", views=views, output_path=shortlist_path,
    )
    with pytest.raises(CanonicalDatasetError):
        make_candidate_freeze(
            candidate_plan_path=plan_path, quality_attribution_gate_path=attribution_path,
            shortlist_path=shortlist_path, live_integration_gate_path=tmp_path / "not-passed.json",
            rationale="Must wait for live integration.", views=views, output_path=tmp_path / "premature-freeze.json",
        )
    subset_path = tmp_path / "integration-subset.json"
    subset_ids = [record.prompt_id for record in views.dev[:1]]
    make_integration_subset_freeze(prompt_ids=subset_ids, views=views, output_path=subset_path)
    per_candidate_result = tmp_path / "live-ridge-results.json"
    per_candidate_result.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    live_evidence = tmp_path / "live-evidence.json"
    live_evidence.write_text(json.dumps({
        "evaluation_mode": "live_end_to_end", "partition": "FROZEN_INTEGRATION_SUBSET",
        "dataset_sha256": views.dataset_sha256, "dev_split_sha256": views.dev_split_sha256,
        "prompt_ids": subset_ids,
        "candidates": [{
            "architecture": "Ridge", "seed": 42, "checkpoint_sha256": sha256_file(checkpoint),
            "candidate_plan_sha256": sha256_file(plan_path), "results_path": str(per_candidate_result.resolve()),
            "results_sha256": sha256_file(per_candidate_result), "prompt_count": len(subset_ids),
            "domains": [views.dev[0].domain], "h_emission_count": 1, "continuation_failure_count": 0,
            "eos_failure_count": 0, "truncation_count": 0, "repetition_count": 0,
            "task_quality": 1.0, "matched_vanilla_task_quality": 1.0, "decode_steps": 9, "vanilla_decode_steps": 10,
        }],
    }), encoding="utf-8")
    checks = {name: {"passed": True, "notes": "Synthetic fixture only."} for name in ("h_emission", "continuation_state", "task_quality", "termination_health", "decode_step_savings")}
    live_review = tmp_path / "live-review.json"
    live_review.write_text(json.dumps({"candidates": [{"architecture": "Ridge", "seed": 42, "checks": checks}]}), encoding="utf-8")
    live_gate_path = tmp_path / "live-gate.json"
    make_live_integration_gate(
        evidence_path=live_evidence, review_path=live_review,
        quality_attribution_gate_path=attribution_path, candidate_plan_path=plan_path,
        shortlist_path=shortlist_path, subset_path=subset_path, views=views, output_path=live_gate_path,
    )
    candidate_path = tmp_path / "candidate-freeze.json"
    candidate = make_candidate_freeze(
        candidate_plan_path=plan_path, quality_attribution_gate_path=attribution_path,
        shortlist_path=shortlist_path, live_integration_gate_path=live_gate_path,
        integration_subset_path=subset_path,
        rationale="Synthetic freeze gate test.", views=views, output_path=candidate_path,
    )
    assert candidate["schema"] == CANDIDATE_FREEZE_SCHEMA
    load_candidate_freeze(candidate_path, views)
    with pytest.raises(FileExistsError):
        make_candidate_freeze(
            candidate_plan_path=plan_path, quality_attribution_gate_path=attribution_path,
            shortlist_path=shortlist_path, live_integration_gate_path=live_gate_path,
            integration_subset_path=subset_path, rationale="Must not overwrite.", views=views, output_path=candidate_path,
        )
    architecture_path = tmp_path / "architecture-freeze.json"
    architecture = make_architecture_freeze(
        bakeoff_path=bakeoff,
        architecture="Ridge",
        seed=42,
        checkpoint_path=checkpoint,
        candidate_freeze_path=candidate_path,
        quality_attribution_gate_path=attribution_path,
        candidate_plan_path=plan_path,
        shortlist_path=shortlist_path,
        live_integration_gate_path=live_gate_path,
        integration_subset_path=subset_path,
        review=_review(),
        rationale="Synthetic freeze gate test.",
        views=views,
        output_path=architecture_path,
    )
    assert architecture["schema"] == ARCHITECTURE_FREEZE_SCHEMA
    load_architecture_freeze(
        architecture_path, views, candidate_path,
        quality_attribution_gate_path=attribution_path,
        candidate_plan_path=plan_path,
        shortlist_path=shortlist_path,
        live_integration_gate_path=live_gate_path,
        integration_subset_path=subset_path,
    )
    with pytest.raises(FinalAccessError, match="--allow-final-eval"):
        issue_final_access_permit(
            allow_final_eval=False,
            quality_attribution_gate_path=attribution_path,
            candidate_plan_path=plan_path,
            candidate_freeze_path=candidate_path,
            shortlist_path=shortlist_path,
            live_integration_gate_path=live_gate_path,
            architecture_freeze_path=architecture_path,
            integration_subset_path=subset_path,
            views=views,
        )
    with pytest.raises(CanonicalDatasetError, match="live integration gate"):
        issue_final_access_permit(
            allow_final_eval=True,
            quality_attribution_gate_path=attribution_path,
            candidate_plan_path=plan_path,
            candidate_freeze_path=candidate_path,
            shortlist_path=shortlist_path,
            live_integration_gate_path=tmp_path / "missing-live-gate.json",
            architecture_freeze_path=architecture_path,
            integration_subset_path=subset_path,
            views=views,
        )
    permit = issue_final_access_permit(
        allow_final_eval=True,
        quality_attribution_gate_path=attribution_path,
        candidate_plan_path=plan_path,
        candidate_freeze_path=candidate_path,
        shortlist_path=shortlist_path,
        live_integration_gate_path=live_gate_path,
        architecture_freeze_path=architecture_path,
        integration_subset_path=subset_path,
        views=views,
    )
    assert len(views.open_final(permit)) == 2
    claim_path = tmp_path / "final.claim.json"
    claim_final_evaluation(claim_path=claim_path, result_path=tmp_path / "final.json", permit=permit)
    with pytest.raises(FileExistsError):
        claim_final_evaluation(claim_path=claim_path, result_path=tmp_path / "final.json", permit=permit)


def test_resume_requires_matching_configuration_and_checkpoint_hash(tmp_path):
    checkpoint = tmp_path / "seed.pkl"
    checkpoint.write_bytes(b"model checkpoint bytes")
    expected_config_hash = sha256_json({"seed": 42, "epochs": 3})
    run_metadata = tmp_path / "seed.json"
    write_json_exclusive(
        run_metadata,
        {
            "run_config_sha256": expected_config_hash,
            "checkpoint_sha256": sha256_file(checkpoint),
        },
    )
    assert validate_resume_pair(checkpoint, run_metadata, expected_config_hash)["run_config_sha256"] == expected_config_hash
    with pytest.raises(CanonicalDatasetError, match="configuration hash"):
        validate_resume_pair(checkpoint, run_metadata, "different-config")
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(CanonicalDatasetError, match="checkpoint hash"):
        validate_resume_pair(checkpoint, run_metadata, expected_config_hash)


def test_training_orchestrator_uses_separate_train_dev_maps_and_never_opens_final(tmp_path, monkeypatch):
    import importlib

    bakeoff = importlib.import_module("experiments.train_predictor_v2_canonical_bakeoff")
    dataset, manifest_path, _ = _write_dataset(tmp_path, _records())
    views, _ = load_canonical_dataset(dataset, manifest_path)
    index, _ = build_index_from_views(views, tokenizer=FakeTokenizer(), config={"min_cooccurrence": 1})
    index_path = tmp_path / "train-index.pkl"
    index.save(str(index_path))
    benchmark = tmp_path / "candidate-benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "scope": "DEV",
                "is_full_dev": True,
                "dataset_sha256": views.dataset_sha256,
                "train_split_sha256": views.train_split_sha256,
                "dev_split_sha256": views.dev_split_sha256,
                "train_index_sha256": sha256_file(index_path),
                "train_index_provenance_sha256": sha256_json(index.provenance),
                "strategy_results": {
                    "baseline": {
                        "256": {
                            "candidate_config": {"strategy": "baseline", "pool_size": 256},
                            "oracle_metrics_by_k": {"32": {"capture_interval": {"lower": 0.1, "upper": 0.2}}},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    attribution_path = _write_attribution_gate(tmp_path, views)
    candidate_plan_path = tmp_path / "candidate-plan.json"
    make_candidate_plan(
        benchmark_path=benchmark,
        strategy="baseline",
        pool_size=256,
        rationale="Temporary synthetic plumbing test; no experimental conclusion.",
        quality_attribution_gate_path=attribution_path,
        views=views,
        output_path=candidate_plan_path,
    )
    monkeypatch.setattr(bakeoff, "load_manifest_tokenizer", lambda manifest: FakeTokenizer())
    monkeypatch.setattr(bakeoff, "_instantiate", lambda name: TinyRanker())
    monkeypatch.setattr(
        CanonicalDatasetViews,
        "open_final",
        lambda self, permit: pytest.fail("architecture selection attempted to open FINAL"),
    )
    result = bakeoff.run_bakeoff(
        dataset_path=str(dataset),
        manifest_path=str(manifest_path),
        index_path=str(index_path),
        quality_attribution_gate_path=str(attribution_path),
        candidate_plan_path=str(candidate_plan_path),
        output_dir=str(tmp_path / "models"),
        out_json=str(tmp_path / "bakeoff.json"),
        epochs=1,
        learning_rate=1e-3,
        seeds=(42, 43, 44),
        time_limit_seconds=2.0,
    )
    assert result["scope"] == "DEV"
    assert result["final_accessed"] is False
    assert result["train_prompts"] == 2
    assert result["dev_prompts"] == 2
    assert set(result["architectures"]) == set(bakeoff.MODEL_NAMES)
    assert all(item["seed_count"] == 3 for name, item in result["architectures"].items() if name != "Ridge")
