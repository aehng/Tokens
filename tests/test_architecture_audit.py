"""CPU/source-level checks for the predictive-fidelity architecture audit.

These tests pin facts about the current tree. They do not load Phi weights
and they do not import zip2zip_compression.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ZIP2ZIP = SRC / "zip2zip"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> ast.Module:
    return ast.parse(_read(path), filename=str(path))


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", 1)[0])
    return names


def _from_import_targets(tree: ast.AST, module_suffix: str) -> set[str]:
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith(module_suffix):
            for alias in node.names:
                targets.add(alias.name)
    return targets


def _function_names(tree: ast.AST) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _class_names(tree: ast.AST) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def test_zip2zip_compression_is_imported_only_by_legacy_modules():
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = _parse(path)
        if "zip2zip_compression" in _imported_names(tree):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert sorted(offenders) == [
        "src/evaluation/lzw_simulator.py",
        "src/zip2zip/codebook.py",
        "src/zip2zip/tokenizer.py",
    ]


def test_static_codebook_and_predictor_v2_do_not_import_lzw():
    static_tree = _parse(ZIP2ZIP / "static_codebook.py")
    pred_init = _parse(ZIP2ZIP / "predictor_v2" / "__init__.py")
    assert "zip2zip_compression" not in _imported_names(static_tree)
    assert "zip2zip_compression" not in _imported_names(pred_init)
    for path in (ZIP2ZIP / "predictor_v2").rglob("*.py"):
        assert "zip2zip_compression" not in _imported_names(_parse(path)), path


def test_package_init_eagerly_imports_legacy_codebook_and_tokenizer():
    tree = _parse(ZIP2ZIP / "__init__.py")
    # Relative imports: from .codebook import CodebookManager
    relative = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            relative.append((node.module, tuple(a.name for a in node.names)))
    assert ("codebook", ("CodebookManager",)) in relative
    assert ("tokenizer", ("Zip2ZipTokenizer",)) in relative
    assert ("model", ("Zip2ZipModel",)) in relative


def test_hyper_modules_type_import_dynamic_codebook_manager():
    embedding = _parse(ZIP2ZIP / "nn" / "embedding.py")
    linear = _parse(ZIP2ZIP / "nn" / "linear.py")
    assert "CodebookManager" in _from_import_targets(embedding, "codebook")
    assert "CodebookManager" in _from_import_targets(linear, "codebook")


def test_zip2zip_model_constructs_dynamic_codebook_manager_in_init():
    source = _read(ZIP2ZIP / "model.py")
    assert "from zip2zip.codebook import CodebookManager" in source
    assert "self.codebook_manager = CodebookManager.from_config(config)" in source
    assert "base_model = PeftModel.from_pretrained(" in source


def test_continuation_consistency_loss_is_documented_but_not_defined():
    path = ZIP2ZIP / "training_objectives.py"
    source = _read(path)
    tree = _parse(path)
    assert "compute_continuation_consistency_loss" in source
    assert "compute_continuation_consistency_loss" not in _function_names(tree)
    assert "compute_reconstruction_loss" in _function_names(tree)
    assert "DifferentiableTrainingManager" in _class_names(tree)
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DifferentiableTrainingManager"
    )
    forward = next(
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_step"
    )
    forward_src = ast.get_source_segment(source, forward)
    assert forward_src is not None
    assert "lm_loss + recon_weight * recon_loss" in forward_src
    assert "kl" not in forward_src.lower()


def test_predictive_pipeline_uses_raw_encode_and_tokenizer_eos():
    source = _read(ZIP2ZIP / "predictive_pipeline.py")
    assert "self.tokenizer.encode(prompt_text, add_special_tokens=False)" in source
    assert "self.tokenizer.encode(response_text, add_special_tokens=False)" in source
    assert "eos_token_id = self.tokenizer.eos_token_id" in source
    assert "response_ids.append(eos_token_id)" in source
    assert "apply_chat_template" not in source


def test_joint_trainer_reads_gold_train_jsonl_and_epfl_checkpoint():
    source = _read(ROOT / "experiments" / "train_predictive_zip2zip.py")
    assert 'train_file = "data/train.jsonl"' in source
    assert "Zip2ZipModel.from_pretrained" in source
    yaml_text = _read(ROOT / "configs" / "predictive_joint_pilot.yaml")
    assert 'name_or_path: "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"' in yaml_text
    assert 'trainable_mode: "joint"' in yaml_text
    assert "reconstruction_weight: 0.1" in yaml_text
    assert "kl" not in yaml_text.lower()


def test_gold_training_data_is_benchmark_references_not_phi_chat():
    first = (ROOT / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert "gsm_8188" in first
    assert "<<10/100*700=70>>" in first
    assert "#### 385000" in first
    assert "<|user|>" not in first
    assert "<|assistant|>" not in first


def test_select_stratified_dev_prompts_is_not_on_the_harness_module():
    harness = _read(ZIP2ZIP / "predictor_v2" / "attribution_harness.py")
    runner = _read(ROOT / "experiments" / "run_phi_attribution_benchmark.py")
    assert "def select_stratified_dev_prompts" not in harness
    assert "def select_stratified_dev_prompts" in runner


def test_pyproject_requires_zip2zip_compression_unconditionally():
    text = _read(ROOT / "pyproject.toml")
    assert "zip2zip-compression>=" in text
    assert "legacy-lzw" not in text


def test_canonical_phi_stop_set_is_chat_end_then_role_then_eot():
    harness = _read(ZIP2ZIP / "predictor_v2" / "attribution_harness.py")
    assert "CANONICAL_EOS_TOKEN_IDS = (32007, 32001, 32000)" in harness
    generator = _read(ROOT / "experiments" / "generate_canonical_phi_continuations.py")
    assert "tokenizer.apply_chat_template" in generator
    manifest = _read(ROOT / "data" / "canonical_phi_continuations.manifest.json")
    assert "microsoft/Phi-3.5-mini-instruct" in manifest
    assert "2fe192450127e6a83f7441aef6e3ca586c338b77" in manifest
