"""Comprehensive Quality & Compute Economics Benchmark Harness.

Compares 4 conditions across the frozen 60-prompt held-out validation set:
A. Original Phi (microsoft/Phi-3.5-mini-instruct, vanilla)
B. Official Reactive Zip2Zip (epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1, native LZW)
C. Our Predictive Step 100 (checkpoint_step_100.pt, K=32 prompt-predicted codebook)
D. Our Predictive Step 150 (checkpoint_step_150.pt, K=32 prompt-predicted codebook)

Outputs to experiments/checkpoints/quality_benchmark/mbpp_signature_v2/:
- raw_results.jsonl (line-by-line streaming)
- aggregate_results.json
- paired_quality_deltas.json
- failure_cases.json
- instruction_blind_review.json
- instruction_blind_review_key.json
- quality_vs_compression.json
- benchmark_report.md
"""

import argparse
import ast
import gc
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import Zip2ZipModel, Zip2ZipTokenizer, StaticCodebookManager
from zip2zip.predictor_policy import CappedPredictorPolicy
from experiments.mbpp_prompt import build_mbpp_prompt
from experiments.load_joint_checkpoint import load_joint_checkpoint
from experiments.load_oracle_predictor import load_oracle_predictor
from experiments.benchmark_provenance import write_json_atomic

VAL_DATA_PATH = "data/cached_pure_pred_val_60.json"
PREDICTOR_PATH = "experiments/checkpoints/oracle_guided_predictor.pkl"
CKPT_100_PATH = "experiments/checkpoints/predictive_joint_pilot/checkpoint_step_100.pt"
CKPT_150_PATH = "experiments/checkpoints/predictive_joint_pilot/checkpoint_step_150.pt"
OUTPUT_DIR = "experiments/checkpoints/quality_benchmark/mbpp_signature_v2"
MAX_NEW_TOKENS = 300
INITIAL_VOCAB = 32011
PHI_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
ZIP2ZIP_MODEL_ID = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
DEFAULT_PHI_REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"
DEFAULT_ZIP2ZIP_REVISION = "11c461733a79d2a5de6b814585c3361ca2aacbe7"
EVALUATOR_VERSION = "phi_quality_evaluator_v2"
PROMPT_FORMATTER_VERSION = "mbpp_task_signature_v2"
GENERATION_RECORD_SCHEMA = "phi_generation_record_v2"

MBPP_ALLOWED_IMPORTS = frozenset(
    {"math", "re", "collections", "heapq", "itertools", "bisect", "string", "functools", "operator"}
)
MBPP_SAFE_BUILTINS = (
    "abs", "all", "any", "bin", "bool", "bytearray", "bytes", "callable", "chr",
    "complex", "dict", "divmod", "enumerate", "filter", "float", "frozenset", "hash",
    "hex", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max",
    "min", "next", "oct", "ord", "pow", "print", "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "zip", "AssertionError",
    "AttributeError", "Exception", "IndexError", "KeyError", "NameError",
    "NotImplementedError", "RuntimeError", "StopIteration", "TypeError", "ValueError",
    "ZeroDivisionError",
)
MBPP_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024

_MBPP_RUNNER = r"""
import bisect as _bisect
import builtins as _builtins
import collections as _collections
import functools as _functools
import heapq as _heapq
import itertools as _itertools
import json as _json
import math as _math
import operator as _operator
import re as _re
import string as _string
import sys as _sys

_allowed_modules = {
    "math": _math,
    "re": _re,
    "collections": _collections,
    "heapq": _heapq,
    "itertools": _itertools,
    "bisect": _bisect,
    "string": _string,
    "functools": _functools,
    "operator": _operator,
}

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name.split(".", 1)[0] not in _allowed_modules:
        raise ImportError("module is not available in the restricted MBPP evaluator")
    return _builtins.__import__(name, globals, locals, fromlist, level)

_SAFE_BUILTIN_NAMES = __SAFE_BUILTIN_NAMES__
_safe_builtins = {name: getattr(_builtins, name) for name in _SAFE_BUILTIN_NAMES}
_safe_builtins["__import__"] = _safe_import
_namespace = {"__builtins__": _safe_builtins, "__name__": "__mbpp__"}
_namespace.update(_allowed_modules)
_payload = _json.loads(_sys.stdin.read())
exec(compile(_payload["code"], "<generated-mbpp-code>", "exec"), _namespace, _namespace)
for _index, _test in enumerate(_payload["assert_statements"]):
    exec(compile(_test, f"<mbpp-assert-{_index}>", "exec"), _namespace, _namespace)
"""


class TimingLogitsProcessor(LogitsProcessor):
    """Accurately measures TTFT (prefill time) and decode step counts, and masks unseeded hypertokens."""

    def __init__(self, t_start: float, static_mgr: Optional[StaticCodebookManager] = None):
        self.t_start = t_start
        self.static_mgr = static_mgr
        self.ttft: Optional[float] = None
        self.step_count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.ttft is None:
            if scores.is_cuda:
                torch.cuda.synchronize(scores.device)
            self.ttft = time.perf_counter() - self.t_start
        self.step_count += 1
        if self.static_mgr is not None:
            scores = self.static_mgr.mask_unused_logits(scores)
        return scores


def synchronize_device(device: torch.device) -> None:
    """Finish queued CUDA work before taking wall-clock timestamps."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _revision_kwargs(revision: Optional[str]) -> Dict[str, str]:
    return {"revision": revision} if revision else {}


def _load_zip2zip_model(
    model_id: str,
    base_revision: Optional[str],
    model_revision: Optional[str],
) -> Zip2ZipModel:
    # Pin the independently-versioned Phi base and Zip2Zip adapter separately.
    base_model = AutoModelForCausalLM.from_pretrained(
        PHI_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **_revision_kwargs(base_revision),
    )
    return Zip2ZipModel.from_pretrained(
        model_id,
        base_model=base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **_revision_kwargs(model_revision),
    )


def load_predictive_model_bundle(
    checkpoint_path: str,
    device: str,
    base_revision: Optional[str] = None,
    model_revision: Optional[str] = None,
    expected_step: Optional[int] = None,
) -> Dict[str, Any]:
    """Load a checkpoint/model/predictor once for paired prompt conditions."""
    base_revision = base_revision or DEFAULT_PHI_REVISION
    model_revision = model_revision or DEFAULT_ZIP2ZIP_REVISION
    device_obj = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        PHI_MODEL_ID, **_revision_kwargs(base_revision)
    )
    model = _load_zip2zip_model(ZIP2ZIP_MODEL_ID, base_revision, model_revision)
    print(f"Loading checkpoint weights from {checkpoint_path}...", flush=True)
    report = load_joint_checkpoint(
        model,
        checkpoint_path,
        expected_step=expected_step,
        expected_model_id=ZIP2ZIP_MODEL_ID,
    )
    print(
        f"Verified Step {report['step']} with {report['checkpoint_loader']}: "
        f"{report['changed_tensor_count']} trained tensors applied; "
        f"missing={sum(map(len, report['missing_keys'].values()))}, "
        f"unexpected={sum(map(len, report['unexpected_keys'].values()))}, "
        f"base_hashes={report['base_hash_status']}.",
        flush=True,
    )
    for name, component in report["components"].items():
        print(
            f"  {name}: sha256 {component['before_sha256'][:12]} -> "
            f"{component['after_sha256'][:12]}, L2 {component['before_l2_norm']:.5g} -> "
            f"{component['after_l2_norm']:.5g}, changed={component['changed_tensor_count']}",
            flush=True,
        )
    model.to(device_obj)
    model.eval()
    raw_predictor = load_oracle_predictor(PREDICTOR_PATH)
    predictor_index = getattr(raw_predictor, "index", raw_predictor)
    policy = CappedPredictorPolicy(
        predictor_index,
        tokenizer,
        budget=32,
        max_structural_slots=0,
        allow_numeric=True,
        filter_bare_punctuation=True,
    )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "policy": policy,
        "device": device_obj,
        "embedding_dim": model.zip2zip_config.encoder.hidden_size,
        "pad_id": tokenizer.pad_token_id or 32000,
        "disabled_ids": list(model.zip2zip_config.compression.disabled_ids),
        "checkpoint_load_report": report,
    }


def _write_generation_record(
    record: Dict[str, Any],
    raw_results_path: str,
    cache_keys: Optional[Mapping[Tuple[str, str], str]],
    cache_root: Optional[str],
    tested_commit: Optional[str],
) -> None:
    record["record_schema"] = GENERATION_RECORD_SCHEMA
    record["evaluator_version"] = EVALUATOR_VERSION
    record["prompt_formatter_version"] = PROMPT_FORMATTER_VERSION
    key = None
    if cache_keys is not None:
        key = cache_keys.get((record["prompt_id"], record["condition"]))
        if not key:
            raise ValueError(
                f"No exact generation cache key for {record['condition']} / {record['prompt_id']}"
            )
        record["generation_cache_key"] = key
        record["generated_from_commit"] = tested_commit

    if key and cache_root:
        write_json_atomic(os.path.join(cache_root, f"{key}.json"), record)
    with open(raw_results_path, "a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_process_rss_gb() -> float:
    p = psutil.Process(os.getpid())
    return round(p.memory_info().rss / (1024 ** 3), 2)


# ==============================================================================
# Quality Evaluation Functions
# ==============================================================================

def extract_python_code(text: str) -> str:
    """Extract python code from markdown or raw text."""
    fence = re.escape(chr(96) * 3)
    m = re.search(fence + r"(?:python)?\s*(.*?)\s*" + fence, text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(
        r"(?m)^\s*(?=(?:import\b|from\b|async\s+def\b|def\b|class\b))",
        text,
    )
    if m:
        text = text[m.start():]
    return re.split(r"(?m)^\s*#\s*Tests\b", text, maxsplit=1)[0].rstrip()


def _mbpp_safety_error(tree: ast.AST) -> Optional[str]:
    forbidden_names = {
        "open", "exec", "eval", "compile", "input", "breakpoint", "globals", "locals",
        "vars", "dir", "getattr", "setattr", "delattr", "help", "exit", "quit",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                return f"dunder identifier {node.id!r} is disallowed"
            if node.id in forbidden_names:
                return f"builtin {node.id!r} is disallowed"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return f"dunder attribute {node.attr!r} is disallowed"
            if node.attr in {"format", "format_map"}:
                return f"dynamic string formatter {node.attr!r} is disallowed"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in MBPP_ALLOWED_IMPORTS:
                    return f"import {alias.name!r} is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if (
                node.level != 0
                or not node.module
                or node.module.split(".", 1)[0] not in MBPP_ALLOWED_IMPORTS
            ):
                return f"import from {node.module!r} is not allowed"
    return None


def _run_restricted_mbpp(code: str, assert_statements: List[str], timeout_s: float) -> dict[str, Any]:
    payload = json.dumps(
        {"code": code, "assert_statements": assert_statements},
        ensure_ascii=True,
    ).encode("utf-8")
    safe_env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR")
        if key in os.environ
    }
    with tempfile.TemporaryDirectory(prefix="tokens-mbpp-") as temp_dir:
        safe_env["TEMP"] = temp_dir
        safe_env["TMP"] = temp_dir
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                _MBPP_RUNNER.replace(
                    "__SAFE_BUILTIN_NAMES__", json.dumps(MBPP_SAFE_BUILTINS)
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=temp_dir,
            env=safe_env,
        )
        assert process.stdin is not None
        try:
            process.stdin.write(payload)
            process.stdin.close()
        except BrokenPipeError:
            process.wait(timeout=1)

        started = time.monotonic()
        memory_exceeded = False
        timed_out = False
        child = psutil.Process(process.pid) if process.poll() is None else None
        while process.poll() is None:
            if time.monotonic() - started > timeout_s:
                timed_out = True
                process.kill()
                break
            try:
                if child is not None and child.memory_info().rss > MBPP_MEMORY_LIMIT_BYTES:
                    memory_exceeded = True
                    process.kill()
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.02)
        process.wait(timeout=2)
        return {
            "return_code": process.returncode,
            "timeout": timed_out,
            "memory_exceeded": memory_exceeded,
            "elapsed_s": round(time.monotonic() - started, 4),
        }


def evaluate_mbpp_code(code_str: str, assert_statements: List[str], timeout_s: float = 5.0) -> Dict[str, Any]:
    """Evaluate MBPP with a restricted, time- and memory-bounded subprocess.

    This is a defense-in-depth harness for benchmark code, not a hardened OS
    sandbox for actively malicious programs.
    """
    code = extract_python_code(code_str)
    try:
        module = ast.parse(code)
        syntax_valid = True
        syntax_error = None
    except SyntaxError as e:
        return {
            "syntax_valid": False,
            "syntax_error": str(e),
            "tests_available": len(assert_statements),
            "tests_passed": 0,
            "problem_pass": False,
            "runtime_error": None,
            "timeout": False,
            "memory_limit_exceeded": False,
            "safety_rejected": False,
            "clean_code": code,
        }

    if not assert_statements:
        return {
            "syntax_valid": True,
            "syntax_error": None,
            "tests_available": 0,
            "tests_passed": 0,
            "problem_pass": False,
            "runtime_error": "No reference assertions supplied",
            "timeout": False,
            "memory_limit_exceeded": False,
            "safety_rejected": False,
            "clean_code": code,
        }

    code_safety_error = _mbpp_safety_error(module)
    if code_safety_error:
        return {
            "syntax_valid": True,
            "syntax_error": None,
            "tests_available": len(assert_statements),
            "tests_passed": 0,
            "problem_pass": False,
            "runtime_error": code_safety_error,
            "timeout": False,
            "memory_limit_exceeded": False,
            "safety_rejected": True,
            "clean_code": code,
        }

    for statement in assert_statements:
        try:
            test_tree = ast.parse(statement)
        except SyntaxError as exc:
            return {
                "syntax_valid": True,
                "syntax_error": None,
                "tests_available": len(assert_statements),
                "tests_passed": 0,
                "problem_pass": False,
                "runtime_error": f"Invalid reference assertion: {exc.msg}",
                "timeout": False,
                "memory_limit_exceeded": False,
                "safety_rejected": True,
                "clean_code": code,
            }
        test_safety_error = _mbpp_safety_error(test_tree)
        if test_safety_error:
            return {
                "syntax_valid": True,
                "syntax_error": None,
                "tests_available": len(assert_statements),
                "tests_passed": 0,
                "problem_pass": False,
                "runtime_error": f"Unsafe reference assertion: {test_safety_error}",
                "timeout": False,
                "memory_limit_exceeded": False,
                "safety_rejected": True,
                "clean_code": code,
            }

    try:
        result = _run_restricted_mbpp(code, assert_statements, timeout_s)
        if result["return_code"] == 0:
            return {
                "syntax_valid": True,
                "syntax_error": None,
                "tests_available": len(assert_statements),
                "tests_passed": len(assert_statements),
                "problem_pass": True,
                "runtime_error": None,
                "timeout": result["timeout"],
                "memory_limit_exceeded": result["memory_exceeded"],
                "safety_rejected": False,
                "evaluation_runtime_s": result["elapsed_s"],
                "clean_code": code,
            }
        return {
            "syntax_valid": True,
            "syntax_error": None,
            "tests_available": len(assert_statements),
            "tests_passed": 0,
            "problem_pass": False,
            "runtime_error": (
                "Memory limit exceeded"
                if result["memory_exceeded"]
                else "TimeoutExpired"
                if result["timeout"]
                else f"Restricted evaluator exit status {result['return_code']}"
            ),
            "timeout": result["timeout"],
            "memory_limit_exceeded": result["memory_exceeded"],
            "safety_rejected": False,
            "evaluation_runtime_s": result["elapsed_s"],
            "clean_code": code,
        }
    except (OSError, subprocess.SubprocessError, psutil.Error) as exc:
        return {
            "syntax_valid": True,
            "syntax_error": None,
            "tests_available": len(assert_statements),
            "tests_passed": 0,
            "problem_pass": False,
            "runtime_error": f"Restricted evaluator failed: {type(exc).__name__}",
            "timeout": False,
            "memory_limit_exceeded": False,
            "safety_rejected": False,
            "clean_code": code,
        }


def extract_gsm_answer(text: str) -> Optional[str]:
    """Extract numeric answer from reasoning text."""
    if not text:
        return None
    # 1. #### <number>
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "").strip()
    # 2. \boxed{<number>}
    m = re.findall(r"\\boxed\{\s*(-?[\d,]+(?:\.\d+)?)\s*\}", text)
    if m:
        return m[-1].replace(",", "").strip()
    # 3. 'answer is ...'
    m = re.findall(r"(?:the answer is|The answer is|is|equals|=)\s*[$]?\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m[-1].replace(",", "").strip()
    # 4. Trailing number
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def normalize_number(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        val = float(s)
        return int(val) if val.is_integer() else val
    except ValueError:
        return None


def evaluate_gsm8k_reasoning(output_text: str, ground_truth_text: str) -> Dict[str, Any]:
    """Evaluate GSM8K final answer exact match."""
    gt_match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ground_truth_text)
    gt_str = gt_match.group(1).replace(",", "").strip() if gt_match else None
    gt_num = normalize_number(gt_str)

    ext_str = extract_gsm_answer(output_text)
    ext_num = normalize_number(ext_str)

    exact_correct = (ext_num is not None) and (gt_num is not None) and (ext_num == gt_num)
    return {
        "ground_truth": gt_str,
        "extracted_answer": ext_str,
        "exact_correct": exact_correct,
        "extraction_failure": ext_num is None,
    }


def severe_repetition_metrics(output_text: str) -> dict[str, Any]:
    """Whitespace-token trigram repetition diagnostic; not a semantic score."""
    words = output_text.casefold().split()
    trigrams = [tuple(words[i : i + 3]) for i in range(max(0, len(words) - 2))]
    counts = Counter(trigrams)
    max_count = max(counts.values(), default=0)
    return {
        "severe_repetition_detected": max_count >= 4,
        "max_trigram_repetitions": max_count,
    }


def evaluate_alpaca_instruction(output_text: str, eos_reached: bool = False) -> Dict[str, Any]:
    """Mechanical failure checks only; does not estimate semantic adherence."""
    text_stripped = output_text.strip()
    words = text_stripped.split()

    empty_output = len(text_stripped) == 0
    very_short = len(words) < 5
    repetition = severe_repetition_metrics(text_stripped)
    trigram_repetition = repetition["severe_repetition_detected"]

    # Malformed unicode (>30% non-ascii)
    non_ascii_count = sum(1 for c in text_stripped if ord(c) > 127)
    malformed = (non_ascii_count / max(len(text_stripped), 1)) > 0.30

    reasons = []
    if empty_output:
        reasons.append("empty_output")
    if very_short:
        reasons.append("very_short")
    if trigram_repetition:
        reasons.append("severe_repetition")
    if malformed:
        reasons.append("malformed_unicode")
    mechanical_failure = bool(reasons)

    return {
        "empty_output": empty_output,
        "very_short": very_short,
        **repetition,
        "malformed": malformed,
        "mechanical_instruction_failure": mechanical_failure,
        "mechanical_instruction_pass": not mechanical_failure,
        "mechanical_instruction_failure_reasons": reasons,
        "word_count": len(words),
        "char_count": len(text_stripped),
    }


def sequence_reached_eos(generated_ids: List[int], eos_token_id: Optional[int]) -> bool:
    """True only when the last generated token is the tokenizer EOS id."""
    if not generated_ids or eos_token_id is None:
        return False
    return generated_ids[-1] == eos_token_id


def generation_health_fields(
    output_text: str,
    generated_ids: List[int],
    eos_token_id: Optional[int],
    max_new_tokens: int,
    response_length_base_tokens: int,
) -> dict[str, Any]:
    eos_reached = sequence_reached_eos(generated_ids, eos_token_id)
    hit_max_length = max_new_tokens > 0 and len(generated_ids) >= max_new_tokens
    return {
        "eos_reached": eos_reached,
        "hit_max_length": hit_max_length,
        "truncated": hit_max_length and not eos_reached,
        **severe_repetition_metrics(output_text),
        "response_length_base_tokens": int(response_length_base_tokens),
        "response_length_chars": len(output_text),
    }


def prepare_prompt_input_ids(
    base_prompt_ids: List[int],
    static_mgr: StaticCodebookManager,
    compress_prompt: bool,
) -> List[int]:
    """Prepare a raw or seeded-codebook prompt and prove compression is lossless."""
    if not compress_prompt:
        return list(base_prompt_ids)
    compressed_ids = static_mgr.segment_sequence(base_prompt_ids)
    if static_mgr.decode_sequence(compressed_ids) != list(base_prompt_ids):
        raise ValueError("Compressed prompt does not round-trip to its original token sequence")
    return compressed_ids


# ==============================================================================
# Model Condition Evaluators
# ==============================================================================

def run_condition_original_phi(
    samples: List[Dict[str, Any]],
    raw_results_path: str,
    completed_keys: Set[Tuple[str, str]],
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = "cpu",
    base_revision: Optional[str] = None,
    generation_cache_keys: Optional[Mapping[Tuple[str, str], str]] = None,
    cache_root: Optional[str] = None,
    tested_commit: Optional[str] = None,
) -> List[Dict[str, Any]]:
    device = torch.device(device)
    condition = "original_phi"
    print(f"\n{'='*80}\nRUNNING CONDITION A: ORIGINAL PHI (Vanilla Phi-3.5-mini-instruct)\n{'='*80}")

    pending_samples = [
        sample for sample in samples if (sample["id"], condition) not in completed_keys
    ]
    if not pending_samples:
        print("All original-Phi records have exact cache hits; model load skipped.", flush=True)
        return []

    tok = AutoTokenizer.from_pretrained(PHI_MODEL_ID, **_revision_kwargs(base_revision))
    model = AutoModelForCausalLM.from_pretrained(
        PHI_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **_revision_kwargs(base_revision),
    )
    model.to(device)
    model.eval()

    results = []
    for idx, s in enumerate(pending_samples, 1):
        prompt_id = s["id"]
        key = (prompt_id, condition)
        if key in completed_keys:
            print(f"[{idx}/{len(samples)}] Skipping already completed {prompt_id} on {condition}")
            continue

        prompt_text = build_mbpp_prompt(s) if s["domain"] == "code" else s["prompt"]
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        base_prompt_len = len(prompt_ids)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        synchronize_device(device)
        t_start = time.perf_counter()
        timing_proc = TimingLogitsProcessor(t_start)
        proc_list = LogitsProcessorList([timing_proc])

        with torch.no_grad():
            out = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                logits_processor=proc_list,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        synchronize_device(device)
        t_total = time.perf_counter() - t_start

        gen_ids = out[0, base_prompt_len:].tolist()
        decode_steps = len(gen_ids)
        expanded_output_tokens = len(gen_ids)
        eos_reached = sequence_reached_eos(gen_ids, tok.eos_token_id)
        hit_max_length = decode_steps >= max_new_tokens

        output_text = tok.decode(gen_ids, skip_special_tokens=True)
        ttft = timing_proc.ttft or 0.0
        decode_time = max(0.0, t_total - ttft)

        rec = {
            "prompt_id": prompt_id,
            "domain": s["domain"],
            "condition": condition,
            "base_prompt_tokens": base_prompt_len,
            "model_prefill_tokens": base_prompt_len,
            "prompt_compression_pct": 0.0,
            "decode_steps": decode_steps,
            "expanded_output_tokens": expanded_output_tokens,
            "tokens_saved": 0,
            "decode_reduction_pct": 0.0,
            "wall_time_s": round(t_total, 3),
            "ttft_s": round(ttft, 3),
            "prefill_time_s": round(ttft, 3),
            "decode_time_s": round(decode_time, 3),
            "predictor_time_s": 0.0,
            "codebook_time_s": 0.0,
            "hyper_setup_time_s": 0.0,
            "throughput_tok_per_s": round(expanded_output_tokens / max(t_total, 0.001), 2),
            "eos_reached": eos_reached,
            "hit_max_length": hit_max_length,
            "hypertokens_count": 0,
            "hypertokens_emitted": [],
            "first_hypertoken_pos": -1,
            "codebook_size": 0,
            "output_text": output_text,
            "process_rss_gb": get_process_rss_gb(),
        }
        rec.update(
            generation_health_fields(
                output_text, gen_ids, tok.eos_token_id, max_new_tokens, expanded_output_tokens
            )
        )

        # Domain quality evaluation
        dom = s["domain"]
        if dom == "code":
            asserts = [line.strip() for line in s["ground_truth_response"].splitlines() if line.strip().startswith("assert")]
            code_eval = evaluate_mbpp_code(output_text, asserts)
            rec.update(code_eval)
        elif dom == "reasoning":
            gsm_eval = evaluate_gsm8k_reasoning(output_text, s["ground_truth_response"])
            rec.update(gsm_eval)
        elif dom == "instruction":
            alp_eval = evaluate_alpaca_instruction(output_text, eos_reached)
            rec.update(alp_eval)

        results.append(rec)
        completed_keys.add(key)
        _write_generation_record(
            rec, raw_results_path, generation_cache_keys, cache_root, tested_commit
        )

        print(
            f"[{idx:2d}/{len(samples)}] {prompt_id:10s} | {dom:11s} | Dec: {decode_steps:3d} steps | "
            f"Time: {rec['wall_time_s']:5.2f}s | TTFT: {rec['ttft_s']:4.2f}s | "
            f"Pass/Acc: {rec.get('problem_pass', rec.get('exact_correct', rec.get('mechanical_instruction_pass')))}",
            flush=True,
        )

    del model
    del tok
    gc.collect()
    return results


def run_condition_official_zip2zip(
    samples: List[Dict[str, Any]],
    raw_results_path: str,
    completed_keys: Set[Tuple[str, str]],
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = "cpu",
    base_revision: Optional[str] = None,
    model_revision: Optional[str] = None,
    generation_cache_keys: Optional[Mapping[Tuple[str, str], str]] = None,
    cache_root: Optional[str] = None,
    tested_commit: Optional[str] = None,
) -> List[Dict[str, Any]]:
    device = torch.device(device)
    condition = "official_zip2zip"
    print(f"\n{'='*80}\nRUNNING CONDITION B: OFFICIAL REACTIVE ZIP2ZIP (Native LZW Path)\n{'='*80}")

    pending_samples = [
        sample for sample in samples if (sample["id"], condition) not in completed_keys
    ]
    if not pending_samples:
        print("All Official Zip2Zip records have exact cache hits; model load skipped.", flush=True)
        return []

    model_id = ZIP2ZIP_MODEL_ID
    tok = Zip2ZipTokenizer.from_pretrained(model_id, **_revision_kwargs(model_revision))
    base_tok = AutoTokenizer.from_pretrained(PHI_MODEL_ID, **_revision_kwargs(base_revision))
    model = _load_zip2zip_model(model_id, base_revision, model_revision)
    model.to(device)
    model.eval()

    results = []
    for idx, s in enumerate(pending_samples, 1):
        prompt_id = s["id"]
        key = (prompt_id, condition)
        if key in completed_keys:
            print(f"[{idx}/{len(samples)}] Skipping already completed {prompt_id} on {condition}")
            continue

        prompt_text = build_mbpp_prompt(s) if s["domain"] == "code" else s["prompt"]
        base_prompt_ids = base_tok.encode(prompt_text, add_special_tokens=False)
        base_prompt_len = len(base_prompt_ids)

        inputs = tok(prompt_text, return_tensors="pt").to(device)
        prefill_len = inputs["input_ids"].shape[1]
        prompt_compression_pct = round((1.0 - prefill_len / max(base_prompt_len, 1)) * 100, 2)

        synchronize_device(device)
        t_start = time.perf_counter()
        timing_proc = TimingLogitsProcessor(t_start)
        proc_list = LogitsProcessorList([timing_proc])

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                logits_processor=proc_list,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        synchronize_device(device)
        t_total = time.perf_counter() - t_start

        full_ids = out[0].tolist()
        gen_ids = out[0, prefill_len:].tolist()
        decode_steps = len(gen_ids)

        # Decompress full generated sequence using official LZW decoder
        try:
            full_base_ids, _ = tok._lzw_decode([full_ids])[0]
            generated_base_ids = full_base_ids[base_prompt_len:]
            output_text = base_tok.decode(generated_base_ids, skip_special_tokens=True)
            expanded_output_tokens = len(generated_base_ids)
        except Exception:
            # Fallback in case of decoding edge case
            full_text = tok.decode(full_ids, skip_special_tokens=True)
            prompt_decoded = tok.decode(inputs["input_ids"][0].tolist(), skip_special_tokens=True)
            output_text = full_text[len(prompt_decoded):] if full_text.startswith(prompt_decoded) else full_text
            expanded_base_ids = base_tok.encode(output_text, add_special_tokens=False)
            expanded_output_tokens = len(expanded_base_ids)

        tokens_saved = max(0, expanded_output_tokens - decode_steps)
        decode_reduction_pct = round((1.0 - decode_steps / max(expanded_output_tokens, 1)) * 100, 2) if expanded_output_tokens > decode_steps else 0.0

        eos_reached = sequence_reached_eos(gen_ids, tok.eos_token_id)
        hit_max_length = decode_steps >= max_new_tokens

        # Check emitted dynamic tokens
        hypertokens = []
        first_hyper_pos = -1
        for pos, tid in enumerate(gen_ids):
            if tid >= INITIAL_VOCAB:
                if first_hyper_pos == -1:
                    first_hyper_pos = pos
                hypertokens.append({"pos": pos, "id": tid})

        ttft = timing_proc.ttft or 0.0
        decode_time = max(0.0, t_total - ttft)

        model.codebook_manager.reset()

        rec = {
            "prompt_id": prompt_id,
            "domain": s["domain"],
            "condition": condition,
            "base_prompt_tokens": base_prompt_len,
            "model_prefill_tokens": prefill_len,
            "prompt_compression_pct": prompt_compression_pct,
            "decode_steps": decode_steps,
            "expanded_output_tokens": expanded_output_tokens,
            "tokens_saved": tokens_saved,
            "decode_reduction_pct": decode_reduction_pct,
            "wall_time_s": round(t_total, 3),
            "ttft_s": round(ttft, 3),
            "prefill_time_s": round(ttft, 3),
            "decode_time_s": round(decode_time, 3),
            "predictor_time_s": 0.0,
            "codebook_time_s": 0.0,
            "hyper_setup_time_s": 0.0,
            "throughput_tok_per_s": round(expanded_output_tokens / max(t_total, 0.001), 2),
            "eos_reached": eos_reached,
            "hit_max_length": hit_max_length,
            "hypertokens_count": len(hypertokens),
            "hypertokens_emitted": hypertokens,
            "first_hypertoken_pos": first_hyper_pos,
            "codebook_size": model.zip2zip_config.compression.max_codebook_size,
            "output_text": output_text,
            "process_rss_gb": get_process_rss_gb(),
        }
        rec.update(
            generation_health_fields(
                output_text, gen_ids, tok.eos_token_id, max_new_tokens, expanded_output_tokens
            )
        )

        dom = s["domain"]
        if dom == "code":
            asserts = [line.strip() for line in s["ground_truth_response"].splitlines() if line.strip().startswith("assert")]
            code_eval = evaluate_mbpp_code(output_text, asserts)
            rec.update(code_eval)
        elif dom == "reasoning":
            gsm_eval = evaluate_gsm8k_reasoning(output_text, s["ground_truth_response"])
            rec.update(gsm_eval)
        elif dom == "instruction":
            alp_eval = evaluate_alpaca_instruction(output_text, eos_reached)
            rec.update(alp_eval)

        results.append(rec)
        completed_keys.add(key)
        _write_generation_record(
            rec, raw_results_path, generation_cache_keys, cache_root, tested_commit
        )

        print(
            f"[{idx:2d}/{len(samples)}] {prompt_id:10s} | {dom:11s} | Dec: {decode_steps:3d} -> Exp: {expanded_output_tokens:3d} "
            f"(Saved: {tokens_saved:2d}) | Time: {rec['wall_time_s']:5.2f}s | "
            f"Pass/Acc: {rec.get('problem_pass', rec.get('exact_correct', rec.get('mechanical_instruction_pass')))}",
            flush=True,
        )

    del model
    del tok
    del base_tok
    gc.collect()
    return results


def run_condition_predictive(
    checkpoint_path: str,
    condition_name: str,
    samples: List[Dict[str, Any]],
    raw_results_path: str,
    completed_keys: Set[Tuple[str, str]],
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = "cpu",
    base_revision: Optional[str] = None,
    model_revision: Optional[str] = None,
    generation_cache_keys: Optional[Mapping[Tuple[str, str], str]] = None,
    cache_root: Optional[str] = None,
    tested_commit: Optional[str] = None,
    compress_prompt: bool = False,
    model_bundle: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    device = torch.device(device)
    print(f"\n{'='*80}\nRUNNING CONDITION {condition_name.upper()} ({checkpoint_path})\n{'='*80}")
    pending_samples = [
        sample for sample in samples if (sample["id"], condition_name) not in completed_keys
    ]
    if not pending_samples:
        print(f"All {condition_name} records have exact cache hits; model load skipped.", flush=True)
        return []

    owns_bundle = model_bundle is None
    bundle = model_bundle or load_predictive_model_bundle(
        checkpoint_path,
        device=str(device),
        base_revision=base_revision,
        model_revision=model_revision,
        expected_step=(
            100 if "predictive_step_100" in condition_name
            else 150 if "predictive_step_150" in condition_name
            else None
        ),
    )
    model = bundle["model"]
    tok = bundle["tokenizer"]
    policy = bundle["policy"]
    dim = bundle["embedding_dim"]
    pad_id = bundle["pad_id"]
    disabled_ids = bundle["disabled_ids"]

    results = []
    for idx, s in enumerate(pending_samples, 1):
        prompt_id = s["id"]
        key = (prompt_id, condition_name)
        if key in completed_keys:
            print(f"[{idx}/{len(samples)}] Skipping already completed {prompt_id} on {condition_name}")
            continue

        prompt_text = build_mbpp_prompt(s) if s["domain"] == "code" else s["prompt"]
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        base_prompt_len = len(prompt_ids)

        # 1. Predictor timing
        t_pred_start = time.perf_counter()
        codebook_dict, _ = policy.select_codebook(prompt_ids)
        predictor_time_s = time.perf_counter() - t_pred_start

        # 2. Hypertoken setup timing
        t_setup_start = time.perf_counter()
        static_mgr = StaticCodebookManager(
            initial_vocab_size=INITIAL_VOCAB,
            max_codebook_size=32,
            max_subtokens=4,
            embedding_dim=dim,
            pad_token_id=pad_id,
            disabled_ids=disabled_ids,
        )
        static_mgr.set_seeded_codebook(codebook_dict, batch_size=1, device=device)
        static_mgr.attach_to_model(model)
        hyper_setup_time_s = time.perf_counter() - t_setup_start
        codebook_time_s = predictor_time_s + hyper_setup_time_s
        codebook_sha256 = hashlib.sha256(
            repr(sorted(codebook_dict.items())).encode("utf-8")
        ).hexdigest()

        model_prompt_ids = prepare_prompt_input_ids(
            prompt_ids, static_mgr, compress_prompt=compress_prompt
        )
        model_prompt_len = len(model_prompt_ids)
        input_tensor = torch.tensor([model_prompt_ids], dtype=torch.long, device=device)

        # 3. Generation timing
        synchronize_device(device)
        t_gen_start = time.perf_counter()
        timing_proc = TimingLogitsProcessor(t_gen_start, static_mgr=static_mgr)
        proc_list = LogitsProcessorList([timing_proc])

        with torch.no_grad():
            out = model.generate(
                input_ids=input_tensor,
                max_new_tokens=max_new_tokens,
                logits_processor=proc_list,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        synchronize_device(device)
        t_gen = time.perf_counter() - t_gen_start
        total_wall_time = codebook_time_s + t_gen

        gen_ids = out[0, model_prompt_len:].tolist()
        decode_steps = len(gen_ids)

        # 4. Decompress / expand hypertokens to constituent base tokens
        hyper_to_tokens = {v: list(k) for k, v in codebook_dict.items()}
        expanded_tokens = []
        hypertokens_emitted = []
        first_hyper_pos = -1

        for pos, tid in enumerate(gen_ids):
            if tid in hyper_to_tokens:
                if first_hyper_pos == -1:
                    first_hyper_pos = pos
                phrase_str = tok.decode(hyper_to_tokens[tid])
                hypertokens_emitted.append({"pos": pos, "id": tid, "phrase": phrase_str, "subtokens": hyper_to_tokens[tid]})
                expanded_tokens.extend(hyper_to_tokens[tid])
            else:
                expanded_tokens.append(tid)

        output_text = tok.decode(expanded_tokens, skip_special_tokens=True)
        expanded_output_tokens = len(expanded_tokens)

        tokens_saved = max(0, expanded_output_tokens - decode_steps)
        decode_reduction_pct = round((1.0 - decode_steps / max(expanded_output_tokens, 1)) * 100, 2) if expanded_output_tokens > decode_steps else 0.0

        eos_reached = sequence_reached_eos(gen_ids, tok.eos_token_id)
        hit_max_length = decode_steps >= max_new_tokens

        ttft = timing_proc.ttft or 0.0
        decode_time = max(0.0, t_gen - ttft)

        static_mgr.detach_from_model(model)
        model.codebook_manager.reset()

        rec = {
            "prompt_id": prompt_id,
            "domain": s["domain"],
            "condition": condition_name,
            "base_prompt_tokens": base_prompt_len,
            "model_prefill_tokens": model_prompt_len,
            "prompt_compression_pct": round(
                100.0 * (1 - model_prompt_len / max(base_prompt_len, 1)), 2
            ),
            "prompt_representation": "predictive_compressed" if compress_prompt else "raw",
            "codebook_sha256": codebook_sha256,
            "decode_steps": decode_steps,
            "expanded_output_tokens": expanded_output_tokens,
            "tokens_saved": tokens_saved,
            "decode_reduction_pct": decode_reduction_pct,
            "wall_time_s": round(total_wall_time, 3),
            "ttft_s": round(ttft, 3),
            "prefill_time_s": round(ttft, 3),
            "decode_time_s": round(decode_time, 3),
            "predictor_time_s": round(predictor_time_s, 4),
            "codebook_time_s": round(codebook_time_s, 4),
            "hyper_setup_time_s": round(hyper_setup_time_s, 4),
            "throughput_tok_per_s": round(expanded_output_tokens / max(total_wall_time, 0.001), 2),
            "eos_reached": eos_reached,
            "hit_max_length": hit_max_length,
            "hypertokens_count": len(hypertokens_emitted),
            "hypertokens_emitted": hypertokens_emitted,
            "first_hypertoken_pos": first_hyper_pos,
            "codebook_size": len(codebook_dict),
            "output_text": output_text,
            "process_rss_gb": get_process_rss_gb(),
        }
        rec.update(
            generation_health_fields(
                output_text, gen_ids, tok.eos_token_id, max_new_tokens, expanded_output_tokens
            )
        )

        dom = s["domain"]
        if dom == "code":
            asserts = [line.strip() for line in s["ground_truth_response"].splitlines() if line.strip().startswith("assert")]
            code_eval = evaluate_mbpp_code(output_text, asserts)
            rec.update(code_eval)
        elif dom == "reasoning":
            gsm_eval = evaluate_gsm8k_reasoning(output_text, s["ground_truth_response"])
            rec.update(gsm_eval)
        elif dom == "instruction":
            alp_eval = evaluate_alpaca_instruction(output_text, eos_reached)
            rec.update(alp_eval)

        results.append(rec)
        completed_keys.add(key)
        _write_generation_record(
            rec, raw_results_path, generation_cache_keys, cache_root, tested_commit
        )

        print(
            f"[{idx:2d}/{len(samples)}] {prompt_id:10s} | {dom:11s} | Dec: {decode_steps:3d} -> Exp: {expanded_output_tokens:3d} "
            f"(Saved: {tokens_saved:2d}, Hypers: {len(hypertokens_emitted):2d}) | Time: {rec['wall_time_s']:5.2f}s | "
            f"Pass/Acc: {rec.get('problem_pass', rec.get('exact_correct', rec.get('mechanical_instruction_pass')))}",
            flush=True,
        )

    if owns_bundle:
        del model
        del tok
        gc.collect()
    return results


# ==============================================================================
# Aggregate & Paired Analytics Generation
# ==============================================================================

def generate_full_benchmark_analytics(output_dir: str, raw_results_path: str):
    print(f"\n{'='*80}\nCOMPUTING BENCHMARK AGGREGATES, PAIRED DELTAS, & REPORTS\n{'='*80}")

    with open(raw_results_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    stale = [
        record.get("prompt_id", "<unknown>")
        for record in records
        if record.get("record_schema") != GENERATION_RECORD_SCHEMA
        or record.get("evaluator_version") != EVALUATOR_VERSION
        or record.get("prompt_formatter_version") != PROMPT_FORMATTER_VERSION
    ]
    if stale:
        raise ValueError(
            "Raw results contain records from a different evaluator or formatter version; "
            f"refusing to mix them (first mismatched prompt: {stale[0]})."
        )

    conditions = ["original_phi", "official_zip2zip", "predictive_step_100", "predictive_step_150"]
    domains = ["code", "reasoning", "instruction"]

    by_cond_dom: Dict[str, Dict[str, List[Dict[str, Any]]]] = {c: {d: [] for d in domains} for c in conditions}
    by_prompt_cond: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for r in records:
        c = r["condition"]
        d = r["domain"]
        p = r["prompt_id"]
        if c in by_cond_dom and d in by_cond_dom[c]:
            by_cond_dom[c][d].append(r)
        if p not in by_prompt_cond:
            by_prompt_cond[p] = {}
        by_prompt_cond[p][c] = r

    # 1. Domain Aggregate Table
    aggregates: Dict[str, Any] = {}
    for c in conditions:
        aggregates[c] = {}
        all_recs = [r for d in domains for r in by_cond_dom[c][d]]

        # Code Pass@1
        code_recs = by_cond_dom[c]["code"]
        code_passed = sum(1 for r in code_recs if r.get("problem_pass", False))
        code_syntax = sum(1 for r in code_recs if r.get("syntax_valid", False))

        # GSM8k Accuracy
        gsm_recs = by_cond_dom[c]["reasoning"]
        gsm_correct = sum(1 for r in gsm_recs if r.get("exact_correct", False))

        # Alpaca Failure Rate
        alp_recs = by_cond_dom[c]["instruction"]
        alp_failed = sum(
            1 for r in alp_recs if r.get("mechanical_instruction_failure", False)
        )

        # Performance & Economics
        tot_decode_steps = sum(r["decode_steps"] for r in all_recs)
        tot_expanded_tokens = sum(r["expanded_output_tokens"] for r in all_recs)
        tot_tokens_saved = sum(r["tokens_saved"] for r in all_recs)
        tot_wall_time = sum(r["wall_time_s"] for r in all_recs)
        tot_prefill_time = sum(r["prefill_time_s"] for r in all_recs)
        tot_decode_time = sum(r["decode_time_s"] for r in all_recs)

        wall_times = [r["wall_time_s"] for r in all_recs]
        ttfts = [r["ttft_s"] for r in all_recs]
        throughputs = [r["throughput_tok_per_s"] for r in all_recs]
        hypers_emitted = [r["hypertokens_count"] for r in all_recs]

        aggregates[c] = {
            "code_pass_rate": round(code_passed / max(len(code_recs), 1), 4),
            "code_pass_count": f"{code_passed}/{len(code_recs)}",
            "code_syntax_valid_rate": round(code_syntax / max(len(code_recs), 1), 4),
            "gsm8k_accuracy": round(gsm_correct / max(len(gsm_recs), 1), 4),
            "gsm8k_correct_count": f"{gsm_correct}/{len(gsm_recs)}",
            "alpaca_mechanical_failure_rate": round(alp_failed / max(len(alp_recs), 1), 4),
            "alpaca_mechanical_failure_count": f"{alp_failed}/{len(alp_recs)}",
            "semantic_instruction_adherence_available": False,
            "total_decode_steps": tot_decode_steps,
            "total_expanded_tokens": tot_expanded_tokens,
            "total_tokens_saved": tot_tokens_saved,
            "micro_decode_reduction_pct": round((tot_tokens_saved / max(tot_expanded_tokens, 1)) * 100, 2),
            "macro_decode_reduction_pct": round(sum(r["decode_reduction_pct"] for r in all_recs) / max(len(all_recs), 1), 2),
            "total_wall_time_s": round(tot_wall_time, 2),
            "median_wall_time_s": round(float(torch.tensor(wall_times).median().item()), 2),
            "mean_wall_time_s": round(sum(wall_times) / max(len(wall_times), 1), 2),
            "mean_ttft_s": round(sum(ttfts) / max(len(ttfts), 1), 2),
            "mean_throughput_tok_per_s": round(sum(throughputs) / max(len(throughputs), 1), 2),
            "total_hypertokens_emitted": sum(hypers_emitted),
            "mean_hypertokens_per_output": round(sum(hypers_emitted) / max(len(hypers_emitted), 1), 2),
            "eos_count": sum(1 for r in all_recs if r.get("eos_reached", False)),
            "hit_generation_cap_count": sum(1 for r in all_recs if r.get("hit_max_length", False)),
            "truncation_count": sum(1 for r in all_recs if r.get("truncated", False)),
            "repetition_count": sum(1 for r in all_recs if r.get("severe_repetition_detected", False)),
            "mean_response_length_base_tokens": round(
                sum(r.get("response_length_base_tokens", 0) for r in all_recs)
                / max(len(all_recs), 1),
                2,
            ),
            "mean_response_length_chars": round(
                sum(r.get("response_length_chars", 0) for r in all_recs)
                / max(len(all_recs), 1),
                1,
            ),
        }

    with open(os.path.join(output_dir, "aggregate_results.json"), "w", encoding="utf-8") as f:
        json.dump(aggregates, f, indent=2)

    # 2. Paired Quality Deltas & Transition Tables
    paired_deltas: Dict[str, Any] = {"code": {}, "reasoning": {}, "instruction": {}}

    for dom in ["code", "reasoning"]:
        for comp in ["official_zip2zip", "predictive_step_100", "predictive_step_150"]:
            tt = {"both_correct": 0, "orig_correct_cand_wrong": 0, "orig_wrong_cand_correct": 0, "both_wrong": 0}
            sample_ids = [r["id"] for r in json.load(open(VAL_DATA_PATH)) if r["domain"] == dom]
            for pid in sample_ids:
                if pid in by_prompt_cond and "original_phi" in by_prompt_cond[pid] and comp in by_prompt_cond[pid]:
                    o_rec = by_prompt_cond[pid]["original_phi"]
                    c_rec = by_prompt_cond[pid][comp]
                    o_pass = o_rec.get("problem_pass", o_rec.get("exact_correct", False))
                    c_pass = c_rec.get("problem_pass", c_rec.get("exact_correct", False))
                    if o_pass and c_pass:
                        tt["both_correct"] += 1
                    elif o_pass and not c_pass:
                        tt["orig_correct_cand_wrong"] += 1
                    elif not o_pass and c_pass:
                        tt["orig_wrong_cand_correct"] += 1
                    else:
                        tt["both_wrong"] += 1
            paired_deltas[dom][comp] = tt

    with open(os.path.join(output_dir, "paired_quality_deltas.json"), "w", encoding="utf-8") as f:
        json.dump(paired_deltas, f, indent=2)

    # 3. Quality-Conditional Compression
    qc_data: Dict[str, Any] = {}
    for c in ["official_zip2zip", "predictive_step_100", "predictive_step_150"]:
        qc_data[c] = {}
        for dom in ["code", "reasoning"]:
            sample_ids = [r["id"] for r in json.load(open(VAL_DATA_PATH)) if r["domain"] == dom]
            preserved_steps = 0
            preserved_expanded = 0
            regressed_steps = 0
            regressed_expanded = 0

            for pid in sample_ids:
                if pid in by_prompt_cond and "original_phi" in by_prompt_cond[pid] and c in by_prompt_cond[pid]:
                    o_rec = by_prompt_cond[pid]["original_phi"]
                    c_rec = by_prompt_cond[pid][c]
                    o_pass = o_rec.get("problem_pass", o_rec.get("exact_correct", False))
                    c_pass = c_rec.get("problem_pass", c_rec.get("exact_correct", False))

                    if c_pass:
                        preserved_steps += c_rec["decode_steps"]
                        preserved_expanded += c_rec["expanded_output_tokens"]
                    if o_pass and not c_pass:
                        regressed_steps += c_rec["decode_steps"]
                        regressed_expanded += c_rec["expanded_output_tokens"]

            qc_data[c][dom] = {
                "preserved_decode_reduction_pct": round((1.0 - preserved_steps / max(preserved_expanded, 1)) * 100, 2) if preserved_expanded > preserved_steps else 0.0,
                "regressed_decode_reduction_pct": round((1.0 - regressed_steps / max(regressed_expanded, 1)) * 100, 2) if regressed_expanded > regressed_steps else 0.0,
                "quality_preserved_realized_savings_tokens": max(0, preserved_expanded - preserved_steps),
                "quality_preserved_expanded_tokens": preserved_expanded,
            }

    with open(os.path.join(output_dir, "quality_vs_compression.json"), "w", encoding="utf-8") as f:
        json.dump(qc_data, f, indent=2)

    # 4. Detailed Failure Case Review
    failures = []
    val_data = json.load(open(VAL_DATA_PATH))
    for s in val_data:
        pid = s["id"]
        dom = s["domain"]
        if dom not in ["code", "reasoning"]:
            continue
        if pid in by_prompt_cond and "original_phi" in by_prompt_cond[pid]:
            o_rec = by_prompt_cond[pid]["original_phi"]
            o_pass = o_rec.get("problem_pass", o_rec.get("exact_correct", False))
            if not o_pass:
                continue  # Only analyze regressions where original was correct

            # Check if any candidate failed
            failed_candidates = []
            for cand in ["official_zip2zip", "predictive_step_100", "predictive_step_150"]:
                if cand in by_prompt_cond[pid]:
                    c_rec = by_prompt_cond[pid][cand]
                    c_pass = c_rec.get("problem_pass", c_rec.get("exact_correct", False))
                    if not c_pass:
                        failed_candidates.append(cand)

            if failed_candidates:
                f_entry = {
                    "prompt_id": pid,
                    "domain": dom,
                    "prompt": s["prompt"],
                    "ground_truth": s["ground_truth_response"][:300],
                    "failed_models": failed_candidates,
                    "original_output": o_rec["output_text"][:300],
                    "official_zip2zip_output": by_prompt_cond[pid].get("official_zip2zip", {}).get("output_text", "")[:300],
                    "predictive_100_output": by_prompt_cond[pid].get("predictive_step_100", {}).get("output_text", "")[:300],
                    "predictive_150_output": by_prompt_cond[pid].get("predictive_step_150", {}).get("output_text", "")[:300],
                    "predictive_100_hypers": by_prompt_cond[pid].get("predictive_step_100", {}).get("hypertokens_emitted", []),
                    "predictive_150_hypers": by_prompt_cond[pid].get("predictive_step_150", {}).get("hypertokens_emitted", []),
                    "taxonomy": "logic_or_semantic_drift",
                }
                failures.append(f_entry)

    with open(os.path.join(output_dir, "failure_cases.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    # 5. Instruction Blind Review Artifact
    blind_reviews = []
    blind_key = {}
    alpaca_samples = [s for s in val_data if s["domain"] == "instruction"]

    for s in alpaca_samples:
        pid = s["id"]
        cond_keys = ["original_phi", "official_zip2zip", "predictive_step_100", "predictive_step_150"]
        random.shuffle(cond_keys)
        labeled_outputs = {}
        prompt_key = {}
        for idx, ck in enumerate(cond_keys):
            label = f"Model_{chr(ord('A') + idx)}"
            labeled_outputs[label] = by_prompt_cond.get(pid, {}).get(ck, {}).get("output_text", "")
            prompt_key[label] = ck

        blind_reviews.append({
            "prompt_id": pid,
            "prompt": s["prompt"],
            "outputs": labeled_outputs,
        })
        blind_key[pid] = prompt_key

    with open(os.path.join(output_dir, "instruction_blind_review.json"), "w", encoding="utf-8") as f:
        json.dump(blind_reviews, f, indent=2)
    with open(os.path.join(output_dir, "instruction_blind_review_key.json"), "w", encoding="utf-8") as f:
        json.dump(blind_key, f, indent=2)

    # 6. Comprehensive Markdown Report
    report_path = os.path.join(output_dir, "benchmark_report.md")
    write_benchmark_markdown_report(report_path, aggregates, paired_deltas, qc_data, failures)
    print(f"\nAll benchmark analytics saved to {output_dir}")


def write_benchmark_markdown_report(
    report_path: str,
    aggregates: Dict[str, Any],
    paired_deltas: Dict[str, Any],
    qc_data: Dict[str, Any],
    failures: List[Dict[str, Any]],
):
    orig = aggregates.get("original_phi", {})
    off = aggregates.get("official_zip2zip", {})
    p100 = aggregates.get("predictive_step_100", {})
    p150 = aggregates.get("predictive_step_150", {})

    def pct_str(val: Optional[float]) -> str:
        return f"{val * 100:.1f}%" if val is not None else "N/A"

    def num_str(val: Optional[float], decimals: int = 2) -> str:
        return f"{val:.{decimals}f}" if val is not None else "N/A"

    # Quality Deltas
    orig_code = orig.get("code_pass_rate", 0)
    off_code = off.get("code_pass_rate", 0)
    p100_code = p100.get("code_pass_rate", 0)
    p150_code = p150.get("code_pass_rate", 0)

    orig_gsm = orig.get("gsm8k_accuracy", 0)
    off_gsm = off.get("gsm8k_accuracy", 0)
    p100_gsm = p100.get("gsm8k_accuracy", 0)
    p150_gsm = p150.get("gsm8k_accuracy", 0)

    orig_alp = orig.get("alpaca_mechanical_failure_rate", 0)
    off_alp = off.get("alpaca_mechanical_failure_rate", 0)
    p100_alp = p100.get("alpaca_mechanical_failure_rate", 0)
    p150_alp = p150.get("alpaca_mechanical_failure_rate", 0)

    lines = [
        "# Quality & Compute Economics Benchmark Report",
        "",
        "Definitive held-out benchmark evaluating Original Phi, Official Reactive Zip2Zip, and Our Predictive Zip2Zip (Steps 100 & 150) across 60 frozen validation samples (20 MBPP code, 20 GSM8K reasoning, 20 Alpaca instruction).",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Original Phi | Official Zip2Zip | Predictive Step 100 | Predictive Step 150 |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **MBPP Code Pass@1** | **{pct_str(orig_code)}** ({orig.get('code_pass_count')}) | {pct_str(off_code)} ({off.get('code_pass_count')}) | {pct_str(p100_code)} ({p100.get('code_pass_count')}) | {pct_str(p150_code)} ({p150.get('code_pass_count')}) |",
        f"| **GSM8K Accuracy** | **{pct_str(orig_gsm)}** ({orig.get('gsm8k_correct_count')}) | {pct_str(off_gsm)} ({off.get('gsm8k_correct_count')}) | {pct_str(p100_gsm)} ({p100.get('gsm8k_correct_count')}) | {pct_str(p150_gsm)} ({p150.get('gsm8k_correct_count')}) |",
        f"| **Alpaca Mechanical Failure Rate** | **{pct_str(orig_alp)}** ({orig.get('alpaca_mechanical_failure_count')}) | {pct_str(off_alp)} ({off.get('alpaca_mechanical_failure_count')}) | {pct_str(p100_alp)} ({p100.get('alpaca_mechanical_failure_count')}) | {pct_str(p150_alp)} ({p150.get('alpaca_mechanical_failure_count')}) |",
        f"| **Micro Decode Reduction** | 0.0% | {num_str(off.get('micro_decode_reduction_pct'))}% | {num_str(p100.get('micro_decode_reduction_pct'))}% | {num_str(p150.get('micro_decode_reduction_pct'))}% |",
        f"| **Macro Decode Reduction** | 0.0% | {num_str(off.get('macro_decode_reduction_pct'))}% | {num_str(p100.get('macro_decode_reduction_pct'))}% | {num_str(p150.get('macro_decode_reduction_pct'))}% |",
        f"| **Total Hypertokens Emitted** | 0 | {off.get('total_hypertokens_emitted', 0)} | {p100.get('total_hypertokens_emitted', 0)} | {p150.get('total_hypertokens_emitted', 0)} |",
        f"| **Mean Hypers / Output** | 0.00 | {num_str(off.get('mean_hypertokens_per_output'))} | {num_str(p100.get('mean_hypertokens_per_output'))} | {num_str(p150.get('mean_hypertokens_per_output'))} |",
        f"| **Mean Wall Time / Req** | {num_str(orig.get('mean_wall_time_s'))}s | {num_str(off.get('mean_wall_time_s'))}s | {num_str(p100.get('mean_wall_time_s'))}s | {num_str(p150.get('mean_wall_time_s'))}s |",
        f"| **Mean TTFT (Prefill)** | {num_str(orig.get('mean_ttft_s'))}s | {num_str(off.get('mean_ttft_s'))}s | {num_str(p100.get('mean_ttft_s'))}s | {num_str(p150.get('mean_ttft_s'))}s |",
        f"| **Mean Throughput (tok/s)**| {num_str(orig.get('mean_throughput_tok_per_s'))} | {num_str(off.get('mean_throughput_tok_per_s'))} | {num_str(p100.get('mean_throughput_tok_per_s'))} | {num_str(p150.get('mean_throughput_tok_per_s'))} |",
        "",
        "---",
        "",
        "## 2. Two-Stage Quality Loss Decomposition",
        "",
        "We separate quality degradation into:",
        "1. **Stage 1 (Baseline Zip2Zip LoRA / LZW Penalty)**: $\\text{Original Phi} \\to \\text{Official Zip2Zip}$",
        "2. **Stage 2 (Predictive Adaptation Delta)**: $\\text{Official Zip2Zip} \\to \\text{Predictive Model}$",
        "",
        "| Task Domain | Metric | Vanilla Phi | Official Zip2Zip | Stage 1 $\\Delta$ (Official Loss) | Pred Step 100 | Stage 2 $\\Delta$ (Step 100) | Pred Step 150 | Stage 2 $\\Delta$ (Step 150) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **MBPP Code** | Pass@1 | {pct_str(orig_code)} | {pct_str(off_code)} | {(off_code - orig_code)*100:+.1f}pp | {pct_str(p100_code)} | {(p100_code - off_code)*100:+.1f}pp | {pct_str(p150_code)} | {(p150_code - off_code)*100:+.1f}pp |",
        f"| **GSM8K Math** | Accuracy | {pct_str(orig_gsm)} | {pct_str(off_gsm)} | {(off_gsm - orig_gsm)*100:+.1f}pp | {pct_str(p100_gsm)} | {(p100_gsm - off_gsm)*100:+.1f}pp | {pct_str(p150_gsm)} | {(p150_gsm - off_gsm)*100:+.1f}pp |",
        f"| **Alpaca** | Mechanical failure rate | {pct_str(orig_alp)} | {pct_str(off_alp)} | {(off_alp - orig_alp)*100:+.1f}pp | {pct_str(p100_alp)} | {(p100_alp - off_alp)*100:+.1f}pp | {pct_str(p150_alp)} | {(p150_alp - off_alp)*100:+.1f}pp |",
        "",
        "---",
        "",
        "## 3. Compute Cost & Economics Analysis",
        "",
        "| Compute Metric | Original Phi | Official Zip2Zip | Predictive Step 100 | Predictive Step 150 |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Total Wall Time (60 prompts)** | {num_str(orig.get('total_wall_time_s'), 1)}s | {num_str(off.get('total_wall_time_s'), 1)}s | {num_str(p100.get('total_wall_time_s'), 1)}s | {num_str(p150.get('total_wall_time_s'), 1)}s |",
        f"| **Median Latency / Prompt** | {num_str(orig.get('median_wall_time_s'))}s | {num_str(off.get('median_wall_time_s'))}s | {num_str(p100.get('median_wall_time_s'))}s | {num_str(p150.get('median_wall_time_s'))}s |",
        f"| **Total Transformer Decode Steps** | {orig.get('total_decode_steps', 0)} | {off.get('total_decode_steps', 0)} | {p100.get('total_decode_steps', 0)} | {p150.get('total_decode_steps', 0)} |",
        f"| **Total Expanded Output Tokens** | {orig.get('total_expanded_tokens', 0)} | {off.get('total_expanded_tokens', 0)} | {p100.get('total_expanded_tokens', 0)} | {p150.get('total_expanded_tokens', 0)} |",
        f"| **Net Decode Steps Saved** | 0 | {off.get('total_tokens_saved', 0)} | {p100.get('total_tokens_saved', 0)} | {p150.get('total_tokens_saved', 0)} |",
        f"| **Micro Decode Step Reduction** | 0.0% | {num_str(off.get('micro_decode_reduction_pct'))}% | {num_str(p100.get('micro_decode_reduction_pct'))}% | {num_str(p150.get('micro_decode_reduction_pct'))}% |",
        f"| **EOS Count** | {orig.get('eos_count', 0)} | {off.get('eos_count', 0)} | {p100.get('eos_count', 0)} | {p150.get('eos_count', 0)} |",
        f"| **Generation Cap Reached** | {orig.get('hit_generation_cap_count', 0)} | {off.get('hit_generation_cap_count', 0)} | {p100.get('hit_generation_cap_count', 0)} | {p150.get('hit_generation_cap_count', 0)} |",
        f"| **True Truncations (Cap Without EOS)** | {orig.get('truncation_count', 0)} | {off.get('truncation_count', 0)} | {p100.get('truncation_count', 0)} | {p150.get('truncation_count', 0)} |",
        f"| **Severe Repetition Count** | {orig.get('repetition_count', 0)} | {off.get('repetition_count', 0)} | {p100.get('repetition_count', 0)} | {p150.get('repetition_count', 0)} |",
        "",
        "---",
        "",
        "## 4. Paired Transition Contingency Tables",
        "",
        "Comparing prompt-by-prompt transitions relative to Original Vanilla Phi:",
        "",
    ]

    for dom in ["code", "reasoning"]:
        lines.append(f"### {dom.capitalize()} Transitions (vs Original Phi)")
        lines.append("| Candidate Condition | Both Correct | Orig Correct, Cand Failed (Regression) | Orig Failed, Cand Correct (Recovery) | Both Failed |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for comp in ["official_zip2zip", "predictive_step_100", "predictive_step_150"]:
            tt = paired_deltas.get(dom, {}).get(comp, {})
            lines.append(
                f"| **{comp}** | {tt.get('both_correct', 0)} | {tt.get('orig_correct_cand_wrong', 0)} | "
                f"{tt.get('orig_wrong_cand_correct', 0)} | {tt.get('both_wrong', 0)} |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 5. Quality-Conditional Compression",
        "",
        "Examines whether compression occurs on problems the model actually answers correctly vs problems where it fails or drifts:",
        "",
        "| Condition | Domain | Preserved Decode Reduction % | Regressed Decode Reduction % | Quality-Preserved Tokens Saved |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ])

    for c in ["official_zip2zip", "predictive_step_100", "predictive_step_150"]:
        for dom in ["code", "reasoning"]:
            d_info = qc_data.get(c, {}).get(dom, {})
            lines.append(
                f"| **{c}** | {dom} | {num_str(d_info.get('preserved_decode_reduction_pct'))}% | "
                f"{num_str(d_info.get('regressed_decode_reduction_pct'))}% | "
                f"{d_info.get('quality_preserved_realized_savings_tokens', 0)} / {d_info.get('quality_preserved_expanded_tokens', 0)} |"
            )

    lines.extend([
        "",
        "---",
        "",
        f"## 6. Failure Mode Review ({len(failures)} Regressed Prompts)",
        "",
        "Representative failure cases where Original Vanilla Phi succeeded but Zip2Zip variants regressed:",
        "",
    ])

    for f_case in failures[:6]:
        lines.append(f"### Sample `{f_case['prompt_id']}` ({f_case['domain']})")
        lines.append(f"- **Prompt**: {f_case['prompt'][:150]}...")
        lines.append(f"- **Failed Models**: {', '.join(f_case['failed_models'])}")
        lines.append(f"- **Original Output Snippet**: `{repr(f_case['original_output'][:120])}`")
        if "official_zip2zip" in f_case["failed_models"]:
            lines.append(f"- **Official Zip2Zip Snippet**: `{repr(f_case['official_zip2zip_output'][:120])}`")
        if "predictive_step_100" in f_case["failed_models"]:
            lines.append(f"- **Pred Step 100 Snippet**: `{repr(f_case['predictive_100_output'][:120])}` (Hypers emitted: {len(f_case['predictive_100_hypers'])})")
        if "predictive_step_150" in f_case["failed_models"]:
            lines.append(f"- **Pred Step 150 Snippet**: `{repr(f_case['predictive_150_output'][:120])}` (Hypers emitted: {len(f_case['predictive_150_hypers'])})")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run 60-prompt quality and economics benchmark.")
    parser.add_argument("--evaluate-only", action="store_true", help="Recompute metrics and reports from existing raw_results.jsonl")
    parser.add_argument("--conditions", nargs="+", default=["original_phi", "official_zip2zip", "predictive_step_100", "predictive_step_150"])
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--device", default="cpu", help="Generation device, for example cpu or cuda:0")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_results_path = os.path.join(OUTPUT_DIR, "raw_results.jsonl")

    completed_keys: Set[Tuple[str, str]] = set()
    if os.path.exists(raw_results_path):
        with open(raw_results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if (
                            r.get("record_schema") != GENERATION_RECORD_SCHEMA
                            or r.get("evaluator_version") != EVALUATOR_VERSION
                            or r.get("prompt_formatter_version") != PROMPT_FORMATTER_VERSION
                        ):
                            raise ValueError(
                                "Existing raw results are from a different evaluator contract. "
                                "Use a fresh output directory; legacy records are not reused."
                            )
                        completed_keys.add((r["prompt_id"], r["condition"]))
                    except json.JSONDecodeError:
                        pass
        print(f"Loaded {len(completed_keys)} previously completed generations from {raw_results_path}")

    if not args.evaluate_only:
        with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
            samples = json.load(f)

        for c in args.conditions:
            if c == "original_phi":
                run_condition_original_phi(samples, raw_results_path, completed_keys, max_new_tokens=args.max_new_tokens, device=args.device)
            elif c == "official_zip2zip":
                run_condition_official_zip2zip(samples, raw_results_path, completed_keys, max_new_tokens=args.max_new_tokens, device=args.device)
            elif c == "predictive_step_100":
                run_condition_predictive(CKPT_100_PATH, c, samples, raw_results_path, completed_keys, max_new_tokens=args.max_new_tokens, device=args.device)
            elif c == "predictive_step_150":
                run_condition_predictive(CKPT_150_PATH, c, samples, raw_results_path, completed_keys, max_new_tokens=args.max_new_tokens, device=args.device)
            else:
                print(f"Unknown condition: {c}")

    # Generate full reports
    generate_full_benchmark_analytics(OUTPUT_DIR, raw_results_path)


if __name__ == "__main__":
    main()
