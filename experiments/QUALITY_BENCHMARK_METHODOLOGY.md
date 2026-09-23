# Quality benchmark methodology

This document defines the evaluator contract for newly generated quality
benchmark records. It applies to the fixed Phi tier-1 comparison, the
quality-benchmark harness, and the K sweep. These runs use validation data;
they must not read or modify the final TEST split.

## Versioned contract

- Evaluator: phi_quality_evaluator_v2
- MBPP formatter: mbpp_task_signature_v2
- Quality generation record: phi_generation_record_v2
- Tier-1 and K-sweep manifests and generation caches use new schema versions.
- Quality benchmark output goes to mbpp_signature_v2; older result folders
  and cache records are not silently adopted as v2 results.

The exact prompt, reference, generation options, evaluator/source hashes,
checkpoint and predictor, model revisions, and runtime remain part of the
run identity or per-generation cache key.

## Prompt disclosure and paired conditions

For MBPP/code items, the formatter appends the single required top-level
function signature to the task description. It does not disclose the reference
function body or hidden assertion tests. The same formatted text is supplied
to every model condition; tokenization or prompt compression may differ by
condition and is measured separately.

The signature is parsed from the reference with Python's AST; reference code is
not imported or executed to construct a prompt. A missing or ambiguous
signature fails closed instead of prompting the model to guess a callable.

## Domain scoring

- **MBPP/code:** Pass@1 requires syntactically valid generated code to pass the
  supplied validation assertions. The runner uses a restricted builtins and
  import allowlist, an isolated Python invocation, a temporary working
  directory, a five-second default timeout, and a 512 MiB process-RSS limit.
  It rejects selected dangerous builtins, dunder access, and unapproved
  imports before execution. This is defense in depth for benchmark output,
  **not** a hardened operating-system sandbox for actively malicious code.
  Reference assertions are checked by the same static restrictions.
- **GSM8K/reasoning:** exact match on the extracted final numeric answer.
- **Alpaca/instruction:** automatic checks are strictly mechanical: empty
  output, fewer than five whitespace-separated words, a whitespace-token
  trigram repeated at least four times, or more than 30% non-ASCII characters.
  These are recorded as mechanical_instruction_failure and
  mechanical_instruction_pass. They do not measure whether an answer followed
  the instruction semantically. Semantic adherence is unavailable to the
  automatic evaluator; tier-1 creates a blinded human-review sheet and leaves
  the semantic score unset until it is rated.

## Generation health

Every newly generated record reports:

- whether its final generated token equals the tokenizer EOS token;
- whether generation reached the configured token cap;
- truncated, defined as reaching the cap without EOS (distinct from merely
  hitting the cap);
- severe whitespace-token trigram repetition and its maximum observed count;
- response length in base-equivalent tokens and decoded characters.

For compressed conditions, response length uses the expanded/base-equivalent
sequence, while generation-cap and EOS checks use the actual generated token
sequence. No single one of these fields substitutes for domain correctness.

## Interpretation

Keep automatic task correctness, mechanical output health, and human semantic
ratings distinct in reports. A mechanical pass is not an instruction-following
claim; valid Python syntax is not MBPP correctness; reaching EOS does not by
itself establish a complete or correct answer. Report denominators and missing
assertions explicitly, and never reinterpret old-version result records as
current evaluator outcomes.
