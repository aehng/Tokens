# Quality benchmark methodology

This document defines the evaluator contract for newly generated quality
benchmark records. It applies to the fixed Phi tier-1 comparison, the
quality-benchmark harness, and any later K recalibration. The immediate
planned GPU comparison and future live runs must use this contract; the
current phase order is in [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md). These
runs use validation data; they must not read or modify the final TEST split.

## Versioned contract

- Evaluator: phi_quality_evaluator_v2
- MBPP formatter: mbpp_task_signature_v2
- Historical quality generation records: phi_generation_record_v2
- New live generation records: phi_generation_record_v3, including the required
  runtime/trajectory diagnostics below.
- Tier-1 and K-sweep manifests and generation caches use new schema versions.
- Historical quality benchmark output remains in mbpp_signature_v2. New
  generation records go to the v3/run-manifest output path; older result
  folders and cache records are not silently adopted as v3 results.
- Diagnostic schema/version and instrumentation source hashes are part of
  future run identity/cache provenance. Never backfill v3 fields into old runs
  or mix v2 and v3 records in a summary.

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

## Required runtime and trajectory diagnostics for new live runs

Collect these fields during the same generation used for quality scoring; do
not launch a duplicate timing suite. Record per prompt, using null when a
measurement is unavailable rather than inventing a value.

### Output and trajectory

- original/base prompt token count, actual compressed prompt position count,
  prompt compression percentage, actual transformer decode iterations, and
  expanded/base-equivalent output token count;
- generated text word and character counts;
- hypertoken count, represented base-token count, net decode steps saved, and
  decode reduction percentage;
- EOS emitted, generation cap reached, truncation, and severe repetition;
- deterministic repetition diagnostics already defined by this evaluator.

### Timing decomposition

Record predictor latency, codebook construction, hyper-weight synthesis/setup
when separable, prompt prefill/TTFT, decode wall time, total generation wall
time, and total request wall time. Distinguish request setup, prefill, decode
iteration count and per-iteration cost, and generation trajectory in every
latency interpretation.

Derive per prompt and in aggregate: transformer steps/second, expanded
base-equivalent tokens/second, wall time per decode step, wall time per
expanded token, and paired output-length and latency ratios versus Vanilla for
matching prompt IDs. Report raw decode reduction and quality-preserved decode
reduction separately. The latter includes only candidate outputs passing the
task-quality gate. Report denominators and null/unmatched paired cases.

Lightweight timestamps at decode-step boundaries may provide mean, median, and
p95 decode-step interval time. On CPU, consecutive generation-callback
timestamps are an approximation of forward-step intervals. Do not synchronize
or profile every step on GPU if that materially changes generation; use
aggregate decode wall time divided by actual decode iterations instead and
mark unavailable distributions null.

### Hypertoken and gate events

For each emitted hypertoken, capture generation position/iteration, phrase
token IDs, phrase length, expanded phrase text, and base positions saved. Do
not dump logits or large tensors. For a gated condition, also record the
first-constituent rank, allowed/rejected decision, threshold/top-N, candidate
count, masked count, and emitted count. Preserve gate totals in the per-prompt
record.

### Answer-to-tail and continuation diagnostics

For GSM8K or other structured tasks, record the first output position where a
valid final answer can be deterministically extracted and the decode/output
length after that point. For instruction or code, use only deterministic safe
task-specific rules. If answer completion cannot be determined reliably, set
the answer position and post-answer tail to null. These are diagnostics, not
semantic completion labels.

Record H count before/after a deterministically detected answer, H density,
distance from the last H emission to EOS/end, and whether repetition starts
soon after an H when deterministically measurable. Treat all H/continuation
relationships as associations, not evidence of causation.

### Generation health

Every newly generated record also reports:

- whether its final generated token equals the tokenizer EOS token;
- whether generation reached the configured token cap;
- truncated, defined as reaching the cap without EOS (distinct from merely
  hitting the cap);
- severe whitespace-token trigram repetition and its maximum observed count;
- response length in base-equivalent tokens and decoded characters.

For compressed conditions, response length uses the expanded/base-equivalent
sequence, while generation-cap and EOS checks use the actual generated token
sequence. No single one of these fields substitutes for domain correctness.

## Required summary interpretation

Summaries must make the following calculations visible:

1. Raw decode reduction = `(expanded_base_equivalent_tokens -
   transformer_decode_iterations) / expanded_base_equivalent_tokens`.
2. Quality-preserved decode reduction uses the same savings numerator and
   expanded-token denominator, restricted to outputs passing the quality gate.
3. Transformer steps/second = decode iterations / decode wall time.
4. Effective expanded tokens/second = expanded tokens / decode wall time.
5. Wall time/decode step = decode wall time / decode iterations.
6. Output-length ratio versus Vanilla = candidate expanded output tokens /
   paired Vanilla expanded output tokens.
7. Latency ratio versus Vanilla = candidate request wall time / paired Vanilla
   request wall time.
8. Post-answer tail length, where its deterministic position is available.

These distinguish expensive decode steps from longer trajectories, added
per-step runtime overhead, and answers that continue after a usable answer.
Never infer that vocabulary size alone explains latency. Instrument the same
generation run; add a dedicated microbenchmark only if these integrated
measurements leave the runtime question ambiguous.

## Interpretation

Keep automatic task correctness, mechanical output health, and human semantic
ratings distinct in reports. A mechanical pass is not an instruction-following
claim; valid Python syntax is not MBPP correctness; reaching EOS does not by
itself establish a complete or correct answer. Report denominators and missing
assertions explicitly, and never reinterpret old-version result records as
current evaluator outcomes.
