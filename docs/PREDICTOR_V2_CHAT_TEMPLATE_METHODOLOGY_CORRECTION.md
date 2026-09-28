# Methodology Correction: Native Phi-3.5 Chat Template Generation Contract

## Executive Summary

During the quality phase scaling of the Tokens Predictor V2 architecture selection (scaling from 60 to 900 prompts), an empirical audit of the canonical continuation pipeline revealed a critical contract discrepancy between raw prompt text presentation and the native conversational contract expected by instruction-tuned causal models (`microsoft/Phi-3.5-mini-instruct`).

This document records the empirical findings, root-cause diagnosis, and the corrected canonical generation contract.

---

## 1. Problem Statement & Anomaly

### Observed Anomaly
In the initial scaled generation run (Version 22 on Kaggle), 100% of instruction (Alpaca) prompts generated exactly 300 new tokens, hitting the generation limit (`max_new_tokens=300`), with an apparent zero-EOS termination rate.

### Upstream Model Specification
The upstream Hugging Face generation configuration for `microsoft/Phi-3.5-mini-instruct` (`rev 2fe19245...`) defines:
- `eos_token_id`: `[32007, 32001, 32000]`
- `pad_token_id`: `32000`

Where the special token IDs correspond to:
- `32000`: `<|endoftext|>` (document separator / pretraining base EOS)
- `32001`: `<|assistant|>` (assistant role turn header)
- `32007`: `<|end|>` (turn terminator / chat EOS)

However, legacy metadata recorded `pad_token_id=eos_token_id=32000`, creating the hypothesis that multi-EOS stopping had been inadvertently overridden.

---

## 2. Empirical Controlled Diagnosis (Kaggle Tesla T4 GPU)

On an Nvidia Tesla T4 GPU in float16, a controlled 5-prompt diagnostic test was executed comparing:
- **Config A** (Version 22 arguments): `eos_token_id=32000`, `pad_token_id=32000`
- **Config B** (Upstream generation config): default `eos_token_id=[32007, 32001, 32000]`
- **Config C** (Explicit multi-EOS): `eos_token_id=[32007, 32001, 32000]`, `pad_token_id=32000`

### Diagnostic Results Table

| Prompt ID | Domain | Len A | Len B | Len C | B == C? | A Prefix == B? | A Continued Past B? | Special Tokens Found in Raw Text |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `alpaca_1` | instruction | 300 | 300 | 300 | True | True | False | None (no 32000, 32001, 32007) |
| `alpaca_2` | instruction | 300 | 300 | 300 | True | True | False | None (no 32000, 32001, 32007) |
| `gsm_1006` | reasoning | 117 | 117 | 117 | True | True | False | Stopped naturally at 116 on 32000 |
| `gsm_1134` | reasoning | 193 | 193 | 193 | True | True | False | Stopped naturally at 192 on 32000 |
| `mbpp_1` | code | 300 | 300 | 300 | True | True | False | None (no 32000, 32001, 32007) |

### Key Discovery: The Raw Prompt Hallucination Phenomenon
1. In raw prompt text mode (`Instruction: ...\nAnswer:`), Config A, B, and C generated **100% token-identical sequences**. Config A did **not** skip token 32007.
2. The reason Alpaca hits 300 tokens is **raw text completion mode**:
   - Because the model is not presented with chat turn boundaries (`<|user|>` / `<|assistant|>`), it treats the prompt as raw text from an instruction dataset.
   - After producing the direct answer (e.g. `"The three primary colors are red, blue, and yellow."`), the model hallucinates synthetic subsequent examples:
     `\n\n\n\nInstruction: Explain why these colors are considered primary.\nAnswer: ...`
   - It will never emit `<|end|>` (32007) because `<|end|>` is reserved strictly for chat turn closure.
3. When `tokenizer.apply_chat_template` is applied:
   - `alpaca_1` stops cleanly at **147 tokens** on token **32007 (`<|end|>`)**.
   - No subsequent synthetic instructions are generated.

---

## 3. Corrected Canonical Methodology

### Previous Methodology (Flawed)
- Raw task text directly encoded and passed to `model.generate()`.
- Result: Model was in raw document-continuation mode, generating multi-example hallucinations and artificially truncating at `max_new_tokens=300`.
- Overrode EOS solely with `tokenizer.eos_token_id` (32000).

### Corrected Methodology (Standardized)
- Preserves each domain's task string exactly:
  - **MBPP**: AST-extracted function signature stub.
  - **GSM8K**: Exact problem text step-by-step.
  - **Alpaca**: Exact instruction text.
- Wraps the task string via `tokenizer.apply_chat_template`:
  ```python
  messages = [{"role": "user", "content": task_prompt}]
  inputs = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt",
      return_dict=True,
  )
  ```
  Rendering structurally as:
  ```text
  <|user|>
  ...task prompt...
  <|end|>
  <|assistant|>
  ```
- Uses native multi-EOS stopping:
  ```python
  eos_token_id = [32007, 32001, 32000]
  pad_token_id = 32000
  ```
- Decodes with both `skip_special_tokens=True` (for clean text) and `skip_special_tokens=False` (raw text), while storing exact raw token IDs and termination metadata (`termination_reason`, `termination_token_id`).

---

## 4. Invalidation of Legacy 60-Prompt Continuations

Because the prompt distribution has changed from raw document completion to native conversational instruction following:
1. The historical 60 canonical continuations in `raw_results.jsonl:original_phi` are designated **legacy raw-prompt data** and deprecated.
2. The entire 900-prompt dataset (540 Train, 180 Dev, 180 Final) is freshly generated under the corrected chat-template contract.
3. Downstream indexes (`TRAIN` association index, candidate retrieval pools) and architecture bake-off models will be trained exclusively on the corrected continuations.
