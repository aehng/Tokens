# Tier-1 (12-Prompt) Authoritative Validation Report

- Device: `Tesla T4` (cuda:0)
- Checkpoint step: `100`
- Static cache capacity: `512` (disable_compile=True)
- Fixed-KV Context Length: `256` (warmups: 20, measured: 100)

## 1. Fixed-KV Microbenchmark (KV=256 Authoritative Decode Steps)

| Condition | Median Forward (ms) | Overhead vs Vanilla (%) | Speedup vs Legacy (%) |
|---|---:|---:|---:|
| Vanilla | 37.44 ms | - | - |
| Predictive Legacy Merged | 37.79 ms | +0.93% | - |
| Predictive Fast Merged | 37.77 ms | +0.89% | +0.04% |

## 2. 12-Prompt Aggregate Decode & Speedup Summary

- Total raw decode iterations: 2794
- Total expanded output tokens: 3134
- Net compression fraction: 10.85%
- Overall break-even decode latency: 42.04 ms
- Total Vanilla equivalent decode time: 117447.6 ms
- Total Legacy decode time: 106016.9 ms (Total with setup: 106392.8 ms)
- Total Fast decode time: 105749.9 ms (Total with setup: 106243.6 ms)
- Fast decode speedup vs Legacy: +0.25%
- Fast total speedup vs Legacy: +0.14%
- Fast decode speedup vs Vanilla: +9.96%
- Fast total speedup vs Vanilla: +9.54%

## 3. Domain Quality & Termination Summary

### MBPP Code Synthesis (4 prompts)
- Vanilla: 3/4 syntax valid, 1/4 passed assertions
- Legacy: 1/4 syntax valid, 0/4 passed assertions
- Fast: 1/4 syntax valid, 0/4 passed assertions

### GSM8K Math Reasoning (4 prompts)
- Vanilla: 3/4 exact correct
- Legacy: 2/4 exact correct
- Fast: 2/4 exact correct

### Alpaca Instruction Following (4 prompts)
- Vanilla: 2/4 mechanical pass
- Legacy: 3/4 mechanical pass
- Fast: 3/4 mechanical pass

## 4. Per-Prompt Matrix Details

| Prompt ID | Domain | Raw / Exp Tokens | Compression | Vanilla ms | Fast ms | Speedup vs Vanilla |
|---|---|---:|---:|---:|---:|---:|
| mbpp_740 | code | 242 / 248 | 2.4% | 9068.3 | 9496.1 | -4.7% |
| mbpp_969 | code | 300 / 345 | 13.0% | 12781.9 | 11242.3 | +12.0% |
| mbpp_542 | code | 223 / 245 | 9.0% | 9214.7 | 8466.7 | +8.1% |
| mbpp_769 | code | 300 / 331 | 9.4% | 12812.2 | 11440.5 | +10.7% |
| gsm_3022 | reasoning | 300 / 351 | 14.5% | 13179.9 | 11335.6 | +14.0% |
| gsm_6613 | reasoning | 300 / 326 | 8.0% | 12142.1 | 11390.8 | +6.2% |
| gsm_2956 | reasoning | 300 / 327 | 8.3% | 12187.4 | 11376.5 | +6.7% |
| gsm_8674 | reasoning | 300 / 346 | 13.3% | 12935.4 | 11364.3 | +12.1% |
| alpaca_1337 | instruction | 300 / 372 | 19.4% | 14032.6 | 11356.3 | +19.1% |
| alpaca_1992 | instruction | 8 / 8 | 0.0% | 301.1 | 338.8 | -12.5% |
| alpaca_55 | instruction | 19 / 23 | 17.4% | 861.9 | 760.0 | +11.8% |
| alpaca_183 | instruction | 202 / 212 | 4.7% | 7930.0 | 7675.6 | +3.2% |

## 5. Model Provenance

```json
{
  "model_class": "Phi3ForCausalLM",
  "base_model_class": "Phi3Model",
  "config_model_type": "phi3",
  "vocab_size": 32064,
  "hidden_size": 3072,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 32,
  "base_model_name_or_path": "microsoft/Phi-3.5-mini-instruct",
  "canonical_base_model_id": "microsoft/Phi-3.5-mini-instruct",
  "canonical_base_revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
  "canonical_zip2zip_revision": "11c461733a79d2a5de6b814585c3361ca2aacbe7",
  "model_architecture_label": "Phi-3.5-mini-instruct",
  "provenance_note": "Prior assistant mentions of 'Phi-1.5' were an assistant labeling error; the active model has always been Microsoft Phi-3.5-mini-instruct with Zip2Zip."
}
```
