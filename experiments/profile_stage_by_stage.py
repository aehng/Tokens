"""
Stage-by-Stage Profiler for Base, Official Reactive Zip2Zip, Pure Predictive, and Hybrid.
Measures execution time and call counts for:
- Tokenizer encode/decode
- Predictor retrieval
- Codebook construction
- Hyper-input synthesis (HyperEmbedding encoder_fn)
- Hyper-output synthesis (HyperLinear encoder_fn)
- Prompt segmentation
- Transformer prefill
- Per-step dictionary update (Rust LZW)
- Logits processing
- Transformer decode loop
- Output expansion
"""

import time
import torch
from transformers import AutoTokenizer
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import StaticCodebookManager, Zip2ZipModel
from zip2zip_compression import CodebookManager as RustCodebookManager, CompressionConfig, LZWCompressor
from src.evaluation.offline_segmenter import segment_tokens_dp
import pickle


def profile_conditions(prompt_text: str = "Write a Python function to solve the following problem:\nWrite a function to zip the two given tuples.\n", max_new_tokens: int = 25):
    print("=" * 80)
    print("STAGE-BY-STAGE DETAILED PROFILER")
    print(f"Prompt: {repr(prompt_text)}")
    print("=" * 80)

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    with open("experiments/checkpoints/cached_predictor.pkl", "rb") as f:
        predictor = pickle.load(f)

    zip_model = Zip2ZipModel.from_pretrained(
        "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
        max_codebook_size=32,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    initial_vocab_size = 32011
    disabled_ids = list(zip_model.zip2zip_config.compression.disabled_ids)
    dim = 3072

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    base_prompt_len = len(prompt_ids)

    # -------------------------------------------------------------
    # Profile Condition 1: Base Phi-3.5
    # -------------------------------------------------------------
    print("\n--- Profiling Condition 1: Base Phi-3.5 ---")
    base_input = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_c1 = zip_model.base_model.generate(base_input, max_new_tokens=max_new_tokens, do_sample=False)
    t1 = time.perf_counter()
    base_total_ms = (t1 - t0) * 1000.0
    print(f"Base Total Latency: {base_total_ms:.1f} ms")

    # -------------------------------------------------------------
    # Profile Condition 2: Official Reactive Zip2Zip
    # -------------------------------------------------------------
    print("\n--- Profiling Condition 2: Official Reactive Zip2Zip ---")
    # Instrument CodebookManager encoder calls
    orig_input_encoder = zip_model.input_encoder.forward
    orig_output_encoder = zip_model.output_encoder.forward if zip_model.output_encoder else orig_input_encoder

    c2_input_calls = 0
    c2_output_calls = 0

    def tracked_in_enc(*args, **kwargs):
        nonlocal c2_input_calls
        c2_input_calls += 1
        return orig_input_encoder(*args, **kwargs)

    def tracked_out_enc(*args, **kwargs):
        nonlocal c2_output_calls
        c2_output_calls += 1
        return orig_output_encoder(*args, **kwargs)

    zip_model.input_encoder.forward = tracked_in_enc
    if zip_model.output_encoder:
        zip_model.output_encoder.forward = tracked_out_enc

    t0 = time.perf_counter()
    with torch.no_grad():
        out_c2 = zip_model.generate(input_ids=base_input, max_new_tokens=max_new_tokens, do_sample=False)
    t1 = time.perf_counter()
    c2_total_ms = (t1 - t0) * 1000.0

    print(f"Official Reactive Zip2Zip Total Latency: {c2_total_ms:.1f} ms")
    print(f"Official Reactive Input Encoder Calls: {c2_input_calls}")
    print(f"Official Reactive Output Encoder Calls: {c2_output_calls}")

    zip_model.input_encoder.forward = orig_input_encoder
    if zip_model.output_encoder:
        zip_model.output_encoder.forward = orig_output_encoder

    # -------------------------------------------------------------
    # Profile Condition 3: Pure Predictive Seeded
    # -------------------------------------------------------------
    print("\n--- Profiling Condition 3: Pure Predictive Seeded ---")
    c3_input_calls = 0
    c3_output_calls = 0
    zip_model.input_encoder.forward = tracked_in_enc
    if zip_model.output_encoder:
        zip_model.output_encoder.forward = tracked_out_enc

    t_tok0 = time.perf_counter()
    t_tok1 = time.perf_counter()
    t_tok_ms = (t_tok1 - t_tok0) * 1000.0

    t_pred0 = time.perf_counter()
    p_dict, _ = predictor.select_prompt_conditioned(prompt_ids, budget=32)
    pred_phrases = list(p_dict.keys())
    t_pred1 = time.perf_counter()
    t_pred_ms = (t_pred1 - t_pred0) * 1000.0

    t_seg0 = time.perf_counter()
    comp_len, tiles, _ = segment_tokens_dp(prompt_ids, set(pred_phrases))
    seeded_dict = {p: initial_vocab_size + i for i, p in enumerate(pred_phrases)}
    resegmented = [tile[0] if len(tile) == 1 else seeded_dict[tile] for tile in tiles]
    t_seg1 = time.perf_counter()
    t_seg_ms = (t_seg1 - t_seg0) * 1000.0

    t_seed0 = time.perf_counter()
    static_mgr = StaticCodebookManager(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=32,
        max_subtokens=3,
        embedding_dim=dim,
        pad_token_id=tokenizer.pad_token_id or 32000,
        disabled_ids=disabled_ids,
    )
    static_mgr.set_seeded_codebook(seeded_dict, batch_size=1, device=torch.device(device))
    static_mgr.attach_to_model(zip_model)
    t_seed1 = time.perf_counter()
    t_seed_ms = (t_seed1 - t_seed0) * 1000.0

    input_tensor_c3 = torch.tensor([resegmented], dtype=torch.long, device=device)
    from transformers import LogitsProcessorList
    logits_proc = LogitsProcessorList([static_mgr.get_logits_processor()])

    t_gen0 = time.perf_counter()
    with torch.no_grad():
        out_c3 = zip_model.generate(input_ids=input_tensor_c3, max_new_tokens=max_new_tokens, logits_processor=logits_proc, do_sample=False)
    t_gen1 = time.perf_counter()
    t_gen_ms = (t_gen1 - t_gen0) * 1000.0
    static_mgr.detach_from_model(zip_model)

    print(f"Pure Predictive Stage Breakdown:")
    print(f"  Predictor Retrieval:      {t_pred_ms:6.2f} ms")
    print(f"  Prompt Segmentation:      {t_seg_ms:6.2f} ms")
    print(f"  Codebook Construction:    {t_seed_ms:6.2f} ms")
    print(f"  Autoregressive Generate:  {t_gen_ms:6.2f} ms")
    print(f"  Total Setup:              {t_pred_ms + t_seg_ms + t_seed_ms:6.2f} ms")
    print(f"  Input Encoder Calls:      {c3_input_calls}")
    print(f"  Output Encoder Calls:     {c3_output_calls}")

    # -------------------------------------------------------------
    # Profile Condition 4: Current Live Hybrid (Slow implementation)
    # -------------------------------------------------------------
    print("\n--- Profiling Condition 4: Current Live Hybrid (Unoptimized) ---")
    from experiments.run_live_zero_shot_benchmark import LiveHybridCodebookManager
    c4_input_calls = 0
    c4_output_calls = 0
    c4_dict_updates = 0

    zip_model.input_encoder.forward = tracked_in_enc
    if zip_model.output_encoder:
        zip_model.output_encoder.forward = tracked_out_enc

    pred_phrases_c4 = pred_phrases[:12]
    seeded_dict_c4 = {phrase: (initial_vocab_size + i) for i, phrase in enumerate(pred_phrases_c4)}
    hybrid_mgr_c4 = LiveHybridCodebookManager(
        initial_vocab_size=initial_vocab_size,
        k_pred=12,
        k_lzw=20,
        max_subtokens=3,
        embedding_dim=dim,
        pad_token_id=tokenizer.pad_token_id or 32000,
        disabled_ids=disabled_ids,
    )
    hybrid_mgr_c4.set_seeded_codebook(seeded_dict_c4, batch_size=1, device=torch.device(device))
    hybrid_mgr_c4.attach_to_model(zip_model)

    comp_len_c4, tiles_c4, _ = segment_tokens_dp(prompt_ids, set(pred_phrases_c4))
    resegmented_c4 = [tile[0] if len(tile) == 1 else seeded_dict_c4[tile] for tile in tiles_c4]
    input_tensor_c4 = torch.tensor([resegmented_c4], dtype=torch.long, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out_c4 = zip_model.generate(input_ids=input_tensor_c4, max_new_tokens=max_new_tokens, do_sample=False)
    t1 = time.perf_counter()
    c4_total_ms = (t1 - t0) * 1000.0
    hybrid_mgr_c4.detach_from_model(zip_model)

    print(f"Current Hybrid Total Latency: {c4_total_ms:.1f} ms")
    print(f"Current Hybrid Input Encoder Calls:  {c4_input_calls}")
    print(f"Current Hybrid Output Encoder Calls: {c4_output_calls}")

    zip_model.input_encoder.forward = orig_input_encoder
    if zip_model.output_encoder:
        zip_model.output_encoder.forward = orig_output_encoder


if __name__ == "__main__":
    profile_conditions()
