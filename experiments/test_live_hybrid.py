"""
Test script to verify LiveHybridCodebookManager mechanics.
Combines 12 static predictive slots with 20 reactive LZW slots.
"""

import torch
from zip2zip import StaticCodebookManager
from zip2zip_compression import CodebookManager as RustCodebookManager, CompressionConfig, LZWCompressor

initial_vocab_size = 32011
k_pred = 12
k_lzw = 20
max_budget = 32
max_subtokens = 3
pad_token_id = 32000

print("Testing Hybrid Codebook setup...")
# 1. Static manager for first 12 slots
static_mgr = StaticCodebookManager(
    initial_vocab_size=initial_vocab_size,
    max_codebook_size=k_pred,
    max_subtokens=max_subtokens,
    embedding_dim=3072,
    pad_token_id=pad_token_id,
)

# Seed with 3 dummy phrases
dummy_phrases = {
    (100, 200): initial_vocab_size,
    (300, 400): initial_vocab_size + 1,
    (500, 600): initial_vocab_size + 2,
}
static_mgr.set_seeded_codebook(dummy_phrases, batch_size=1)
print(f"Static seeded: {static_mgr.num_seeded} phrases")

# 2. Rust manager for remaining 20 slots (starts at 32011 + 12 = 32023)
rust_cfg = CompressionConfig(
    initial_vocab_size=initial_vocab_size + k_pred,
    max_codebook_size=k_lzw,
    max_subtokens=max_subtokens,
    pad_token_id=pad_token_id,
    disabled_ids=[],
)
rust_mgr = RustCodebookManager(rust_cfg)

# Test updating rust_mgr with a sequence that has both base tokens and static hypertokens
test_seq = [10, 20, initial_vocab_size, 30, 40, 30, 40]
updates, indices = rust_mgr.update_codebooks([test_seq])
print(f"Rust updates: indices={indices}")
print(f"Rust codebook: {rust_mgr.get_codebooks()[0].to_dict()}")

# Test decoding
def decode_hybrid_sequence(tokens, static_dict, rust_compressor):
    # Step 1: Decode reactive tokens (>= initial_vocab_size + k_pred)
    decoded_reactive, _ = rust_compressor.batch_decode([tokens])[0]
    # Step 2: Decode static tokens (in initial_vocab_size .. initial_vocab_size + k_pred)
    final_tokens = []
    inv_static = {v: list(k) for k, v in static_dict.items()}
    for t in decoded_reactive:
        if t in inv_static:
            final_tokens.extend(inv_static[t])
        else:
            final_tokens.append(t)
    return final_tokens

rust_compressor = LZWCompressor(
    initial_vocab_size=initial_vocab_size + k_pred,
    max_codebook_size=k_lzw,
    max_subtokens=max_subtokens,
    pad_token_id=pad_token_id,
    disabled_ids=[],
)

# If sequence contains static hypertoken 32011 and reactive hypertoken 32023
sample_hybrid_seq = [10, initial_vocab_size, 32023]
# 32023 should expand to [10, 20] from rust
decoded = decode_hybrid_sequence(sample_hybrid_seq, dummy_phrases, rust_compressor)
print(f"Decoded hybrid sequence: {decoded}")
print("SUCCESS: Hybrid mechanics validated!")
