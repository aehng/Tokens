"""CPU checks for the vLLM proof harness.

No vLLM import. The GPU runner uses these helpers to size the preemption
pool and to stop a manual engine loop.
"""

from __future__ import annotations

NULL_KV_BLOCKS = 1
DEFAULT_BLOCK_SIZE = 16
MAX_ENGINE_STEPS = 500


def blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    if num_tokens < 0 or block_size <= 0:
        raise ValueError(
            f"num_tokens={num_tokens} block_size={block_size} is not usable"
        )
    if num_tokens == 0:
        return 0
    return (num_tokens + block_size - 1) // block_size


def new_tokens_to_cross_block(prompt_len: int, block_size: int) -> int:
    """Smallest generation length that allocates one block past the prompt."""
    if prompt_len <= 0 or block_size <= 0:
        raise ValueError(
            f"prompt_len={prompt_len} block_size={block_size} is not usable"
        )
    remainder = prompt_len % block_size
    if remainder == 0:
        return 1
    return block_size - remainder + 1


def preemption_block_budget(
    prompt_a: int,
    prompt_b: int,
    block_size: int,
    *,
    null_blocks: int = NULL_KV_BLOCKS,
) -> dict[str, int]:
    """Size a KV pool so each prompt fits and the pair cannot stay resident.

    vLLM 0.30.0 full attention with prefix caching off:

    - the block pool holds one unusable null block
    - a waiting request is admitted only when its prompt fits
    - a later token that crosses a block boundary allocates one more block
    - a running request that cannot allocate preempts another running request

    Usable blocks equal the sum of the two prompt block counts. Generation
    is long enough that each sequence needs one extra block, so the pair
    does not fit and one request is preempted.
    """
    if null_blocks < 1:
        raise ValueError("the null block must be reserved")
    a_prompt_blocks = blocks_for_tokens(prompt_a, block_size)
    b_prompt_blocks = blocks_for_tokens(prompt_b, block_size)
    if a_prompt_blocks < 1 or b_prompt_blocks < 1:
        raise ValueError("both prompts must occupy at least one KV block")
    max_new = max(
        new_tokens_to_cross_block(prompt_a, block_size),
        new_tokens_to_cross_block(prompt_b, block_size),
    )
    a_full = blocks_for_tokens(prompt_a + max_new, block_size)
    b_full = blocks_for_tokens(prompt_b + max_new, block_size)
    usable = a_prompt_blocks + b_prompt_blocks
    max_model_len = max(prompt_a, prompt_b) + max_new
    if blocks_for_tokens(max_model_len, block_size) > usable:
        raise ValueError("max_model_len does not fit in the usable block pool")
    if a_full > usable or b_full > usable:
        raise ValueError("a request alone does not fit in the usable block pool")
    if a_full + b_full <= usable:
        raise ValueError("both full sequences fit; preemption is not forced")
    return {
        "block_size": block_size,
        "null_blocks": null_blocks,
        "usable_blocks": usable,
        "num_gpu_blocks": usable + null_blocks,
        "prompt_a_tokens": prompt_a,
        "prompt_b_tokens": prompt_b,
        "a_prompt_blocks": a_prompt_blocks,
        "b_prompt_blocks": b_prompt_blocks,
        "max_new_tokens": max_new,
        "a_full_blocks": a_full,
        "b_full_blocks": b_full,
        "combined_full_blocks": a_full + b_full,
        "max_model_len": max_model_len,
    }


def engine_step_decision(step_index: int, max_steps: int, unfinished: bool) -> str:
    """Decide whether a manual ``engine.step`` loop may continue.

    ``step_index`` is the number of steps already taken. ``max_steps``
    unfinished iterations return ``budget_exceeded``.
    """
    if step_index < 0 or max_steps < 1:
        raise ValueError(
            f"step_index={step_index} max_steps={max_steps} is not usable"
        )
    if not unfinished:
        return "stop"
    if step_index >= max_steps:
        return "budget_exceeded"
    return "continue"


def preemption_cycle(events: list[dict], request_id: str) -> bool:
    """True when ``request_id`` was added, removed, then added again."""
    seen_add = False
    seen_remove = False
    for event in events:
        if event.get("req_id") != request_id:
            continue
        kind = event.get("event")
        if kind == "add" and not seen_add:
            seen_add = True
        elif kind == "remove" and seen_add:
            seen_remove = True
        elif kind == "add" and seen_remove:
            return True
    return False


def preempted_request_ids(events: list[dict]) -> list[str]:
    """Request ids whose admission log shows add, remove, re-add."""
    ordered: list[str] = []
    for event in events:
        req_id = event.get("req_id")
        if req_id is not None and req_id not in ordered:
            ordered.append(req_id)
    return [req_id for req_id in ordered if preemption_cycle(events, req_id)]
