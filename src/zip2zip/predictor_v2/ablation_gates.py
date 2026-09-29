"""Small, deterministic gate calculations for Phi attribution ablations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F


def token_equivalence_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_prompt_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Require exact A/B0 token, termination, and generation-contract equality."""
    indexed: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for record in records:
        prompt_id = record.get("prompt_id")
        condition = record.get("condition")
        if prompt_id and condition in {"A_vanilla", "B0_tokens_vanilla_weights"}:
            indexed.setdefault(str(prompt_id), {})[str(condition)] = record

    pairs = [(rows["A_vanilla"], rows["B0_tokens_vanilla_weights"]) for rows in indexed.values()
             if "A_vanilla" in rows and "B0_tokens_vanilla_weights" in rows]
    expected = {str(prompt_id) for prompt_id in expected_prompt_ids} if expected_prompt_ids is not None else None
    paired_ids = {str(vanilla.get("prompt_id")) for vanilla, _ in pairs}
    missing_prompt_ids = sorted(expected - paired_ids) if expected is not None else []
    compared = []
    for vanilla, wrapped in pairs:
        v_runtime = vanilla.get("runtime", {})
        b_runtime = wrapped.get("runtime", {})
        v_ids = vanilla.get("generated_token_ids")
        b_ids = wrapped.get("generated_token_ids")
        compared.append({
            "prompt_id": vanilla.get("prompt_id"),
            "exact_token_ids": v_ids == b_ids,
            "same_generated_length": len(v_ids or []) == len(b_ids or []),
            "same_termination_reason": vanilla.get("termination_reason") == wrapped.get("termination_reason"),
            "same_termination_token": vanilla.get("termination_token_id") == wrapped.get("termination_token_id"),
            "same_base_weight_hash": bool(v_runtime.get("base_phi_weight_sha256")) and
                                     v_runtime.get("base_phi_weight_sha256") == b_runtime.get("base_phi_weight_sha256"),
            "same_generation_policy": bool(v_runtime.get("generation_policy_sha256")) and
                                      v_runtime.get("generation_policy_sha256") == b_runtime.get("generation_policy_sha256"),
            "same_input_tokens": bool(v_runtime.get("input_token_ids_sha256")) and
                                 v_runtime.get("input_token_ids_sha256") == b_runtime.get("input_token_ids_sha256"),
            "same_dtype": bool(v_runtime.get("torch_dtype")) and v_runtime.get("torch_dtype") == b_runtime.get("torch_dtype"),
            "same_tokenizer_revision": bool(v_runtime.get("tokenizer_revision")) and
                                       v_runtime.get("tokenizer_revision") == b_runtime.get("tokenizer_revision"),
            "same_tokenizer_id": bool(v_runtime.get("tokenizer_id")) and
                                 v_runtime.get("tokenizer_id") == b_runtime.get("tokenizer_id"),
            "same_tokenizer_template": bool(v_runtime.get("tokenizer_chat_template_sha256")) and
                                       v_runtime.get("tokenizer_chat_template_sha256") == b_runtime.get("tokenizer_chat_template_sha256"),
            "same_attention_implementation": v_runtime.get("attention_implementation") == b_runtime.get("attention_implementation"),
            "same_rendered_prompt": bool(v_runtime.get("rendered_prompt_sha256")) and
                                    v_runtime.get("rendered_prompt_sha256") == b_runtime.get("rendered_prompt_sha256"),
            "same_eos_ids": v_runtime.get("eos_token_ids") == b_runtime.get("eos_token_ids"),
            "same_max_new_tokens": v_runtime.get("max_new_tokens") == b_runtime.get("max_new_tokens"),
            "same_generation_config": v_runtime.get("generation_config") == b_runtime.get("generation_config"),
            "b0_checkpoint_not_applied": b_runtime.get("checkpoint_applied") is False,
            "b0_adapter_disabled": b_runtime.get("active_lora_during_generation") is False
                                     and b_runtime.get("adapter_disabled_context_used") is True,
            "b0_h_disabled": b_runtime.get("h_seeded_slots") == 0
                              and b_runtime.get("h_logits_masked_by_static_manager") is True,
            "b0_generated_no_h_ids": not any(32011 <= int(token_id) < 32043 for token_id in (b_ids or [])),
        })

    fields = (
        "exact_token_ids", "same_generated_length", "same_termination_reason", "same_termination_token",
        "same_base_weight_hash", "same_generation_policy", "same_input_tokens", "same_dtype",
        "same_tokenizer_revision", "same_rendered_prompt", "same_eos_ids", "same_max_new_tokens",
        "same_tokenizer_id", "same_tokenizer_template", "same_attention_implementation", "same_generation_config",
        "b0_checkpoint_not_applied", "b0_adapter_disabled", "b0_h_disabled", "b0_generated_no_h_ids",
    )
    field_pass = {field: bool(compared) and all(row[field] for row in compared) for field in fields}
    return {
        "status": "PASS" if bool(compared) and not missing_prompt_ids and all(field_pass.values()) else ("FAIL" if compared or missing_prompt_ids else "NOT_TESTED"),
        "paired_prompt_count": len(compared),
        "expected_prompt_count": len(expected) if expected is not None else None,
        "missing_prompt_ids": missing_prompt_ids,
        "all_exact_token_sequences": field_pass["exact_token_ids"],
        "checks": field_pass,
        "per_prompt": compared,
    }


def normalize_wrapper_logits(
    logits: torch.Tensor,
    *,
    base_vocab_size: int,
    initial_vocab_size: int,
    inserted_h_count: int,
) -> torch.Tensor:
    """Remove inserted H rows so wrapper logits align with native Phi IDs."""
    width = int(logits.shape[-1])
    if width == base_vocab_size:
        return logits
    if width != base_vocab_size + inserted_h_count:
        raise ValueError(
            f"wrapper logit width {width} is neither native {base_vocab_size} nor "
            f"native+H {base_vocab_size + inserted_h_count}"
        )
    return torch.cat(
        (logits[..., :initial_vocab_size], logits[..., initial_vocab_size + inserted_h_count:]),
        dim=-1,
    )


def logit_parity_metrics(
    vanilla_logits: torch.Tensor,
    wrapped_logits: torch.Tensor,
    *,
    eos_token_ids: Sequence[int],
    special_token_ids: Sequence[int] = (),
    top_k: int = 5,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Compare paired next-token logits, with a strict numerical and top-1 gate."""
    a = torch.as_tensor(vanilla_logits).detach().to(device="cpu", dtype=torch.float32)
    b = torch.as_tensor(wrapped_logits).detach().to(device="cpu", dtype=torch.float32)
    if a.ndim == 1:
        a = a.unsqueeze(0)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    if a.shape != b.shape:
        return {
            "status": "FAIL", "shape_match": False,
            "vanilla_shape": list(a.shape), "wrapped_shape": list(b.shape),
            "reason": "logit shapes differ after native-vocabulary normalization",
        }
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] < 2:
        raise ValueError(f"expected non-empty [states, vocab] logits; got {tuple(a.shape)}")

    differences = (a - b).abs()
    a_top1 = a.argmax(dim=-1)
    b_top1 = b.argmax(dim=-1)
    k = min(max(1, int(top_k)), int(a.shape[-1]))
    a_topk = a.topk(k, dim=-1).indices
    b_topk = b.topk(k, dim=-1).indices
    topk_overlap = [len(set(x.tolist()) & set(y.tolist())) / k for x, y in zip(a_topk, b_topk)]
    a_two = a.topk(2, dim=-1).values
    b_two = b.topk(2, dim=-1).values
    a_margin = a_two[:, 0] - a_two[:, 1]
    b_margin = b_two[:, 0] - b_two[:, 1]
    special_ids = sorted({int(value) for value in (*eos_token_ids, *special_token_ids)})
    special = {
        str(int(token_id)): {
            "vanilla": float(a[:, int(token_id)].mean().item()),
            "wrapped": float(b[:, int(token_id)].mean().item()),
            "mean_abs_difference": float((a[:, int(token_id)] - b[:, int(token_id)]).abs().mean().item()),
            "max_abs_difference": float((a[:, int(token_id)] - b[:, int(token_id)]).abs().max().item()),
        }
        for token_id in special_ids
        if 0 <= int(token_id) < a.shape[-1]
    }
    top1_all = bool(torch.equal(a_top1, b_top1))
    max_abs = float(differences.max().item())
    tolerance_pass = bool(torch.allclose(a, b, rtol=0.0, atol=atol))
    a_log_probs = F.log_softmax(a, dim=-1)
    b_log_probs = F.log_softmax(b, dim=-1)
    a_probs = a_log_probs.exp()
    kl_a_to_b = (a_probs * (a_log_probs - b_log_probs)).sum(dim=-1)
    return {
        "status": "PASS" if top1_all and tolerance_pass else "FAIL",
        "shape_match": True,
        "checked_state_count": int(a.shape[0]),
        "native_vocab_width": int(a.shape[1]),
        "max_abs_logit_difference": max_abs,
        "mean_abs_logit_difference": float(differences.mean().item()),
        "mean_kl_reference_to_candidate_nats": float(kl_a_to_b.mean().item()),
        "max_kl_reference_to_candidate_nats": float(kl_a_to_b.max().item()),
        "top1_agreement_count": int((a_top1 == b_top1).sum().item()),
        "top1_agreement_rate": float((a_top1 == b_top1).float().mean().item()),
        "top1_all_positions_agree": top1_all,
        "top_k": k,
        "top_k_overlap_mean": float(sum(topk_overlap) / len(topk_overlap)),
        "top_k_exact_agreement_count": sum(1 for x, y in zip(a_topk, b_topk) if torch.equal(x, y)),
        "mean_abs_top1_margin_difference": float((a_margin - b_margin).abs().mean().item()),
        "max_abs_top1_margin_difference": float((a_margin - b_margin).abs().max().item()),
        "eos_special_logit_differences": special,
        "absolute_tolerance": float(atol),
        "numerical_tolerance_pass": tolerance_pass,
    }


def checkpoint_isolation_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_prompt_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Verify B0/B1 share base weights and B1 loads a known non-empty delta."""
    indexed: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for record in records:
        prompt_id = record.get("prompt_id")
        condition = record.get("condition")
        if prompt_id and condition in {"B0_tokens_vanilla_weights", "B_h_disabled"}:
            indexed.setdefault(str(prompt_id), {})[str(condition)] = record
    pairs = [rows for rows in indexed.values() if "B0_tokens_vanilla_weights" in rows and "B_h_disabled" in rows]
    expected = {str(prompt_id) for prompt_id in expected_prompt_ids} if expected_prompt_ids is not None else None
    paired_ids = {str(rows["B0_tokens_vanilla_weights"].get("prompt_id")) for rows in pairs}
    missing_prompt_ids = sorted(expected - paired_ids) if expected is not None else []
    pair_checks = []
    for rows in pairs:
        b0_runtime = rows["B0_tokens_vanilla_weights"].get("runtime", {})
        b1_runtime = rows["B_h_disabled"].get("runtime", {})
        changes = b1_runtime.get("checkpoint_changed_parameters", {})
        lora_names = changes.get("lora", {}).get("names", []) if isinstance(changes, Mapping) else []
        lora_shapes = changes.get("lora", {}).get("shapes", {}) if isinstance(changes, Mapping) else {}
        expected_groups = {"lora", "input_encoder", "output_encoder"}
        full_inventory_known = isinstance(changes, Mapping) and set(changes) == expected_groups
        if full_inventory_known:
            for component in changes.values():
                names = component.get("names", [])
                shapes = component.get("shapes", {})
                full_inventory_known = full_inventory_known and (
                    set(names) == set(shapes)
                    and len(names) == int(component.get("changed_tensor_count", -1))
                )
        pair_checks.append({
            "prompt_id": rows["B0_tokens_vanilla_weights"].get("prompt_id"),
            "same_base_hash": bool(b0_runtime.get("base_phi_weight_sha256")) and
                              b0_runtime.get("base_phi_weight_sha256") == b1_runtime.get("base_phi_weight_sha256"),
            "b0_checkpoint_not_applied": b0_runtime.get("checkpoint_applied") is False,
            "b0_step100_h_state_not_loaded": b0_runtime.get("step100_h_encoder_checkpoint_loaded") is False,
            "b0_adapter_disabled_and_h_masked": (
                b0_runtime.get("active_lora_during_generation") is False
                and b0_runtime.get("adapter_disabled_context_used") is True
                and b0_runtime.get("h_seeded_slots") == 0
                and b0_runtime.get("h_logits_masked_by_static_manager") is True
            ),
            "b1_lora_verified": bool(b1_runtime.get("lora_delta_verified")) and bool(lora_names),
            "b1_lora_active_during_generation": b1_runtime.get("active_lora_during_generation") is True,
            "b1_changed_parameter_shapes_known": full_inventory_known
                                                 and bool(lora_names)
                                                 and set(lora_names) == set(lora_shapes),
        })
    passed = bool(pair_checks) and not missing_prompt_ids and all(all(v for k, v in row.items() if k != "prompt_id") for row in pair_checks)
    return {
        "status": "PASS" if passed else ("FAIL" if pair_checks or missing_prompt_ids else "NOT_TESTED"),
        "paired_prompt_count": len(pair_checks),
        "expected_prompt_count": len(expected) if expected is not None else None,
        "missing_prompt_ids": missing_prompt_ids,
        "checks": pair_checks,
    }


def forced_h_representation_gates(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_prompt_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Aggregate exact expansion, semantic-position/cache, and immediate continuation checks."""
    cf_records = [r for r in records if r.get("condition") == "CF_forced_oracle"]
    expected = {str(prompt_id) for prompt_id in expected_prompt_ids} if expected_prompt_ids is not None else None
    cf_ids = {str(r.get("prompt_id")) for r in cf_records}
    missing_prompt_ids = sorted(expected - cf_ids) if expected is not None else []

    def aggregate(field: str, detail_field: str | None = None) -> Dict[str, Any]:
        values = [r.get(field) for r in cf_records]
        tested = [value for value in values if value is not None]
        passed = bool(tested) and len(tested) == len(values) and all(bool(value) for value in tested)
        return {
            "status": "PASS" if passed else ("FAIL" if tested else "NOT_TESTED"),
            "passed_count": sum(bool(value) for value in tested),
            "tested_count": len(tested),
            "record_count": len(cf_records),
            "details": [r.get("runtime", {}).get(detail_field) for r in cf_records] if detail_field else None,
        }

    expansion = aggregate("forced_oracle_roundtrip_ok")
    positions = aggregate("cf_semantic_positions_ok", "cf_semantic_position_checks")
    continuation = aggregate("cf_continuation_stable", "cf_continuation_checks")
    overall_pass = all(gate["status"] == "PASS" for gate in (expansion, positions, continuation)) and not missing_prompt_ids
    return {
        "status": "PASS" if overall_pass else ("FAIL" if cf_records or missing_prompt_ids else "NOT_TESTED"),
        "expected_prompt_count": len(expected) if expected is not None else None,
        "missing_prompt_ids": missing_prompt_ids,
        "forced_h_exact_expansion": expansion,
        "forced_h_semantic_position_and_state": positions,
        "forced_h_continuation_stability": continuation,
    }
