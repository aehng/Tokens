"""Small, deterministic gate calculations for Phi attribution ablations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F


def adapter_modules_to_save_are_empty(modules_to_save: Any) -> bool:
    """Require a present adapter inventory with no saved full-module weights."""
    return (
        isinstance(modules_to_save, Mapping)
        and bool(modules_to_save)
        and all(isinstance(modules, (list, tuple)) and not modules for modules in modules_to_save.values())
    )


def token_equivalence_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_prompt_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Require exact A/B0 token, termination, and generation-contract equality."""
    indexed: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    pair_counts: Dict[tuple[str, str], int] = {}
    for record in records:
        prompt_id = record.get("prompt_id")
        condition = record.get("condition")
        if prompt_id and condition in {"A_vanilla", "B0_tokens_vanilla_weights"}:
            pair = (str(prompt_id), str(condition))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            indexed.setdefault(str(prompt_id), {})[str(condition)] = record

    duplicate_pairs = sorted(
        [
            {"prompt_id": prompt_id, "condition": condition, "count": count}
            for (prompt_id, condition), count in pair_counts.items()
            if count > 1
        ],
        key=lambda row: (row["prompt_id"], row["condition"]),
    )

    pairs = [(rows["A_vanilla"], rows["B0_tokens_vanilla_weights"]) for rows in indexed.values()
             if "A_vanilla" in rows and "B0_tokens_vanilla_weights" in rows]
    expected = {str(prompt_id) for prompt_id in expected_prompt_ids} if expected_prompt_ids is not None else None
    paired_ids = {str(vanilla.get("prompt_id")) for vanilla, _ in pairs}
    missing_prompt_ids = sorted(expected - paired_ids) if expected is not None else []
    unexpected_prompt_ids = sorted(paired_ids - expected) if expected is not None else []
    compared = []
    for vanilla, wrapped in pairs:
        v_runtime = vanilla.get("runtime", {})
        b_runtime = wrapped.get("runtime", {})
        v_ids = vanilla.get("generated_token_ids")
        b_ids = wrapped.get("generated_token_ids")
        compared.append({
            "prompt_id": vanilla.get("prompt_id"),
            "exact_token_ids": isinstance(v_ids, list) and isinstance(b_ids, list) and v_ids == b_ids,
            "same_generated_length": isinstance(v_ids, list) and isinstance(b_ids, list)
                                     and vanilla.get("generated_token_count") == len(v_ids)
                                     and wrapped.get("generated_token_count") == len(b_ids)
                                     and len(v_ids) == len(b_ids),
            "same_expanded_token_ids": isinstance(vanilla.get("expanded_token_ids"), list)
                                       and isinstance(wrapped.get("expanded_token_ids"), list)
                                       and vanilla.get("expanded_token_ids") == wrapped.get("expanded_token_ids"),
            "same_expanded_length": isinstance(vanilla.get("expanded_token_ids"), list)
                                    and isinstance(wrapped.get("expanded_token_ids"), list)
                                    and vanilla.get("expanded_token_count") == len(vanilla.get("expanded_token_ids"))
                                    and wrapped.get("expanded_token_count") == len(wrapped.get("expanded_token_ids"))
                                    and len(vanilla.get("expanded_token_ids")) == len(wrapped.get("expanded_token_ids")),
            "same_termination_reason": vanilla.get("termination_reason") == wrapped.get("termination_reason"),
            "same_termination_token": vanilla.get("termination_token_id") == wrapped.get("termination_token_id"),
            "same_eos_reached": vanilla.get("eos_reached") is not None
                                and vanilla.get("eos_reached") == wrapped.get("eos_reached"),
            "same_model_id": bool(vanilla.get("model_id")) and vanilla.get("model_id") == wrapped.get("model_id"),
            "same_model_revision": bool(vanilla.get("model_revision")) and vanilla.get("model_revision") == wrapped.get("model_revision"),
            "same_base_weight_hash": bool(v_runtime.get("base_phi_weight_sha256")) and
                                     v_runtime.get("base_phi_weight_sha256") == b_runtime.get("base_phi_weight_sha256"),
            "same_generation_policy": bool(v_runtime.get("generation_policy_sha256")) and
                                      v_runtime.get("generation_policy_sha256") == b_runtime.get("generation_policy_sha256"),
            "same_input_tokens": bool(v_runtime.get("input_token_ids_sha256")) and
                                 v_runtime.get("input_token_ids_sha256") == b_runtime.get("input_token_ids_sha256") and
                                 isinstance(v_runtime.get("input_token_ids"), list) and
                                 bool(v_runtime.get("input_token_ids")) and
                                 v_runtime.get("input_token_ids") == b_runtime.get("input_token_ids"),
            "same_dtype": bool(v_runtime.get("torch_dtype")) and v_runtime.get("torch_dtype") == b_runtime.get("torch_dtype"),
            "same_tokenizer_revision": bool(v_runtime.get("tokenizer_revision")) and
                                       v_runtime.get("tokenizer_revision") == b_runtime.get("tokenizer_revision"),
            "same_tokenizer_id": bool(v_runtime.get("tokenizer_id")) and
                                 v_runtime.get("tokenizer_id") == b_runtime.get("tokenizer_id"),
            "same_tokenizer_template": bool(v_runtime.get("tokenizer_chat_template_sha256")) and
                                       v_runtime.get("tokenizer_chat_template_sha256") == b_runtime.get("tokenizer_chat_template_sha256"),
            "same_attention_implementation": bool(v_runtime.get("attention_implementation")) and
                                             v_runtime.get("attention_implementation") == b_runtime.get("attention_implementation"),
            "same_rendered_prompt": bool(v_runtime.get("rendered_prompt_sha256")) and
                                    v_runtime.get("rendered_prompt_sha256") == b_runtime.get("rendered_prompt_sha256") and
                                    bool(v_runtime.get("rendered_prompt_text")) and
                                    v_runtime.get("rendered_prompt_text") == b_runtime.get("rendered_prompt_text"),
            "same_git_commit": bool(v_runtime.get("git_commit")) and v_runtime.get("git_commit") == b_runtime.get("git_commit"),
            "same_git_branch": bool(v_runtime.get("git_branch")) and v_runtime.get("git_branch") == b_runtime.get("git_branch"),
            "same_worktree_dirty_status": v_runtime.get("git_worktree_dirty") is not None and
                                          v_runtime.get("git_worktree_dirty") == b_runtime.get("git_worktree_dirty"),
            "same_git_status_hash": bool(v_runtime.get("git_status_sha256")) and
                                    v_runtime.get("git_status_sha256") == b_runtime.get("git_status_sha256"),
            "same_source_archive": bool(v_runtime.get("source_archive_sha256")) and
                                   v_runtime.get("source_archive_sha256") == b_runtime.get("source_archive_sha256"),
            "same_dataset_manifest": bool(v_runtime.get("dataset_manifest_sha256")) and
                                     v_runtime.get("dataset_manifest_sha256") == b_runtime.get("dataset_manifest_sha256"),
            "same_canonical_dataset": bool(v_runtime.get("canonical_dataset_sha256")) and
                                      v_runtime.get("canonical_dataset_sha256") == b_runtime.get("canonical_dataset_sha256"),
            "same_cuda_version": bool(v_runtime.get("cuda_version")) and
                                 v_runtime.get("cuda_version") == b_runtime.get("cuda_version"),
            "same_gpu": bool(v_runtime.get("gpu_name")) and v_runtime.get("gpu_name") == b_runtime.get("gpu_name"),
            "same_package_versions": v_runtime.get("package_versions_complete") is True and
                                    b_runtime.get("package_versions_complete") is True and
                                    bool(v_runtime.get("package_versions")) and
                                    v_runtime.get("package_versions") == b_runtime.get("package_versions"),
            "same_eos_ids": bool(v_runtime.get("eos_token_ids")) and
                            v_runtime.get("eos_token_ids") == b_runtime.get("eos_token_ids"),
            "same_max_new_tokens": v_runtime.get("max_new_tokens") is not None and
                                   v_runtime.get("max_new_tokens") == b_runtime.get("max_new_tokens"),
            "same_generation_config": bool(v_runtime.get("generation_config")) and
                                      v_runtime.get("generation_config") == b_runtime.get("generation_config"),
            "b0_checkpoint_not_applied": b_runtime.get("checkpoint_applied") is False,
            "b0_step100_not_loaded": b_runtime.get("step100_h_encoder_checkpoint_loaded") is False,
            "b0_checkpoint_named_none": wrapped.get("checkpoint_name") == "NONE" and not wrapped.get("checkpoint_sha256"),
            "b0_adapter_not_installed": b_runtime.get("adapter_load_requested") is False
                                         and b_runtime.get("active_lora_during_generation") is False
                                         and b_runtime.get("adapter_disabled_context_used") is False
                                         and b_runtime.get("adapters_disabled_state_during_generation") is None
                                         and b_runtime.get("adapter_present_in_wrapper") is False
                                         and not b_runtime.get("active_adapter_names")
                                         and not b_runtime.get("adapter_modules_to_save"),
            "b0_h_disabled": b_runtime.get("h_seeded_slots") == 0
                               and b_runtime.get("h_logits_masked_by_static_manager") is True
                               and b_runtime.get("codebook_backend") == "static",
            "b0_span_aware_positions": b_runtime.get("effective_position_mode") == "base_token_end",
            "b0_generated_no_h_ids": not any(32011 <= int(token_id) < 32043 for token_id in (b_ids or [])),
            "b0_expanded_no_h_ids": not any(
                32011 <= int(token_id) < 32043 for token_id in (wrapped.get("expanded_token_ids") or [])
            ),
        })

    fields = (
        "exact_token_ids", "same_generated_length", "same_expanded_token_ids", "same_expanded_length",
        "same_termination_reason", "same_termination_token", "same_eos_reached",
        "same_model_id", "same_model_revision", "same_base_weight_hash", "same_generation_policy", "same_input_tokens", "same_dtype",
        "same_tokenizer_revision", "same_rendered_prompt", "same_eos_ids", "same_max_new_tokens",
        "same_tokenizer_id", "same_tokenizer_template", "same_attention_implementation", "same_generation_config",
        "same_git_commit", "same_git_branch", "same_worktree_dirty_status", "same_git_status_hash",
        "same_source_archive", "same_dataset_manifest", "same_canonical_dataset", "same_cuda_version",
        "same_gpu", "same_package_versions", "b0_checkpoint_not_applied", "b0_step100_not_loaded",
        "b0_checkpoint_named_none", "b0_adapter_not_installed", "b0_h_disabled", "b0_span_aware_positions",
        "b0_generated_no_h_ids", "b0_expanded_no_h_ids",
    )
    field_pass = {field: bool(compared) and all(row[field] for row in compared) for field in fields}
    return {
        "status": "PASS" if bool(compared) and not duplicate_pairs and not missing_prompt_ids and not unexpected_prompt_ids and all(field_pass.values()) else ("FAIL" if compared or duplicate_pairs or missing_prompt_ids or unexpected_prompt_ids else "NOT_TESTED"),
        "paired_prompt_count": len(compared),
        "expected_prompt_count": len(expected) if expected is not None else None,
        "duplicate_pairs": duplicate_pairs,
        "missing_prompt_ids": missing_prompt_ids,
        "unexpected_prompt_ids": unexpected_prompt_ids,
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


def b0_b1_adapter_isolation_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_prompt_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Verify B0 and B1 share base weights and differ only by the pinned upstream adapter."""
    indexed: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for record in records:
        prompt_id = record.get("prompt_id")
        condition = record.get("condition")
        if prompt_id and condition in {
            "B0_tokens_vanilla_weights",
            "B1_upstream_epfl_adapter_h_disabled",
        }:
            indexed.setdefault(str(prompt_id), {})[str(condition)] = record

    pairs = [
        rows
        for rows in indexed.values()
        if "B0_tokens_vanilla_weights" in rows
        and "B1_upstream_epfl_adapter_h_disabled" in rows
    ]
    expected = {str(prompt_id) for prompt_id in expected_prompt_ids} if expected_prompt_ids is not None else None
    paired_ids = {str(rows["B0_tokens_vanilla_weights"].get("prompt_id")) for rows in pairs}
    missing_prompt_ids = sorted(expected - paired_ids) if expected is not None else []
    pair_checks = []
    for rows in pairs:
        b0_rec = rows["B0_tokens_vanilla_weights"]
        b1_rec = rows["B1_upstream_epfl_adapter_h_disabled"]
        b0_runtime = b0_rec.get("runtime", {})
        b1_runtime = b1_rec.get("runtime", {})
        b0_gen = b0_rec.get("generated_token_ids") or []
        b1_gen = b1_rec.get("generated_token_ids") or []
        b0_exp = b0_rec.get("expanded_token_ids") or []
        b1_exp = b1_rec.get("expanded_token_ids") or []

        pair_checks.append({
            "prompt_id": b0_rec.get("prompt_id"),
            "same_base_hash": (
                bool(b0_runtime.get("base_phi_weight_sha256"))
                and b0_runtime.get("base_phi_weight_sha256") == b1_runtime.get("base_phi_weight_sha256")
            ),
            "b0_adapter_absent": (
                b0_runtime.get("adapter_load_requested") is False
                and b0_runtime.get("adapter_present_in_wrapper") is False
                and not b0_runtime.get("active_adapter_names")
                and not b0_runtime.get("adapter_modules_to_save")
                and b0_runtime.get("active_lora_during_generation") is False
                and b0_runtime.get("checkpoint_applied") is False
                and b0_runtime.get("step100_h_encoder_checkpoint_loaded") is False
            ),
            "b1_upstream_adapter_active": (
                b1_runtime.get("adapter_load_requested") is True
                and b1_runtime.get("adapter_present_in_wrapper") is True
                and b1_runtime.get("active_lora_during_generation") is True
                and bool(b1_runtime.get("active_adapter_names"))
                and int(b1_runtime.get("adapter_param_count", 0)) > 0
                and b1_runtime.get("upstream_adapter_id") == "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
                and b1_runtime.get("upstream_adapter_revision") == "11c461733a79d2a5de6b814585c3361ca2aacbe7"
            ),
            "b1_step100_not_loaded": (
                b1_runtime.get("checkpoint_applied") is False
                and b1_runtime.get("step100_h_encoder_checkpoint_loaded") is False
                and b1_runtime.get("checkpoint_step") is None
            ),
            "h_disabled_in_both": (
                b0_runtime.get("h_seeded_slots") == 0
                and b0_runtime.get("h_logits_masked_by_static_manager") is True
                and b1_runtime.get("h_seeded_slots") == 0
                and b1_runtime.get("h_logits_masked_by_static_manager") is True
            ),
            "no_h_emitted_in_b0": (
                not any(32011 <= int(tid) < 32043 for tid in b0_gen)
                and not any(32011 <= int(tid) < 32043 for tid in b0_exp)
            ),
            "no_h_emitted_in_b1": (
                not any(32011 <= int(tid) < 32043 for tid in b1_gen)
                and not any(32011 <= int(tid) < 32043 for tid in b1_exp)
            ),
            "prompt_and_generation_contract_match": (
                b0_runtime.get("input_token_ids_sha256") == b1_runtime.get("input_token_ids_sha256")
                and b0_runtime.get("rendered_prompt_sha256") == b1_runtime.get("rendered_prompt_sha256")
                and b0_runtime.get("generation_policy_sha256") == b1_runtime.get("generation_policy_sha256")
                and b0_runtime.get("max_new_tokens") == b1_runtime.get("max_new_tokens")
                and b0_runtime.get("eos_token_ids") == b1_runtime.get("eos_token_ids")
            ),
        })

    passed = (
        bool(pair_checks)
        and not missing_prompt_ids
        and all(all(v for k, v in row.items() if k != "prompt_id") for row in pair_checks)
    )
    return {
        "status": "PASS" if passed else ("FAIL" if pair_checks or missing_prompt_ids else "NOT_TESTED"),
        "paired_prompt_count": len(pair_checks),
        "expected_prompt_count": len(expected) if expected is not None else None,
        "missing_prompt_ids": missing_prompt_ids,
        "checks": pair_checks,
    }


def b1_b2_checkpoint_isolation_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_prompt_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Verify B1 and B2 share base weights and B2 loads the Step-100 parameter delta."""
    indexed: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for record in records:
        prompt_id = record.get("prompt_id")
        condition = record.get("condition")
        if prompt_id and condition in {
            "B1_upstream_epfl_adapter_h_disabled",
            "B2_step100_h_disabled",
            "B_h_disabled",
        }:
            indexed.setdefault(str(prompt_id), {})[str(condition)] = record

    pairs = [
        rows
        for rows in indexed.values()
        if "B1_upstream_epfl_adapter_h_disabled" in rows
        and ("B2_step100_h_disabled" in rows or "B_h_disabled" in rows)
    ]
    expected = {str(prompt_id) for prompt_id in expected_prompt_ids} if expected_prompt_ids is not None else None
    paired_ids = {str(rows["B1_upstream_epfl_adapter_h_disabled"].get("prompt_id")) for rows in pairs}
    missing_prompt_ids = sorted(expected - paired_ids) if expected is not None else []
    pair_checks = []
    for rows in pairs:
        b1_runtime = rows["B1_upstream_epfl_adapter_h_disabled"].get("runtime", {})
        b2_row = rows.get("B2_step100_h_disabled") or rows.get("B_h_disabled")
        b2_runtime = b2_row.get("runtime", {})
        changes = b2_runtime.get("checkpoint_changed_parameters", {})
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
            "prompt_id": rows["B1_upstream_epfl_adapter_h_disabled"].get("prompt_id"),
            "same_base_hash": (
                bool(b1_runtime.get("base_phi_weight_sha256"))
                and b1_runtime.get("base_phi_weight_sha256") == b2_runtime.get("base_phi_weight_sha256")
            ),
            "b1_step100_not_applied": b1_runtime.get("checkpoint_applied") is False,
            "b2_step100_applied": (
                b2_runtime.get("checkpoint_applied") is True
                and b2_runtime.get("step100_h_encoder_checkpoint_loaded") is True
                and int(b2_runtime.get("checkpoint_step", 0)) == 100
            ),
            "b2_lora_delta_verified": bool(b2_runtime.get("lora_delta_verified")) and bool(lora_names),
            "b2_changed_parameter_shapes_known": (
                full_inventory_known and bool(lora_names) and set(lora_names) == set(lora_shapes)
            ),
            "h_disabled_in_both": (
                b1_runtime.get("h_seeded_slots") == 0
                and b1_runtime.get("h_logits_masked_by_static_manager") is True
                and b2_runtime.get("h_seeded_slots") == 0
                and b2_runtime.get("h_logits_masked_by_static_manager") is True
            ),
        })
    passed = bool(pair_checks) and not missing_prompt_ids and all(all(v for k, v in row.items() if k != "prompt_id") for row in pair_checks)
    return {
        "status": "PASS" if passed else ("FAIL" if pair_checks or missing_prompt_ids else "NOT_TESTED"),
        "paired_prompt_count": len(pair_checks),
        "expected_prompt_count": len(expected) if expected is not None else None,
        "missing_prompt_ids": missing_prompt_ids,
        "checks": pair_checks,
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
        if prompt_id and condition in {"B0_tokens_vanilla_weights", "B_h_disabled", "B2_step100_h_disabled"}:
            indexed.setdefault(str(prompt_id), {})[str(condition)] = record
    pairs = [
        rows for rows in indexed.values()
        if "B0_tokens_vanilla_weights" in rows
        and ("B_h_disabled" in rows or "B2_step100_h_disabled" in rows)
    ]
    expected = {str(prompt_id) for prompt_id in expected_prompt_ids} if expected_prompt_ids is not None else None
    paired_ids = {str(rows["B0_tokens_vanilla_weights"].get("prompt_id")) for rows in pairs}
    missing_prompt_ids = sorted(expected - paired_ids) if expected is not None else []
    pair_checks = []
    for rows in pairs:
        b0_runtime = rows["B0_tokens_vanilla_weights"].get("runtime", {})
        b_trained_row = rows.get("B_h_disabled") or rows.get("B2_step100_h_disabled")
        b1_runtime = b_trained_row.get("runtime", {})
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
            "b0_adapter_absent_and_h_masked": (
                b0_runtime.get("active_lora_during_generation") is False
                and b0_runtime.get("adapter_load_requested") is False
                and b0_runtime.get("adapter_present_in_wrapper") is False
                and b0_runtime.get("adapter_disabled_context_used") is False
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
