"""Tau3 instance activation extraction."""
from __future__ import annotations

from typing import Any

import numpy as np

from ccer.mechanism.activation_store import save_activation_npz
from ccer.mechanism.line_a_instance_activations import LINE_A_INSTANCE_POSITIONS
from ccer.mechanism.tau3.cohort import Tau3InstanceRow, instance_npz_path
from ccer.mechanism.tau3.domain import Tau3DomainConfig, get_tau3_domain
from ccer.mechanism.tau3.replay import (
    resolve_tau3_quote_tokens,
    tau2_to_chat_messages,
    teacher_force_tau2_trajectory,
    tokenize_tau2_trajectory,
)
from ccer.replay.hf_forward import teacher_force_on_ids


def extract_tau3_instance_trajectory(
    *,
    model: Any,
    tokenizer: Any,
    trajectory: dict[str, Any],
    instances: list[Tau3InstanceRow],
    layer_indices: list[int],
    cfg: Tau3DomainConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or get_tau3_domain("telecom")
    """One teacher-forced forward per tau2 trajectory; one npz per instance."""
    messages = tau2_to_chat_messages(trajectory.get("messages") or [])
    tok_pack = tokenize_tau2_trajectory(messages, tokenizer)
    full_ids = tok_pack["full_ids"]
    if not full_ids:
        return {"n_instances": len(instances), "n_written": 0, "errors": ["empty_full_ids"]}

    fwd = teacher_force_on_ids(
        model,
        tokenizer,
        full_ids,
        prompt_token_count=0,
        output_ids=full_ids,
    )
    hidden_states = fwd["hidden_states"]

    def _vec_at(idx: int, li: int) -> np.ndarray | None:
        if idx is None or li + 1 >= len(hidden_states):
            return None
        return hidden_states[li + 1][0, int(idx)].detach().cpu().float().numpy().astype(np.float16)

    written = 0
    skipped = 0
    errors: list[str] = []
    for inst in instances:
        out_path = instance_npz_path(inst.instance_audit_key, inst.trajectory_id, cfg=cfg)
        if out_path.is_file():
            skipped += 1
            continue
        inst_positions = resolve_tau3_quote_tokens(
            serialized_text=tok_pack["serialized_text"],
            offset_mapping=tok_pack["offset_mapping"],
            quote_start=inst.quote_start,
            quote_end=inst.quote_end,
            response_quote=inst.response_quote,
        )
        positions: dict[str, int | None] = {
            "prompt_end": None,
            "claim_onset": inst_positions.get("claim_onset"),
            "pre_value": inst_positions.get("pre_value"),
        }
        vectors: dict[str, dict[str, np.ndarray]] = {}
        for pos in LINE_A_INSTANCE_POSITIONS:
            idx = positions.get(pos)
            if idx is None:
                continue
            layer_map: dict[str, np.ndarray] = {}
            for li in layer_indices:
                v = _vec_at(idx, li)
                if v is not None:
                    layer_map[f"layer_{li}"] = v
            if layer_map:
                vectors[pos] = layer_map

        if not vectors.get("claim_onset"):
            errors.append(f"missing_claim_onset:{inst.instance_audit_key}")
            continue

        save_activation_npz(
            out_path,
            trajectory_id=inst.trajectory_id,
            condition_id="tau3_line_a_instance",
            cohort="tau3_line_a_instance",
            layer_indices=layer_indices,
            positions={k: v for k, v in positions.items() if v is not None},
            vectors=vectors,
            metadata={
                "instance_audit_key": inst.instance_audit_key,
                "quote_start": inst.quote_start,
                "quote_end": inst.quote_end,
                "response_quote": inst.response_quote,
                "gold_verdict": inst.gold_verdict,
                "domain": cfg.domain,
            },
        )
        written += 1

    return {
        "n_instances": len(instances),
        "n_written": written,
        "n_skipped_existing": skipped,
        "errors": errors,
        "layer_indices": layer_indices,
    }
