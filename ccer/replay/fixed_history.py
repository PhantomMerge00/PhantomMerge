"""fixed_history replay harness (§6)."""
from __future__ import annotations

from typing import Any

from ccer.counterfactual.base import OperatorResult, filter_held_fixed_present, validate_held_fixed
from ccer.io_utils import sha256_json
from ccer.replay.answer_utils import extract_answer, validate_replay_output
from ccer.replay.vllm_replay import vllm_generate

from ccer.replay.hf_forward import teacher_force_score

__all__ = ["extract_answer", "run_fixed_history", "teacher_force_score"]


def run_fixed_history(
    trajectory: dict[str, Any],
    op_result: OperatorResult,
    *,
    estimand: str = "fixed_history_final_synthesis",
) -> dict[str, Any]:
    base_messages = trajectory.get("messages_final_call") or []
    base_hash = trajectory.get("input_hash") or sha256_json(base_messages)

    if op_result.invalid_reason or op_result.semantic_validation == "invalid":
        cf = op_result.to_counterfactual_record(
            root_id=trajectory["root_id"],
            pair_id=f"{trajectory['root_id']}:{op_result.condition_id}",
            estimand=estimand,
            split=trajectory.get("split", "unassigned"),
            base_hash=base_hash,
        )
        return {
            "status": "invalid",
            "counterfactual": cf,
            "generation": None,
        }

    held = filter_held_fixed_present(base_messages, op_result.held_fixed)
    ok, fails = validate_held_fixed(base_messages, op_result.messages, held)
    if not ok and op_result.held_fixed:
        op_result.semantic_validation = "invalid"
        op_result.invalid_reason = ";".join(fails)
        cf = op_result.to_counterfactual_record(
            root_id=trajectory["root_id"],
            pair_id=f"{trajectory['root_id']}:{op_result.condition_id}",
            estimand=estimand,
            split=trajectory.get("split", "unassigned"),
            base_hash=base_hash,
        )
        return {"status": "invalid", "counterfactual": cf, "generation": None}

    gen_text, meta = vllm_generate(op_result.messages, final_synthesis=True)
    _, harness_err = validate_replay_output(gen_text)
    cf = op_result.to_counterfactual_record(
        root_id=trajectory["root_id"],
        pair_id=f"{trajectory['root_id']}:{op_result.condition_id}",
        estimand=estimand,
        split=trajectory.get("split", "unassigned"),
        base_hash=base_hash,
    )
    orig_answer = extract_answer(str((trajectory.get("metadata") or {}).get("final_answer") or ""))
    new_answer = extract_answer(gen_text)
    return {
        "status": "ok",
        "counterfactual": cf,
        "generation": {
            "text": gen_text,
            "answer": new_answer,
            "original_answer": orig_answer,
            "latency": meta,
            "harness_valid": harness_err is None,
            "harness_error": harness_err,
        },
    }


def sequence_score_placeholder(*, messages: list[dict[str, str]], candidate_value: str) -> dict[str, Any]:
    """Backward-compatible alias; delegates to HF teacher_force_score."""
    return teacher_force_score(messages=messages, candidate_value=candidate_value)
