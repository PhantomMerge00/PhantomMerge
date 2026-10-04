"""End-to-end mitigation dispatch: each system uses its own detection gate."""

from __future__ import annotations

from typing import Any, Literal

from ccer.mechanism.baseline_cove_official import apply_cove_official_rewrite
from ccer.mechanism.baseline_rarr_official import apply_rarr_official_rewrite
from ccer.mechanism.e2e_row_builder import build_e2e_inference_row
from ccer.mechanism.line_l_baseline_copy import apply_copy_baseline_rewrite
from ccer.mechanism.prism_l_plus import apply_prism_l_plus

GateSource = Literal[
    "probe_delete",
    "rarr_native",
    "cove_native",
    "copy_all_claims",
    "prism_l_plus_a",
    "prism_l_plus_b",
]

E2EMethodId = Literal[
    "track_k_e2e",
    "b1_rarr_e2e",
    "b2_cove_e2e",
    "b3_copy_e2e",
    "prism_l_plus_a_e2e",
    "prism_l_plus_b_e2e",
]

METHOD_TO_GATE: dict[E2EMethodId, GateSource] = {
    "track_k_e2e": "probe_delete",
    "b1_rarr_e2e": "rarr_native",
    "b2_cove_e2e": "cove_native",
    "b3_copy_e2e": "copy_all_claims",
    "prism_l_plus_a_e2e": "prism_l_plus_a",
    "prism_l_plus_b_e2e": "prism_l_plus_b",
}


def _probe_delete(
    row: dict[str, Any],
    *,
    arm: str,
    tau: float,
) -> dict[str, Any]:
    p_pm = float(row.get("p_pm", 0.0))
    quote = str(row.get("response_quote") or "")
    if p_pm <= tau:
        return {
            "arm": arm,
            "branch": "e2e_probe_keep",
            "action": "keep",
            "quote_before": quote,
            "quote_after": quote,
            "method": "track_k_e2e",
            "gate_source": "probe_delete",
        }
    return {
        "arm": arm,
        "branch": "e2e_probe_delete",
        "action": "delete",
        "quote_before": quote,
        "quote_after": None,
        "method": "track_k_e2e",
        "gate_source": "probe_delete",
    }


def apply_e2e_mitigation(
    eval_row: dict[str, Any],
    traj: dict[str, Any],
    *,
    method: E2EMethodId,
    arm: str,
    tau: float,
    anchor_pid: str,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Run E2E mitigation: heuristic parse + native gate per method."""
    inference = build_e2e_inference_row(eval_row, traj)
    inference["tau"] = tau
    gate = METHOD_TO_GATE[method]
    quote = str(inference.get("response_quote") or "")

    if gate == "probe_delete":
        out = _probe_delete(inference, arm=arm, tau=tau)
    elif gate == "rarr_native":
        out = apply_rarr_official_rewrite(
            {**inference, "p_pm": 1.0},
            traj,
            anchor_pid=anchor_pid,
            arm=arm,
            use_llm=use_llm,
            allow_extractive_shortcut=False,
        )
        out = {**out, "method": "b1_rarr_e2e", "gate_source": "rarr_native"}
        if str(out.get("branch") or "") == "skip_unflagged":
            out["branch"] = "rarr_native_no_edit"
            out["action"] = "keep"
            out["quote_after"] = quote
    elif gate == "cove_native":
        out = apply_cove_official_rewrite(
            {**inference, "p_pm": 1.0},
            traj,
            anchor_pid=anchor_pid,
            arm=arm,
            use_llm=use_llm,
            allow_extractive_shortcut=False,
        )
        out = {**out, "method": "b2_cove_e2e", "gate_source": "cove_native"}
        if str(out.get("branch") or "") == "skip_unflagged":
            out["branch"] = "cove_native_no_edit"
            out["action"] = "keep"
            out["quote_after"] = quote
    elif gate == "copy_all_claims":
        out = apply_copy_baseline_rewrite(
            {**inference, "p_pm": 1.0},
            traj,
            anchor_pid=anchor_pid,
            arm=arm,
            use_gold_cem_hint=False,
        )
        out = {**out, "method": "b3_copy_e2e", "gate_source": "copy_all_claims"}
        if str(out.get("branch") or "") == "skip_unflagged":
            out["branch"] = "copy_no_candidate"
            out["action"] = "keep"
            out["quote_after"] = quote
    elif gate == "prism_l_plus_a":
        out = apply_prism_l_plus(
            inference,
            traj,
            anchor_pid=anchor_pid,
            arm=arm,
            probe_only_mode="delete_always",
            use_llm=use_llm,
            gate_enabled=True,
        )
        out = {**out, "method": "prism_l_plus_a_e2e", "gate_source": "prism_l_plus_a"}
    elif gate == "prism_l_plus_b":
        out = apply_prism_l_plus(
            inference,
            traj,
            anchor_pid=anchor_pid,
            arm=arm,
            probe_only_mode="y0_delete_y1_rarr",
            use_llm=use_llm,
            gate_enabled=True,
        )
        out = {**out, "method": "prism_l_plus_b_e2e", "gate_source": "prism_l_plus_b"}
    else:
        raise ValueError(f"unknown gate: {gate}")

    return {
        **eval_row,
        **out,
        "anchor_pid": anchor_pid,
        "anchor_pid_source": "heuristic",
        "e2e_mode": True,
    }
