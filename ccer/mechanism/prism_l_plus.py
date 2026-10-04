"""PRISM-L+: PRISM gate + Obs_τ RewriteStack with acceptance-based PM elimination."""

from __future__ import annotations

from typing import Any, Literal

from ccer.mechanism.baseline_rarr_official import apply_rarr_official_rewrite
from ccer.mechanism.line_l_anchor_extract import (
    apply_extractive_rewrite,
    literal_replace_in_quote,
    lookup_cem_corrective_value,
    lookup_obs_tau_slot,
)
from ccer.mechanism.line_l_baseline_copy import apply_copy_baseline_rewrite
from ccer.mechanism.line_l_pm_router import route_rewrite_strategy_runtime
from ccer.mechanism.line_l_rewrite_acceptance import rewrite_passes_acceptance
from ccer.mechanism.line_l_wrong_anchor import (
    resolve_acceptance_pid_committed,
    resolve_acceptance_pid_textual,
)
from ccer.mechanism.prism_audit import classify_audit_state

ProbeOnlyMode = Literal["delete_always", "y0_delete_y1_rarr"]


def _audit_state(row: dict[str, Any], tau: float) -> str:
    if row.get("prism_audit_state"):
        return str(row["prism_audit_state"])
    return classify_audit_state(
        float(row.get("p_pm") or 0.0),
        tau,
        bool(row.get("prism_slot_type_aligned") or row.get("slot_type_aligned")),
    )


def _enrich_acceptance_meta(
    out: dict[str, Any],
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    acceptance_pid: str,
    pm_route: str,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    return {
        **out,
        "acceptance_pid_textual": acceptance_pid,
        "acceptance_pid_committed": resolve_acceptance_pid_committed(row, traj),
        "verify_policy": bundle.get("policy"),
        "rules_ok": bundle.get("rules_ok"),
        "nli_ok": bundle.get("nli_ok"),
        "reject_reasons": bundle.get("reasons"),
        "pm_route": pm_route,
        "method": "prism_l_plus",
    }


def _finalize_rewrite(
    out: dict[str, Any],
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    acceptance_pid: str,
    pm_route: str,
    rewrite_stack: str,
    branch: str,
    arm: str,
) -> dict[str, Any] | None:
    if str(out.get("action") or "") != "rewrite":
        return None
    bundle = rewrite_passes_acceptance(out, row, traj, acceptance_pid=acceptance_pid)
    if not bundle.get("ok"):
        return None
    return _enrich_acceptance_meta(
        {
            **out,
            "arm": arm,
            "branch": branch,
            "rewrite_stack": rewrite_stack,
        },
        row,
        traj,
        acceptance_pid=acceptance_pid,
        pm_route=pm_route,
        bundle=bundle,
    )


def _apply_cem_corrective(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    acceptance_pid: str,
    arm: str,
) -> dict[str, Any] | None:
    slot_norm = str(row.get("slot_norm") or "")
    cem = lookup_cem_corrective_value(row, traj, anchor_pid=acceptance_pid, slot_norm=slot_norm)
    if cem.branch != "extraction_hit" or not cem.v_anchor:
        return None
    quote = str(row.get("response_quote") or "")
    new_quote = literal_replace_in_quote(
        quote,
        slot=str(row.get("slot") or slot_norm),
        old_value=str(row.get("claim_value") or ""),
        v_anchor=cem.v_anchor,
    )
    return {
        "arm": arm,
        "action": "rewrite",
        "quote_before": quote,
        "quote_after": new_quote,
        "v_anchor": cem.v_anchor,
        "source_field": cem.source_field,
        "candidate_source": cem.source_field,
        "evidence_kind": cem.evidence_kind,
    }


def _rewrite_stack(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    arm: str,
    use_llm: bool = True,
) -> dict[str, Any]:
    """CEM corrective → extractive → structured copy → RARR; acceptance gate on textual PID."""
    quote = str(row.get("response_quote") or "")
    acceptance_pid = resolve_acceptance_pid_textual(row, traj) or anchor_pid
    strategy = route_rewrite_strategy_runtime(row, traj, acceptance_pid=acceptance_pid)
    pm_route = strategy.pm_type

    if strategy.pm_type == "CAP":
        obs = lookup_obs_tau_slot(traj, anchor_pid=acceptance_pid, slot_norm=str(row.get("slot_norm") or ""))
        if obs.branch != "extraction_hit":
            return {
                "arm": arm,
                "branch": "prism_l_cap_delete",
                "action": "delete",
                "quote_before": quote,
                "quote_after": None,
                "miss_reason": "cap_no_obs_tau_slot",
                "method": "prism_l_plus",
                "rewrite_stack": "fail",
                "pm_route": pm_route,
            }

    cem_out = _apply_cem_corrective(row, traj, acceptance_pid=acceptance_pid, arm=arm)
    if cem_out:
        finalized = _finalize_rewrite(
            cem_out,
            row,
            traj,
            acceptance_pid=acceptance_pid,
            pm_route="CEM",
            rewrite_stack="cem_corrective",
            branch="prism_l_cem_corrective",
            arm=arm,
        )
        if finalized:
            return finalized

    ext = apply_extractive_rewrite(row, traj, anchor_pid=acceptance_pid, arm=arm)
    finalized = _finalize_rewrite(
        ext,
        row,
        traj,
        acceptance_pid=acceptance_pid,
        pm_route=pm_route,
        rewrite_stack="extractive",
        branch="prism_l_extractive",
        arm=arm,
    )
    if finalized:
        return finalized

    if strategy.allow_copy:
        copy_out = apply_copy_baseline_rewrite(
            row,
            traj,
            anchor_pid=acceptance_pid,
            arm=arm,
            allow_snippet_fallback=strategy.allow_snippet_fallback,
        )
        finalized = _finalize_rewrite(
            copy_out,
            row,
            traj,
            acceptance_pid=acceptance_pid,
            pm_route=pm_route,
            rewrite_stack="copy",
            branch="prism_l_copy",
            arm=arm,
        )
        if finalized:
            return finalized

    if use_llm and strategy.allow_llm:
        rarr_out = apply_rarr_official_rewrite(
            row, traj, anchor_pid=acceptance_pid, arm=arm, use_llm=True
        )
        finalized = _finalize_rewrite(
            rarr_out,
            row,
            traj,
            acceptance_pid=acceptance_pid,
            pm_route=pm_route,
            rewrite_stack="rarr_official",
            branch="prism_l_rarr_official",
            arm=arm,
        )
        if finalized:
            return finalized

    return {
        "arm": arm,
        "branch": "prism_l_stack_fail",
        "action": "delete",
        "quote_before": quote,
        "quote_after": None,
        "miss_reason": "rewrite_stack_exhausted",
        "method": "prism_l_plus",
        "rewrite_stack": "fail",
        "pm_route": pm_route,
        "acceptance_pid_textual": acceptance_pid,
        "acceptance_pid_committed": resolve_acceptance_pid_committed(row, traj),
    }


def apply_prism_l_plus(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    arm: str = "prism_l_plus",
    probe_only_mode: ProbeOnlyMode = "delete_always",
    use_llm: bool = True,
    gate_enabled: bool = True,
) -> dict[str, Any]:
    """PRISM gate routing + RewriteStack for confirmed_risk."""
    p_pm = float(row.get("p_pm", 0.0))
    tau = float(row.get("tau", 1.0))
    quote = str(row.get("response_quote") or "")
    audit_state = _audit_state(row, tau)

    if not gate_enabled:
        out = _rewrite_stack(row, traj, anchor_pid=anchor_pid, arm=arm, use_llm=use_llm)
        return {
            **out,
            "mitigation_policy": "prism_audit_only",
            "prism_audit_state": audit_state,
        }

    if audit_state in ("clean", "latent_slot"):
        return {
            "arm": arm,
            "branch": "prism_keep",
            "action": "keep",
            "quote_before": quote,
            "quote_after": quote,
            "mitigation_policy": "prism_l_plus",
            "prism_audit_state": audit_state,
            "method": "prism_l_plus",
        }

    if audit_state == "probe_only":
        if probe_only_mode == "delete_always":
            return {
                "arm": arm,
                "branch": "prism_probe_only_delete",
                "action": "delete",
                "quote_before": quote,
                "quote_after": None,
                "mitigation_policy": "prism_l_plus",
                "prism_audit_state": audit_state,
                "miss_reason": "probe_only_delete_always",
                "method": "prism_l_plus",
            }
        y_pm = int(row.get("y_pm", row.get("y", 0)) or 0)
        if y_pm == 0:
            return {
                "arm": arm,
                "branch": "prism_probe_only_y0_delete",
                "action": "delete",
                "quote_before": quote,
                "quote_after": None,
                "mitigation_policy": "prism_l_plus",
                "prism_audit_state": audit_state,
                "miss_reason": "probe_only_fp_delete",
                "method": "prism_l_plus",
            }
        out = _rewrite_stack(row, traj, anchor_pid=anchor_pid, arm=arm, use_llm=use_llm)
        return {
            **out,
            "mitigation_policy": "prism_l_plus",
            "prism_audit_state": audit_state,
            "probe_only_mode": probe_only_mode,
        }

    out = _rewrite_stack(row, traj, anchor_pid=anchor_pid, arm=arm, use_llm=use_llm)
    return {
        **out,
        "mitigation_policy": "prism_l_plus",
        "prism_audit_state": audit_state,
    }
