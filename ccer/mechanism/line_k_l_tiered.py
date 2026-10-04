"""Track K+L unified mitigation: rewrite all flagged claims; delete is Track-K-only baseline."""

from __future__ import annotations

from typing import Any, Literal

from ccer.mechanism.line_l_anchor_extract import apply_extractive_rewrite
from ccer.mechanism.line_l_baseline_copy import apply_copy_baseline_rewrite
from ccer.mechanism.baseline_cove_official import apply_cove_official_rewrite
from ccer.mechanism.baseline_rarr_official import apply_rarr_official_rewrite
from ccer.mechanism.line_l_baseline_cove import apply_cove_baseline_rewrite
from ccer.mechanism.line_l_baseline_rarr import apply_rarr_baseline_rewrite
from ccer.mechanism.prism_l_plus import apply_prism_l_plus
from ccer.mechanism.line_l_generative_rewrite import RewriteFallback, apply_fallback_rewrite
from ccer.mechanism.line_l_pm_router import route_rewrite_strategy
from ccer.mechanism.prism_audit import classify_audit_state

MitigationTier = Literal["keep", "delete", "rewrite"]
MitigationPolicy = Literal[
    "delete_or_rewrite_v1",
    "rewrite_all_flagged_v2",
    "selective_grounded_v3",
    "prism_gated_v1",
    "prism_audit_only",
]

MethodId = Literal[
    "b1_rarr",
    "b2_cove",
    "b1_rarr_official",
    "b2_cove_official",
    "b3_copy",
    "ours_v3",
    "prism_l_plus_a",
    "prism_l_plus_b",
    "prism_audit_only",
    "extractive_only",
    "wrong_anchor_extractive",
    "eval_e1_v2_stub",
]


def outcome_to_tier(outcome: dict[str, Any]) -> MitigationTier:
    action = str(outcome.get("action") or "keep")
    if action == "rewrite":
        return "rewrite"
    if action == "delete":
        return "delete"
    return "keep"


def apply_method_rewrite(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    arm: str,
    method: MethodId,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Dispatch to literature baseline or Ours v3."""
    if method == "b1_rarr":
        return apply_rarr_baseline_rewrite(row, traj, anchor_pid=anchor_pid, arm=arm, use_llm=use_llm)
    if method == "b2_cove":
        return apply_cove_baseline_rewrite(row, traj, anchor_pid=anchor_pid, arm=arm, use_llm=use_llm)
    if method == "b1_rarr_official":
        return apply_rarr_official_rewrite(row, traj, anchor_pid=anchor_pid, arm=arm, use_llm=use_llm)
    if method == "b2_cove_official":
        return apply_cove_official_rewrite(row, traj, anchor_pid=anchor_pid, arm=arm, use_llm=use_llm)
    if method == "prism_l_plus_a":
        return apply_prism_l_plus(
            row, traj, anchor_pid=anchor_pid, arm=arm, probe_only_mode="delete_always", use_llm=use_llm
        )
    if method == "prism_l_plus_b":
        return apply_prism_l_plus(
            row,
            traj,
            anchor_pid=anchor_pid,
            arm=arm,
            probe_only_mode="y0_delete_y1_rarr",
            use_llm=use_llm,
        )
    if method == "prism_audit_only":
        return apply_prism_l_plus(
            row, traj, anchor_pid=anchor_pid, arm=arm, gate_enabled=False, use_llm=use_llm
        )
    if method == "b3_copy":
        return apply_copy_baseline_rewrite(row, traj, anchor_pid=anchor_pid, arm=arm)
    if method in ("extractive_only", "wrong_anchor_extractive"):
        return apply_extractive_rewrite(row, traj, anchor_pid=anchor_pid, arm=arm)
    if method == "ours_v3":
        return apply_selective_grounded_v3(
            row, traj, anchor_pid=anchor_pid, arm=arm, use_llm=use_llm
        )
    raise ValueError(f"unknown method: {method}")


def apply_selective_grounded_v3(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    arm: str = "ours_v3",
    use_llm: bool = True,
) -> dict[str, Any]:
    """Ours: PM-type router + extractive → copy → RARR (AH only) → delete."""
    strategy = route_rewrite_strategy(row, traj, anchor_pid=anchor_pid)
    if strategy.skip_rewrite:
        p_pm = float(row.get("p_pm", 0.0))
        tau = float(row.get("tau", 1.0))
        quote = str(row.get("response_quote") or "")
        if p_pm <= tau:
            return {
                "arm": arm,
                "branch": "skip_unflagged",
                "action": "keep",
                "quote_before": quote,
                "quote_after": quote,
                "mitigation_policy": "selective_grounded_v3",
                "method": "ours_v3",
                "pm_route": strategy.pm_type,
            }
        return {
            "arm": arm,
            "branch": "skip_routed",
            "action": "delete" if strategy.on_miss == "delete" else "keep",
            "quote_before": quote,
            "quote_after": None if strategy.on_miss == "delete" else quote,
            "miss_reason": strategy.reason,
            "mitigation_policy": "selective_grounded_v3",
            "method": "ours_v3",
            "pm_route": strategy.pm_type,
        }

    outcome = apply_extractive_rewrite(row, traj, anchor_pid=anchor_pid, arm=arm)
    p_pm = float(row.get("p_pm", 0.0))
    tau = float(row.get("tau", 1.0))
    if p_pm <= tau or str(outcome.get("action") or "") != "delete":
        return {
            **outcome,
            "mitigation_tier": outcome_to_tier(outcome),
            "mitigation_policy": "selective_grounded_v3",
            "method": "ours_v3",
            "pm_route": strategy.pm_type,
        }

    if strategy.allow_copy:
        copy_out = apply_copy_baseline_rewrite(row, traj, anchor_pid=anchor_pid, arm=arm)
        if str(copy_out.get("action") or "") == "rewrite":
            return {
                **copy_out,
                "mitigation_tier": "rewrite",
                "mitigation_policy": "selective_grounded_v3",
                "method": "ours_v3",
                "pm_route": strategy.pm_type,
            }

    if strategy.allow_llm and use_llm and strategy.on_miss == "rarr":
        rarr_out = apply_rarr_baseline_rewrite(row, traj, anchor_pid=anchor_pid, arm=arm, use_llm=True)
        if str(rarr_out.get("action") or "") == "rewrite":
            return {
                **rarr_out,
                "mitigation_tier": "rewrite",
                "mitigation_policy": "selective_grounded_v3",
                "method": "ours_v3",
                "pm_route": strategy.pm_type,
            }

    return {
        **outcome,
        "mitigation_tier": "delete",
        "mitigation_policy": "selective_grounded_v3",
        "method": "ours_v3",
        "pm_route": strategy.pm_type,
    }


def _prism_audit_state_for_row(row: dict[str, Any], *, tau: float) -> str:
    if row.get("prism_audit_state"):
        return str(row["prism_audit_state"])
    return classify_audit_state(
        float(row.get("p_pm") or 0.0),
        tau,
        bool(row.get("prism_slot_type_aligned") or row.get("slot_type_aligned")),
    )


def apply_prism_gated_v1(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    arm: str = "prism_gated_v1",
    rewrite_fallback: RewriteFallback = "stub",
    use_llm: bool = True,
) -> dict[str, Any]:
    """Dual-channel gate via PRISM-L+ RewriteStack (official RARR, strict verify)."""
    out = apply_prism_l_plus(
        row,
        traj,
        anchor_pid=anchor_pid,
        arm=arm,
        probe_only_mode="delete_always",
        use_llm=use_llm,
    )
    tier = outcome_to_tier(out)
    return {
        **out,
        "mitigation_tier": tier,
        "mitigation_policy": "prism_gated_v1",
    }


def apply_tiered_mitigation(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    arm: str = "tiered_rewrite_all",
    policy: MitigationPolicy = "rewrite_all_flagged_v2",
    rewrite_fallback: RewriteFallback = "stub",
    method: MethodId | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    if method is not None:
        return apply_method_rewrite(
            row, traj, anchor_pid=anchor_pid, arm=arm, method=method, use_llm=use_llm
        )
    if policy == "selective_grounded_v3":
        return apply_selective_grounded_v3(
            row, traj, anchor_pid=anchor_pid, arm=arm, use_llm=use_llm
        )
    if policy == "prism_gated_v1":
        return apply_prism_gated_v1(
            row, traj, anchor_pid=anchor_pid, arm=arm, rewrite_fallback=rewrite_fallback
        )
    if policy == "prism_audit_only":
        out = apply_tiered_mitigation(
            row,
            traj,
            anchor_pid=anchor_pid,
            arm=arm,
            policy="rewrite_all_flagged_v2",
            rewrite_fallback=rewrite_fallback,
        )
        return {**out, "mitigation_policy": "prism_audit_only"}
    """
    Unified K+L policy (v2 default):
    - unflagged        -> keep
    - flagged + hit    -> evidence-grounded extractive rewrite
    - flagged + miss   -> stub or LLM rewrite (never delete)

    v1 ``delete_or_rewrite`` retained for ablation: miss -> delete.
    """
    outcome = apply_extractive_rewrite(row, traj, anchor_pid=anchor_pid, arm=arm)
    p_pm = float(row.get("p_pm", 0.0))
    tau = float(row.get("tau", 1.0))

    if (
        policy == "rewrite_all_flagged_v2"
        and p_pm > tau
        and str(outcome.get("action") or "") == "delete"
    ):
        return apply_fallback_rewrite(
            row,
            traj,
            anchor_pid=anchor_pid,
            arm=arm,
            miss_reason=str(outcome.get("miss_reason") or ""),
            prior_branch=str(outcome.get("branch") or ""),
            fallback=rewrite_fallback,
        )

    tier = outcome_to_tier(outcome)
    pol = policy
    if tier == "rewrite" and policy == "rewrite_all_flagged_v2":
        outcome = {
            **outcome,
            "evidence_kind": outcome.get("evidence_kind") or "extractive",
        }
    return {
        **outcome,
        "mitigation_tier": tier,
        "mitigation_policy": pol,
    }


def compute_tiered_routing_stats(rows: list[dict[str, Any]], *, tau: float) -> dict[str, Any]:
    """Summarize keep/delete/rewrite counts for flagged claims."""
    pm_labels = {
        "cross_object_merge": "CEM",
        "constraint_projection": "CAP",
        "anchored_hallucination": "AH",
        "trajectory_clean": "clean_expansion",
    }
    flagged = [r for r in rows if float(r.get("p_pm", 0.0)) > float(tau)]
    by_tier: dict[str, int] = {"keep": 0, "delete": 0, "rewrite": 0}
    by_branch: dict[str, int] = {}
    by_pm: dict[str, dict[str, int]] = {}

    for r in rows:
        tier = str(r.get("mitigation_tier") or outcome_to_tier(r))
        by_tier[tier] = by_tier.get(tier, 0) + 1
        branch = str(r.get("branch") or "")
        if branch:
            by_branch[branch] = by_branch.get(branch, 0) + 1
        if float(r.get("p_pm", 0.0)) <= float(tau):
            continue
        pm = pm_labels.get(str(r.get("gold_verdict") or ""), "other")
        bucket = by_pm.setdefault(pm, {"flagged": 0, "delete": 0, "rewrite": 0})
        bucket["flagged"] += 1
        if tier in ("delete", "rewrite"):
            bucket[tier] += 1

    n_flagged = len(flagged)
    n_rewrite = by_tier.get("rewrite", 0)
    n_extractive = by_branch.get("extraction_hit", 0)
    n_fallback = by_branch.get("fallback_rewrite", 0)
    return {
        "tau": tau,
        "n_total": len(rows),
        "n_flagged": n_flagged,
        "n_keep": by_tier.get("keep", 0),
        "n_delete": by_tier.get("delete", 0),
        "n_rewrite": n_rewrite,
        "n_extractive_rewrite": n_extractive,
        "n_fallback_rewrite": n_fallback,
        "rewrite_rate_among_flagged": (n_rewrite / n_flagged) if n_flagged else None,
        "delete_rate_among_flagged": (by_tier.get("delete", 0) / n_flagged) if n_flagged else None,
        "by_pm_type_flagged": by_pm,
        "by_branch": by_branch,
    }
