"""P2 repair evaluation metrics (§10)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any


def extract_claims_from_answer(answer: str, *, anchor_pid: str | None) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for line in answer.splitlines():
        line = line.strip()
        if not line.startswith("-") and ":" not in line:
            continue
        m = re.match(r"[-*]\s*([^:]+):\s*(.+)", line)
        if not m:
            continue
        claims.append(
            {
                "slot_norm": m.group(1).strip(),
                "value": m.group(2).strip(),
                "referent_ids": [anchor_pid] if anchor_pid else [],
                "mode": "assertion",
            }
        )
    return claims


def evaluate_repair_output(
    trajectory: dict[str, Any],
    *,
    method_id: str,
    generated_answer: str,
    latency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    anchor = (trajectory.get("commitment") or {}).get("action_anchor")
    orig_claims = trajectory.get("claims") or []
    new_claims = extract_claims_from_answer(generated_answer, anchor_pid=anchor)
    orig_pm = sum(1 for c in orig_claims if c.get("normalized_axes", {}).get("pm_binary") == "PM")
    # Heuristic PM on new output: unsupported assertion vs E_seen plan
    from ccer.repair.baselines import build_evidence_plan_from_trajectory

    plan = build_evidence_plan_from_trajectory(trajectory)
    new_pm = 0
    for nc in new_claims:
        slot = nc.get("slot_norm") or ""
        st = (plan.get("slots") or {}).get(slot, {}).get("status")
        if st != "supported" and nc.get("value"):
            new_pm += 1
    supported_retained = 0
    supported_total = 0
    for oc in orig_claims:
        if oc.get("legacy_label") == "correct_binding":
            supported_total += 1
            slot = oc.get("slot_norm") or ""
            if any(c.get("slot_norm") == slot for c in new_claims):
                supported_retained += 1
    unknown_phrases = ("not confirmed", "unknown", "未确认", "not verified")
    has_unknown = any(p in generated_answer.lower() for p in unknown_phrases)
    return {
        "method_id": method_id,
        "trajectory_id": trajectory.get("trajectory_id"),
        "orig_pm_claims": orig_pm,
        "new_pm_claims_heuristic": new_pm,
        "orig_claim_count": len(orig_claims),
        "new_claim_count": len(new_claims),
        "supported_fact_retention": supported_retained / supported_total if supported_total else None,
        "unknown_recovery_signal": has_unknown,
        "latency": latency,
    }


def aggregate_repair_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(r["method_id"], []).append(r)
    summary = {}
    for mid, rs in by_method.items():
        summary[mid] = {
            "n": len(rs),
            "mean_new_pm": sum(r["new_pm_claims_heuristic"] for r in rs) / len(rs),
            "mean_new_claims": sum(r["new_claim_count"] for r in rs) / len(rs),
            "mean_supported_retention": _mean([r["supported_fact_retention"] for r in rs if r["supported_fact_retention"] is not None]),
            "latency_p50": _median([r.get("latency", {}).get("latency_sec") for r in rs if r.get("latency")]),
        }
    return summary


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[len(xs) // 2]
