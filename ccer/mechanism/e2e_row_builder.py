"""Build inference rows for E2E mitigation (evaluation labels kept separate)."""

from __future__ import annotations

from typing import Any

from ccer.mechanism.anchor_resolve_heuristic import (
    resolve_heuristic_anchor_pid,
    resolve_heuristic_donor_pids,
)
from ccer.mechanism.claim_segmentation import match_segmented_to_quote, parse_bullet_quote, segment_attribute_claims
from ccer.mechanism.prism_audit import classify_audit_state

_ORACLE_FIELDS = frozenset(
    {
        "committed_anchor_pid",
        "anchor_evidence_quote",
        "donor_owners",
        "gold_verdict",
    }
)


def build_e2e_inference_row(
    eval_row: dict[str, Any],
    traj: dict[str, Any],
    *,
    p_pm: float | None = None,
    prism_slot_type_aligned: bool | None = None,
) -> dict[str, Any]:
    """Strip adjudication oracle fields; populate heuristic parse + diagnostics."""
    quote = str(eval_row.get("response_quote") or "")
    segments = segment_attribute_claims(traj)
    seg = match_segmented_to_quote(segments, quote)
    if seg:
        slot, slot_norm, claim_value = seg.slot, seg.slot_norm, seg.claim_value
        section = seg.section
    else:
        parsed = parse_bullet_quote(quote)
        if parsed:
            slot, slot_norm, claim_value = parsed
        else:
            slot = str(eval_row.get("slot") or "")
            slot_norm = str(eval_row.get("slot_norm") or "")
            claim_value = str(eval_row.get("claim_value") or "")
        section = "other"

    anchor_pid = resolve_heuristic_anchor_pid(traj)
    donors = resolve_heuristic_donor_pids(traj, anchor_pid=anchor_pid)
    donor_owners = [{"pid": d} for d in donors]

    p = float(p_pm if p_pm is not None else eval_row.get("p_pm") or 0.0)
    tau = float(eval_row.get("tau") or 0.05)
    slot_aligned = bool(
        prism_slot_type_aligned
        if prism_slot_type_aligned is not None
        else eval_row.get("prism_slot_type_aligned") or eval_row.get("slot_type_aligned")
    )

    row = {k: v for k, v in eval_row.items() if k not in _ORACLE_FIELDS}
    row.update(
        {
            "slot": slot,
            "slot_norm": slot_norm,
            "claim_value": claim_value,
            "claim_section": section,
            "anchor_pid_heuristic": anchor_pid,
            "donor_owners": donor_owners,
            "p_pm": p,
            "prism_slot_type_aligned": slot_aligned,
            "prism_audit_state": classify_audit_state(p, tau, slot_aligned),
            "e2e_mode": True,
        }
    )
    return row
