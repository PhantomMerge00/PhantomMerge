"""Line L: wrong-anchor PID resolution (entity only, no activation vectors)."""

from __future__ import annotations

from typing import Any

from ccer.adjudication.loader import resolve_donor_pid
from ccer.counterfactual.bundles import resolve_cem_rival_value
from ccer.replay.evidence_mask import _rival_pids_from_evidence


def _user_content(traj: dict[str, Any]) -> str:
    for m in traj.get("messages_final_call") or []:
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def resolve_committed_anchor_pid(row: dict[str, Any], traj: dict[str, Any]) -> str:
    return str(
        row.get("committed_anchor_pid")
        or (traj.get("commitment") or {}).get("action_anchor")
        or (traj.get("text_anchor") or {}).get("selected_pid")
        or row.get("textual_selected_pid")
        or ""
    ).strip()


def resolve_acceptance_pid_textual(row: dict[str, Any], traj: dict[str, Any]) -> str:
    """User-visible selected product for rewrite acceptance (primary runtime gate)."""
    return str(
        row.get("textual_selected_pid")
        or (traj.get("commitment") or {}).get("action_anchor")
        or (traj.get("text_anchor") or {}).get("selected_pid")
        or row.get("committed_anchor_pid")
        or ""
    ).strip()


def resolve_acceptance_pid_committed(row: dict[str, Any], traj: dict[str, Any]) -> str:
    """Committed / attribution anchor (evaluation second track)."""
    return resolve_committed_anchor_pid(row, traj)


def resolve_acceptance_pid_textual(row: dict[str, Any], traj: dict[str, Any]) -> str:
    """Acceptance corpus PID for rewrite gate (textual selection first)."""
    return str(
        row.get("textual_selected_pid")
        or (traj.get("commitment") or {}).get("action_anchor")
        or (traj.get("text_anchor") or {}).get("selected_pid")
        or row.get("committed_anchor_pid")
        or ""
    ).strip()


def resolve_acceptance_pid_committed(row: dict[str, Any], traj: dict[str, Any]) -> str:
    """Acceptance corpus PID for committed-anchor evaluation track."""
    return resolve_committed_anchor_pid(row, traj)


def _donor_pids_from_row(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for d in row.get("donor_owners") or []:
        if isinstance(d, dict):
            pid = str(d.get("pid") or "").strip()
        else:
            pid = str(d).strip()
        if pid:
            out.append(pid)
    return out


def resolve_wrong_anchor_pid(row: dict[str, Any], traj: dict[str, Any]) -> str | None:
    """Pick non-anchor entity PID for wrong_anchor_extraction control."""
    anchor_pid = resolve_committed_anchor_pid(row, traj)
    user = _user_content(traj)

    for pid in _donor_pids_from_row(row):
        if pid != anchor_pid:
            return pid

    try:
        donor = resolve_donor_pid(row, user=user)
        if donor and donor != anchor_pid:
            return donor
    except (AttributeError, TypeError):
        pass

    resolved = resolve_cem_rival_value(traj)
    if resolved:
        _, rival_pid = resolved
        if rival_pid and str(rival_pid) != anchor_pid:
            return str(rival_pid)

    for rpid in _rival_pids_from_evidence(traj):
        if rpid and rpid != anchor_pid:
            return str(rpid)

    compared = str((traj.get("text_anchor") or {}).get("compared_pid") or "")
    if compared and compared != anchor_pid:
        return compared

    for pid in _donor_pids_from_row(row):
        if pid != anchor_pid:
            return pid
    return None
