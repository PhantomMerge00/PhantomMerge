"""Heuristic anchor / donor resolution from trajectory only (no adjudication oracle)."""

from __future__ import annotations

from typing import Any

from ccer.replay.answer_utils import extract_selected_product_id, normalize_final_synthesis_text
from ccer.replay.evidence_mask import _rival_pids_from_evidence


def _final_assistant_text(traj: dict[str, Any]) -> str:
    for m in reversed(traj.get("messages_final_call") or []):
        if m.get("role") == "assistant":
            return normalize_final_synthesis_text(str(m.get("content") or ""))
    return ""


def _evidence_records(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """Shopping list evidence vs τ²/τ³ ``evidence.E_seen`` nested lists."""
    ev = traj.get("evidence")
    if isinstance(ev, list):
        return [x for x in ev if isinstance(x, dict)]
    if isinstance(ev, dict):
        for key in ("E_seen", "items", "records"):
            raw = ev.get(key)
            if isinstance(raw, list):
                return [x for x in raw if isinstance(x, dict)]
    return []


def resolve_heuristic_anchor_pid(traj: dict[str, Any]) -> str:
    """Committed anchor from trajectory metadata + final answer Selected PID."""
    for key in (
        lambda t: (t.get("commitment") or {}).get("action_anchor"),
        lambda t: (t.get("text_anchor") or {}).get("selected_pid"),
    ):
        pid = str(key(traj) or "").strip()
        if pid:
            return pid
    text = _final_assistant_text(traj)
    pid = extract_selected_product_id(text)
    if pid:
        return str(pid)
    for ev in _evidence_records(traj):
        hint = str(ev.get("entity_hint") or "").strip()
        if hint:
            return hint
        ents = ev.get("entity_ids") or ev.get("source_id") or []
        if isinstance(ents, str):
            ents = [ents]
        for e in ents:
            s = str(e).strip()
            if s.isdigit() and len(s) >= 8:
                return s
            if s and not s.isdigit():
                return s
    return ""


def resolve_heuristic_donor_pids(traj: dict[str, Any], *, anchor_pid: str) -> list[str]:
    """Rival PIDs from evidence / compared section — no gold donor_owners."""
    out: list[str] = []
    compared = str((traj.get("text_anchor") or {}).get("compared_pid") or "").strip()
    if compared and compared != anchor_pid:
        out.append(compared)
    for rpid in _rival_pids_from_evidence(traj):
        if rpid and rpid != anchor_pid and rpid not in out:
            out.append(rpid)
    for ev in _evidence_records(traj):
        hint = str(ev.get("entity_hint") or "").strip()
        if hint and hint != anchor_pid and hint not in out:
            out.append(hint)
        ents = ev.get("entity_ids") or ev.get("source_id") or []
        if isinstance(ents, str):
            ents = [ents]
        for e in ents:
            pid = str(e).strip()
            if pid and pid != anchor_pid and pid not in out:
                out.append(pid)
    return out[:5]
