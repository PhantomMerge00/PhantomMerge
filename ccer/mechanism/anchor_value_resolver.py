"""Resolve anchor evidence value s_{a_tau} per claim (Eq.2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ccer.mechanism.claim_segmentation import parse_bullet_quote
from ccer.mechanism.line_l_anchor_extract import lookup_obs_tau_slot

Provenance = Literal["obs_tau_dict", "missing"]


@dataclass(frozen=True)
class AnchorValueResolution:
    v_anchor: str | None
    provenance: Provenance
    miss_reason: str | None = None
    slot_norm: str | None = None
    anchor_pid: str | None = None


def _anchor_pid(
    *,
    y_pm: int,
    adjudication_row: dict[str, Any] | None,
    traj: dict[str, Any] | None,
) -> str:
    if int(y_pm) == 0:
        commitment = (traj or {}).get("commitment") or {}
        return str(commitment.get("action_anchor") or "").strip()
    adj = adjudication_row or {}
    return str(adj.get("committed_anchor_pid") or "").strip()


def _slot_norm(
    *,
    y_pm: int,
    response_quote: str,
    adjudication_row: dict[str, Any] | None,
) -> str:
    adj = adjudication_row or {}
    if adj.get("slot_norm"):
        return str(adj["slot_norm"]).strip()
    parsed = parse_bullet_quote(str(response_quote or ""))
    if parsed:
        return str(parsed[1]).strip()
    return ""


def resolve_anchor_value(
    *,
    y_pm: int,
    response_quote: str,
    adjudication_row: dict[str, Any] | None,
    traj: dict[str, Any] | None,
) -> AnchorValueResolution:
    """
    Paper-aligned resolver: v_a(c) = Obs_τ(a_τ)[slot(c)].

    If the anchor attribute value is not recoverable from structured tool
    observations, returns ``missing`` — downstream fusion uses z_rep only.
    """
    slot_norm = _slot_norm(
        y_pm=y_pm,
        response_quote=response_quote,
        adjudication_row=adjudication_row,
    )
    anchor_pid = _anchor_pid(
        y_pm=y_pm,
        adjudication_row=adjudication_row,
        traj=traj,
    )
    if not traj or not anchor_pid or not slot_norm:
        return AnchorValueResolution(
            v_anchor=None,
            provenance="missing",
            miss_reason="missing_traj_anchor_or_slot",
            slot_norm=slot_norm or None,
            anchor_pid=anchor_pid or None,
        )

    hit = lookup_obs_tau_slot(
        traj,
        anchor_pid=anchor_pid,
        slot_norm=slot_norm,
    )
    if hit.branch == "extraction_hit" and hit.v_anchor:
        return AnchorValueResolution(
            v_anchor=str(hit.v_anchor).strip(),
            provenance="obs_tau_dict",
            slot_norm=slot_norm,
            anchor_pid=anchor_pid,
        )
    return AnchorValueResolution(
        v_anchor=None,
        provenance="missing",
        miss_reason=str(hit.miss_reason or "obs_tau_slot_missing"),
        slot_norm=slot_norm,
        anchor_pid=anchor_pid,
    )
