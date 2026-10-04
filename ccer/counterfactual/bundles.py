"""CEM/CAP/AH dev bundles (§6) — adjudication-aware v2."""
from __future__ import annotations

from typing import Any

from ccer.adjudication.loader import load_adjudication_index, resolve_ah_swap_target, resolve_cem_swap_target
from ccer.counterfactual import operators as ops
from ccer.counterfactual.base import find_pid_blocks


def _user(traj: dict[str, Any]) -> str:
    for m in traj.get("messages_final_call") or []:
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def resolve_cem_rival_value(trajectory: dict[str, Any]) -> tuple[str, str] | None:
    """Return (value, donor_pid) using cem_ah_strict_v2.1 adjudication when available."""
    target = resolve_cem_swap_target(trajectory)
    if target:
        return target.claim_value, target.donor_pid

    # Legacy fallback (pre-adjudication cohort tooling)
    text = trajectory.get("text_anchor") or {}
    compared = str(text.get("compared_pid") or "")
    if not compared:
        return None
    cem_claim = next(
        (c for c in (trajectory.get("claims") or []) if c.get("legacy_label") == "cross_object_merge"),
        None,
    )
    if not cem_claim:
        return None
    value = str(cem_claim.get("value") or "").strip()
    if not value:
        return None
    user = _user(trajectory)
    blocks = find_pid_blocks(user, compared)
    if not blocks:
        return None
    for _, _, snippet in blocks:
        if value in snippet:
            return value, compared
        for sep in (",", ";", "/"):
            if sep in snippet:
                for part in snippet.split(sep):
                    if value == part.strip() or value in part:
                        return value, compared
    if value in user:
        return value, compared
    return None


def build_cem_bundle(trajectory: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    """Return list of (bundle_name, operator_fn, kwargs)."""
    target = resolve_cem_swap_target(trajectory)
    commitment = trajectory.get("commitment") or {}
    text_anchor = trajectory.get("text_anchor") or {}
    anchor = (
        (target.committed_anchor_pid if target else None)
        or commitment.get("action_anchor")
        or text_anchor.get("selected_pid")
    )
    bundle: list[tuple[str, Any, dict[str, Any]]] = [
        ("original", None, {}),
    ]
    resolved = resolve_cem_rival_value(trajectory)
    if resolved:
        value, donor_pid = resolved
        bundle.append(
            (
                "rival_value_swap",
                ops.rival_value_swap,
                {
                    "rival_pid": str(donor_pid),
                    "slot_value_old": str(value),
                    "slot_value_new": f"CF_{value}",
                    "anchor_pid": str(anchor) if anchor else None,
                },
            )
        )
        bundle.append(
            ("source_null", ops.source_null, {"source_pid": str(donor_pid)}),
        )
    bundle.append(("position_permutation", ops.position_permutation, {}))
    return bundle


def build_cap_bundle(trajectory: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    claims = trajectory.get("claims") or []
    cap_claim = next((c for c in claims if c.get("legacy_label") == "constraint_projection"), None)
    value = str((cap_claim or {}).get("value") or "")
    bundle: list[tuple[str, Any, dict[str, Any]]] = [("original", None, {})]
    if value:
        bundle.append(
            (
                "query_value_swap",
                ops.query_value_swap,
                {"old_value": value, "new_value": f"CF_{value}"},
            )
        )
        bundle.append(("position_permutation", ops.position_permutation, {}))
    bundle.append(("instruction_pressure_neutral", ops.instruction_pressure_neutral, {}))
    return bundle


def build_ah_bundle(trajectory: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    """Tier-1 specificity control: AH claim-level swap should not directionally follow."""
    target = resolve_ah_swap_target(trajectory)
    claims = trajectory.get("claims") or []
    ah_claim = next((c for c in claims if c.get("legacy_label") == "anchored_hallucination"), None)
    value = str((target.claim_value if target else "") or (ah_claim or {}).get("value") or "")
    anchor = (
        (target.anchor_pid if target else None)
        or (trajectory.get("commitment") or {}).get("action_anchor")
        or (trajectory.get("text_anchor") or {}).get("selected_pid")
    )
    bundle: list[tuple[str, Any, dict[str, Any]]] = [("original", None, {})]
    if value and anchor:
        bundle.append(
            (
                "anchor_value_swap",
                ops.anchor_value_swap,
                {
                    "anchor_pid": str(anchor),
                    "old_value": value,
                    "new_value": f"CF_{value}",
                },
            )
        )
    return bundle


def adjudication_mechanism_pools(*, split: str = "dev") -> dict[str, list[str]]:
    """Trajectory IDs for mechanism pools from strict v2.1 adjudication."""
    from ccer.io_utils import load_json
    from ccer.paths import SPLIT_MANIFEST_JSON

    index = load_adjudication_index()
    splits = load_json(SPLIT_MANIFEST_JSON).get("splits", {})
    cem: list[str] = []
    ah: list[str] = []
    for tid in index.by_trajectory:
        if splits.get(tid) != split:
            continue
        if index.cem_trajectory_status.get(tid) == "confirmed":
            cem.append(tid)
        if index.ah_trajectory_status.get(tid) == "confirmed":
            ah.append(tid)
    return {"CEM": sorted(cem), "AH": sorted(ah)}
