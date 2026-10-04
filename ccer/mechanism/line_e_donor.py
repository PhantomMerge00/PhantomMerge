"""Line E v4 ultimate: donor selection, strata, and strict bilateral activation gates."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

import numpy as np

from ccer.io_utils import load_json
from ccer.mechanism.activation_store import activation_path, fast_npz_roi_ok, get_vector, load_activation_npz
from ccer.mechanism.pair_select import _query_slot_from_trajectory, load_trajectory_index
from ccer.mechanism.steering_vector import (
    collect_layer_activations,
    paired_activation_vectors,
    paired_clean_trajectory_id,
)
from ccer.paths import COHORT_MANIFEST_JSON
from ccer.replay.live_position import activation_passes_line_d_v3_gate, pair_has_symmetric_claim_anchor

LineEStratum = Literal["commitment_mismatch", "commitment_consistent", "other_pm", "unknown"]


def trajectory_commitment_stratum(traj: dict[str, Any]) -> LineEStratum:
    rel = str((traj.get("commitment") or {}).get("commitment_relation") or "unknown")
    if rel == "mismatch":
        return "commitment_mismatch"
    if rel == "consistent":
        return "commitment_consistent"
    if rel in ("unobservable", "unknown", "ambiguous"):
        return "other_pm"
    return "other_pm"


def _action_anchor(traj: dict[str, Any]) -> str | None:
    anchor = (traj.get("commitment") or {}).get("action_anchor")
    return str(anchor) if anchor else None


def _consistent_pool_by_slot() -> dict[str, list[dict[str, Any]]]:
    cohort = load_json(COHORT_MANIFEST_JSON)
    rows = load_trajectory_index()
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tid in cohort.get("cohorts", {}).get("CEM", []):
        traj = rows.get(tid)
        if not traj:
            continue
        rel = (traj.get("commitment") or {}).get("commitment_relation")
        if rel != "consistent":
            continue
        pool[_query_slot_from_trajectory(traj)].append(
            {
                "trajectory_id": tid,
                "slot_norm": _query_slot_from_trajectory(traj),
                "action_anchor": _action_anchor(traj),
            }
        )
    return pool


def _anchor_aligned_pool() -> dict[str, list[str]]:
    cohort = load_json(COHORT_MANIFEST_JSON)
    rows = load_trajectory_index()
    by_anchor: dict[str, list[str]] = defaultdict(list)
    for tid in cohort.get("cohorts", {}).get("CEM", []):
        traj = rows.get(tid)
        if not traj:
            continue
        if (traj.get("commitment") or {}).get("commitment_relation") != "consistent":
            continue
        anchor = _action_anchor(traj)
        if anchor:
            by_anchor[anchor].append(tid)
    return by_anchor


def bilateral_activation_gate(
    pm_trajectory_id: str,
    donor_trajectory_id: str,
    *,
    layer: int,
    position: str,
) -> tuple[bool, str]:
    """Both trajectories must have symmetric Line D v3 activations at ROI — no silent fallback."""
    for tid, role in ((pm_trajectory_id, "pm"), (donor_trajectory_id, "donor")):
        ok, reason = fast_npz_roi_ok(tid, position=position, layer=layer)
        if not ok:
            return False, f"{reason.split(':', 1)[0]}:{role}:{tid}" if ":" in reason else f"{reason}:{role}:{tid}"
    return True, "ok"


def slot_consistent_donor_id(recipient_trajectory_id: str) -> str | None:
    rows = load_trajectory_index()
    recipient = rows.get(recipient_trajectory_id)
    if not recipient:
        return None
    slot = _query_slot_from_trajectory(recipient)
    pool = _consistent_pool_by_slot().get(slot) or []
    if not pool:
        return None
    recipient_anchor = _action_anchor(recipient)
    # Prefer donor with different anchor (restoration toward consistent binding mode).
    for donor in pool:
        if donor.get("action_anchor") != recipient_anchor:
            return str(donor["trajectory_id"])
    return str(pool[0]["trajectory_id"])


def anchor_aligned_donor_id(recipient_trajectory_id: str) -> str | None:
    rows = load_trajectory_index()
    recipient = rows.get(recipient_trajectory_id)
    if not recipient:
        return None
    anchor = _action_anchor(recipient)
    if not anchor:
        return None
    candidates = [
        tid
        for tid in _anchor_aligned_pool().get(anchor, [])
        if tid != recipient_trajectory_id
    ]
    return candidates[0] if candidates else None


def donor_activation_vector(
    donor_trajectory_id: str,
    *,
    layer: int,
    position: str,
) -> np.ndarray | None:
    vecs, used = collect_layer_activations([donor_trajectory_id], layer=layer, position=position)
    if not vecs or not used:
        return None
    return np.asarray(vecs[0], dtype=np.float32).reshape(-1)


def build_slot_consistent_bundle(
    recipient_trajectory_id: str,
    *,
    layer: int,
    position: str,
) -> dict[str, Any]:
    donor_tid = slot_consistent_donor_id(recipient_trajectory_id)
    if not donor_tid:
        return {"error": "no_slot_consistent_donor", "recipient_trajectory_id": recipient_trajectory_id}
    ok, reason = bilateral_activation_gate(recipient_trajectory_id, donor_tid, layer=layer, position=position)
    if not ok:
        return {
            "error": "bilateral_gate_failed",
            "reason": reason,
            "recipient_trajectory_id": recipient_trajectory_id,
            "donor_trajectory_id": donor_tid,
        }
    pm_vecs, _ = collect_layer_activations([recipient_trajectory_id], layer=layer, position=position)
    donor_vecs, _ = collect_layer_activations([donor_tid], layer=layer, position=position)
    if not pm_vecs or not donor_vecs:
        return {"error": "missing_slot_consistent_activation", "donor_trajectory_id": donor_tid}
    pm_vec = pm_vecs[0]
    donor_vec = donor_vecs[0]
    return {
        "recipient_trajectory_id": recipient_trajectory_id,
        "donor_trajectory_id": donor_tid,
        "donor_kind": "slot_consistent",
        "pm_vec": pm_vec,
        "donor_vec": donor_vec,
        "mitigation_vector": (pm_vec - donor_vec).astype(np.float32),
        "layer": layer,
        "position": position,
    }


def build_anchor_aligned_bundle(
    recipient_trajectory_id: str,
    *,
    layer: int,
    position: str,
) -> dict[str, Any]:
    donor_tid = anchor_aligned_donor_id(recipient_trajectory_id)
    if not donor_tid:
        return {"error": "no_anchor_aligned_donor", "recipient_trajectory_id": recipient_trajectory_id}
    ok, reason = bilateral_activation_gate(recipient_trajectory_id, donor_tid, layer=layer, position=position)
    if not ok:
        return {
            "error": "bilateral_gate_failed",
            "reason": reason,
            "recipient_trajectory_id": recipient_trajectory_id,
            "donor_trajectory_id": donor_tid,
        }
    pm_vecs, _ = collect_layer_activations([recipient_trajectory_id], layer=layer, position=position)
    donor_vecs, _ = collect_layer_activations([donor_tid], layer=layer, position=position)
    if not pm_vecs or not donor_vecs:
        return {"error": "missing_anchor_aligned_activation", "donor_trajectory_id": donor_tid}
    pm_vec = pm_vecs[0]
    donor_vec = donor_vecs[0]
    return {
        "recipient_trajectory_id": recipient_trajectory_id,
        "donor_trajectory_id": donor_tid,
        "donor_kind": "anchor_aligned",
        "pm_vec": pm_vec,
        "donor_vec": donor_vec,
        "mitigation_vector": (pm_vec - donor_vec).astype(np.float32),
        "layer": layer,
        "position": position,
    }


def build_paired_cross_bundle(
    pm_trajectory_id: str,
    *,
    layer: int,
    position: str,
) -> dict[str, Any]:
    clean_tid = paired_clean_trajectory_id(pm_trajectory_id)
    if not clean_tid:
        return {"error": "no_paired_clean", "pm_trajectory_id": pm_trajectory_id}
    ok, reason = bilateral_activation_gate(pm_trajectory_id, clean_tid, layer=layer, position=position)
    if not ok:
        return {
            "error": "bilateral_gate_failed",
            "reason": reason,
            "pm_trajectory_id": pm_trajectory_id,
            "clean_trajectory_id": clean_tid,
        }
    paired = paired_activation_vectors(pm_trajectory_id, layer=layer, position=position)
    if paired.get("error"):
        return paired
    return {
        **paired,
        "donor_kind": "cross_pair_clean",
        "donor_trajectory_id": clean_tid,
        "donor_vec": paired["clean_vec"],
    }


def list_missing_bilateral_pairs(
    pm_trajectory_ids: list[str],
    *,
    layer: int,
    position: str,
) -> list[dict[str, Any]]:
    from ccer.mechanism.mechanism_pool import build_cem_mechanism_cross_pairs

    clean_by_pm = {
        str(cp["pm_trajectory_id"]): str(cp["clean_trajectory_id"])
        for cp in build_cem_mechanism_cross_pairs(pool="mechanism_research")["pm_clean_cross_pairs"]
    }
    missing: list[dict[str, Any]] = []
    for tid in pm_trajectory_ids:
        clean_tid = clean_by_pm.get(tid)
        if not clean_tid:
            missing.append({"pm_trajectory_id": tid, "reason": "no_paired_clean"})
            continue
        ok, reason = bilateral_activation_gate(tid, clean_tid, layer=layer, position=position)
        if not ok:
            missing.append(
                {
                    "pm_trajectory_id": tid,
                    "clean_trajectory_id": clean_tid,
                    "reason": reason,
                }
            )
    return missing


def ultimate_eligible_trajectory(
    traj: dict[str, Any],
    *,
    layer: int,
    position: str,
) -> tuple[bool, str]:
    if not pair_has_symmetric_claim_anchor(traj):
        return False, "no_symmetric_claim_anchor"
    tid = str(traj.get("trajectory_id") or "")
    if not tid:
        return False, "missing_trajectory_id"
    clean_tid = paired_clean_trajectory_id(tid)
    if not clean_tid:
        return False, "no_paired_clean"
    ok, reason = bilateral_activation_gate(tid, clean_tid, layer=layer, position=position)
    if not ok:
        return False, reason
    return True, "ok"
