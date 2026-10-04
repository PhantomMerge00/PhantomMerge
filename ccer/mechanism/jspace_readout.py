"""Track M: J-lens readout (legacy pair-level). Owner-axis scoring is deprecated.

For claim-level PRISM audit use ``prism_audit.score_slot_alignment`` instead.
"""
from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np
import torch

from ccer.adjudication.loader import resolve_cem_swap_target
from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.jspace_intervention import token_id_variants
from ccer.mechanism.line_l_anchor_extract import lookup_from_anchor_product_json
from ccer.mechanism.pair_select import load_trajectory_index
from ccer.mechanism.rank_sweep import _cross_pairs, _line_d_eligible_cross_pairs
from ccer.mechanism.stats import wilson_ci
from ccer.mechanism.value_token_match import norm_value, numeric_parts, value_tokens_match
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

OwnerAxis = Literal["anchor", "distractor", "ambiguous", "unreadable"]

_SLOT_HINTS: dict[str, tuple[str, ...]] = {
    "price": ("price", "价格", "php", "usd", "eur", "cost", "价"),
    "capacity": ("oz", "ml", "liter", "capacity", "size", "容量", "体积"),
    "blades": ("blade", "blades", "刀", "刃"),
    "color": ("color", "colour", "颜色", "色"),
    "weight": ("weight", "kg", "lb", "重量", "克"),
    "rating": ("rating", "star", "评分", "星"),
}


def _norm_value(text: str) -> str:
    return norm_value(text)


def _value_tokens_match(token_text: str, value: str) -> bool:
    return value_tokens_match(token_text, value)


def _slot_hint_match(token_text: str, slot_norm: str) -> bool:
    hints = _SLOT_HINTS.get(_norm_value(slot_norm), ())
    tok_l = str(token_text or "").lower()
    return any(h in tok_l for h in hints)


def readout_topk(
    lens: JacobianLens,
    jlens_model: LensModel,
    tokenizer: Any,
    h: np.ndarray,
    *,
    layer: int,
    k: int = 10,
    use_jacobian: bool = True,
) -> list[dict[str, Any]]:
    """Decode top-k tokens from hidden state via J-lens (or logit-lens baseline)."""
    h_t = torch.tensor(np.asarray(h, dtype=np.float32).reshape(1, -1))
    if use_jacobian:
        transported = lens.transport(h_t, layer)
    else:
        transported = h_t
    logits = jlens_model.unembed(transported)[0].detach().cpu().float()
    topk = torch.topk(logits, min(k, logits.numel()))
    rows: list[dict[str, Any]] = []
    for score, tid in zip(topk.values.tolist(), topk.indices.tolist()):
        rows.append(
            {
                "token_id": int(tid),
                "token": tokenizer.decode([int(tid)]),
                "logit": float(score),
                "rank": len(rows) + 1,
            }
        )
    return rows


def _anchor_value_for_traj(traj: dict[str, Any], slot_norm: str, anchor_pid: str) -> str | None:
    hit = lookup_from_anchor_product_json(traj, anchor_pid=anchor_pid, slot_norm=slot_norm)
    if hit.branch == "extraction_hit" and hit.v_anchor:
        return str(hit.v_anchor)
    return None


def _distractor_value(
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    donor_pid: str,
    slot_norm: str,
) -> str | None:
    hit = lookup_from_anchor_product_json(traj, anchor_pid=donor_pid, slot_norm=slot_norm)
    if hit.branch == "extraction_hit" and hit.v_anchor:
        return str(hit.v_anchor)
    return None


def score_readout(
    topk: list[dict[str, Any]],
    *,
    anchor_value: str | None,
    distractor_value: str | None,
    claim_value: str | None,
    slot_norm: str = "",
    traj_role: str = "pm",
) -> dict[str, Any]:
    """Legacy pair-level scoring. ``owner_axis`` is deprecated — use PRISM slot alignment."""
    value_hits: list[str] = []
    anchor_hits = 0
    distractor_hits = 0
    claim_hits = 0
    slot_hits = 0
    for row in topk:
        tok = str(row.get("token") or "")
        if anchor_value and _value_tokens_match(tok, anchor_value):
            anchor_hits += 1
            value_hits.append("anchor")
        if distractor_value and _value_tokens_match(tok, distractor_value):
            distractor_hits += 1
            value_hits.append("distractor")
        if claim_value and _value_tokens_match(tok, claim_value):
            claim_hits += 1
            value_hits.append("claim")
        if _slot_hint_match(tok, slot_norm):
            slot_hits += 1
            value_hits.append("slot_hint")

    value_readable = bool(value_hits) or claim_hits > 0 or slot_hits > 0
    owner_axis: OwnerAxis
    anchor_vs_distractor_same = bool(
        anchor_value
        and distractor_value
        and _norm_value(anchor_value) == _norm_value(distractor_value)
    )
    if traj_role == "pm":
        # PM: claim_value encodes erroneous binding; anchor_value is correct.
        if claim_hits > anchor_hits:
            owner_axis = "distractor"
        elif anchor_hits > claim_hits:
            owner_axis = "anchor"
        elif anchor_vs_distractor_same:
            owner_axis = "ambiguous"
        elif anchor_hits == distractor_hits == claim_hits == 0:
            owner_axis = "unreadable" if not value_readable else "ambiguous"
        else:
            owner_axis = "ambiguous"
    else:
        if anchor_hits > claim_hits and anchor_hits >= distractor_hits:
            owner_axis = "anchor"
        elif distractor_hits > anchor_hits:
            owner_axis = "distractor"
        elif anchor_hits == distractor_hits == claim_hits == 0:
            owner_axis = "unreadable" if not value_readable else "ambiguous"
        else:
            owner_axis = "ambiguous"

    return {
        "value_readable": value_readable,
        "owner_axis": owner_axis,
        "anchor_hits_in_topk": anchor_hits,
        "distractor_hits_in_topk": distractor_hits,
        "claim_hits_in_topk": claim_hits,
        "slot_hits_in_topk": slot_hits,
        "value_hit_labels": value_hits,
    }


def run_track_m_readout(
    lens: JacobianLens,
    jlens_model: LensModel,
    tokenizer: Any,
    *,
    layer: int = 49,
    position: str = "claim_onset",
    track: str = "CEM",
    topk: int = 10,
) -> dict[str, Any]:
    """Run J-lens readout on all eligible cross-pairs."""
    rows_index = load_trajectory_index()
    cross_pairs = _line_d_eligible_cross_pairs(
        _cross_pairs(track.upper()),
        position=position,
        track_u=track.upper(),
        rows_index=rows_index,
    )
    pair_details: list[dict[str, Any]] = []
    for cp in cross_pairs:
        pm_tid = str(cp.get("pm_trajectory_id") or cp.get("cap_trajectory_id"))
        clean_tid = str(cp["clean_trajectory_id"])
        pm_traj = rows_index.get(pm_tid)
        clean_traj = rows_index.get(clean_tid)
        if not pm_traj or not clean_traj:
            continue
        target = resolve_cem_swap_target(pm_traj)
        if not target:
            continue
        slot_norm = str(target.slot_norm or cp.get("slot_norm") or "")
        anchor_pid = str(target.committed_anchor_pid or "")
        donor_pid = str(target.donor_pid or "")
        claim_value = str(target.claim_value or "")
        anchor_value = _anchor_value_for_traj(pm_traj, slot_norm, anchor_pid)
        if not anchor_value and clean_traj:
            anchor_value = _anchor_value_for_traj(clean_traj, slot_norm, anchor_pid)
        distractor_value = _distractor_value(
            pm_traj, anchor_pid=anchor_pid, donor_pid=donor_pid, slot_norm=slot_norm
        )
        if not distractor_value and clean_traj:
            distractor_value = _distractor_value(
                clean_traj, anchor_pid=anchor_pid, donor_pid=donor_pid, slot_norm=slot_norm
            )

        row: dict[str, Any] = {
            "pm_trajectory_id": pm_tid,
            "clean_trajectory_id": clean_tid,
            "slot_norm": slot_norm,
            "claim_value": claim_value,
            "anchor_value": anchor_value,
            "distractor_value": distractor_value,
            "anchor_pid": anchor_pid,
            "donor_pid": donor_pid,
        }
        for tid, traj_role in ((pm_tid, "pm"), (clean_tid, "clean")):
            npz = load_activation_npz(activation_path(tid, "original"))
            h = get_vector(npz, position=position, layer=layer)
            if h is None:
                row[f"{traj_role}_error"] = "missing_activation"
                continue
            j_topk = readout_topk(
                lens, jlens_model, tokenizer, h, layer=layer, k=topk, use_jacobian=True
            )
            logit_topk = readout_topk(
                lens, jlens_model, tokenizer, h, layer=layer, k=topk, use_jacobian=False
            )
            scores = score_readout(
                j_topk,
                anchor_value=anchor_value,
                distractor_value=distractor_value,
                claim_value=claim_value,
                slot_norm=slot_norm,
                traj_role=traj_role,
            )
            row[f"{traj_role}_jlens_topk"] = j_topk
            row[f"{traj_role}_logit_lens_topk"] = logit_topk
            row[f"{traj_role}_scores"] = scores
        pair_details.append(row)

    pm_distractor = sum(
        1
        for r in pair_details
        if (r.get("pm_scores") or {}).get("owner_axis") == "distractor"
    )
    clean_anchor = sum(
        1
        for r in pair_details
        if (r.get("clean_scores") or {}).get("owner_axis") == "anchor"
    )
    pm_readable = sum(
        1 for r in pair_details if (r.get("pm_scores") or {}).get("value_readable")
    )
    n = len(pair_details)
    return {
        "schema": "ccer_track_m_jspace_readout_v1",
        "layer": layer,
        "position": position,
        "n_pairs": n,
        "pm_distractor_owner_rate": pm_distractor / n if n else 0.0,
        "pm_distractor_owner_ci95": wilson_ci(pm_distractor, n) if n else [0.0, 0.0],
        "clean_anchor_owner_rate": clean_anchor / n if n else 0.0,
        "clean_anchor_owner_ci95": wilson_ci(clean_anchor, n) if n else [0.0, 0.0],
        "pm_value_readable_rate": pm_readable / n if n else 0.0,
        "pair_details": pair_details,
    }
