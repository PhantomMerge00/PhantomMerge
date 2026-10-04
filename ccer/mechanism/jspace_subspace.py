"""J-space subspace construction for Track N coordinate-swap."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.jspace_intervention import (
    lens_direction_for_token,
    orthogonalize_columns,
    token_id_variants,
)
from ccer.mechanism.pair_select import load_trajectory_index
from ccer.mechanism.rank_sweep import _cross_pairs, _line_d_eligible_cross_pairs
from ccer.adjudication.loader import load_adjudication_index, resolve_cem_swap_target
from ccer.mechanism.line_l_anchor_extract import lookup_from_anchor_product_json
from jlens.lens import JacobianLens
from jlens.protocol import LensModel


def gradient_pursuit(
    h: np.ndarray,
    dictionary: np.ndarray,
    *,
    k: int = 25,
    nonneg: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Greedy matching pursuit; returns coefficients and active column indices."""
    D = np.asarray(dictionary, dtype=np.float64)
    h_vec = np.asarray(h, dtype=np.float64).reshape(-1)
    residual = h_vec.copy()
    active: list[int] = []
    coeffs = np.zeros(D.shape[1], dtype=np.float64)
    for _ in range(min(k, D.shape[1])):
        projections = D.T @ residual
        if nonneg:
            projections = np.maximum(projections, 0.0)
        j = int(np.argmax(np.abs(projections)))
        if j in active:
            break
        active.append(j)
        D_active = D[:, active]
        c, _, _, _ = np.linalg.lstsq(D_active, h_vec, rcond=None)
        if nonneg:
            c = np.maximum(c, 0.0)
        coeffs[active] = c
        residual = h_vec - D_active @ c
        if float(np.linalg.norm(residual)) < 1e-4 * float(np.linalg.norm(h_vec) + 1e-8):
            break
    return coeffs.astype(np.float32), active


def _distractor_slot_value(
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    donor_pid: str,
    slot_norm: str,
) -> str | None:
    anchor_hit = lookup_from_anchor_product_json(traj, anchor_pid=anchor_pid, slot_norm=slot_norm)
    donor_hit = lookup_from_anchor_product_json(traj, anchor_pid=donor_pid, slot_norm=slot_norm)
    if donor_hit.branch == "extraction_hit" and donor_hit.v_anchor:
        return str(donor_hit.v_anchor)
    if anchor_hit.branch == "extraction_hit" and anchor_hit.v_anchor:
        return None
    return None


def build_lens_dictionary(
    lens: JacobianLens,
    jlens_model: LensModel,
    tokenizer: Any,
    token_texts: list[str],
    *,
    layer: int,
) -> tuple[np.ndarray, list[str]]:
    """Build dictionary matrix [d_model, n_dirs] from value-relevant token lens directions."""
    cols: list[np.ndarray] = []
    labels: list[str] = []
    seen: set[int] = set()
    for text in token_texts:
        if not text:
            continue
        for tid in token_id_variants(tokenizer, text):
            if tid in seen:
                continue
            seen.add(tid)
            direction = lens_direction_for_token(lens, jlens_model, layer, tid)
            cols.append(direction)
            labels.append(f"{text}::tid{tid}")
    if not cols:
        return np.zeros((jlens_model.d_model, 0), dtype=np.float32), labels
    D = np.stack(cols, axis=1).astype(np.float32)
    return D, labels


def collect_value_texts_for_pair(
    pm_traj: dict[str, Any],
    clean_traj: dict[str, Any],
    *,
    slot_norm: str,
) -> list[str]:
    target = resolve_cem_swap_target(pm_traj)
    if not target:
        return []
    texts: list[str] = []
    claim = str(target.claim_value or "")
    if claim:
        texts.append(claim)
    anchor_pid = str(target.committed_anchor_pid or "")
    donor_pid = str(target.donor_pid or "")
    for traj, pid in ((pm_traj, anchor_pid), (clean_traj, anchor_pid), (pm_traj, donor_pid)):
        hit = lookup_from_anchor_product_json(traj, anchor_pid=pid, slot_norm=slot_norm)
        if hit.branch == "extraction_hit" and hit.v_anchor:
            texts.append(str(hit.v_anchor))
    distractor = _distractor_slot_value(
        pm_traj, anchor_pid=anchor_pid, donor_pid=donor_pid, slot_norm=slot_norm
    )
    if distractor:
        texts.append(distractor)
    return list(dict.fromkeys(t for t in texts if t))


def fit_jspace_subspace(
    lens: JacobianLens,
    jlens_model: LensModel,
    tokenizer: Any,
    *,
    layer: int = 49,
    position: str = "claim_onset",
    track: str = "CEM",
    k: int = 25,
    cross_pairs_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fit pooled J-space subspace V from sparse decompositions of eligible activations."""
    rows_index = load_trajectory_index()
    cross_pairs = (
        list(cross_pairs_override)
        if cross_pairs_override is not None
        else _line_d_eligible_cross_pairs(
            _cross_pairs(track.upper()),
            position=position,
            track_u=track.upper(),
            rows_index=rows_index,
        )
    )
    all_texts: list[str] = []
    vectors: list[np.ndarray] = []
    pair_meta: list[dict[str, Any]] = []
    for cp in cross_pairs:
        pm_tid = str(cp.get("pm_trajectory_id") or cp.get("cap_trajectory_id"))
        clean_tid = str(cp["clean_trajectory_id"])
        pm_traj = rows_index.get(pm_tid)
        clean_traj = rows_index.get(clean_tid)
        if not pm_traj or not clean_traj:
            continue
        slot_norm = str(cp.get("slot_norm") or "")
        all_texts.extend(collect_value_texts_for_pair(pm_traj, clean_traj, slot_norm=slot_norm))
        for tid, role in ((pm_tid, "pm"), (clean_tid, "clean")):
            npz = load_activation_npz(activation_path(tid, "original"))
            vec = get_vector(npz, position=position, layer=layer)
            if vec is not None:
                vectors.append(np.asarray(vec, dtype=np.float32))
                pair_meta.append({"trajectory_id": tid, "role": role, "pm_tid": pm_tid})

    unique_texts = list(dict.fromkeys(t for t in all_texts if t))
    dictionary, dict_labels = build_lens_dictionary(
        lens, jlens_model, tokenizer, unique_texts, layer=layer
    )
    if dictionary.shape[1] == 0:
        return {"error": "empty_lens_dictionary", "U_owner": None, "rank": 0}

    active_union: set[int] = set()
    decomp_rows: list[dict[str, Any]] = []
    for h_vec, meta in zip(vectors, pair_meta):
        coeffs, active = gradient_pursuit(h_vec, dictionary, k=k, nonneg=True)
        active_union.update(active)
        decomp_rows.append({**meta, "active_indices": active, "n_active": len(active)})

    if not active_union:
        active_union = set(range(min(k, dictionary.shape[1])))

    active_list = sorted(active_union)[:k]
    V = orthogonalize_columns(dictionary[:, active_list])
    return {
        "U_owner": V,
        "rank": int(V.shape[1]),
        "n_vectors": len(vectors),
        "n_pairs": len(cross_pairs),
        "dictionary_size": int(dictionary.shape[1]),
        "active_directions": [dict_labels[i] for i in active_list if i < len(dict_labels)],
        "decompositions": decomp_rows,
        "subspace_source": "jspace",
        "layer": layer,
        "position": position,
    }
