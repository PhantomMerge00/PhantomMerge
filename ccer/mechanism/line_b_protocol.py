"""Line B protocol defaults and pair-quality helpers (root-cause fixes v2)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ccer.paths import REPORTS
from ccer.replay.hf_forward import coarse_layer_indices, dense_layer_indices

LINE_B_PROTOCOL_VERSION = "line_b_v3_root_cause_closed"
DEFAULT_MIN_PAIR_QUALITY = 0.35
DEFAULT_MIN_EVAL_PAIR_QUALITY = 0.35
PAIR_AUDIT_JSONL = REPORTS / "manual_verification" / "round12_cem_cross_pair_audit.jsonl"
LINE_A_PROBE_SUMMARY = Path("${PHANTOM_MERGE_ROOT}/results/line_a/v3/probe_vs_bow_summary.json")

# Line A v3 dev-selected ROI (probe_vs_bow_summary.json)
DEFAULT_LAYER = 49
DEFAULT_POSITION = "claim_onset"


def line_b_roi_layer_indices(n_layers: int, *, center: int | None = None, radius: int = 2) -> list[int]:
    """Layers to store for Line B v2: coarse scan + dense ROI around Line A peak (default L49)."""
    center = int(center if center is not None else DEFAULT_LAYER)
    coarse = coarse_layer_indices(n_layers)
    dense = dense_layer_indices(n_layers, center, radius=radius)
    return sorted(set(coarse) | set(dense))


def line_a_roi_defaults() -> dict[str, Any]:
    """Primary ROI aligned with Line A v3 (L49 / claim_onset)."""
    layer, position = DEFAULT_LAYER, DEFAULT_POSITION
    if LINE_A_PROBE_SUMMARY.is_file():
        try:
            summary = json.loads(LINE_A_PROBE_SUMMARY.read_text(encoding="utf-8"))
            roi = summary.get("primary_roi") or {}
            layer = int(roi.get("layer") or summary.get("layer") or layer)
            position = str(roi.get("position") or summary.get("position") or position)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return {
        "layer": layer,
        "position": position,
        "source": "line_a_v3_probe_vs_bow_summary",
        "protocol_version": LINE_B_PROTOCOL_VERSION,
    }


def load_pair_audit_index() -> dict[str, dict[str, Any]]:
    if not PAIR_AUDIT_JSONL.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in PAIR_AUDIT_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pm = str(row.get("pm_trajectory_id") or "")
        if pm:
            out[pm] = row
    return out


def pair_quality_score(
    audit_row: dict[str, Any] | None,
    *,
    cross_pair_row: dict[str, Any] | None = None,
) -> float:
    """Higher = better training/eval pair (less cohort shell noise)."""
    if not audit_row:
        score = 0.5
    else:
        score = 1.0
    if cross_pair_row and cross_pair_row.get("pairing_method") == "slot_norm_match":
        score += 0.25
    if audit_row:
        if audit_row.get("pairing_method") == "cohort_index_fallback":
            score -= 0.35
        if audit_row.get("anchor_title_quality") == "shell_pid_only":
            score -= 0.45
        if audit_row.get("owner_uniqueness") == "unique":
            score += 0.15
        if audit_row.get("has_pins_not_viewed"):
            score -= 0.1
    return max(0.0, min(1.0, score))


def filter_cross_pairs_by_quality(
    cross_pairs: list[dict[str, Any]],
    *,
    min_quality: float = 0.35,
    require_real_anchor: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop low-quality cohort-fallback / shell-anchor pairs for subspace training."""
    audit = load_pair_audit_index()
    kept: list[dict[str, Any]] = []
    dropped_shell = dropped_fallback = 0
    for cp in cross_pairs:
        pm = str(cp.get("pm_trajectory_id") or cp.get("cap_trajectory_id") or "")
        row = audit.get(pm)
        q = pair_quality_score(row, cross_pair_row=cp)
        if require_real_anchor and row and row.get("anchor_title_quality") == "shell_pid_only":
            dropped_shell += 1
            continue
        if q < min_quality:
            if row and row.get("anchor_title_quality") == "shell_pid_only":
                dropped_shell += 1
            elif row and row.get("pairing_method") == "cohort_index_fallback":
                dropped_fallback += 1
            continue
        kept.append({**cp, "pair_quality_score": q})
    meta = {
        "n_input": len(cross_pairs),
        "n_kept": len(kept),
        "n_dropped_shell_anchor": dropped_shell,
        "n_dropped_low_quality": len(cross_pairs) - len(kept),
        "min_quality": min_quality,
        "require_real_anchor": require_real_anchor,
    }
    return kept, meta


def line_b_select_roi_layer(
    *,
    n_layers: int,
    position: str,
    track: str = "CEM",
    pool: str = "mechanism_research",
    pm_trajectory_allowlist: set[str] | None = None,
    radius: int = 2,
) -> tuple[int, dict[str, Any]]:
    """Dense layer scan around Line A peak (root #2): pick strongest cross-vector signal."""
    from ccer.mechanism.owner_pca import collect_diff_vectors

    roi = line_a_roi_defaults()
    center = int(roi["layer"])
    max_layer = max(0, int(n_layers * 0.85) - 1)
    candidates = [
        int(l)
        for l in dense_layer_indices(n_layers, center, radius=radius)
        if int(l) <= max_layer
    ]
    if center not in candidates:
        candidates.append(center)
    candidates = sorted(set(candidates))

    best_layer = center
    best_score = -1.0
    scan_rows: list[dict[str, Any]] = []
    for layer in candidates:
        bundle = collect_diff_vectors(
            position=position,
            layer=layer,
            track=track,  # type: ignore[arg-type]
            pool=pool,
            pm_trajectory_allowlist=pm_trajectory_allowlist,
            cross_pair_quality_min=DEFAULT_MIN_PAIR_QUALITY,
        )
        cross = bundle["cross_vectors"]
        if len(cross) < 3:
            scan_rows.append({"layer": layer, "n_cross": len(cross), "score": 0.0, "skipped": True})
            continue
        norms = [float(np.linalg.norm(v)) for v in cross]
        score = float(np.mean(norms)) * len(cross)
        scan_rows.append(
            {
                "layer": layer,
                "n_cross": len(cross),
                "mean_delta_norm": float(np.mean(norms)),
                "score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_layer = layer

    meta = {
        "center_layer_line_a": center,
        "candidates": candidates,
        "selected_layer": best_layer,
        "selected_score": best_score,
        "scan_rows": scan_rows,
        "method": "cross_vector_mean_norm_x_count",
    }
    return best_layer, meta


def resolve_wrong_owner_donor_vec(
    *,
    pm_tid: str,
    paired_clean_tid: str,
    cross_pairs: list[dict[str, Any]],
    layer: int,
    position: str,
) -> tuple[np.ndarray | None, str]:
    """
    Wrong-owner donor for specificity control (root #7):
    1) rival_value_swap activation on PM trajectory
    2) wrong clean donor (different clean than paired)
    3) caller falls back to orthogonal_complement subspace control
    """
    from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz

    swap_path = activation_path(pm_tid, "rival_value_swap")
    if swap_path.is_file():
        rival_vec = get_vector(load_activation_npz(swap_path), position=position, layer=layer)
        if rival_vec is not None:
            return np.asarray(rival_vec, dtype=np.float32), "rival_value_swap"

    alt_cleans = [
        str(cp["clean_trajectory_id"])
        for cp in cross_pairs
        if str(cp.get("pm_trajectory_id") or "") != pm_tid
        and str(cp.get("clean_trajectory_id") or "") != paired_clean_tid
    ]
    seen: set[str] = set()
    for alt_tid in alt_cleans:
        if alt_tid in seen:
            continue
        seen.add(alt_tid)
        alt_path = activation_path(alt_tid, "original")
        if not alt_path.is_file():
            continue
        alt_vec = get_vector(load_activation_npz(alt_path), position=position, layer=layer)
        if alt_vec is not None:
            return np.asarray(alt_vec, dtype=np.float32), "wrong_clean_donor"

    return None, "orthogonal_complement_fallback"


def pending_roi_layer_extract_ids(
    *,
    layer: int | None = None,
    n_layers: int = 64,
    skip_trajectory_ids: dict[str, str] | None = None,
) -> list[str]:
    """Trajectory IDs missing any required ROI layer in on-disk activations."""
    from ccer.mechanism.activation_store import activation_path, missing_layers
    from ccer.mechanism.mechanism_pool import build_cem_mechanism_cross_pairs
    from ccer.mechanism.round8_verify import cross_pair_trajectory_ids

    roi = line_a_roi_defaults()
    layer = int(layer if layer is not None else roi["layer"])
    required_layers = line_b_roi_layer_indices(n_layers, center=layer)
    skip_ids = skip_trajectory_ids or {}
    cross = build_cem_mechanism_cross_pairs(pool="mechanism_research")["pm_clean_cross_pairs"]
    tids: set[str] = set()
    for cp in cross:
        tids.add(str(cp["pm_trajectory_id"]))
        tids.add(str(cp["clean_trajectory_id"]))
    tids |= cross_pair_trajectory_ids("CEM")
    pending: list[str] = []
    for tid in sorted(tids):
        if tid in skip_ids:
            continue
        path = activation_path(tid, "original")
        if not path.is_file() or missing_layers(path, required_layers):
            pending.append(tid)
    return pending


def pending_symmetric_v3_refresh_ids(
    *,
    layer: int | None = None,
    position: str | None = None,
    n_layers: int = 64,
    skip_trajectory_ids: dict[str, str] | None = None,
) -> list[str]:
    """Trajectory IDs needing symmetric Line D v3 refresh and/or ROI layer merge."""
    from ccer.mechanism.activation_store import activation_path, load_activation_npz, missing_layers
    from ccer.mechanism.mechanism_pool import build_cem_mechanism_cross_pairs
    from ccer.mechanism.round8_verify import cross_pair_trajectory_ids
    from ccer.replay.live_position import ANSWER_REGION_POSITIONS, activation_passes_line_d_v3_gate

    roi = line_a_roi_defaults()
    layer = int(layer if layer is not None else roi["layer"])
    position = str(position if position is not None else roi["position"])
    required_layers = line_b_roi_layer_indices(n_layers, center=layer)
    skip_ids = skip_trajectory_ids or {}
    cross = build_cem_mechanism_cross_pairs(pool="mechanism_research")["pm_clean_cross_pairs"]
    tids: set[str] = set()
    for cp in cross:
        tids.add(str(cp["pm_trajectory_id"]))
        tids.add(str(cp["clean_trajectory_id"]))
    tids |= cross_pair_trajectory_ids("CEM")
    pending: list[str] = []
    for tid in sorted(tids):
        if tid in skip_ids:
            continue
        path = activation_path(tid, "original")
        if not path.is_file():
            pending.append(tid)
            continue
        loaded = load_activation_npz(path)
        if position in ANSWER_REGION_POSITIONS and not activation_passes_line_d_v3_gate(
            loaded, position=position
        ):
            pending.append(tid)
            continue
        if missing_layers(path, required_layers):
            pending.append(tid)
    return pending


def preflight_root_cause_audit(
    *,
    pool: str = "mechanism_research",
    layer: int | None = None,
    position: str | None = None,
    n_layers: int = 64,
) -> dict[str, Any]:
    """CPU audit of all seven root-cause closures before GPU sweep (debug mode)."""
    from ccer.mechanism.activation_store import activation_path, load_activation_npz
    from ccer.mechanism.round12_line_b import _eligible_cross_pairs
    from ccer.replay.live_position import ANSWER_REGION_POSITIONS, cross_pair_passes_line_d_v3_gate

    roi = line_a_roi_defaults()
    layer = int(layer if layer is not None else roi["layer"])
    position = str(position if position is not None else roi["position"])
    pool_name = "mechanism_research" if pool == "mechanism_research" else "dev"

    all_eligible = _eligible_cross_pairs(
        pool_name, layer=layer, position=position, min_pair_quality=0.0,
    )
    train_pairs, train_meta = filter_cross_pairs_by_quality(
        all_eligible, min_quality=DEFAULT_MIN_PAIR_QUALITY,
    )
    eval_pairs, eval_meta = filter_cross_pairs_by_quality(
        all_eligible, min_quality=DEFAULT_MIN_EVAL_PAIR_QUALITY,
    )

    wrong_impl: dict[str, int] = {}
    for cp in eval_pairs:
        pm_tid = str(cp["pm_trajectory_id"])
        clean_tid = str(cp["clean_trajectory_id"])
        _, impl = resolve_wrong_owner_donor_vec(
            pm_tid=pm_tid,
            paired_clean_tid=clean_tid,
            cross_pairs=all_eligible,
            layer=layer,
            position=position,
        )
        wrong_impl[impl] = wrong_impl.get(impl, 0) + 1

    selected_layer, layer_scan = line_b_select_roi_layer(
        n_layers=n_layers,
        position=position,
        track="CEM",
        pool=pool_name,
        pm_trajectory_allowlist={str(cp["pm_trajectory_id"]) for cp in train_pairs} or None,
    )

    pending_extract = pending_symmetric_v3_refresh_ids(layer=layer, n_layers=n_layers)
    pending_roi_only = pending_roi_layer_extract_ids(layer=layer, n_layers=n_layers)

    eval_v3_pairs = list(eval_pairs)
    if position in ANSWER_REGION_POSITIONS:
        eval_v3_pairs = []
        for cp in eval_pairs:
            pm_tid = str(cp["pm_trajectory_id"])
            clean_tid = str(cp["clean_trajectory_id"])
            pm_npz = load_activation_npz(activation_path(pm_tid, "original"))
            clean_npz = load_activation_npz(activation_path(clean_tid, "original"))
            if cross_pair_passes_line_d_v3_gate(pm_npz, clean_npz, position):
                eval_v3_pairs.append(cp)

    coverage_note = "blocked" if not eval_v3_pairs else "ok"
    blocked_reason: str | None = None
    if not eval_v3_pairs:
        if pending_extract:
            blocked_reason = (
                f"No eval pairs passing line_d_v3 symmetric gate @ {position} — "
                f"run extract-activations ({len(pending_extract)} trajectories pending symmetric v3 refresh)"
            )
        else:
            blocked_reason = (
                f"No eval pairs passing line_d_v3 symmetric gate @ {position} — "
                "check skip-list / symmetric claim anchor resolution"
            )

    report = {
        "schema_version": "ccer_line_b_preflight_v3",
        "protocol_version": LINE_B_PROTOCOL_VERSION,
        "roi": {"layer": layer, "position": position, **roi},
        "layer_scan_selected": selected_layer,
        "layer_scan": layer_scan,
        "n_all_activation_eligible": len(all_eligible),
        "n_train_quality": len(train_pairs),
        "train_quality_meta": train_meta,
        "n_eval_quality": len(eval_pairs),
        "eval_quality_meta": eval_meta,
        "n_eval_v3_gate": len(eval_v3_pairs),
        "n_pending_roi_extract": len(pending_roi_only),
        "n_pending_symmetric_v3_refresh": len(pending_extract),
        "wrong_owner_impl_counts": wrong_impl,
        "root_cause_status": {
            "1_training_eval_alignment": "das_behavioral_margin_loss_v3",
            "2_roi_layer_scan": f"L{selected_layer}_selected",
            "3_pair_quality_filter": f"train={len(train_pairs)} eval={len(eval_pairs)} v3_gate={len(eval_v3_pairs)}",
            "4_use_cache_symmetric": "use_cache=False",
            "5_steering_apply": "dynamic_answer_anchor@claim_onset",
            "6_full_vector_ceiling": "per_rank_in_sweep",
            "7_wrong_owner_donor": wrong_impl,
        },
        "coverage_gate": coverage_note,
        "blocked_reason": blocked_reason if not eval_v3_pairs else None,
    }
    return report


def subspace_projection_stats(
    u: np.ndarray,
    h_recipient: np.ndarray,
    h_donor: np.ndarray,
) -> dict[str, float]:
    """Fraction of donor-recipient delta captured by rank-k subspace (root #6 diagnostic)."""
    u = np.asarray(u, dtype=np.float64)
    hr = np.asarray(h_recipient, dtype=np.float64).reshape(-1)
    hd = np.asarray(h_donor, dtype=np.float64).reshape(-1)
    delta = hd - hr
    d_norm = float(np.linalg.norm(delta))
    if d_norm < 1e-12 or u.size == 0:
        return {"delta_norm": d_norm, "proj_fraction": 0.0, "residual_norm": d_norm}
    proj = u @ (u.T @ delta)
    p_norm = float(np.linalg.norm(proj))
    resid = float(np.linalg.norm(delta - proj))
    return {
        "delta_norm": d_norm,
        "proj_fraction": p_norm / d_norm,
        "residual_norm": resid,
        "proj_norm": p_norm,
    }
