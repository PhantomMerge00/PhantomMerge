"""Owner difference PCA + two-stage scan + DAS robustness (P3a, expert_recorrect §3.1)."""
from __future__ import annotations

from typing import Any

import numpy as np

from ccer.io_utils import load_jsonl, write_json
from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.das_train import compare_pca_vs_das, fit_das_subspace
from ccer.mechanism.pair_select import TrackName, select_track_pairs
from ccer.paths import P1_DIR, P3_BINDING_ROI, P3_U_OWNER_NPZ
from ccer.replay.hf_forward import coarse_layer_indices, dense_layer_indices
from ccer.replay.position_registry import POSITION_NAMES

RANK_CANDIDATES = [2, 4, 8, 16]
MIN_CROSS_FOR_INTERCHANGE = 10
MAX_LAYER_FRAC_FOR_ROI = 0.85  # exclude last ~15% layers (ROUND4: weak causal leverage)


def _follow_map(track: TrackName = "CAP") -> dict[str, bool | None]:
    cond = "query_value_swap" if track == "CAP" else "rival_value_swap"
    out: dict[str, bool | None] = {}
    for row in load_jsonl(P1_DIR / "source_effects_rows.jsonl"):
        if row.get("cohort") != track or row.get("condition_id") != cond:
            continue
        if not row.get("scorable"):
            continue
        out[str(row["trajectory_id"])] = row.get("source_following")
    return out


def _separation_score(vectors: list[np.ndarray], labels: list[int]) -> float:
    if len(vectors) < 4 or len(set(labels)) < 2:
        return 0.0
    x = np.stack(vectors, axis=0)
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    between = 0.0
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        if not idx:
            continue
        cluster = xc[idx].mean(axis=0)
        between += len(idx) * float(np.dot(cluster, cluster))
    within = float(np.mean(np.sum(xc * xc, axis=1)))
    return between / max(within, 1e-8)


def collect_diff_vectors(
    *,
    position: str,
    layer: int,
    track: TrackName = "CAP",
    cond_swap: str | None = None,
    pool: str = "dev",
    pm_trajectory_allowlist: set[str] | None = None,
    cross_pair_quality_min: float = 0.35,
    require_line_d_v3: bool = False,
) -> dict[str, Any]:
    if track == "CEM" and pool == "mechanism_research":
        from ccer.mechanism.mechanism_pool import build_cem_mechanism_cross_pairs

        selection = build_cem_mechanism_cross_pairs(pool="mechanism_research")
        selection["pairs"] = selection["cem_pairs"]
    else:
        selection = select_track_pairs(track)
    follow = _follow_map(track)
    within: list[np.ndarray] = []
    within_labels: list[int] = []
    cross: list[np.ndarray] = []
    cond_swap = cond_swap or ("query_value_swap" if track == "CAP" else "rival_value_swap")
    cross_key = "cap_trajectory_id" if track == "CAP" else "pm_trajectory_id"

    if require_line_d_v3:
        from ccer.replay.live_position import activation_passes_line_d_v3_gate, cross_pair_passes_line_d_v3_gate

    n_cross_skipped_v3 = 0
    n_within_skipped_v3 = 0

    for pair in selection["pairs"]:
        tid = pair["trajectory_id"]
        if pm_trajectory_allowlist is not None and str(tid) not in pm_trajectory_allowlist:
            continue
        orig_path = activation_path(tid, "original")
        swap_path = activation_path(tid, cond_swap)
        if not orig_path.is_file() or not swap_path.is_file():
            continue
        orig = load_activation_npz(orig_path)
        swap = load_activation_npz(swap_path)
        if require_line_d_v3:
            if not activation_passes_line_d_v3_gate(orig, position=position):
                n_within_skipped_v3 += 1
                continue
            if not activation_passes_line_d_v3_gate(swap, position=position):
                n_within_skipped_v3 += 1
                continue
        h_orig = get_vector(orig, position=position, layer=layer)
        h_swap = get_vector(swap, position=position, layer=layer)
        if h_orig is None or h_swap is None:
            continue
        within.append(h_orig - h_swap)
        within_labels.append(1 if follow.get(tid) else 0)

    audit_index: dict[str, Any] = {}
    if pool == "mechanism_research":
        from ccer.mechanism.line_b_protocol import load_pair_audit_index, pair_quality_score

        audit_index = load_pair_audit_index()

    n_cross_skipped_quality = 0

    for cp in selection["pm_clean_cross_pairs"]:
        pm_tid = cp.get(cross_key) or cp.get("cap_trajectory_id")
        clean_tid = cp["clean_trajectory_id"]
        if not pm_tid:
            continue
        if pm_trajectory_allowlist is not None and str(pm_tid) not in pm_trajectory_allowlist:
            continue
        if audit_index:
            q = pair_quality_score(audit_index.get(str(pm_tid)), cross_pair_row=cp)
            if q < cross_pair_quality_min:
                n_cross_skipped_quality += 1
                continue
        pm_path = activation_path(str(pm_tid), "original")
        clean_path = activation_path(str(clean_tid), "original")
        if not pm_path.is_file() or not clean_path.is_file():
            continue
        pm = load_activation_npz(pm_path)
        clean = load_activation_npz(clean_path)
        if require_line_d_v3:
            if not cross_pair_passes_line_d_v3_gate(pm, clean, position):
                n_cross_skipped_v3 += 1
                continue
        h_pm = get_vector(pm, position=position, layer=layer)
        h_clean = get_vector(clean, position=position, layer=layer)
        if h_pm is None or h_clean is None:
            continue
        cross.append(h_pm - h_clean)

    return {
        "within_vectors": within,
        "within_labels": within_labels,
        "cross_vectors": cross,
        "n_cross_skipped_quality": n_cross_skipped_quality,
        "n_cross_skipped_v3": n_cross_skipped_v3 if require_line_d_v3 else 0,
        "n_within_skipped_v3": n_within_skipped_v3 if require_line_d_v3 else 0,
        "require_line_d_v3": require_line_d_v3,
    }


POSITION_CONTROL_POSITIONS = ("prompt_end", "commitment", "claim_onset", "pre_value")


def position_comparison_table(*, track: TrackName = "CAP", layer: int | None = None) -> list[dict[str, Any]]:
    staged = two_stage_layer_scan(track=track)
    layer = int(layer or staged.get("peak_layer") or 0)
    rows: list[dict[str, Any]] = []
    for position in POSITION_CONTROL_POSITIONS:
        bundle = collect_diff_vectors(position=position, layer=layer, track=track)
        within, labels, cross = bundle["within_vectors"], bundle["within_labels"], bundle["cross_vectors"]
        cross_labels = ([0] * (len(cross) // 2) + [1] * (len(cross) - len(cross) // 2)) if len(cross) >= 4 else []
        row = {
            "layer": layer,
            "position": position,
            "n_within": len(within),
            "n_cross": len(cross),
            "follow_separation": _separation_score(within, labels),
            "cross_separation": _separation_score(cross, cross_labels) if cross_labels else 0.0,
        }
        row["combined_score"] = row["follow_separation"] + 0.5 * row["cross_separation"]
        rows.append(row)
    rows.sort(key=lambda r: r["combined_score"], reverse=True)
    return rows


def position_permutation_control(
    *,
    layer: int,
    position: str,
    track: TrackName = "CAP",
    confound_ratio_threshold: float = 0.5,
) -> dict[str, Any]:
    owner_cond = "query_value_swap" if track == "CAP" else "rival_value_swap"
    owner = collect_diff_vectors(position=position, layer=layer, track=track, cond_swap=owner_cond)
    perm = collect_diff_vectors(position=position, layer=layer, track=track, cond_swap="position_permutation")
    owner_vecs = owner["within_vectors"]
    perm_vecs = perm["within_vectors"]
    owner_sep = _separation_score(owner_vecs, owner["within_labels"])
    perm_labels = [1] * len(perm_vecs) if perm_vecs else []
    perm_sep = _separation_score(perm_vecs, perm_labels) if len(set(perm_labels)) >= 1 and len(perm_vecs) >= 4 else (
        float(np.linalg.norm(np.mean(np.stack(perm_vecs, axis=0), axis=0))) if len(perm_vecs) >= 2 else 0.0
    )
    ratio = perm_sep / max(owner_sep, 1e-8)
    recency_confounded = ratio >= confound_ratio_threshold and owner_sep > 0
    return {
        "layer": layer,
        "position": position,
        "owner_condition": owner_cond,
        "owner_separation": owner_sep,
        "permutation_separation": perm_sep,
        "permutation_to_owner_ratio": ratio,
        "recency_confounded": recency_confounded,
        "n_owner_within": len(owner_vecs),
        "n_perm_within": len(perm_vecs),
        "passes_control": not recency_confounded,
    }


def _n_layers(track: TrackName) -> int:
    selection = select_track_pairs(track)
    sample_tid = selection["pairs"][0]["trajectory_id"] if selection["pairs"] else None
    if sample_tid and activation_path(sample_tid, "original").is_file():
        return max(load_activation_npz(activation_path(sample_tid, "original"))["layer_indices"]) + 1
    return 64


def _pick_peak_scan_row(rows: list[dict[str, Any]], *, min_cross: int = MIN_CROSS_FOR_INTERCHANGE) -> dict[str, Any]:
    if not rows:
        return {"layer": 0, "position": "prompt_end", "combined_score": 0.0, "n_cross": 0}
    eligible = [r for r in rows if int(r.get("n_cross") or 0) >= min_cross]
    pool = eligible if eligible else rows
    return max(pool, key=lambda r: float(r.get("combined_score") or 0.0))


def _eligible_layers(n_layers: int) -> list[int]:
    max_layer = max(0, int(n_layers * MAX_LAYER_FRAC_FOR_ROI) - 1)
    return [l for l in coarse_layer_indices(n_layers) if l <= max_layer]


def layer_position_scan(*, track: TrackName = "CAP") -> list[dict[str, Any]]:
    n_layers = _n_layers(track)
    layers = _eligible_layers(n_layers)
    scan_positions = [p for p in POSITION_NAMES if p not in ("query_owner",)]
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for position in scan_positions:
            bundle = collect_diff_vectors(position=position, layer=layer, track=track)
            within, labels, cross = bundle["within_vectors"], bundle["within_labels"], bundle["cross_vectors"]
            cross_labels = ([0] * (len(cross) // 2) + [1] * (len(cross) - len(cross) // 2)) if len(cross) >= 4 else []
            row = {
                "layer": layer,
                "position": position,
                "n_within": len(within),
                "n_cross": len(cross),
                "follow_separation": _separation_score(within, labels),
                "cross_separation": _separation_score(cross, cross_labels) if cross_labels else 0.0,
            }
            row["combined_score"] = row["follow_separation"] + 0.5 * row["cross_separation"]
            rows.append(row)
    rows.sort(key=lambda r: r["combined_score"], reverse=True)
    return rows


def two_stage_layer_scan(*, track: TrackName = "CAP") -> dict[str, Any]:
    coarse = layer_position_scan(track=track)
    if not coarse:
        return {"coarse_scan": [], "dense_scan": [], "peak_layer": 0, "peak_position": "prompt_end"}
    peak = _pick_peak_scan_row(coarse)
    peak_layer = int(peak["layer"])
    peak_pos = str(peak["position"])
    max_layer = max(0, int(_n_layers(track) * MAX_LAYER_FRAC_FOR_ROI) - 1)
    dense_layers = [l for l in dense_layer_indices(_n_layers(track), peak_layer, radius=2) if l <= max_layer]
    dense_rows: list[dict[str, Any]] = []
    for layer in dense_layers:
        bundle = collect_diff_vectors(position=peak_pos, layer=layer, track=track)
        within, labels, cross = bundle["within_vectors"], bundle["within_labels"], bundle["cross_vectors"]
        cross_labels = ([0] * (len(cross) // 2) + [1] * (len(cross) - len(cross) // 2)) if len(cross) >= 4 else []
        row = {
            "layer": layer,
            "position": peak_pos,
            "follow_separation": _separation_score(within, labels),
            "cross_separation": _separation_score(cross, cross_labels) if cross_labels else 0.0,
        }
        row["combined_score"] = row["follow_separation"] + 0.5 * row["cross_separation"]
        dense_rows.append(row)
    dense_rows.sort(key=lambda r: r["combined_score"], reverse=True)
    return {
        "coarse_scan": coarse[:20],
        "dense_scan": dense_rows,
        "peak_layer": int(dense_rows[0]["layer"]) if dense_rows else peak_layer,
        "peak_position": peak_pos,
    }


def fit_owner_pca(
    *,
    layer: int,
    position: str,
    rank: int,
    seed: int = 42,
    track: TrackName = "CAP",
    pool: str = "dev",
    pm_trajectory_allowlist: set[str] | None = None,
    require_line_d_v3: bool = False,
    cross_only: bool = False,
) -> dict[str, Any]:
    bundle = collect_diff_vectors(
        position=position,
        layer=layer,
        track=track,
        pool=pool,
        pm_trajectory_allowlist=pm_trajectory_allowlist,
        require_line_d_v3=require_line_d_v3,
    )
    if cross_only:
        vectors = bundle["cross_vectors"]
    else:
        vectors = bundle["within_vectors"] + bundle["cross_vectors"]
    if len(vectors) < rank:
        return {
            "error": "insufficient_vectors",
            "n_vectors": len(vectors),
            "rank": rank,
            "n_cross": len(bundle["cross_vectors"]),
            "n_within": len(bundle["within_vectors"]),
            "require_line_d_v3": require_line_d_v3,
            "cross_only": cross_only,
        }
    x = np.stack(vectors, axis=0)
    x = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    r = min(rank, vt.shape[0])
    q, _ = np.linalg.qr(vt[:r].T)
    explained = (s[:r] ** 2) / max(float(np.sum(s**2)), 1e-8)
    return {
        "rank": r,
        "U_owner": q.astype(np.float32),
        "pc_variance_explained": float(np.sum(explained)),
        "seed": seed,
        "n_vectors": len(vectors),
        "n_cross": len(bundle["cross_vectors"]),
        "n_within": len(bundle["within_vectors"]),
        "require_line_d_v3": require_line_d_v3,
        "cross_only": cross_only,
    }


def run_owner_pca_analysis(
    *,
    seed: int = 42,
    track: TrackName = "CAP",
    peak_layer_override: int | None = None,
    peak_pos_override: str | None = None,
    iia_probe_table: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    staged = two_stage_layer_scan(track=track)
    peak_layer = int(peak_layer_override if peak_layer_override is not None else staged["peak_layer"])
    peak_pos = str(peak_pos_override if peak_pos_override is not None else staged["peak_position"])
    best_rank = 4
    best_var = -1.0
    rank_results: dict[str, Any] = {}
    for rank in RANK_CANDIDATES:
        fit = fit_owner_pca(layer=peak_layer, position=peak_pos, rank=rank, seed=seed, track=track)
        rank_results[str(rank)] = {k: v for k, v in fit.items() if k != "U_owner"}
        var = float(fit.get("pc_variance_explained") or 0.0)
        if fit.get("U_owner") is not None and var >= best_var:
            best_var = var
            best_rank = rank

    pos_table = position_comparison_table(track=track, layer=peak_layer)
    pos_control = position_permutation_control(layer=peak_layer, position=peak_pos, track=track)
    primary_passes = bool(pos_control.get("passes_control"))
    if not primary_passes and pos_table:
        passing: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for cand in pos_table:
            ctrl = position_permutation_control(
                layer=int(cand["layer"]),
                position=str(cand["position"]),
                track=track,
            )
            if ctrl.get("passes_control"):
                passing.append((cand, ctrl))
        if passing:
            interchange_ready = [
                (cand, ctrl)
                for cand, ctrl in passing
                if int(cand.get("n_cross") or 0) >= MIN_CROSS_FOR_INTERCHANGE
            ]
            pick_pool = interchange_ready if interchange_ready else passing
            pick_pool.sort(key=lambda item: float(item[0].get("combined_score") or 0.0), reverse=True)
            cand, pos_control = pick_pool[0]
            peak_layer = int(cand["layer"])
            peak_pos = str(cand["position"])
            primary_passes = True
            pos_table = position_comparison_table(track=track, layer=peak_layer)

    top = next((r for r in pos_table if r["position"] == peak_pos and r["layer"] == peak_layer), pos_table[0] if pos_table else {})
    if int(top.get("n_cross") or 0) < MIN_CROSS_FOR_INTERCHANGE:
        upgrade = _pick_peak_scan_row(staged.get("coarse_scan") or [])
        if int(upgrade.get("n_cross") or 0) >= MIN_CROSS_FOR_INTERCHANGE:
            ctrl = position_permutation_control(
                layer=int(upgrade["layer"]),
                position=str(upgrade["position"]),
                track=track,
            )
            if ctrl.get("passes_control"):
                peak_layer = int(upgrade["layer"])
                peak_pos = str(upgrade["position"])
                pos_control = ctrl
                primary_passes = True
                top = upgrade

    primary_fit = fit_owner_pca(layer=peak_layer, position=peak_pos, rank=best_rank, seed=seed, track=track)
    u_pca = primary_fit.get("U_owner")
    vecs = collect_diff_vectors(position=peak_pos, layer=peak_layer, track=track)
    all_vecs = vecs["within_vectors"] + vecs["cross_vectors"]
    if all_vecs:
        dim = int(np.asarray(all_vecs[0]).reshape(-1).shape[0])
        all_vecs = [np.asarray(v).reshape(-1) for v in all_vecs if np.asarray(v).reshape(-1).shape[0] == dim]
    das_fit = fit_das_subspace(
        all_vecs,
        rank=best_rank,
        seed=seed,
        layer=peak_layer,
        position=peak_pos,
        track=track,
    )
    u_das = das_fit.get("U_das")
    das_compare = compare_pca_vs_das(
        u_pca=u_pca if u_pca is not None else np.array([]),
        u_das=u_das if u_das is not None else np.array([]),
    )
    n_cross = int(top.get("n_cross") or 0)
    selection = select_track_pairs(track)
    roi: dict[str, Any] = {
        "schema_version": "ccer_binding_roi_v2",
        "track": track,
        "primary_roi": {
            "layer": peak_layer,
            "position": peak_pos,
            "rank": best_rank,
            "pc_variance_explained": primary_fit.get("pc_variance_explained"),
            "combined_score": top.get("combined_score"),
            "n_cross": n_cross,
            "interchange_ready": n_cross >= MIN_CROSS_FOR_INTERCHANGE,
            "roi_status": "frozen" if primary_passes else "candidate_recency_confounded",
            "selection_method": "iia_probe" if iia_probe_table else "pca_scan",
            "iia_probe_output_diff_rate": (
                float(iia_probe_table[0].get("output_diff_rate") or 0.0) if iia_probe_table else None
            ),
        },
        "iia_probe_table": iia_probe_table or [],
        "two_stage_scan": staged,
        "position_comparison_table": pos_table,
        "factorial_disentanglement_table": pos_control,
        "rank_sweep": rank_results,
        "U_owner_path": str(P3_U_OWNER_NPZ),
        "das_robustness": {k: v for k, v in das_fit.items() if k != "U_das"},
        "pca_vs_das": das_compare,
        "selection_summary": {
            "n_pairs": len(selection.get("pairs") or []),
            "below_32_threshold": selection.get("below_32_threshold"),
            "below_32_reason": selection.get("below_32_reason"),
        },
        "signal_present": float(top.get("combined_score") or 0.0) > 0.05,
        "pca_triage": {"requires_das_before_negative_claim": True, "das_fitted": "U_das" in das_fit},
    }
    return roi, u_pca, u_das


def write_binding_roi(roi: dict[str, Any], *, u_pca: np.ndarray | None = None, u_das: np.ndarray | None = None) -> None:
    P3_BINDING_ROI.parent.mkdir(parents=True, exist_ok=True)
    if u_pca is not None or u_das is not None:
        np.savez_compressed(P3_U_OWNER_NPZ, U_pca=u_pca, U_das=u_das)
    write_json(P3_BINDING_ROI, roi)
