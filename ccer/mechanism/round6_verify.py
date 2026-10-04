"""ROUND6 verification: subspace projection check + dual-metric contrast."""
from __future__ import annotations

from typing import Any

import numpy as np

from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.controls import build_control_specs
from ccer.mechanism.interchange import _patch_hidden
from ccer.mechanism.iia_roi_probe import _live_token_idx, _outputs_differ
from ccer.mechanism.owner_pca import fit_owner_pca
from ccer.mechanism.pair_select import load_trajectory_index, messages_for_condition
from ccer.mechanism.rank_sweep import _cross_pairs, measure_diff_rate
from ccer.mechanism.stats import wilson_ci
from ccer.replay.answer_utils import (
    answers_differ,
    extract_selected_product_id,
    full_response_excerpt_differ,
    selected_product_blocks_differ,
)


def _subspace_residual(patch: np.ndarray, u: np.ndarray) -> dict[str, float]:
    """Fraction of patch norm outside column space of orthonormal U (d, r)."""
    proj = u @ (u.T @ patch)
    residual = patch - proj
    total = float(np.linalg.norm(patch))
    resid = float(np.linalg.norm(residual))
    if total < 1e-12:
        return {
            "patch_l2": total,
            "residual_l2": resid,
            "residual_frac": 0.0,
            "in_subspace_frac": 1.0,
        }
    return {
        "patch_l2": total,
        "residual_l2": resid,
        "residual_frac": resid / total,
        "in_subspace_frac": float(np.linalg.norm(proj)) / total,
    }


def verify_subspace_projection(
    *,
    layer: int = 32,
    position: str = "commitment",
    rank: int = 16,
    track: str = "CEM",
    pair_indices: list[int] | None = None,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """§一: check interchange patch Δ lies in rank-r PCA subspace (CPU, no model)."""
    pairs = _cross_pairs(track)
    if pair_indices is None:
        pair_indices = [0, 5, 10, 15, 21]
    pair_indices = [i for i in pair_indices if 0 <= i < len(pairs)]

    fit = fit_owner_pca(layer=layer, position=position, rank=rank, track=track)
    u = fit.get("U_owner")
    if u is None:
        return {"error": fit.get("error", "pca_fit_failed")}

    rows_index = load_trajectory_index()
    roi_stub = {"primary_roi": {"layer": layer, "position": position}, "U_owner": u.tolist()}
    pair_rows: list[dict[str, Any]] = []
    utu = u.T @ u
    ortho_err = float(np.linalg.norm(utu - np.eye(utu.shape[0]), ord="fro"))

    for idx in pair_indices:
        cp = pairs[idx]
        pm_tid = str(cp["pm_trajectory_id"])
        clean_tid = str(cp["clean_trajectory_id"])
        pm_traj = rows_index.get(pm_tid)
        if not pm_traj:
            continue
        pm_npz = load_activation_npz(activation_path(pm_tid, "original"))
        clean_npz = load_activation_npz(activation_path(clean_tid, "original"))
        pm_vec = get_vector(pm_npz, position=position, layer=layer)
        clean_vec = get_vector(clean_npz, position=position, layer=layer)
        pos_idx = int((pm_npz.get("positions") or {}).get(position) or -1)
        if pm_vec is None or clean_vec is None or pos_idx < 0:
            continue

        delta = clean_vec.astype(np.float32) - pm_vec.astype(np.float32)
        analytical_patch = alpha * (u @ (u.T @ delta))
        full_patch = alpha * delta
        delta_capture = float(np.linalg.norm(u @ (u.T @ delta))) / max(float(np.linalg.norm(delta)), 1e-12)

        spec = build_control_specs(
            control_id="target_interchange",
            roi=roi_stub,
            donor_vec=clean_vec,
            recipient_vec=pm_vec,
            position_token_idx=pos_idx,
            alpha=alpha,
        )
        import torch

        h = torch.tensor(pm_vec, dtype=torch.float32)
        h_new = _patch_hidden(h, spec)
        hook_patch = (h_new - h).detach().cpu().numpy()

        anal_res = _subspace_residual(analytical_patch, u)
        hook_res = _subspace_residual(hook_patch, u)

        pair_rows.append(
            {
                "pair_index": idx,
                "pm_trajectory_id": pm_tid,
                "clean_trajectory_id": clean_tid,
                "delta_l2": float(np.linalg.norm(delta)),
                "delta_frac_in_subspace": delta_capture,
                "analytical_patch": anal_res,
                "hook_patch": hook_res,
                "hook_matches_analytical_l2": float(np.linalg.norm(hook_patch - analytical_patch)),
                "full_vector_patch_l2": float(np.linalg.norm(full_patch)),
                "interchange_vs_full_vector_ratio": anal_res["patch_l2"] / max(float(np.linalg.norm(full_patch)), 1e-12),
            }
        )

    max_resid = max((r["hook_patch"]["residual_frac"] for r in pair_rows), default=0.0)
    return {
        "schema_version": "ccer_round6_subspace_v1",
        "layer": layer,
        "position": position,
        "rank": int(fit.get("rank") or rank),
        "pc_variance_explained": fit.get("pc_variance_explained"),
        "u_orthonormal_fro_err": ortho_err if pair_rows else None,
        "pairs_checked": len(pair_rows),
        "max_hook_residual_frac": max_resid,
        "passes_subspace_check": max_resid < 1e-4,
        "pair_rows": pair_rows,
    }


def verify_dual_metrics_from_texts(
    pair_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """§三: tiered metrics on pre-generated ni/ti texts.

    Primary: product_id_diff. Secondary: selected_product_block (no Compared).
    Diagnostic: full_response_excerpt (legacy loose extract_answer).
    """
    n = 0
    raw_diff = 0
    answer_diff = 0
    product_id_diff = 0
    selected_block_diff = 0
    full_excerpt_diff = 0
    compared_only = 0
    raw_only = 0
    answer_only = 0
    details: list[dict[str, Any]] = []

    for row in pair_rows:
        ni = str(row.get("ni_text") or "")
        ti = str(row.get("ti_text") or "")
        if not ni and not ti:
            continue
        n += 1
        raw = _outputs_differ(ni, ti)
        ans = answers_differ(ni, ti)
        sel_diff = selected_product_blocks_differ(ni, ti)
        full_diff = full_response_excerpt_differ(ni, ti)
        pid_ni = extract_selected_product_id(ni)
        pid_ti = extract_selected_product_id(ti)
        pid_diff = bool(pid_ni and pid_ti and pid_ni != pid_ti)
        if raw:
            raw_diff += 1
        if ans:
            answer_diff += 1
        if pid_diff:
            product_id_diff += 1
        if sel_diff:
            selected_block_diff += 1
        if full_diff:
            full_excerpt_diff += 1
        if full_diff and not ans:
            compared_only += 1
        if raw and not ans:
            raw_only += 1
        if ans and not raw:
            answer_only += 1
        details.append(
            {
                "pm_trajectory_id": row.get("pm_trajectory_id"),
                "clean_trajectory_id": row.get("clean_trajectory_id"),
                "outputs_differ": raw,
                "answer_diff": ans,
                "extract_answer_diff": ans,
                "selected_product_block_diff": sel_diff,
                "full_response_excerpt_diff": full_diff,
                "product_id_ni": pid_ni,
                "product_id_ti": pid_ti,
                "product_id_diff": pid_diff,
                "compared_section_only_diff": full_diff and not ans,
            }
        )

    return {
        "n_pairs": n,
        "primary_metric": "product_id_diff",
        "product_id_diff": {
            "k": product_id_diff,
            "rate": product_id_diff / n if n else 0.0,
            "ci95": wilson_ci(product_id_diff, n),
        },
        "selected_product_block_diff": {
            "k": selected_block_diff,
            "rate": selected_block_diff / n if n else 0.0,
            "ci95": wilson_ci(selected_block_diff, n),
        },
        "answer_diff": {
            "k": answer_diff,
            "rate": answer_diff / n if n else 0.0,
            "ci95": wilson_ci(answer_diff, n),
        },
        "outputs_differ": {"k": raw_diff, "rate": raw_diff / n if n else 0.0, "ci95": wilson_ci(raw_diff, n)},
        "full_response_excerpt_diff": {
            "k": full_excerpt_diff,
            "rate": full_excerpt_diff / n if n else 0.0,
            "ci95": wilson_ci(full_excerpt_diff, n),
        },
        "extract_answer_diff": {
            "k": answer_diff,
            "rate": answer_diff / n if n else 0.0,
            "ci95": wilson_ci(answer_diff, n),
        },
        "compared_section_only_diff": compared_only,
        "raw_only_not_answer": raw_only,
        "answer_only_not_raw": answer_only,
        "metrics_agree": raw_diff == answer_diff,
        "pair_details": details,
    }


def run_dual_metric_sweep(
    *,
    model: Any,
    tokenizer: Any,
    layer: int = 32,
    position: str = "commitment",
    rank: int = 16,
    track: str = "CEM",
    probe_max_tokens: int = 96,
    pair_limit: int | None = None,
) -> dict[str, Any]:
    """Re-run interchange@rank and collect both metrics (needs GPU)."""
    row = measure_diff_rate(
        model=model,
        tokenizer=tokenizer,
        track=track,
        layer=layer,
        position=position,
        rank=rank,
        mode="interchange",
        pair_limit=pair_limit,
        probe_max_tokens=probe_max_tokens,
        save_texts=True,  # noqa: FBT003
    )
    return {
        "schema_version": "ccer_round6_dual_metric_v1",
        "layer": layer,
        "position": position,
        "rank": rank,
        "probe_max_tokens": probe_max_tokens,
        "round5_outputs_differ_rate": row.get("output_diff_rate"),
        "dual_metrics": verify_dual_metrics_from_texts(row.get("pair_details") or []),
    }


def rank_wilson_table(rank_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """§二: Wilson CI for each rank condition."""
    out = []
    for row in rank_rows:
        k = int(row.get("n_output_diff") or 0)
        n = int(row.get("n_pairs") or 0)
        lo, hi = wilson_ci(k, n)
        out.append(
            {
                "rank": row.get("rank"),
                "k": k,
                "n": n,
                "rate": row.get("output_diff_rate"),
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
        )
    return out
