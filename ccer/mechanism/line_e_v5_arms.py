"""Build Line E v5 intervention specs (B-style interchange + probe steer)."""
from __future__ import annotations

from typing import Any

import numpy as np

from ccer.mechanism.controls import build_control_specs
from ccer.mechanism.interchange import InterchangeSpec, build_steering_spec
from ccer.mechanism.line_b_protocol import resolve_wrong_owner_donor_vec
from ccer.mechanism.line_e_donor import build_paired_cross_bundle
from ccer.mechanism.line_e_v5_protocol import LineEV5Config, line_e_v5_config
from ccer.mechanism.owner_pca import fit_owner_pca
from ccer.mechanism.pair_select import cem_valid_pair_ids
from ccer.mechanism.steering_vector import paired_activation_vectors


def build_v5_pca_roi(
    cfg: LineEV5Config | None = None,
    *,
    track: str = "CEM",
    pool: str = "mechanism_research",
) -> dict[str, Any]:
    cfg = cfg or line_e_v5_config()
    fit = fit_owner_pca(
        layer=cfg.layer,
        position=cfg.position,
        rank=cfg.pca_rank,
        track=track,
        pool=pool,
        require_line_d_v3=True,
        cross_only=True,
    )
    if fit.get("error"):
        return {"error": fit["error"], **{k: fit.get(k) for k in ("n_vectors", "rank")}}
    u = fit["U_owner"]
    return {
        "primary_roi": {"layer": cfg.layer, "position": cfg.position},
        "U_owner": np.asarray(u, dtype=np.float32).tolist(),
        "inject_site": "residual",
        "steering_apply": cfg.steering_apply,
        "pca_fit": {
            "rank": fit.get("rank"),
            "n_vectors": fit.get("n_vectors"),
            "pc_variance_explained": fit.get("pc_variance_explained"),
        },
    }


def _cross_pairs() -> list[dict[str, Any]]:
    return cem_valid_pair_ids(include_matched_clean=True)["pm_clean_cross_pairs"]


def build_v5_interchange_spec(
    *,
    pm_trajectory_id: str,
    control_label: str,
    control_id: str,
    mode: str,
    alpha: float,
    roi: dict[str, Any],
    position_token_idx: int,
    cfg: LineEV5Config | None = None,
) -> dict[str, Any]:
    """Return InterchangeSpec or error dict."""
    cfg = cfg or line_e_v5_config()
    paired = build_paired_cross_bundle(pm_trajectory_id, layer=cfg.layer, position=cfg.position)
    if paired.get("error"):
        return paired
    pm_vec = paired["pm_vec"]
    clean_vec = paired["clean_vec"]
    clean_tid = str(paired["clean_trajectory_id"])
    wrong_impl: str | None = None

    if control_id == "wrong_owner_donor":
        wrong_vec, wrong_impl = resolve_wrong_owner_donor_vec(
            pm_tid=pm_trajectory_id,
            paired_clean_tid=clean_tid,
            cross_pairs=_cross_pairs(),
            layer=cfg.layer,
            position=cfg.position,
        )
        if wrong_vec is not None and wrong_impl != "orthogonal_complement_fallback":
            spec = build_control_specs(
                control_id="target_interchange",
                roi=roi,
                donor_vec=wrong_vec,
                recipient_vec=pm_vec,
                position_token_idx=position_token_idx,
                alpha=alpha,
            )
        else:
            spec = build_control_specs(
                control_id="orthogonal_complement",
                roi=roi,
                donor_vec=clean_vec,
                recipient_vec=pm_vec,
                position_token_idx=position_token_idx,
                alpha=alpha,
            )
            wrong_impl = "orthogonal_complement"
    else:
        spec = build_control_specs(
            control_id=control_id,
            roi=roi,
            donor_vec=clean_vec,
            recipient_vec=pm_vec,
            position_token_idx=position_token_idx,
            alpha=alpha,
        )

    if mode == "full_vector":
        spec.mode = "full_vector"
    spec.control_id = control_label
    spec.inject_site = str(roi.get("inject_site") or "residual")
    spec.steering_apply = cfg.steering_apply
    return {
        "spec": spec,
        "pm_vec": pm_vec,
        "clean_vec": clean_vec,
        "clean_trajectory_id": clean_tid,
        "wrong_owner_impl": wrong_impl,
        "paired_norm": paired.get("paired_norm"),
    }


def build_v5_probe_steer_spec(
    *,
    probe_direction: np.ndarray,
    position_token_idx: int,
    alpha: float,
    cfg: LineEV5Config | None = None,
) -> InterchangeSpec:
    cfg = cfg or line_e_v5_config()
    return build_steering_spec(
        control_id="probe_steer",
        layer=cfg.layer,
        position_token_idx=position_token_idx,
        steering_vec=probe_direction,
        alpha=alpha,
        steering_direction="subtract",
        steering_apply=cfg.steering_apply,
    )


def trajectory_paired_vectors(
    pm_trajectory_id: str,
    *,
    layer: int,
    position: str,
) -> dict[str, Any]:
    return paired_activation_vectors(pm_trajectory_id, layer=layer, position=position)
