"""Control variants for interchange interventions (P3b)."""
from __future__ import annotations

from typing import Any

import numpy as np

from ccer.mechanism.interchange import InterchangeSpec, build_spec_from_roi


def random_orthogonal_basis(u: np.ndarray, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = u.shape[1] if u.ndim == 2 and u.shape[1] else 4
    d = u.shape[0] if u.ndim == 2 and u.shape[0] else 5120
    a = rng.standard_normal((d, r))
    q, _ = np.linalg.qr(a)
    return q[:, :r].astype(np.float32)


def build_control_specs(
    *,
    control_id: str,
    roi: dict[str, Any],
    donor_vec: np.ndarray,
    recipient_vec: np.ndarray,
    position_token_idx: int,
    alpha: float = 1.0,
    wrong_donor_vec: np.ndarray | None = None,
    seed: int = 0,
) -> InterchangeSpec:
    primary = roi.get("primary_roi") or {}
    layer = int(primary.get("layer") or 0)
    u = np.array(roi.get("U_owner") or [], dtype=np.float32)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    rank = u.shape[1] if u.size else 4

    if control_id == "no_intervention":
        return build_spec_from_roi(
            control_id=control_id,
            roi=roi,
            donor_vec=donor_vec,
            recipient_vec=recipient_vec,
            position_token_idx=position_token_idx,
            alpha=0.0,
        )
    if control_id == "self_donor":
        return build_spec_from_roi(
            control_id=control_id,
            roi=roi,
            donor_vec=recipient_vec,
            recipient_vec=recipient_vec,
            position_token_idx=position_token_idx,
            alpha=alpha,
        )
    if control_id == "random_same_rank":
        ur = random_orthogonal_basis(u if u.size else np.eye(len(donor_vec), rank), seed=seed)
        return build_spec_from_roi(
            control_id=control_id,
            roi=roi,
            donor_vec=donor_vec,
            recipient_vec=recipient_vec,
            position_token_idx=position_token_idx,
            alpha=alpha,
            U_override=ur,
        )
    if control_id == "orthogonal_complement":
        if u.size:
            d = u.shape[0]
            comp = np.eye(d, dtype=np.float32) - u @ u.T
            q, _ = np.linalg.qr(comp + 1e-6 * np.eye(d))
            uc = q[:, :rank]
        else:
            uc = random_orthogonal_basis(np.eye(len(donor_vec), rank), seed=seed + 1)
        return build_spec_from_roi(
            control_id=control_id,
            roi=roi,
            donor_vec=donor_vec,
            recipient_vec=recipient_vec,
            position_token_idx=position_token_idx,
            alpha=alpha,
            U_override=uc,
        )
    if control_id == "wrong_owner_donor":
        w = wrong_donor_vec if wrong_donor_vec is not None else donor_vec * -1.0
        return build_spec_from_roi(
            control_id=control_id,
            roi=roi,
            donor_vec=w,
            recipient_vec=recipient_vec,
            position_token_idx=position_token_idx,
            alpha=alpha,
        )
    if control_id == "unrelated_layer":
        n_layers = max(layer + 2, 8)
        alt = min(n_layers - 1, layer + max(1, n_layers // 8))
        if alt == layer:
            alt = max(0, layer - 1)
        return build_spec_from_roi(
            control_id=control_id,
            roi=roi,
            donor_vec=donor_vec,
            recipient_vec=recipient_vec,
            position_token_idx=position_token_idx,
            alpha=alpha,
            layer_override=alt,
        )
    if control_id == "path_ablation":
        return build_spec_from_roi(
            control_id=control_id,
            roi=roi,
            donor_vec=donor_vec,
            recipient_vec=recipient_vec,
            position_token_idx=position_token_idx,
            alpha=alpha,
            mode="ablation",
        )
    if control_id == "path_rescue":
        return build_spec_from_roi(
            control_id=control_id,
            roi=roi,
            donor_vec=donor_vec,
            recipient_vec=recipient_vec,
            position_token_idx=position_token_idx,
            alpha=alpha,
            mode="rescue",
        )
    return build_spec_from_roi(
        control_id="target_interchange",
        roi=roi,
        donor_vec=donor_vec,
        recipient_vec=recipient_vec,
        position_token_idx=position_token_idx,
        alpha=alpha,
    )


CONTROL_IDS = [
    "no_intervention",
    "self_donor",
    "target_interchange",
    "random_same_rank",
    "orthogonal_complement",
    "wrong_owner_donor",
    "unrelated_layer",
    "path_ablation",
    "path_rescue",
]

ALPHA_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]
