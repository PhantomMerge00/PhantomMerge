"""Line A–aligned probe direction for Line E v5 hidden steering."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.steering_vector import (
    mechanism_research_clean_for_build,
    mechanism_research_pm_with_activations,
)


def _activation_vector(trajectory_id: str, *, layer: int, position: str) -> np.ndarray | None:
    path = activation_path(trajectory_id, "original")
    if not path.is_file():
        return None
    vec = get_vector(load_activation_npz(path), position=position, layer=layer)
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def fit_probe_steering_direction(
    *,
    layer: int,
    position: str,
    pool: str = "mechanism_research",
    seed: int = 42,
    min_samples: int = 12,
) -> dict[str, Any]:
    """
    Fit logistic probe on mechanism_research activations; return unit direction
    (positive coef → PM). Line E subtract steering moves opposite (toward clean).
    """
    if pool != "mechanism_research":
        return {"error": "unsupported_pool", "pool": pool}

    pm_ids = mechanism_research_pm_with_activations(layer=layer, position=position)
    clean_ids = mechanism_research_clean_for_build(pm_ids, layer=layer, position=position)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for tid in pm_ids:
        v = _activation_vector(tid, layer=layer, position=position)
        if v is not None:
            xs.append(v)
            ys.append(1)
    for tid in clean_ids:
        v = _activation_vector(tid, layer=layer, position=position)
        if v is not None:
            xs.append(v)
            ys.append(0)

    if len(xs) < min_samples or len(set(ys)) < 2:
        return {
            "error": "insufficient_probe_data",
            "n_pm": sum(ys),
            "n_clean": len(ys) - sum(ys),
            "n_total": len(xs),
        }

    x = np.stack(xs, axis=0)
    y = np.array(ys, dtype=int)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs", random_state=seed)
    clf.fit(x_scaled, y)
    raw = clf.coef_.reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(raw))
    if norm < 1e-8:
        return {"error": "degenerate_probe_coef", "n_total": len(xs)}
    direction = raw / norm
    return {
        "direction": direction,
        "norm": norm,
        "n_pm": int(y.sum()),
        "n_clean": int(len(y) - y.sum()),
        "layer": layer,
        "position": position,
        "pool": pool,
        "source": "line_a_logistic_probe_mechanism_research",
    }
