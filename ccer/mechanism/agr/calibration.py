"""AGR Track C: ECE and affine recalibration (Eq.3/4)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from ccer.mechanism.bind_surprise import detection_metrics, try_auroc


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def expected_calibration_error(
    y: np.ndarray,
    probs: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "equal_width",
) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    if strategy == "equal_mass":
        order = np.argsort(p)
        bins = np.array_split(order, n_bins)
        bin_defs = [(p[idx].min(), p[idx].max(), idx) for idx in bins if len(idx)]
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_defs = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            idx = np.where((p >= lo) & (p < hi if i < n_bins - 1 else p <= hi))[0]
            if len(idx):
                bin_defs.append((lo, hi, idx))

    ece = 0.0
    diagram: list[dict[str, Any]] = []
    n = len(y)
    for lo, hi, idx in bin_defs:
        acc = float(y[idx].mean())
        conf = float(p[idx].mean())
        w = len(idx) / n
        ece += w * abs(acc - conf)
        diagram.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": int(len(idx)),
                "mean_pred": conf,
                "emp_freq": acc,
            }
        )
    return {"ece": float(ece), "n_bins": n_bins, "strategy": strategy, "reliability": diagram}


def fit_affine_calibration(
    z: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Eq.4: z' = a*z + c via logistic regression on scalar z."""
    z = np.asarray(z, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    clf = LogisticRegression(max_iter=4000, solver="lbfgs")
    clf.fit(z, y)
    a = float(clf.coef_[0, 0])
    c = float(clf.intercept_[0])
    return {"a": a, "c": c}


def apply_affine(z: np.ndarray, params: dict[str, float]) -> np.ndarray:
    return params["a"] * np.asarray(z, dtype=np.float64) + params["c"]


def fit_threshold_f1(rows: list[dict[str, Any]], score_key: str) -> float:
    """Dev F1-optimal threshold for arbitrary score column."""
    if not rows:
        return 0.0
    scores = np.array([float(r.get(score_key) or 0.0) for r in rows], dtype=np.float64)
    labels = np.array([int(r.get("y_pm") or 0) for r in rows], dtype=np.int32)
    candidates = np.unique(scores)
    if len(candidates) > 400:
        candidates = np.quantile(scores, np.linspace(0.02, 0.98, 200))
    best_tau = float(candidates[0])
    best_f1 = -1.0
    for tau in candidates:
        pred = scores >= float(tau)
        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        fn = int(np.sum((~pred) & (labels == 1)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return best_tau


def calibration_report(
    rows: list[dict[str, Any]],
    *,
    z_key: str,
    split: str,
) -> dict[str, Any]:
    sub = [r for r in rows if r.get("agr_split") == split]
    if z_key == "z2":
        sub = [r for r in sub if r.get("z2_available") and math.isfinite(float(r.get("z2") or float("nan")))]
    y = np.array([int(r["y_pm"]) for r in sub], dtype=int)
    z = np.array([float(r[z_key]) for r in sub], dtype=float)
    p = sigmoid(z)
    before = {
        "equal_width": expected_calibration_error(y, p, strategy="equal_width"),
        "equal_mass": expected_calibration_error(y, p, strategy="equal_mass"),
    }
    aff = fit_affine_calibration(z, y)
    z_cal = apply_affine(z, aff)
    p_after = sigmoid(z_cal)
    after = {
        "equal_width": expected_calibration_error(y, p_after, strategy="equal_width"),
        "equal_mass": expected_calibration_error(y, p_after, strategy="equal_mass"),
    }
    delta_proxy = float(np.mean(z) - np.log((y.mean() + 1e-9) / (1 - y.mean() + 1e-9)))
    return {
        "z_key": z_key,
        "split": split,
        "n": len(sub),
        "affine": aff,
        "delta_proxy": delta_proxy,
        "ece_before": before,
        "ece_after": after,
        "ece_primary_before": max(before["equal_width"]["ece"], before["equal_mass"]["ece"]),
        "ece_primary_after": max(after["equal_width"]["ece"], after["equal_mass"]["ece"]),
        "auroc": try_auroc(sub, z_key),
    }
