"""AGR significance: cluster bootstrap AUROC CI and DeLong paired test."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ccer.mechanism.bind_surprise import detection_metrics


def _auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _cluster_indices(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        cid = str(r.get("trajectory_id") or i)
        out.setdefault(cid, []).append(i)
    return out


def cluster_bootstrap_auroc(
    rows: list[dict[str, Any]],
    score_key: str,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Trajectory-cluster bootstrap CI for a single AUROC."""
    y = np.array([int(r["y_pm"]) for r in rows], dtype=int)
    scores = np.array([float(r[score_key]) for r in rows], dtype=float)
    point = _auroc(y, scores)
    clusters = _cluster_indices(rows)
    cluster_ids = sorted(clusters)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx: list[int] = []
        for cid in sampled:
            idx.extend(clusters[cid])
        if len(idx) < 5 or len(np.unique(y[idx])) < 2:
            continue
        boots.append(_auroc(y[idx], scores[idx]))
    if not boots:
        return {"auroc": point, "ci_low": None, "ci_high": None, "n_boot_effective": 0}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "auroc": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot_effective": len(boots),
        "n_boot_requested": n_boot,
    }


def cluster_bootstrap_ap(
    rows: list[dict[str, Any]],
    score_key: str,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Trajectory-cluster bootstrap CI for average precision (claim-level)."""
    y = np.array([int(r["y_pm"]) for r in rows], dtype=int)
    scores = np.array([float(r[score_key]) for r in rows], dtype=float)
    if y.sum() == 0:
        return {"ap": None, "ci_low": None, "ci_high": None, "n_boot_effective": 0}
    point = float(average_precision_score(y, scores))
    clusters = _cluster_indices(rows)
    cluster_ids = sorted(clusters)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx: list[int] = []
        for cid in sampled:
            idx.extend(clusters[cid])
        if len(idx) < 5 or np.unique(y[idx]).size < 2:
            continue
        boots.append(float(average_precision_score(y[idx], scores[idx])))
    if not boots:
        return {"ap": point, "ci_low": None, "ci_high": None, "n_boot_effective": 0}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "ap": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot_effective": len(boots),
        "n_boot_requested": n_boot,
        "method": "cluster_bootstrap_trajectory",
    }


def cluster_bootstrap_detection_at_threshold(
    rows: list[dict[str, Any]],
    score_key: str,
    threshold: float,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap CI for precision / recall / F1 at a fixed threshold (trajectory-cluster)."""
    point = detection_metrics(rows, score_key=score_key, threshold=float(threshold))
    clusters = _cluster_indices(rows)
    cluster_ids = sorted(clusters)
    rng = np.random.default_rng(seed)
    precs: list[float] = []
    recs: list[float] = []
    f1s: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx: list[int] = []
        for cid in sampled:
            idx.extend(clusters[cid])
        if len(idx) < 5:
            continue
        sub = [rows[i] for i in idx]
        m = detection_metrics(sub, score_key=score_key, threshold=float(threshold))
        precs.append(m["precision"])
        recs.append(m["recall"])
        f1s.append(m["f1"])
    if not precs:
        return {**point, "precision_ci": [None, None], "recall_ci": [None, None], "f1_ci": [None, None], "n_boot_effective": 0}
    return {
        **point,
        "precision_ci": [float(np.percentile(precs, 2.5)), float(np.percentile(precs, 97.5))],
        "recall_ci": [float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5))],
        "f1_ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "n_boot_effective": len(precs),
        "n_boot_requested": n_boot,
        "method": "cluster_bootstrap_trajectory",
    }


def cluster_bootstrap_auroc_difference(
    rows: list[dict[str, Any]],
    score_key_a: str,
    score_key_b: str,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Trajectory-cluster bootstrap CI for AUROC(A) − AUROC(B) on the same claims."""
    y = np.array([int(r["y_pm"]) for r in rows], dtype=int)
    sa = np.array([float(r[score_key_a]) for r in rows], dtype=float)
    sb = np.array([float(r[score_key_b]) for r in rows], dtype=float)
    point_a = _auroc(y, sa)
    point_b = _auroc(y, sb)
    point_diff = point_a - point_b
    clusters = _cluster_indices(rows)
    cluster_ids = sorted(clusters)
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx: list[int] = []
        for cid in sampled:
            idx.extend(clusters[cid])
        if len(idx) < 5 or len(np.unique(y[idx])) < 2:
            continue
        diffs.append(_auroc(y[idx], sa[idx]) - _auroc(y[idx], sb[idx]))
    if not diffs:
        return {
            "auroc_a": point_a,
            "auroc_b": point_b,
            "diff": point_diff,
            "diff_ci_low": None,
            "diff_ci_high": None,
            "significant_at_05": None,
            "n_boot_effective": 0,
        }
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "auroc_a": point_a,
        "auroc_b": point_b,
        "diff": point_diff,
        "diff_ci_low": float(lo),
        "diff_ci_high": float(hi),
        "significant_at_05": bool(lo > 0 or hi < 0),
        "n_boot_effective": len(diffs),
        "n_boot_requested": n_boot,
        "method": "cluster_bootstrap_trajectory",
    }


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n, dtype=float)
    out[sorted_idx] = midranks + 1
    return out


def _structural_components(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y == 1]
    neg = scores[y == 0]
    m = len(pos)
    n = len(neg)
    if m == 0 or n == 0:
        return np.full(len(y), np.nan)
    tx = _compute_midrank(pos)
    ty = _compute_midrank(neg)
    tz = _compute_midrank(scores)
    v_pos = (tz[y == 1] - tx) / n
    v_neg = 1.0 - (tz[y == 0] - ty) / m
    out = np.empty(len(y), dtype=float)
    out[y == 1] = v_pos
    out[y == 0] = v_neg
    return out


def delong_test(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict[str, Any]:
    """
    DeLong test for correlated ROC curves (same labels, two score vectors).

    Returns AUROC for each scorer and two-sided p-value for H0: AUC_a = AUC_b.
    """
    y = np.asarray(y, dtype=int)
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)
    if len(np.unique(y)) < 2:
        return {"auroc_a": None, "auroc_b": None, "pvalue": None, "method": "delong"}
    auc_a = _auroc(y, sa)
    auc_b = _auroc(y, sb)
    v_a = _structural_components(y, sa)
    v_b = _structural_components(y, sb)
    if not np.all(np.isfinite(v_a)) or not np.all(np.isfinite(v_b)):
        return {"auroc_a": auc_a, "auroc_b": auc_b, "pvalue": None, "method": "delong"}
    diff = v_a - v_b
    var_diff = float(np.var(diff, ddof=1) / len(diff))
    if var_diff <= 0:
        return {
            "auroc_a": auc_a,
            "auroc_b": auc_b,
            "diff": auc_a - auc_b,
            "pvalue": 1.0,
            "method": "delong",
        }
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    from scipy.stats import norm

    p = float(2 * norm.sf(abs(z)))
    return {
        "auroc_a": auc_a,
        "auroc_b": auc_b,
        "diff": auc_a - auc_b,
        "z": float(z),
        "pvalue": p,
        "method": "delong",
    }


def compare_scores_on_rows(
    rows: list[dict[str, Any]],
    score_key_a: str,
    score_key_b: str,
    *,
    label_a: str,
    label_b: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    y = np.array([int(r["y_pm"]) for r in rows], dtype=int)
    sa = np.array([float(r[score_key_a]) for r in rows], dtype=float)
    sb = np.array([float(r[score_key_b]) for r in rows], dtype=float)
    boot = cluster_bootstrap_auroc_difference(
        rows, score_key_a, score_key_b, n_boot=n_boot, seed=seed
    )
    delong = delong_test(y, sa, sb)
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n": len(rows),
        "n_pos": int(y.sum()),
        "bootstrap_diff": boot,
        "delong": delong,
        "interpretation_note": (
            "CI excludes 0 or p<0.05 → statistically significant difference; "
            "overlapping CI and p≥0.05 → no evidence of difference"
        ),
    }
