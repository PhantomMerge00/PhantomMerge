"""AGR Track F: fusion, ablations, complementarity (Eq.5/8/9)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from ccer.mechanism.agr.calibration import apply_affine, fit_affine_calibration, sigmoid
from ccer.mechanism.agr.calibration import fit_threshold_f1
from ccer.mechanism.bind_surprise import detection_metrics, try_auroc
from ccer.mechanism.agr.calibration import expected_calibration_error


def _y(rows: list[dict]) -> np.ndarray:
    return np.array([int(r["y_pm"]) for r in rows], dtype=int)


def _clusters(rows: list[dict]) -> np.ndarray:
    return np.array([str(r["trajectory_id"]) for r in rows], dtype=str)


def _z2_available(r: dict[str, Any]) -> bool:
    return bool(r.get("z2_available", True))


def _rep_score(r: dict[str, Any], *, calibrated: bool) -> float:
    if calibrated:
        return float(r.get("z1_prime") if r.get("z1_prime") is not None else r["z1"])
    return float(r["z1"])


def fit_fusion_calibrations(
    rows: list[dict[str, Any]],
    *,
    cal_split: str = "D_c",
) -> dict[str, Any]:
    sub = [r for r in rows if r.get("agr_split") == cal_split]
    y = _y(sub)
    z1 = np.array([float(r["z1"]) for r in sub])
    z2_sub = [r for r in sub if _z2_available(r)]
    z2_cal = (
        fit_affine_calibration(np.array([float(r["z2"]) for r in z2_sub]), _y(z2_sub))
        if z2_sub
        else {"a": 1.0, "c": 0.0}
    )
    return {
        "z1": fit_affine_calibration(z1, y),
        "z2": z2_cal,
        "cal_split": cal_split,
        "z2_cal_n": len(z2_sub),
    }


def enrich_calibrated(rows: list[dict[str, Any]], cal: dict[str, Any]) -> None:
    for r in rows:
        r["z1_prime"] = float(apply_affine(np.array([r["z1"]]), cal["z1"])[0])
        if _z2_available(r) and math.isfinite(float(r.get("z2") or float("nan"))):
            r["z2_prime"] = float(apply_affine(np.array([r["z2"]]), cal["z2"])[0])
        else:
            r["z2_prime"] = float("nan")


def fit_mle_fusion(
    rows: list[dict[str, Any]],
    *,
    fit_split: str = "D_f",
    use_calibrated: bool = True,
    interaction: bool = False,
) -> dict[str, Any]:
    sub = [r for r in rows if r.get("agr_split") == fit_split and _z2_available(r)]
    if not sub:
        return {
            "w1": 1.0,
            "w2": 0.0,
            "b": 0.0,
            "interaction": interaction,
            "use_calibrated": use_calibrated,
            "fit_split": fit_split,
            "n_fit": 0,
        }
    y = _y(sub)
    k1 = "z1_prime" if use_calibrated else "z1"
    k2 = "z2_prime" if use_calibrated else "z2"
    z1 = np.array([float(r[k1]) for r in sub]).reshape(-1, 1)
    z2 = np.array([float(r[k2]) for r in sub]).reshape(-1, 1)
    if interaction:
        X = np.hstack([z1, z2, z1 * z2])
    else:
        X = np.hstack([z1, z2])
    clf = LogisticRegression(max_iter=4000, solver="lbfgs")
    clf.fit(X, y)
    out: dict[str, Any] = {
        "w1": float(clf.coef_[0, 0]),
        "w2": float(clf.coef_[0, 1]),
        "b": float(clf.intercept_[0]),
        "interaction": interaction,
        "use_calibrated": use_calibrated,
        "fit_split": fit_split,
    }
    if interaction:
        out["w3"] = float(clf.coef_[0, 2])
    return out


def fusion_score(
    r: dict[str, Any],
    *,
    mode: str,
    cal: dict[str, Any] | None = None,
    weights: dict[str, Any] | None = None,
) -> float:
    calibrated = cal is not None
    has_z2 = _z2_available(r)
    z1 = _rep_score(r, calibrated=calibrated)
    if mode == "probe_only":
        return z1
    if not has_z2:
        # Paper: missing anchor readout → representational evidence alone.
        if mode == "causal_only":
            return float("nan")
        if mode in ("calibrated_equal", "full_fusion"):
            return z1
        if mode == "naive_sum":
            return float(r["z1"])
        if mode == "bind_surprise_equal":
            return float(r["z1"])
        return z1
    z2 = float(r.get("z2_prime" if calibrated else "z2") or r["z2"])
    if mode == "causal_only":
        return z2
    if mode == "naive_sum":
        z2_legacy = float(r.get("z2_log_mass") or r["z2"])
        return float(r["z1"]) + z2_legacy
    if mode == "calibrated_equal":
        return float(r["z1_prime"]) + float(r["z2_prime"])
    if mode == "full_fusion":
        w = weights or {"w1": 1.0, "w2": 1.0, "b": 0.0}
        return w["w1"] * float(r["z1_prime"]) + w["w2"] * float(r["z2_prime"]) + w.get("b", 0.0)
    if mode == "bind_surprise_equal":
        z2_legacy = float(r.get("z2_log_mass") or r["z2"])
        return float(r["z1"]) + z2_legacy
    raise ValueError(mode)


def eval_mode(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    cal: dict[str, Any] | None = None,
    weights: dict[str, Any] | None = None,
    threshold: float | None = None,
    all_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pool = all_rows if all_rows is not None else rows
    for r in pool:
        r["_fusion_score"] = float(fusion_score(r, mode=mode, cal=cal, weights=weights))
    scores = np.array([float(r["_fusion_score"]) for r in rows], dtype=float)
    y = _y(rows)
    if threshold is None:
        fit_rows = [r for r in pool if r.get("agr_split") == "D_f"]
        thr = fit_threshold_f1(fit_rows, "_fusion_score")
    else:
        thr = threshold
    m = detection_metrics(rows, score_key="_fusion_score", threshold=float(thr))
    probs = sigmoid(scores)
    ece = expected_calibration_error(y, probs, strategy="equal_width")
    return {
        "mode": mode,
        "threshold": float(thr),
        "auroc": try_auroc(rows, "_fusion_score"),
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "ece_equal_width": ece["ece"],
        **m,
    }


def complementarity_2x2(
    rows: list[dict[str, Any]],
    *,
    tau1: float,
    tau2: float,
    use_prime: bool = True,
) -> dict[str, Any]:
    k1 = "z1_prime" if use_prime else "z1"
    k2 = "z2_prime" if use_prime else "z2"
    cells = {"z1_ok_z2_ok": 0, "z1_ok_z2_wrong": 0, "z1_wrong_z2_ok": 0, "z1_wrong_z2_wrong": 0}
    for r in rows:
        if not _z2_available(r):
            continue
        y = int(r["y_pm"])
        p1 = float(r[k1]) >= tau1
        p2 = float(r[k2]) >= tau2
        ok1 = p1 == bool(y)
        ok2 = p2 == bool(y)
        if ok1 and ok2:
            cells["z1_ok_z2_ok"] += 1
        elif ok1 and not ok2:
            cells["z1_ok_z2_wrong"] += 1
        elif not ok1 and ok2:
            cells["z1_wrong_z2_ok"] += 1
        else:
            cells["z1_wrong_z2_wrong"] += 1
    return cells


def residual_correlation(
    rows: list[dict[str, Any]],
    *,
    use_prime: bool = True,
    rho_proxy: str = "binary_label",
) -> dict[str, float]:
    """
    Correlation of signal residuals ε_i' = z_i' − ρ*(c).

    ``binary_label`` uses ±log((y+ε)/(1−y+ε)) as ρ* proxy (systematically inflates r).
    Prefer ``oof_logistic`` via :func:`residual_correlation_oof`.
    """
    k1 = "z1_prime" if use_prime else "z1"
    k2 = "z2_prime" if use_prime else "z2"
    y = _y(rows)
    log_odds = np.log((y + EPS) / (1 - y + EPS))
    e1 = np.array([float(r[k1]) for r in rows]) - log_odds
    e2 = np.array([float(r[k2]) for r in rows]) - log_odds
    r = float(np.corrcoef(e1, e2)[0, 1]) if len(y) > 2 else float("nan")
    return {
        "residual_corr_z1_z2": r,
        "n": len(rows),
        "rho_proxy": rho_proxy,
        "warning": "binary_label proxy inflates correlation; do not cite in paper",
    }


def _oof_log_odds_matrix(
    sub: list[dict[str, Any]],
    *,
    use_prime: bool,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return z1, z2, log_odds1, log_odds2, n_splits for D_f subset."""
    from sklearn.model_selection import GroupKFold

    k1 = "z1_prime" if use_prime else "z1"
    k2 = "z2_prime" if use_prime else "z2"
    y = _y(sub)
    groups = _clusters(sub)
    z1 = np.array([float(r[k1]) for r in sub]).reshape(-1, 1)
    z2 = np.array([float(r[k2]) for r in sub]).reshape(-1, 1)
    n_unique = len(set(groups.tolist()))
    n_splits = min(n_splits, n_unique)
    if n_splits < 2:
        nan = np.full(len(sub), np.nan)
        return z1, z2, nan, nan, n_splits
    gkf = GroupKFold(n_splits=n_splits)
    oof1 = np.full(len(sub), np.nan)
    oof2 = np.full(len(sub), np.nan)
    for train_idx, test_idx in gkf.split(z1, y, groups):
        for X, oof in ((z1, oof1), (z2, oof2)):
            clf = LogisticRegression(max_iter=4000, solver="lbfgs")
            clf.fit(X[train_idx], y[train_idx])
            oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    log_odds1 = np.log((oof1 + EPS) / (1 - oof1 + EPS))
    log_odds2 = np.log((oof2 + EPS) / (1 - oof2 + EPS))
    return z1, z2, log_odds1, log_odds2, n_splits


def residual_correlation_oof(
    rows: list[dict[str, Any]],
    *,
    fit_split: str = "D_f",
    use_prime: bool = True,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """OOF logistic ρ* proxy per signal (grouped by trajectory_id)."""
    sub = [r for r in rows if r.get("agr_split") == fit_split and _z2_available(r)]
    if len(sub) < 10:
        return {"residual_corr_z1_z2": float("nan"), "n": len(sub), "rho_proxy": "oof_logistic"}
    z1, z2, log_odds1, log_odds2, n_splits = _oof_log_odds_matrix(
        sub, use_prime=use_prime, n_splits=n_splits
    )
    e1 = z1.ravel() - log_odds1
    e2 = z2.ravel() - log_odds2
    mask = np.isfinite(e1) & np.isfinite(e2)
    r = float(np.corrcoef(e1[mask], e2[mask])[0, 1]) if mask.sum() > 2 else float("nan")
    return {
        "residual_corr_z1_z2": r,
        "n": int(mask.sum()),
        "rho_proxy": "oof_logistic",
        "n_splits": n_splits,
        "seed": seed,
    }


def bootstrap_signal_variances_label_proxy(
    rows: list[dict[str, Any]],
    *,
    fit_split: str = "D_f",
    use_prime: bool = True,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Legacy bootstrap using binary label as ρ* (diagnostic only)."""
    sub = [r for r in rows if r.get("agr_split") == fit_split]
    k1 = "z1_prime" if use_prime else "z1"
    k2 = "z2_prime" if use_prime else "z2"
    y = _y(sub)
    log_odds = np.log((y + EPS) / (1 - y + EPS))
    clusters = _clusters(sub)
    cluster_ids = sorted(set(clusters.tolist()))
    c2i: dict[str, list[int]] = {c: [] for c in cluster_ids}
    for i, c in enumerate(clusters.tolist()):
        c2i[c].append(i)
    rng = np.random.default_rng(seed)
    s1: list[float] = []
    s2: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx: list[int] = []
        for c in sampled:
            idx.extend(c2i[c])
        if len(idx) < 5:
            continue
        e1 = np.array([float(sub[i][k1]) for i in idx]) - log_odds[idx]
        e2 = np.array([float(sub[i][k2]) for i in idx]) - log_odds[idx]
        s1.append(float(np.var(e1)))
        s2.append(float(np.var(e2)))
    sigma1 = float(np.mean(s1)) if s1 else float("nan")
    sigma2 = float(np.mean(s2)) if s2 else float("nan")
    w_theory = sigma2 / sigma1 if sigma1 > 0 else float("nan")
    pooled = 1.0 / (1.0 / sigma1 + 1.0 / sigma2) if sigma1 > 0 and sigma2 > 0 else float("nan")
    return {
        "sigma1_sq": sigma1,
        "sigma2_sq": sigma2,
        "w1_over_w2_theory": w_theory,
        "variance_upper_bound_eq9": pooled,
        "n_boot_effective": len(s1),
        "rho_proxy": "binary_label",
        "warning": "do_not_cite",
    }


def bootstrap_signal_variances(
    rows: list[dict[str, Any]],
    *,
    fit_split: str = "D_f",
    use_prime: bool = True,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap σ² using OOF logistic ρ* proxy (trajectory-cluster resampling)."""
    sub = [r for r in rows if r.get("agr_split") == fit_split and _z2_available(r)]
    k1 = "z1_prime" if use_prime else "z1"
    k2 = "z2_prime" if use_prime else "z2"
    z1, z2, log_odds1, log_odds2, _ = _oof_log_odds_matrix(sub, use_prime=use_prime)
    e1_full = z1.ravel() - log_odds1
    e2_full = z2.ravel() - log_odds2
    clusters = _clusters(sub)
    cluster_ids = sorted(set(clusters.tolist()))
    c2i: dict[str, list[int]] = {c: [] for c in cluster_ids}
    for i, c in enumerate(clusters.tolist()):
        c2i[c].append(i)
    rng = np.random.default_rng(seed)
    s1: list[float] = []
    s2: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx: list[int] = []
        for c in sampled:
            idx.extend(c2i[c])
        if len(idx) < 5:
            continue
        e1 = e1_full[idx]
        e2 = e2_full[idx]
        mask = np.isfinite(e1) & np.isfinite(e2)
        if mask.sum() < 5:
            continue
        s1.append(float(np.var(e1[mask])))
        s2.append(float(np.var(e2[mask])))
    sigma1 = float(np.mean(s1)) if s1 else float("nan")
    sigma2 = float(np.mean(s2)) if s2 else float("nan")
    w_theory = sigma2 / sigma1 if sigma1 > 0 else float("nan")
    pooled = 1.0 / (1.0 / sigma1 + 1.0 / sigma2) if sigma1 > 0 and sigma2 > 0 else float("nan")
    return {
        "sigma1_sq": sigma1,
        "sigma2_sq": sigma2,
        "w1_over_w2_theory": w_theory,
        "variance_upper_bound_eq9": pooled,
        "n_boot_effective": len(s1),
        "rho_proxy": "oof_logistic",
    }


EPS = 1e-9


def nonlinear_ceiling(
    rows: list[dict[str, Any]],
    *,
    fit_split: str = "D_f",
    eval_split: str = "test",
) -> dict[str, Any]:
    """Diagnostic upper bound on D_f → test. Uses only (z1', z2'); no auxiliary features."""
    fit = [r for r in rows if r.get("agr_split") == fit_split and _z2_available(r)]
    ev = [r for r in rows if r.get("agr_split") == eval_split and _z2_available(r)]
    if len(fit) < 5 or len(ev) < 5:
        return {"gbm": {"auroc": float("nan")}, "mlp": {"auroc": float("nan")}}
    X_fit = np.array([[r["z1_prime"], r["z2_prime"]] for r in fit])
    y_fit = _y(fit)
    X_ev = np.array([[r["z1_prime"], r["z2_prime"]] for r in ev])
    y_ev = _y(ev)
    models: dict[str, Any] = {}
    for name, mdl in [
        ("gbm", GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=42)),
    ]:
        mdl.fit(X_fit, y_fit)
        if hasattr(mdl, "predict_proba"):
            p = mdl.predict_proba(X_ev)[:, 1]
        else:
            p = mdl.predict(X_ev)
        from ccer.mechanism.supervised_probe import _auroc

        models[name] = {"auroc": _auroc(y_ev, p)}
    return models
