"""AGR Track P: probe Eq.1 validation."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from ccer.mechanism.agr.signals import (
    ASF_VERDICT,
    PM_TYPES,
    AgrProbeModel,
    build_agr_signal_rows,
    train_agr_bow,
    train_agr_probe,
)
from ccer.mechanism.supervised_probe import _auroc, normalize_quote_for_bow
from ccer.mechanism.agr.calibration import fit_threshold_f1
from ccer.mechanism.bind_surprise import detection_metrics


def gaussian_audit(probe: AgrProbeModel, rows: list[dict[str, Any]]) -> dict[str, Any]:
    dp = [r for r in rows if r.get("agr_split") == "D_p"]
    h_pm: list[float] = []
    h_cl: list[float] = []
    w = probe.w_p
    for r in dp:
        # use stored z1 as projection proxy when h not stored — recompute from z1 only 1d
        proj = float(r["z1"])
        if int(r["y_pm"]):
            h_pm.append(proj)
        else:
            h_cl.append(proj)
    sh_pm = stats.shapiro(h_pm) if len(h_pm) >= 3 else None
    sh_cl = stats.shapiro(h_cl) if len(h_cl) >= 3 else None
    lev = stats.levene(h_pm, h_cl) if len(h_pm) >= 2 and len(h_cl) >= 2 else None
    return {
        "n_D_p": len(dp),
        "shapiro_pm": {"statistic": float(sh_pm.statistic), "pvalue": float(sh_pm.pvalue)} if sh_pm else None,
        "shapiro_clean": {"statistic": float(sh_cl.statistic), "pvalue": float(sh_cl.pvalue)} if sh_cl else None,
        "levene_proj": {"statistic": float(lev.statistic), "pvalue": float(lev.pvalue)} if lev else None,
        "var_pm": float(np.var(h_pm)),
        "var_clean": float(np.var(h_cl)),
        "note": "Projection along fitted probe direction (z1 on D_p)",
    }


def subtype_metrics(rows: list[dict[str, Any]], *, split: str = "test") -> dict[str, Any]:
    sub = [r for r in rows if r.get("agr_split") == split]
    out: dict[str, Any] = {"split": split}
    subtype_verdict = {
        "CEM": "cross_object_merge",
        "CAP": "constraint_projection",
        "ASF": ASF_VERDICT,
    }
    dc = [r for r in rows if r.get("agr_split") == "D_c"]
    tau = fit_threshold_f1(dc, "z1")
    y_all = np.array([int(r["y_pm"]) for r in sub], dtype=int)
    scores_all = np.array([float(r["z1"]) for r in sub], dtype=float)
    m_all = detection_metrics(sub, score_key="z1", threshold=tau)
    out["overall"] = {
        "n": len(sub),
        "auroc": _auroc(y_all, scores_all),
        "f1": m_all["f1"],
        "precision": m_all["precision"],
        "recall": m_all["recall"],
        "tau_z1": tau,
    }
    for name, verdict in subtype_verdict.items():
        y_sub = np.array([1 if r.get("gold_verdict") == verdict else 0 for r in sub], dtype=int)
        n_pos = int(y_sub.sum())
        if n_pos < 3 or n_pos == len(sub):
            out[name] = {"n_pos": n_pos, "n": len(sub), "skipped": True}
            continue
        out[name] = {
            "n_pos": n_pos,
            "n": len(sub),
            "auroc_ovr": _auroc(y_sub, scores_all),
            "note": f"one-vs-rest for {verdict}",
        }
    return out


def bow_control(rows: list[dict[str, Any]], *, split: str = "test") -> dict[str, Any]:
    vec, clf = train_agr_bow(train_split="D_p")
    sub = [r for r in rows if r.get("agr_split") == split]
    texts = [normalize_quote_for_bow(str(r.get("response_quote") or "")) for r in sub]
    probs = clf.predict_proba(vec.transform(texts))[:, 1]
    y = np.array([int(r["y_pm"]) for r in sub], dtype=int)
    dc = [r for r in rows if r.get("agr_split") == "D_c"]
    dc_texts = [normalize_quote_for_bow(str(r.get("response_quote") or "")) for r in dc]
    dc_probs = clf.predict_proba(vec.transform(dc_texts))[:, 1]
    bow_dc = [{"y_pm": r["y_pm"], "p_bow": float(p)} for r, p in zip(dc, dc_probs)]
    tau = fit_threshold_f1(bow_dc, "p_bow")
    flagged = probs >= tau
    tp = int(((flagged) & (y == 1)).sum())
    fp = int(((flagged) & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "split": split,
        "auroc": _auroc(y, probs),
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "tau_bow": tau,
    }


def run_track_p(rows: list[dict[str, Any]], probe: AgrProbeModel) -> dict[str, Any]:
    return {
        "gaussian_audit": gaussian_audit(probe, rows),
        "discrimination": subtype_metrics(rows, split="test"),
        "bow_control": bow_control(rows, split="test"),
        "probe_weights": {"w_p_norm": float(np.linalg.norm(probe.w_p)), "b_p": probe.b_p},
    }
