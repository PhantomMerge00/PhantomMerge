"""BindSurprise: coarse binding-risk score B = ℓ_pm + log π_J(T_slot).

Reported alongside AGR ρ(c) as a second, simpler formulation. Empirically stronger
on shopping (AUROC ~0.96 vs AGR ~0.90) despite weaker theoretical grounding.
AGR Method § uses calibrated z_rep + z_fun fusion; BindSurprise is the pragmatic variant.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

import numpy as np
from scipy.special import logsumexp

from ccer.mechanism.value_token_match import value_tokens_match

MechanismQuadrant = Literal["grounded_risk", "ungrounded_alarm", "latent_slot", "clean"]

LOG_MASS_FLOOR = -12.0

# Slot-type lexicon T_s (slot-conditional workspace). Keys are normalized slot_norm.
_SLOT_LEXICON: dict[str, tuple[str, ...]] = {
    "price": ("price", "价格", "php", "usd", "eur", "cost", "价", "peso"),
    "capacity": ("oz", "ml", "liter", "litre", "capacity", "容量", "体积", "volume"),
    "blades": ("blade", "blades", "刀", "刃"),
    "color": ("color", "colour", "颜色", "色"),
    "weight": ("weight", "kg", "lb", "重量", "克", "gram"),
    "rating": ("rating", "star", "评分", "星"),
    "length": ("length", "floor", "长", "inch", "cm"),
    "fit": ("fit", "regular", "slim", "loose"),
    "title": ("title", "name", "名称", "product"),
    "size": ("size", "尺码", "尺寸"),
    "material": ("material", "fabric", "cotton", "polyester", "材质", "面料", "leather", "nylon", "wool", "silk"),
    "brand": ("brand", "品牌"),
    "formulation": ("formulation", "fragrance", "free", "配方", "成分", "ingredient"),
    "interface_port": ("port", "interface", "usb", "hdmi", "接口", "type-c", "connector", "thunderbolt"),
    "design": ("design", "style", "款式", "设计", "pattern"),
    "quantity": ("quantity", "count", "pack", "数量"),
    "description": ("description", "desc", "描述"),
    "compatibility": ("compatible", "compatibility", "兼容"),
    "tip_size": ("tip", "size", "笔尖"),
    "cut_style": ("cut", "style", "切割"),
    "surge_protection": ("surge", "protection", "防雷"),
    "product_configuration": ("configuration", "config", "配置"),
    "shop_id": ("shop", "store", "店铺"),
    # tau3 telecom / airline coarse slots (BindSurprise T_slot)
    "line_id": ("line", "line_id", "l1001", "l1002", "subscriber", "mobile"),
    "line_status": ("line", "status", "active", "suspended", "suspend"),
    "roaming_enabled": ("roaming", "roam", "enabled", "international", "abroad"),
    "data_usage": ("data", "usage", "gb", "limit", "cycle", "billing"),
    "data_used_gb": ("data", "used", "gb", "gigabyte", "limit"),
    "data_refueling": ("data", "refuel", "refueling", "gb", "add"),
    "data_refueling_gb": ("data", "refuel", "gb", "amount"),
    "data_refuel": ("data", "refuel", "gb"),
    "bill_status": ("bill", "status", "due", "overdue", "paid", "draft"),
    "bill_id": ("bill", "invoice", "b1001", "b1002", "payment"),
    "phone_number": ("phone", "number", "555", "mobile", "digits"),
    "customer_id": ("customer", "c1001", "account", "client"),
    "device_status": ("device", "status", "sim", "network"),
    "reservation_id": ("reservation", "booking", "confirm", "pnr"),
    "flight_route": ("flight", "route", "origin", "destination", "airport"),
    "flight_number": ("flight", "number", "airline", "carrier"),
    "user_id": ("user", "passenger", "customer", "member"),
    "order_id": ("order", "order_id", "w", "purchase", "delivery"),
    "order_status": ("order", "status", "delivered", "pending", "shipped"),
    "order_items": ("order", "item", "product", "sku", "quantity"),
    "item_id": ("item", "product", "sku", "variant", "id"),
    "item_name": ("item", "name", "product", "title"),
    "payment_method_id": ("payment", "card", "method", "visa", "mastercard"),
    "item_options": ("option", "size", "color", "variant"),
}


def norm_slot(slot_norm: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(slot_norm or "").strip().lower()).strip("_")


def slot_lexicon(slot_norm: str) -> tuple[str, ...]:
    key = norm_slot(slot_norm)
    if key in _SLOT_LEXICON:
        return _SLOT_LEXICON[key]
    # Fallback: tokenize slot name itself (e.g. "Interface/Port" -> interface, port)
    parts = [p for p in re.split(r"[^a-z0-9]+", key) if len(p) >= 3]
    return tuple(parts) if parts else (key,) if key else ()


def token_in_slot_lexicon(token_text: str, slot_norm: str) -> bool:
    hints = slot_lexicon(slot_norm)
    tok_l = str(token_text or "").lower().strip()
    if not tok_l:
        return False
    for h in hints:
        if h in tok_l or tok_l in h:
            return True
    return False


def probe_logit(p_pm: float, *, eps: float = 1e-9) -> float:
    p = float(min(max(p_pm, eps), 1.0 - eps))
    return math.log(p / (1.0 - p))


def compute_slot_workspace_mass(
    topk: list[dict[str, Any]],
    slot_norm: str,
) -> dict[str, Any]:
    """
    log π_J(slot=s | h) ≈ log ∑_{t ∈ T_s ∩ topk} softmax_topk(t).

    Approximates slot-conditional workspace mass from J-lens readout top-k.
    """
    if not topk:
        return {
            "log_slot_mass": float("-inf"),
            "slot_mass": 0.0,
            "slot_lexicon_hits": [],
            "slot_rank": None,
        }

    logits = np.array([float(r.get("logit") or 0.0) for r in topk], dtype=np.float64)
    log_z = float(logsumexp(logits))

    slot_rows: list[dict[str, Any]] = []
    for row in topk:
        tok = str(row.get("token") or "")
        if token_in_slot_lexicon(tok, slot_norm):
            slot_rows.append(row)

    if not slot_rows:
        return {
            "log_slot_mass": float("-inf"),
            "slot_mass": 0.0,
            "slot_lexicon_hits": [],
            "slot_rank": None,
        }

    slot_logits = np.array([float(r.get("logit") or 0.0) for r in slot_rows], dtype=np.float64)
    log_mass = float(logsumexp(slot_logits) - log_z)
    best_rank = min(int(r.get("rank") or 99) for r in slot_rows)
    return {
        "log_slot_mass": log_mass,
        "slot_mass": float(math.exp(log_mass)),
        "slot_lexicon_hits": [str(r.get("token") or "") for r in slot_rows],
        "slot_rank": best_rank,
    }


def compute_anchor_value_workspace_mass(
    topk: list[dict[str, Any]],
    anchor_value: str,
) -> dict[str, Any]:
    """
    log π_J(s_{a_τ} | h) ≈ log ∑_{t ∈ topk matching anchor_value} softmax_topk(t).

    T_s is the set of top-k tokens whose text matches the anchor evidence value.
    """
    if not topk or not str(anchor_value or "").strip():
        return {
            "log_anchor_value_mass": float("-inf"),
            "anchor_value_mass": 0.0,
            "anchor_value_hits": [],
            "anchor_value_rank": None,
        }

    logits = np.array([float(r.get("logit") or 0.0) for r in topk], dtype=np.float64)
    log_z = float(logsumexp(logits))

    value_rows: list[dict[str, Any]] = []
    for row in topk:
        tok = str(row.get("token") or "")
        if value_tokens_match(tok, anchor_value):
            value_rows.append(row)

    if not value_rows:
        return {
            "log_anchor_value_mass": float("-inf"),
            "anchor_value_mass": 0.0,
            "anchor_value_hits": [],
            "anchor_value_rank": None,
        }

    value_logits = np.array([float(r.get("logit") or 0.0) for r in value_rows], dtype=np.float64)
    log_mass = float(logsumexp(value_logits) - log_z)
    best_rank = min(int(r.get("rank") or 99) for r in value_rows)
    return {
        "log_anchor_value_mass": log_mass,
        "anchor_value_mass": float(math.exp(log_mass)),
        "anchor_value_hits": [str(r.get("token") or "") for r in value_rows],
        "anchor_value_rank": best_rank,
    }


def compute_bind_surprise(
    p_pm: float,
    log_slot_mass: float,
    *,
    log_mass_floor: float = -12.0,
) -> dict[str, Any]:
    """
    BindSurprise B(h,s) = ℓ_pm(h) + log π_J(slot=s | h).

    Unified slot-conditional binding likelihood (log-domain factorization).
    """
    ell_pm = probe_logit(p_pm)
    log_sm = float(log_slot_mass) if math.isfinite(log_slot_mass) else log_mass_floor
    log_sm = max(log_sm, log_mass_floor)
    score = ell_pm + log_sm
    return {
        "probe_logit": ell_pm,
        "log_slot_mass": log_sm,
        "bind_surprise": score,
    }


def classify_mechanism_quadrant(
    p_pm: float,
    tau_probe: float,
    log_slot_mass: float,
    log_slot_threshold: float,
) -> MechanismQuadrant:
    """Interpretable decomposition of BindSurprise factors (not the detector itself)."""
    probe_high = float(p_pm) > float(tau_probe)
    slot_high = float(log_slot_mass) > float(log_slot_threshold)
    if probe_high and slot_high:
        return "grounded_risk"
    if probe_high and not slot_high:
        return "ungrounded_alarm"
    if not probe_high and slot_high:
        return "latent_slot"
    return "clean"


def bind_surprise_rationale(
    quadrant: MechanismQuadrant,
    *,
    slot_norm: str,
    bind_surprise: float,
    probe_logit: float,
    log_slot_mass: float,
    jlens_top5: list[str],
) -> str:
    top_s = ", ".join(jlens_top5[:5]) if jlens_top5 else "(none)"
    if quadrant == "grounded_risk":
        return (
            f"BindSurprise={bind_surprise:.2f}: probe binding risk (ℓ={probe_logit:.2f}) "
            f"with slot-{slot_norm} workspace mass (log π={log_slot_mass:.2f}); "
            f"grounded binding error at claim_onset (top-5: {top_s})."
        )
    if quadrant == "ungrounded_alarm":
        return (
            f"BindSurprise={bind_surprise:.2f}: probe risk (ℓ={probe_logit:.2f}) without "
            f"slot-{slot_norm} workspace support (log π={log_slot_mass:.2f}); "
            f"ungrounded alarm (top-5: {top_s})."
        )
    if quadrant == "latent_slot":
        return (
            f"BindSurprise={bind_surprise:.2f}: slot-{slot_norm} workspace active "
            f"(log π={log_slot_mass:.2f}) below probe threshold (ℓ={probe_logit:.2f})."
        )
    return (
        f"BindSurprise={bind_surprise:.2f}: low probe risk and low slot-{slot_norm} workspace mass."
    )


def fit_tau_b(
    dev_rows: list[dict[str, Any]],
    *,
    metric: str = "f1",
) -> dict[str, Any]:
    """Select τ_B on dev to maximize F1 (frozen for test)."""
    if not dev_rows:
        return {"tau_b": 0.0, "dev_f1": None, "dev_precision": None, "dev_recall": None}

    scores = np.array([float(r["bind_surprise"]) for r in dev_rows], dtype=np.float64)
    labels = np.array([int(r["y_pm"]) for r in dev_rows], dtype=np.int32)
    candidates = np.unique(scores)
    if len(candidates) > 400:
        candidates = np.quantile(scores, np.linspace(0.02, 0.98, 200))

    best_tau = float(candidates[0])
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0

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
            best_prec = prec
            best_rec = rec

    return {
        "tau_b": best_tau,
        "dev_f1": best_f1,
        "dev_precision": best_prec,
        "dev_recall": best_rec,
        "metric": metric,
    }


def fit_log_slot_threshold(dev_rows: list[dict[str, Any]]) -> float:
    """Median log slot mass among dev positives (for quadrant decomposition)."""
    pos = [float(r["log_slot_mass"]) for r in dev_rows if int(r.get("y_pm") or 0) == 1]
    if not pos:
        return -2.0
    return float(np.median(pos))


def detection_metrics(
    rows: list[dict[str, Any]],
    *,
    score_key: str = "bind_surprise",
    threshold: float,
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for r in rows:
        y = int(r.get("y_pm") or 0)
        pred = float(r.get(score_key) or 0.0) >= float(threshold)
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif not pred and y:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "threshold": threshold,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def fit_tau_b_at_recall(
    dev_rows: list[dict[str, Any]],
    *,
    target_recall: float = 0.92,
    score_key: str = "bind_surprise",
) -> dict[str, Any]:
    """Select τ_B on dev to achieve target recall (for matched-recall operating point)."""
    if not dev_rows:
        return {"tau_b": 0.0, "dev_recall": None, "dev_precision": None}

    scores = np.array([float(r[score_key]) for r in dev_rows], dtype=np.float64)
    labels = np.array([int(r["y_pm"]) for r in dev_rows], dtype=np.int32)
    candidates = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 400)))

    best_tau = float(candidates[0])
    best_prec = 0.0
    best_rec = 0.0

    for tau in candidates:
        pred = scores >= float(tau)
        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        fn = int(np.sum((~pred) & (labels == 1)))
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        if rec >= float(target_recall) and prec >= best_prec:
            best_prec = prec
            best_rec = rec
            best_tau = float(tau)

    return {
        "tau_b": best_tau,
        "target_recall": target_recall,
        "dev_recall": best_rec,
        "dev_precision": best_prec,
        "score_key": score_key,
    }


def precision_at_recall(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    target_recall: float,
) -> dict[str, Any]:
    """Precision at fixed recall via score ranking (threshold-free on this split)."""
    y = np.array([int(r["y_pm"]) for r in rows], dtype=np.int32)
    scores = np.array([float(r.get(score_key) or 0.0) for r in rows], dtype=np.float64)
    if y.sum() == 0:
        return {"precision": None, "recall": None, "n_flagged": 0}

    order = np.argsort(-scores)
    y_sorted = y[order]
    n_pos = int(y.sum())
    for k in range(1, len(y_sorted) + 1):
        rec = float(y_sorted[:k].sum()) / n_pos
        if rec >= float(target_recall):
            prec = float(y_sorted[:k].sum()) / k
            return {"precision": prec, "recall": rec, "n_flagged": k, "target_recall": target_recall}
    return {
        "precision": float(y.sum()) / len(y),
        "recall": 1.0,
        "n_flagged": len(y),
        "target_recall": target_recall,
    }


def pr_dominance_report(
    rows: list[dict[str, Any]],
    *,
    probe_key: str = "p_pm",
    bind_key: str = "bind_surprise",
    recall_grid: tuple[float, ...] = (0.80, 0.85, 0.90, 0.917, 0.95, 1.0),
) -> dict[str, Any]:
    """
    Compare probe vs BindSurprise on PR-style operating points.

    Key reviewer metric: precision at matched recall (ranking-based, no test threshold tuning).
    """
    report: dict[str, Any] = {
        "auroc_probe": try_auroc(rows, probe_key),
        "auroc_bind_surprise": try_auroc(rows, bind_key),
        "ap_probe": _try_ap(rows, probe_key),
        "ap_bind_surprise": _try_ap(rows, bind_key),
        "precision_at_recall": {},
        "pareto_dominates_probe": None,
    }

    deltas: list[float] = []
    for target in recall_grid:
        mp = precision_at_recall(rows, score_key=probe_key, target_recall=target)
        mb = precision_at_recall(rows, score_key=bind_key, target_recall=target)
        delta = None
        if mp.get("precision") is not None and mb.get("precision") is not None:
            delta = float(mb["precision"]) - float(mp["precision"])
            deltas.append(delta)
        report["precision_at_recall"][str(target)] = {
            "probe": mp,
            "bind_surprise": mb,
            "precision_delta": delta,
        }

    report["pareto_dominates_probe"] = bool(deltas) and all(d >= 0 for d in deltas)
    report["mean_precision_delta_at_recall"] = float(np.mean(deltas)) if deltas else None
    return report


def export_pr_curve(
    rows: list[dict[str, Any]],
    *,
    probe_key: str = "p_pm",
    bind_key: str = "bind_surprise",
) -> dict[str, Any]:
    """Export PR curve points for probe vs BindSurprise (paper Figure data)."""
    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
    except ImportError:
        return {"error": "sklearn not available"}

    y = np.array([int(r["y_pm"]) for r in rows], dtype=np.int32)
    if len(set(y.tolist())) < 2:
        return {"error": "single class"}

    probe_scores = np.array([float(r.get(probe_key) or 0.0) for r in rows], dtype=np.float64)
    bind_scores = np.array([float(r.get(bind_key) or 0.0) for r in rows], dtype=np.float64)

    prec_p, rec_p, _ = precision_recall_curve(y, probe_scores)
    prec_b, rec_b, _ = precision_recall_curve(y, bind_scores)

    return {
        "split": rows[0].get("split") if rows else None,
        "n_claims": len(rows),
        "n_positive": int(y.sum()),
        "probe": {
            "recall": [float(x) for x in rec_p.tolist()],
            "precision": [float(x) for x in prec_p.tolist()],
            "auroc": float(roc_auc_score(y, probe_scores)),
            "ap": float(average_precision_score(y, probe_scores)),
        },
        "bind_surprise": {
            "recall": [float(x) for x in rec_b.tolist()],
            "precision": [float(x) for x in prec_b.tolist()],
            "auroc": float(roc_auc_score(y, bind_scores)),
            "ap": float(average_precision_score(y, bind_scores)),
        },
        "dominance": pr_dominance_report(rows, probe_key=probe_key, bind_key=bind_key),
    }


def _try_ap(rows: list[dict[str, Any]], score_key: str) -> float | None:
    try:
        from sklearn.metrics import average_precision_score

        y = [int(r["y_pm"]) for r in rows]
        if len(set(y)) < 2:
            return None
        scores = [float(r.get(score_key) or 0.0) for r in rows]
        return float(average_precision_score(y, scores))
    except Exception:
        return None


def try_auroc(rows: list[dict[str, Any]], score_key: str) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        y = [int(r["y_pm"]) for r in rows]
        if len(set(y)) < 2:
            return None
        scores = [float(r.get(score_key) or 0.0) for r in rows]
        return float(roc_auc_score(y, scores))
    except Exception:
        return None
