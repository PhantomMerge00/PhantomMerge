"""TF-IDF-only control for pooled L49 probe: same train pool, fixed test splits."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from ccer.audit.tau3_pooled_probe_sweep import (
    POOL_BUILDERS,
    _collect_shopping,
    _collect_tau3,
    _register_pools,
)
from ccer.audit.tau3_quote_template_split import quote_fingerprint
from ccer.io_utils import write_json
from ccer.mechanism.bind_surprise import try_auroc
from ccer.mechanism.line_a_cohort_v3 import build_probe_dataset_v3
from ccer.mechanism.line_a_instance_activations import instance_npz_path as shopping_npz_path
from ccer.mechanism.tau3.agr_slot_eval import _load_split_maps
from ccer.mechanism.tau3.cohort import instance_npz_path as tau3_npz_path, tau3_instance_rows
from ccer.mechanism.tau3.domain import get_tau3_domain
from ccer.paths import TAU3_DIR

EVAL_DOMAINS = ("shopping", "telecom", "airline", "retail")
VERDICT_SHORT = {
    "cross_object_merge": "CEM",
    "constraint_projection": "CAP",
    "anchored_hallucination": "AH",
}


@dataclass(frozen=True)
class TextRow:
    text: str
    y: int
    domain: str
    split: str
    verdict: str
    instance_audit_key: str


def _shopping_text(*, splits: set[str], require_activation: bool = True) -> list[TextRow]:
    ds = build_probe_dataset_v3(require_activation=require_activation)
    out: list[TextRow] = []
    for row in ds.instances:
        sp = str(row.get("split") or "")
        if sp not in splits:
            continue
        text = str(row.get("response_quote") or "").strip()
        if not text:
            continue
        out.append(
            TextRow(
                text=text,
                y=int(row["y"]),
                domain="shopping",
                split=sp,
                verdict=str(row.get("gold_verdict") or ""),
                instance_audit_key=str(row["instance_audit_key"]),
            )
        )
    return out


def _tau3_text(cfg, *, inst_splits: set[str], require_activation: bool = True) -> list[TextRow]:
    _, inst_split = _load_split_maps(cfg)
    out: list[TextRow] = []
    for row in tau3_instance_rows(require_activation=require_activation, cfg=cfg):
        sp = inst_split.get(row.instance_audit_key, "unassigned")
        if sp not in inst_splits:
            continue
        text = str(row.response_quote or "").strip()
        if not text:
            continue
        out.append(
            TextRow(
                text=text,
                y=row.y,
                domain=cfg.domain,
                split=sp,
                verdict=row.gold_verdict,
                instance_audit_key=row.instance_audit_key,
            )
        )
    return out


def _pool_shopping_train() -> list[TextRow]:
    return _shopping_text(splits={"train"})


def _pool_pooled_shop_air_tel() -> list[TextRow]:
    return (
        _shopping_text(splits={"train"})
        + _tau3_text(get_tau3_domain("airline"), inst_splits={"D_p"})
        + _tau3_text(get_tau3_domain("telecom"), inst_splits={"D_p"})
    )


def _test_rows(domain: str) -> list[TextRow]:
    if domain == "shopping":
        return _shopping_text(splits={"test"})
    return _tau3_text(get_tau3_domain(domain), inst_splits={"test"})


def _fit_eval_tfidf(train: list[TextRow], test: list[TextRow]) -> dict[str, Any]:
    if not train or not test:
        return {"error": "empty", "n_train": len(train), "n_test": len(test)}
    y_tr = [r.y for r in train]
    y_te = [r.y for r in test]
    if len(set(y_tr)) < 2:
        return {"error": "single_class_train", "n_train": len(train)}
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tr = vec.fit_transform([r.text for r in train])
    X_te = vec.transform([r.text for r in test])
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    scored = [{"y_pm": int(y), "p_pm": float(p)} for y, p in zip(y_te, probs)]
    return {
        "n_train": len(train),
        "n_test": len(test),
        "n_pm_test": sum(y_te),
        "n_clean_test": len(y_te) - sum(y_te),
        "test_auroc": try_auroc(scored, "p_pm"),
        "test_f1": float(f1_score(y_te, preds, zero_division=0)),
    }


def _template_overlap(test_rows: list[TextRow]) -> dict[str, Any]:
    pm = {quote_fingerprint(r.text) for r in test_rows if r.y == 1}
    clean = {quote_fingerprint(r.text) for r in test_rows if r.y == 0}
    return {
        "pm_unique_templates": len(pm),
        "clean_unique_templates": len(clean),
        "pm_clean_intersection": len(pm & clean),
    }


def _verdict_mix(rows: list[TextRow]) -> dict[str, int]:
    c = Counter()
    for r in rows:
        if r.y == 1:
            c[VERDICT_SHORT.get(r.verdict, r.verdict)] += 1
    return dict(c)


def run_control() -> dict[str, Any]:
    _register_pools()
    train_pools = {
        "shopping_only": _pool_shopping_train(),
        "pooled_shop_air_tel": _pool_pooled_shop_air_tel(),
        "telecom_Dp_only": _tau3_text(get_tau3_domain("telecom"), inst_splits={"D_p"}),
        "airline_Dp_only": _tau3_text(get_tau3_domain("airline"), inst_splits={"D_p"}),
    }

    # L49 pooled probe reference (same test splits)
    sweep_path = TAU3_DIR / "pooled_probe_sweep.json"
    l49_ref: dict[str, Any] = {}
    if sweep_path.is_file():
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        pooled = sweep["strategies"]["shop_plus_airline_plus_telecom"]["eval"]
        shop_frozen = sweep["baselines"]["shopping_frozen"]["eval"]
        l49_ref = {
            "pooled_L49_probe": {d: pooled[d]["probe_auroc"] for d in EVAL_DOMAINS},
            "pooled_L49_bind": {d: pooled[d]["bind_surprise_auroc"] for d in EVAL_DOMAINS},
            "shopping_frozen_L49_probe": {d: shop_frozen[d]["probe_auroc"] for d in EVAL_DOMAINS},
        }

    out: dict[str, Any] = {
        "schema": "tau3_pooled_tfidf_control_v1",
        "train_pools": {
            name: {"n": len(rows), "n_pm": sum(r.y for r in rows), "verdict_pm": _verdict_mix(rows)}
            for name, rows in train_pools.items()
        },
        "eval": {},
        "l49_reference": l49_ref,
        "prior_telecom_quote_template_tfidf": 0.9772727272727273,
    }

    for train_name, train_rows in train_pools.items():
        out["eval"][train_name] = {}
        for dom in EVAL_DOMAINS:
            test_rows = _test_rows(dom)
            out["eval"][train_name][dom] = {
                **_fit_eval_tfidf(train_rows, test_rows),
                "template_overlap": _template_overlap(test_rows),
            }

    # decision logic for pooled vs shopping-only delta
    pooled = out["eval"]["pooled_shop_air_tel"]
    shop_only = out["eval"]["shopping_only"]
    deltas = {}
    for dom in EVAL_DOMAINS:
        pt = pooled[dom].get("test_auroc")
        st = shop_only[dom].get("test_auroc")
        l49 = (l49_ref.get("pooled_L49_probe") or {}).get(dom)
        deltas[dom] = {
            "tfidf_pooled_minus_shopping_only": None if pt is None or st is None else round(pt - st, 4),
            "l49_pooled_probe": l49,
            "tfidf_pooled": pt,
            "tfidf_shopping_only": st,
            "l49_minus_tfidf_pooled": None if l49 is None or pt is None else round(l49 - pt, 4),
        }
    out["delta_analysis"] = deltas

    # global narrative gate
    tau3_domains = ("telecom", "airline", "retail")
    tfidf_jumps = [
        deltas[d]["tfidf_pooled_minus_shopping_only"]
        for d in tau3_domains
        if deltas[d]["tfidf_pooled_minus_shopping_only"] is not None
    ]
    l49_gaps = [
        deltas[d]["l49_minus_tfidf_pooled"]
        for d in tau3_domains
        if deltas[d]["l49_minus_tfidf_pooled"] is not None
    ]
    avg_tfidf_jump = sum(tfidf_jumps) / len(tfidf_jumps) if tfidf_jumps else None
    avg_l49_gap = sum(l49_gaps) / len(l49_gaps) if l49_gaps else None

    tel = deltas.get("telecom", {})
    air = deltas.get("airline", {})
    shop = deltas.get("shopping", {})
    tel_l49_gap = tel.get("l49_minus_tfidf_pooled")
    tel_tfidf = tel.get("tfidf_pooled")
    air_tfidf_jump = air.get("tfidf_pooled_minus_shopping_only")
    shop_l49_gap = shop.get("l49_minus_tfidf_pooled")

    if tel_tfidf is not None and tel_tfidf >= 0.95 and (tel_l49_gap or 0) < 0.05:
        gate = "REJECT_tau3_shared_representation"
        gate_reason = (
            "Telecom: pooled TF-IDF≈1.0 alongside L49≈1.0 (Δ<0.05); "
            "shopping-only TF-IDF already 0.955 — perfect τ³ separation is structure-driven, not L49-specific"
        )
    elif avg_l49_gap is not None and avg_l49_gap >= 0.10:
        gate = "SUPPORT_shared_representation"
        gate_reason = "L49 pooled >> TF-IDF pooled on τ³ → representation gain beyond surface text"
    elif air_tfidf_jump is not None and air_tfidf_jump >= 0.5:
        gate = "REJECT_tau3_shared_representation"
        gate_reason = "Airline: pooled TF-IDF jumps ~1.0 with pooling — L49 gain not distinguishable from text shortcut"
    elif shop_l49_gap is not None and shop_l49_gap >= 0.10:
        gate = "SHOPPING_ONLY_representation_gap"
        gate_reason = "Shopping: L49 >> TF-IDF; τ³ claims unsupported — keep main evidence on shopping in-domain"
    else:
        gate = "MIXED_review_required"
        gate_reason = "TF-IDF and L49 gaps ambiguous; report both with template overlap caveats"

    out["narrative_gate"] = {
        "verdict": gate,
        "reason": gate_reason,
        "avg_tfidf_jump_tau3": avg_tfidf_jump,
        "avg_l49_minus_tfidf_tau3": avg_l49_gap,
    }
    return out


def write_report(payload: dict[str, Any], path: Path) -> None:
    gate = payload["narrative_gate"]
    lines = [
        "# Pooled TF-IDF Control (vs L49 Pooled Probe)",
        "",
        "> 同一训练池（shopping train + airline/telecom D_p）与固定 test；纯 claim `response_quote` TF-IDF+LogReg。",
        "",
        f"**叙事门禁**: `{gate['verdict']}` — {gate['reason']}",
        "",
        "## TF-IDF test AUROC（行=训练池，列=测试域）",
        "",
        "| Train pool | shopping | telecom | airline | retail |",
        "|------------|---------:|--------:|--------:|-------:|",
    ]
    for train_name in ("shopping_only", "pooled_shop_air_tel", "telecom_Dp_only", "airline_Dp_only"):
        row = f"| {train_name} |"
        for dom in EVAL_DOMAINS:
            au = payload["eval"][train_name][dom].get("test_auroc")
            row += f" {au:.3f} |" if au is not None else " — |"
        lines.append(row)

    lines += [
        "",
        "## L49 pooled probe vs pooled TF-IDF（同 test）",
        "",
        "| Domain | L49 probe | TF-IDF pooled | Δ(L49−TFIDF) | TF-IDF Δ(pooled−shop-only) | template PM∩clean |",
        "|--------|----------:|--------------:|-------------:|---------------------------:|------------------:|",
    ]
    ref = payload.get("l49_reference", {})
    for dom in EVAL_DOMAINS:
        d = payload["delta_analysis"][dom]
        tmpl = payload["eval"]["pooled_shop_air_tel"][dom]["template_overlap"]
        l49 = (ref.get("pooled_L49_probe") or {}).get(dom)
        tf = d["tfidf_pooled"]
        lines.append(
            f"| {dom} | {l49:.3f} | {tf:.3f} | {d['l49_minus_tfidf_pooled']} | "
            f"{d['tfidf_pooled_minus_shopping_only']} | {tmpl['pm_clean_intersection']} |"
        )

    lines += [
        "",
        "## 解读",
        "",
        "- telecom test 模板 PM∩clean=0 时 TF-IDF 仍高 → 结构/词面捷径未消除",
        "- 仅当 **L49 pooled ≫ TF-IDF pooled** 且 TF-IDF 未随 pooling 同步暴涨，才可写 shared binding signature",
        f"- retail n_pm={payload['eval']['pooled_shop_air_tel']['retail'].get('n_pm_test')} — 仅 diagnostic",
        "",
        f"Prior telecom quote-template TF-IDF (结构修复后): {payload.get('prior_telecom_quote_template_tfidf')}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    payload = run_control()
    out_json = TAU3_DIR / "pooled_tfidf_control.json"
    out_md = TAU3_DIR / "TAU3_POOLED_TFIDF_CONTROL_REPORT.md"
    write_json(out_json, payload)
    write_report(payload, out_md)
    print(json.dumps(payload["narrative_gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
