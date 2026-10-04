"""Post structure-matched clean fix: quote-template-split probe + TF-IDF + zero-shot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from ccer.io_utils import write_json
from ccer.mechanism.bind_surprise import try_auroc
from ccer.mechanism.tau3.domain import Tau3DomainConfig, get_tau3_domain
from ccer.mechanism.tau3.frozen_probe import load_shopping_frozen_probe
from ccer.paths import TAU3_DIR

from ccer.audit.tau3_agr_red_flag_audit import _XY, _rows_from_probe, _sign_flip_auroc
from ccer.audit.tau3_quote_template_split import (
    _collect_xy_with_manifest,
    build_quote_template_split_manifest,
    evaluate_probe_on_quote_template_split,
    quote_fingerprint,
)
from ccer.mechanism.tau3.cohort import tau3_instance_rows


def _records_for_manifest(cfg: Tau3DomainConfig, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    inst_split = manifest.get("instance_split") or {}
    out: list[dict[str, Any]] = []
    for row in tau3_instance_rows(require_activation=True, cfg=cfg):
        sp = inst_split.get(row.instance_audit_key, "unassigned")
        if sp not in {"D_p", "test"}:
            continue
        out.append(
            {
                "split": sp,
                "y_pm": int(row.y),
                "response_quote": row.response_quote,
                "quote_template": quote_fingerprint(row.response_quote),
            }
        )
    return out


def _tfidf_on_manifest(cfg: Tau3DomainConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    records = _records_for_manifest(cfg, manifest)
    train_meta = [r for r in records if r["split"] == "D_p"]
    test_meta = [r for r in records if r["split"] == "test"]
    if not train_meta or not test_meta:
        return {"error": "empty_split"}
    y_tr = [r["y_pm"] for r in train_meta]
    y_te = [r["y_pm"] for r in test_meta]
    if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        return {"error": "single_class"}
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tr = vec.fit_transform([r["response_quote"] for r in train_meta])
    X_te = vec.transform([r["response_quote"] for r in test_meta])
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    scored = [{"y_pm": int(y), "p_pm": float(p)} for y, p in zip(y_te, probs)]
    return {
        "n_train": len(train_meta),
        "n_test": len(test_meta),
        "test_auroc": try_auroc(scored, "p_pm"),
        "test_f1": float(f1_score(y_te, preds)),
    }


def _zeroshot_on_manifest(cfg: Tau3DomainConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    probe = load_shopping_frozen_probe()
    X_test, y_test, groups, meta_test = _collect_xy_with_manifest(cfg, manifest, splits={"test"})
    if len(y_test) == 0:
        return {"error": "empty_test"}
    rows = _rows_from_probe(probe, _XY(X_test, y_test, groups, meta_test))
    return {
        "n_test": len(y_test),
        "zeroshot_auroc": try_auroc(rows, "p_pm"),
        "sign_flip": _sign_flip_auroc(rows),
    }


def _template_overlap_stats(cfg: Tau3DomainConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    records = _records_for_manifest(cfg, manifest)
    dp_templates = {r["quote_template"] for r in records if r["split"] == "D_p"}
    test_rows = [r for r in records if r["split"] == "test"]
    pm_templates = {r["quote_template"] for r in test_rows if r["y_pm"] == 1}
    clean_templates = {r["quote_template"] for r in test_rows if r["y_pm"] == 0}
    leak = manifest.get("leak_check") or {}
    return {
        "dp_unique_templates": len(dp_templates),
        "test_pm_unique_templates": len(pm_templates),
        "test_clean_unique_templates": len(clean_templates),
        "pm_clean_template_intersection": len(pm_templates & clean_templates),
        "test_dp_template_overlap_templates": leak.get("test_intersect_D_p_templates"),
    }


def run_domain_reval(cfg: Tau3DomainConfig) -> dict[str, Any]:
    manifest = build_quote_template_split_manifest(cfg)
    manifest_path = cfg.report.parent / "split_manifest_quote_template.json"
    write_json(manifest_path, manifest)

    probe_eval = evaluate_probe_on_quote_template_split(cfg, manifest)
    tfidf = _tfidf_on_manifest(cfg, manifest)
    zeroshot = _zeroshot_on_manifest(cfg, manifest)
    overlap = _template_overlap_stats(cfg, manifest)

    train_auroc = probe_eval.get("train_D_p_auroc")
    test_auroc = probe_eval.get("test_auroc")
    gap = None
    if train_auroc is not None and test_auroc is not None:
        gap = round(test_auroc - train_auroc, 4)

    if test_auroc is not None and tfidf.get("test_auroc") is not None:
        if test_auroc >= 0.85 and tfidf["test_auroc"] < test_auroc - 0.15 and (train_auroc or 0) < 0.999:
            scenario = "A_optimistic"
        else:
            scenario = "B_pessimistic"
    else:
        scenario = "unknown"

    return {
        "schema": "tau3_structure_fix_reval_v1",
        "domain": cfg.domain,
        "manifest_path": str(manifest_path),
        "quote_template_overlap": overlap,
        "probe_retrained_quote_template_split": {
            "train_D_p_auroc": train_auroc,
            "test_auroc": test_auroc,
            "test_f1": probe_eval.get("test_f1"),
            "train_test_gap": gap,
            "n_train": probe_eval.get("n_train"),
            "n_test": probe_eval.get("n_test"),
            "n_pm_test": probe_eval.get("n_pm_test"),
            "n_clean_test": probe_eval.get("n_clean_test"),
        },
        "tfidf_quote_only": tfidf,
        "zeroshot_frozen_probe": zeroshot,
        "decision_scenario": scenario,
        "headline_four_numbers": {
            "test_auroc": test_auroc,
            "test_f1": probe_eval.get("test_f1"),
            "train_D_p_auroc": train_auroc,
            "tfidf_test_auroc": tfidf.get("test_auroc"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="both", choices=["telecom", "airline", "both"])
    args = ap.parse_args(argv)
    domains = ["telecom", "airline"] if args.domain == "both" else [args.domain]
    results: dict[str, Any] = {}
    for d in domains:
        cfg = get_tau3_domain(d)
        r = run_domain_reval(cfg)
        out = cfg.report.parent / "structure_fix_reval.json"
        write_json(out, r)
        results[d] = r
        print(json.dumps({d: r["headline_four_numbers"]}, indent=2))

    report = TAU3_DIR / "TAU3_STRUCTURE_FIX_REVAL_REPORT.md"
    lines = [
        "# Tau3 Structure-Matched Clean Fix — Re-evaluation",
        "",
        "Clean 负样本已改为 tool-JSON / tool-observation 结构匹配 anchor（非 final_answer 散文）。",
        "探针指标均在 **quote-template 零重叠切分** 上报告。",
        "",
    ]
    for d, r in results.items():
        h = r["headline_four_numbers"]
        ov = r["quote_template_overlap"]
        zs = r["zeroshot_frozen_probe"]
        lines += [
            f"## {d.capitalize()}",
            "",
            f"- **test AUROC / F1**: {h.get('test_auroc')} / {h.get('test_f1')}",
            f"- **train AUROC (D_p)**: {h.get('train_D_p_auroc')}",
            f"- **TF-IDF-only test AUROC**: {h.get('tfidf_test_auroc')}",
            f"- **PM/clean 模板交集**: {ov.get('pm_clean_template_intersection')}",
            f"- **zero-shot AUROC**: {zs.get('zeroshot_auroc')} (sign-flip: {(zs.get('sign_flip') or {}).get('auroc_sign_flipped')})",
            f"- **决策情形**: {r.get('decision_scenario')}",
            "",
        ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
