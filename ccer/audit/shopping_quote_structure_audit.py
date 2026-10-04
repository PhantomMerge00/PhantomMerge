"""Step 0: Shopping headline sanity check — quote-structure / TF-IDF baseline."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ccer.io_utils import write_json
from ccer.mechanism.bind_surprise import try_auroc
from ccer.mechanism.line_a_cohort_v3 import build_probe_dataset_v3
from ccer.mechanism.line_a_instance_activations import activation_matrix_for_instances
from ccer.mechanism.supervised_probe import normalize_quote_for_bow
from ccer.paths import ARTIFACTS

_JSON_LIKE = re.compile(r"^\s*[\{\[]")
_ATTR = re.compile(r"^[A-Za-z][^:]{0,40}:\s")


def quote_structure(q: str) -> str:
    s = str(q or "").strip()
    if _JSON_LIKE.match(s):
        return "json_like"
    if _ATTR.match(s):
        return "attr_colon"
    if s.lower().startswith("selected product"):
        return "selected_pid"
    return "prose"


def quote_fingerprint(q: str) -> str:
    t = normalize_quote_for_bow(q).lower()
    t = re.sub(r"\b\d+(\.\d+)?\b", "<NUM>", t)
    t = re.sub(r"\b[a-z]{2,}\d+[a-z0-9]*\b", "<ID>", t)
    return re.sub(r"\s+", " ", t).strip()


def _fit_probe_auroc(instances: list[dict], *, train_split: str = "train", test_split: str = "test") -> dict:
    train = [r for r in instances if r["split"] == train_split]
    test = [r for r in instances if r["split"] == test_split]
    if not train or not test:
        return {"error": "empty"}
    y_tr = np.array([r["y"] for r in train])
    y_te = np.array([r["y"] for r in test])
    if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        return {"error": "single_class"}
    X_tr, vidx_tr = activation_matrix_for_instances(train, position="claim_onset", layer=49)
    X_te, vidx_te = activation_matrix_for_instances(test, position="claim_onset", layer=49)
    if X_tr is None or X_te is None:
        return {"error": "missing_activation"}
    y_tr = y_tr[vidx_tr]
    y_te = y_te[vidx_te]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(Xtr, y_tr)
    probs = clf.predict_proba(Xte)[:, 1]
    scored = [{"y_pm": int(y), "p_pm": float(p)} for y, p in zip(y_te, probs)]
    tr_probs = clf.predict_proba(Xtr)[:, 1]
    tr_scored = [{"y_pm": int(y), "p_pm": float(p)} for y, p in zip(y_tr, tr_probs)]
    return {
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "train_auroc": try_auroc(tr_scored, "p_pm"),
        "test_auroc": try_auroc(scored, "p_pm"),
    }


def _tfidf_auroc(instances: list[dict], *, train_split: str = "train", test_split: str = "test") -> dict:
    train = [r for r in instances if r["split"] == train_split]
    test = [r for r in instances if r["split"] == test_split]
    y_tr = [r["y"] for r in train]
    y_te = [r["y"] for r in test]
    if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        return {"error": "single_class"}
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    Xtr = vec.fit_transform([normalize_quote_for_bow(r["response_quote"]) for r in train])
    Xte = vec.transform([normalize_quote_for_bow(r["response_quote"]) for r in test])
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(Xtr, y_tr)
    probs = clf.predict_proba(Xte)[:, 1]
    scored = [{"y_pm": int(y), "p_pm": float(p)} for y, p in zip(y_te, probs)]
    return {"n_test": len(y_te), "test_auroc": try_auroc(scored, "p_pm")}


def run_shopping_audit() -> dict[str, Any]:
    ds = build_probe_dataset_v3(require_activation=True)
    inst = ds.instances
    pm = [r for r in inst if r["y"] == 1]
    clean = [r for r in inst if r["y"] == 0]

    pm_tmpl = {quote_fingerprint(r["response_quote"]) for r in pm}
    cl_tmpl = {quote_fingerprint(r["response_quote"]) for r in clean}
    train_pm = {quote_fingerprint(r["response_quote"]) for r in pm if r["split"] == "train"}
    test_pm = {quote_fingerprint(r["response_quote"]) for r in pm if r["split"] == "test"}
    train_cl = {quote_fingerprint(r["response_quote"]) for r in clean if r["split"] == "train"}
    test_cl = {quote_fingerprint(r["response_quote"]) for r in clean if r["split"] == "test"}

    test_rows = [r for r in inst if r["split"] == "test"]
    test_overlap = sum(
        1 for r in test_rows if quote_fingerprint(r["response_quote"]) in train_pm | train_cl
    )

    probe = _fit_probe_auroc(inst)
    tfidf = _tfidf_auroc(inst)
    tfidf_test_only_pm_vs_clean = _tfidf_auroc(inst)  # same split

    headline_auroc = 0.962  # from prism_summary shopping ref
    tfidf_val = tfidf.get("test_auroc")
    gate = "SHOPPING_CLEAN"
    if tfidf_val is not None and tfidf_val >= 0.90:
        gate = "SHOPPING_CONTAMINATED_STOP"
    elif tfidf_val is not None and tfidf_val >= 0.75:
        gate = "SHOPPING_MARGINAL_REVIEW"
    else:
        gate = "SHOPPING_CLEAN"

    payload = {
        "schema": "shopping_quote_structure_audit_v0",
        "n_instances": len(inst),
        "n_pm": len(pm),
        "n_clean": len(clean),
        "pm_verdict_mix": dict(Counter(r["gold_verdict"] for r in pm)),
        "quote_structure": {
            "pm": dict(Counter(quote_structure(r["response_quote"]) for r in pm)),
            "clean": dict(Counter(quote_structure(r["response_quote"]) for r in clean)),
        },
        "template_counts": {
            "pm_unique": len(pm_tmpl),
            "clean_unique": len(cl_tmpl),
            "pm_clean_intersection": len(pm_tmpl & cl_tmpl),
        },
        "train_test_template_overlap": {
            "test_claims_overlapping_train_templates": test_overlap,
            "test_claims_overlapping_train_templates_rate": test_overlap / max(len(test_rows), 1),
            "test_pm_templates_in_train_pm": len(test_pm & train_pm),
            "test_clean_templates_in_train_clean": len(test_cl & train_cl),
        },
        "L49_probe": probe,
        "tfidf_quote_only": tfidf,
        "headline_reference_auroc": headline_auroc,
        "decision_gate": gate,
        "decision_rules": {
            "SHOPPING_CLEAN": "tfidf_test_auroc < 0.75 — proceed tau3 fix only",
            "SHOPPING_MARGINAL_REVIEW": "0.75 <= tfidf < 0.90 — shopping likely ok but monitor",
            "SHOPPING_CONTAMINATED_STOP": "tfidf >= 0.90 — stop, fix shopping first",
        },
    }
    out = ARTIFACTS / "ccer/shopping/shopping_quote_structure_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, payload)
    return payload


def main() -> int:
    p = run_shopping_audit()
    print(json.dumps(p, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
