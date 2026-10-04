"""Line A: supervised linear probe vs TF-IDF BoW baseline on incremental adjudication."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from ccer.io_utils import load_jsonl
from ccer.paths import INCREMENTAL_ADJUDICATION_JSONL, NORMALIZED_SHOPPING, SPLIT_MANIFEST_JSON

# Adjudication quote attribute prefixes (CAP/AH/CEM span format); stripped for BoW.
_QUOTE_ATTR_PREFIX = re.compile(
    r"(?i)^(size|color|material|brand|type|style|model|capacity|voltage|power|weight|"
    r"length|width|height|dimensions|project type|hull material|fastener type|"
    r"mesh opening size|season|neckline|print|includes|usage|size range|fit|price|"
    r"formulation|title):\s*"
)

LINE_A_POSITION = "prompt_end"

PM_VERDICTS = frozenset(
    {
        "cross_object_merge",
        "constraint_projection",
        "anchored_hallucination",
    }
)
CLEAN_VERDICT = "correct_binding"


@dataclass
class ProbeDataset:
    instances: list[dict[str, Any]]
    activation_layers: dict[str, list[int]] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)


def _load_split_manifest() -> dict[str, str]:
    payload = json.loads(SPLIT_MANIFEST_JSON.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in (payload.get("splits") or {}).items()}


def line_a_skip_trajectory_ids() -> dict[str, str]:
    """Explicit extraction skips (tid -> reason)."""
    from ccer.paths import LINE_A_SKIP_TRAJECTORIES

    if not LINE_A_SKIP_TRAJECTORIES.is_file():
        return {}
    payload = json.loads(LINE_A_SKIP_TRAJECTORIES.read_text(encoding="utf-8"))
    raw = payload.get("trajectory_ids") or {}
    return {str(k): str(v) for k, v in raw.items()}


def line_a_trajectory_ids(*, exclude_skipped: bool = True) -> list[str]:
    """Trajectories with commitment-eligible PM or correct_binding instances."""
    skips = line_a_skip_trajectory_ids() if exclude_skipped else {}
    tids: set[str] = set()
    for row in load_jsonl(INCREMENTAL_ADJUDICATION_JSONL):
        if not row.get("commitment_eligible"):
            continue
        verdict = row.get("gold_verdict")
        if verdict in PM_VERDICTS or verdict == CLEAN_VERDICT:
            tid = str(row["trajectory_id"])
            if tid not in skips:
                tids.add(tid)
    return sorted(tids)


def normalize_quote_for_bow(text: str | None) -> str:
    """De-template adjudication quote for BoW: strip attr prefix, mask numbers, lower."""
    q = str(text or "").strip()
    q = _QUOTE_ATTR_PREFIX.sub("", q)
    q = re.sub(r"\s+", " ", q).strip().lower()
    q = re.sub(r"\d+(?:\.\d+)?", "<num>", q)
    return q


def _bow_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 1),
        min_df=2,
        token_pattern=r"(?u)\b[\w<>]+\b",
    )


def _auroc(y: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def bootstrap_auroc_ci(
    y: np.ndarray,
    scores: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    clusters = np.asarray(clusters, dtype=str)
    point = _auroc(y, scores)
    cluster_ids = sorted(set(clusters.tolist()))
    cluster_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, cid in enumerate(clusters.tolist()):
        cluster_to_idx[cid].append(i)

    rng = np.random.default_rng(seed)
    boot_aurocs: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx: list[int] = []
        for cid in sampled:
            idx.extend(cluster_to_idx[cid])
        if len(idx) < 2:
            continue
        yb = y[idx]
        sb = scores[idx]
        a = _auroc(yb, sb)
        if a is not None:
            boot_aurocs.append(a)
    if not boot_aurocs:
        return {"auroc": point, "auroc_ci_95": None, "n_boot_effective": 0}
    lo, hi = np.percentile(boot_aurocs, [2.5, 97.5])
    return {
        "auroc": point,
        "auroc_ci_95": [float(lo), float(hi)],
        "n_boot_effective": len(boot_aurocs),
    }


def bootstrap_diff_ci(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    clusters = np.asarray(clusters, dtype=str)
    point_a = _auroc(y, scores_a)
    point_b = _auroc(y, scores_b)
    diff = None if point_a is None or point_b is None else point_a - point_b
    cluster_ids = sorted(set(clusters.tolist()))
    cluster_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, cid in enumerate(clusters.tolist()):
        cluster_to_idx[cid].append(i)
    rng = np.random.default_rng(seed)
    boot_diffs: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx: list[int] = []
        for cid in sampled:
            idx.extend(cluster_to_idx[cid])
        if len(idx) < 2:
            continue
        a = _auroc(y[idx], scores_a[idx])
        b = _auroc(y[idx], scores_b[idx])
        if a is not None and b is not None:
            boot_diffs.append(a - b)
    if not boot_diffs:
        return {"diff": diff, "diff_ci_95": None, "n_boot_effective": 0}
    lo, hi = np.percentile(boot_diffs, [2.5, 97.5])
    return {
        "diff": diff,
        "diff_ci_95": [float(lo), float(hi)],
        "n_boot_effective": len(boot_diffs),
    }


def _subset_indices(
    instances: list[dict[str, Any]],
    *,
    split_filter: Iterable[str] | None = None,
) -> list[int]:
    allowed = set(split_filter) if split_filter is not None else None
    out: list[int] = []
    for i, row in enumerate(instances):
        if allowed is not None and row["split"] not in allowed:
            continue
        out.append(i)
    return out


def _split_stats(y: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    return {
        "n": int(len(y)),
        "n_pm": int((y == 1).sum()),
        "n_clean": int((y == 0).sum()),
    }


def train_layer_position_probe(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[str],
    trajectory_ids: list[str],
    *,
    eval_splits: set[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y[train_idx])

    def _eval(idxs: list[int], key: str) -> dict[str, Any]:
        if not idxs:
            return {"auroc": None, "n": 0}
        Xe = scaler.transform(X[idxs])
        probs = clf.predict_proba(Xe)[:, 1]
        y_e = y[idxs]
        clusters = np.array([trajectory_ids[i] for i in idxs], dtype=str)
        ci = bootstrap_auroc_ci(y_e, probs, clusters, n_boot=n_boot, seed=seed)
        return {**_split_stats(y_e), **ci, "roc": _roc_payload(y_e, probs)}

    out: dict[str, Any] = {
        "train": _eval(train_idx, "train"),
        "eval": {},
    }
    for split_name in sorted(eval_splits):
        idxs = [i for i, s in enumerate(splits) if s == split_name]
        out["eval"][split_name] = _eval(idxs, split_name)
    return out


def train_bow_baseline(
    texts: list[str],
    y: np.ndarray,
    splits: list[str],
    trajectory_ids: list[str],
    *,
    eval_splits: set[str],
    n_boot: int = 2000,
    seed: int = 42,
    normalize_quotes: bool = True,
) -> dict[str, Any]:
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    if normalize_quotes:
        texts = [normalize_quote_for_bow(t) for t in texts]
    vec = _bow_vectorizer()
    X_train = vec.fit_transform([texts[i] for i in train_idx])
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y[train_idx])

    def _eval(idxs: list[int]) -> dict[str, Any]:
        if not idxs:
            return {"auroc": None, "n": 0}
        Xe = vec.transform([texts[i] for i in idxs])
        probs = clf.predict_proba(Xe)[:, 1]
        y_e = y[idxs]
        clusters = np.array([trajectory_ids[i] for i in idxs], dtype=str)
        ci = bootstrap_auroc_ci(y_e, probs, clusters, n_boot=n_boot, seed=seed)
        return {**_split_stats(y_e), **ci, "roc": _roc_payload(y_e, probs)}

    out: dict[str, Any] = {
        "train": _eval(train_idx),
        "eval": {},
    }
    for split_name in sorted(eval_splits):
        idxs = [i for i, s in enumerate(splits) if s == split_name]
        out["eval"][split_name] = _eval(idxs)
    return out


def _roc_payload(y: np.ndarray, scores: np.ndarray) -> dict[str, Any] | None:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, scores)
    return {
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr],
    }


def experiment_readiness(ds: ProbeDataset) -> dict[str, Any]:
    """Return gating checks for a valid Line A holdout experiment."""
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ds.instances:
        by_split[row["split"]].append(row)

    def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        y = [r["y"] for r in rows]
        return {
            "n_instances": len(rows),
            "n_trajectories": len({r["trajectory_id"] for r in rows}),
            "n_pm": sum(1 for v in y if v == 1),
            "n_clean": sum(1 for v in y if v == 0),
            "both_classes": len(set(y)) >= 2,
        }

    train = _stats(by_split.get("train", []))
    test = _stats(by_split.get("test", []))
    dev = _stats(by_split.get("dev", []))
    issues: list[str] = []
    if train["n_trajectories"] < 20:
        issues.append(
            f"train trajectories {train['n_trajectories']} < 20 (need trajectory-level train split)"
        )
    if not train["both_classes"]:
        issues.append("train split lacks both PM and clean instances")
    if test["n_instances"] < 10:
        issues.append(f"test instances {test['n_instances']} < 10 (primary holdout underpowered)")
    if not test["both_classes"]:
        issues.append("test split lacks both PM and clean instances")
    return {
        "ready": not issues,
        "issues": issues,
        "train": train,
        "test": test,
        "dev": dev,
    }


def _holdout_scores_probe(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[str],
    trajectory_ids: list[str],
    eval_splits: set[str],
) -> np.ndarray:
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    holdout_idx = [i for i, s in enumerate(splits) if s in eval_splits]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y[train_idx])
    X_h = scaler.transform(X[holdout_idx])
    return clf.predict_proba(X_h)[:, 1]


def _holdout_scores_bow(
    texts: list[str],
    y: np.ndarray,
    splits: list[str],
    trajectory_ids: list[str],
    eval_splits: set[str],
    *,
    normalize_quotes: bool = True,
) -> np.ndarray:
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    holdout_idx = [i for i, s in enumerate(splits) if s in eval_splits]
    if normalize_quotes:
        texts = [normalize_quote_for_bow(t) for t in texts]
    vec = _bow_vectorizer()
    X_train = vec.fit_transform([texts[i] for i in train_idx])
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y[train_idx])
    X_h = vec.transform([texts[i] for i in holdout_idx])
    return clf.predict_proba(X_h)[:, 1]


def _find_winners(layer_results: list[dict[str, Any]]) -> dict[str, Any]:
    primary: list[dict[str, Any]] = []
    supplement: list[dict[str, Any]] = []
    for row in layer_results:
        d_test = row.get("delta_vs_bow_claim_test") or {}
        d_devtest = row.get("delta_vs_bow_claim_dev_test") or {}
        probe_test = ((row.get("probe") or {}).get("eval") or {}).get("test") or {}
        if d_test.get("diff") is not None and d_test.get("diff", -1) >= 0.03:
            ci = d_test.get("diff_ci_95")
            if ci and ci[0] > 0:
                primary.append(
                    {
                        "position": row["position"],
                        "layer": row["layer"],
                        "probe_auroc": probe_test.get("auroc"),
                        "delta_auroc": d_test.get("diff"),
                        "delta_ci_95": ci,
                    }
                )
        if d_devtest.get("diff") is not None and d_devtest.get("diff", -1) >= 0.03:
            ci = d_devtest.get("diff_ci_95")
            if ci and ci[0] > 0:
                supplement.append(
                    {
                        "position": row["position"],
                        "layer": row["layer"],
                        "delta_auroc": d_devtest.get("diff"),
                        "delta_ci_95": ci,
                    }
                )
    primary.sort(key=lambda r: r["delta_auroc"], reverse=True)
    supplement.sort(key=lambda r: r["delta_auroc"], reverse=True)
    return {
        "primary_passed": bool(primary),
        "primary_winners": primary,
        "supplement_winners": supplement,
    }
