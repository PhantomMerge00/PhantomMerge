"""Track K: probe-guided fixed-anchor claim filtering using Line A v3 probe."""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ccer.mechanism.claim_filter_audit import (
    audit_gating,
    audit_rewrite,
    bootstrap_traj_pm,
    instances_to_dataframe,
    kept_counts_per_group,
    random_retention_matched_eval,
)
from ccer.mechanism.line_a_cohort_v3 import build_probe_dataset_v3
from ccer.mechanism.line_a_instance_activations import activation_matrix_for_instances
from ccer.mechanism.line_a_probe_v3 import LINE_A_PRIMARY_POSITION
from ccer.paths import (
    LINE_A_V3_PROBE_SUMMARY,
    LINE_K_DEV_THRESHOLDS,
    LINE_K_PROBE_SCORES,
    LINE_K_SUMMARY,
)
from ccer.mechanism.supervised_probe import (
    _bow_vectorizer,
    experiment_readiness,
    normalize_quote_for_bow,
)

REWRITE_STUB = (
    "[Evidence boundary: this attribute cannot be verified from the selected product's anchor evidence alone.]"
)

DEFAULT_TAUS = np.asarray([round(x, 2) for x in np.arange(0.05, 0.96, 0.05)], dtype=float)
BALANCED_MIN_CB = 0.85


def _fit_probe_scores(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[str],
) -> np.ndarray:
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y[train_idx])
    return clf.predict_proba(scaler.transform(X))[:, 1]


def _fit_bow_scores(
    quotes: list[str],
    y: np.ndarray,
    splits: list[str],
) -> np.ndarray:
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    texts = [normalize_quote_for_bow(q) for q in quotes]
    vec = _bow_vectorizer()
    X_train = vec.fit_transform([texts[i] for i in train_idx])
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y[train_idx])
    return clf.predict_proba(vec.transform(texts))[:, 1]


def _score_drop_factory(score_col: str, tau: float) -> Callable[[dict], bool]:
    return lambda r, t=tau, col=score_col: float(r.get(col, 0.0)) > t


def _select_aggressive_tau(df_dev: pd.DataFrame, score_col: str, taus: np.ndarray) -> float:
    best_tau = float(taus[0])
    best_pm = float("inf")
    for tau in taus:
        audit = audit_gating(df_dev, _score_drop_factory(score_col, float(tau)))
        if audit.gated_pm_rate < best_pm:
            best_pm = audit.gated_pm_rate
            best_tau = float(tau)
    return best_tau


def _select_balanced_tau(
    df_dev: pd.DataFrame,
    score_col: str,
    taus: np.ndarray,
    min_cb: float = BALANCED_MIN_CB,
) -> float:
    best: dict[str, Any] | None = None
    for tau in taus:
        audit = audit_gating(df_dev, _score_drop_factory(score_col, float(tau)))
        if audit.cb_retention_rate + 1e-12 < min_cb:
            continue
        rec = {
            "tau": float(tau),
            "pm_reduction": audit.baseline_pm_rate - audit.gated_pm_rate,
            "cb_retention_rate": audit.cb_retention_rate,
            "gated_pm_rate": audit.gated_pm_rate,
        }
        if best is None or (rec["pm_reduction"], rec["cb_retention_rate"]) > (
            best["pm_reduction"],
            best["cb_retention_rate"],
        ):
            best = rec
    if best is not None:
        return float(best["tau"])
    return _select_aggressive_tau(df_dev, score_col, taus)


def _val_pareto(df_val: pd.DataFrame, score_col: str, taus: np.ndarray, tag: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tau in taus:
        audit = audit_gating(df_val, _score_drop_factory(score_col, float(tau)))
        rows.append(
            {
                "method": tag,
                "tau": float(tau),
                "gated_pm_rate": audit.gated_pm_rate,
                "pm_reduction": audit.baseline_pm_rate - audit.gated_pm_rate,
                "cb_retention_rate": audit.cb_retention_rate,
                "claims_retained_mean": audit.claims_retained_mean,
            }
        )
    return rows


def _eval_method(
    df_test: pd.DataFrame,
    score_col: str,
    tau: float,
    name: str,
    *,
    boot_B: int = 2000,
    seed: int = 42,
    random_match: dict[str, int] | None = None,
) -> dict[str, Any]:
    drop_fn = _score_drop_factory(score_col, tau)
    audit = audit_gating(df_test, drop_fn)
    row = audit.to_dict()
    row["method"] = name
    row["tau"] = tau
    row.update(bootstrap_traj_pm(df_test, drop_fn, B=boot_B, seed=seed))
    if random_match is not None:
        row["random_retention_matched"] = random_retention_matched_eval(
            df_test, random_match, n_seeds=200, base_seed=seed
        )
    return row


def _eval_rewrite(
    df_test: pd.DataFrame,
    score_col: str,
    tau: float,
    name: str,
) -> dict[str, Any]:
    rewrite_fn = _score_drop_factory(score_col, tau)
    audit = audit_rewrite(df_test, rewrite_fn)
    row = audit.to_dict()
    row["method"] = name
    row["tau"] = tau
    row["intervention"] = "anchor_stub_rewrite"
    return row


def export_frozen_probe_scores(
    sub_instances: list[dict[str, Any]],
    probe_map: dict[str, float],
    bow_map: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for inst in sub_instances:
        iak = str(inst["instance_audit_key"])
        rows.append(
            {
                "instance_audit_key": iak,
                "trajectory_id": str(inst["trajectory_id"]),
                "split": str(inst["split"]),
                "p_pm": float(probe_map.get(iak, 0.0)),
                "p_bow": float(bow_map.get(iak, 0.0)),
                "y": int(inst["y"]),
                "gold_verdict": str(inst.get("gold_verdict", "")),
                "response_quote": str(inst.get("response_quote", "")),
            }
        )
    return rows


def load_frozen_probe_scores() -> list[dict[str, Any]]:
    from ccer.io_utils import load_jsonl

    if not LINE_K_PROBE_SCORES.is_file():
        raise FileNotFoundError(f"Missing frozen probe scores: {LINE_K_PROBE_SCORES}")
    return load_jsonl(LINE_K_PROBE_SCORES)


def load_frozen_dev_thresholds() -> dict[str, float]:
    if LINE_K_DEV_THRESHOLDS.is_file():
        import json

        return {k: float(v) for k, v in json.loads(LINE_K_DEV_THRESHOLDS.read_text(encoding="utf-8")).items()}
    if LINE_K_SUMMARY.is_file():
        payload = json.loads(LINE_K_SUMMARY.read_text(encoding="utf-8"))
        return {k: float(v) for k, v in (payload.get("dev_thresholds") or {}).items()}
    raise FileNotFoundError("Missing dev thresholds; run Track K first.")


def run_line_k_claim_filter_experiment(
    *,
    n_boot: int = 2000,
    seed: int = 42,
    balanced_min_cb: float = BALANCED_MIN_CB,
    require_ready: bool = True,
) -> dict[str, Any]:
    ds = build_probe_dataset_v3(require_activation=True)
    readiness = experiment_readiness(ds)
    if require_ready and not readiness["ready"]:
        return {
            "schema_version": "ccer_line_k_v1",
            "experiment_status": "blocked",
            "readiness": readiness,
        }

    instances = ds.instances
    y_all = np.array([r["y"] for r in instances], dtype=int)
    splits = [r["split"] for r in instances]
    quotes = [str(r.get("response_quote", "")) for r in instances]
    iaks = [str(r["instance_audit_key"]) for r in instances]

    layer_pick: dict[str, Any] = {"layer": 49, "selection_fallback": True}
    if LINE_A_V3_PROBE_SUMMARY.is_file():
        probe_summary = json.loads(LINE_A_V3_PROBE_SUMMARY.read_text(encoding="utf-8"))
        protocol = probe_summary.get("protocol") or {}
        layer_pick = protocol.get("layer_selected_on_dev") or layer_pick
        if protocol.get("layer_reported") is not None:
            layer_pick = {**layer_pick, "layer": int(protocol["layer_reported"])}
    layer = int(layer_pick["layer"])

    X, valid_idx = activation_matrix_for_instances(instances, position=LINE_A_PRIMARY_POSITION, layer=layer)
    if X is None or len(valid_idx) < 20:
        return {"schema_version": "ccer_line_k_v1", "experiment_status": "blocked", "message": "no vectors"}

    y = y_all[valid_idx]
    sub_splits = [splits[i] for i in valid_idx]
    sub_quotes = [quotes[i] for i in valid_idx]
    sub_instances = [instances[i] for i in valid_idx]

    probe_scores = _fit_probe_scores(X, y, sub_splits)
    bow_scores = _fit_bow_scores(sub_quotes, y, sub_splits)

    probe_map = {sub_instances[i]["instance_audit_key"]: float(probe_scores[i]) for i in range(len(valid_idx))}
    bow_map = {sub_instances[i]["instance_audit_key"]: float(bow_scores[i]) for i in range(len(valid_idx))}

    df_probe = instances_to_dataframe(sub_instances, probe_map)
    df_probe["p_bow"] = df_probe["claim_id"].map(lambda c: bow_map.get(str(c), 0.0))

    df_dev = df_probe[df_probe["split"] == "dev"].reset_index(drop=True)
    df_test = df_probe[df_probe["split"] == "test"].reset_index(drop=True)

    taus = DEFAULT_TAUS

    probe_tau_agg = _select_aggressive_tau(df_dev, "p_pm", taus)
    probe_tau_bal = _select_balanced_tau(df_dev, "p_pm", taus, min_cb=balanced_min_cb)
    bow_tau_agg = _select_aggressive_tau(df_dev, "p_bow", taus)
    bow_tau_bal = _select_balanced_tau(df_dev, "p_bow", taus, min_cb=balanced_min_cb)

    probe_agg_drop = _score_drop_factory("p_pm", probe_tau_agg)
    target_kept = kept_counts_per_group(df_test, probe_agg_drop)

    test_rows: list[dict[str, Any]] = []
    test_rows.append(
        _eval_method(
            df_test,
            "p_pm",
            probe_tau_agg,
            "line_a_probe_aggressive",
            boot_B=n_boot,
            seed=seed,
            random_match=target_kept,
        )
    )
    test_rows.append(
        _eval_method(
            df_test,
            "p_pm",
            probe_tau_bal,
            "line_a_probe_balanced",
            boot_B=n_boot,
            seed=seed,
        )
    )
    test_rows.append(
        _eval_method(df_test, "p_bow", bow_tau_agg, "quote_bow_aggressive", boot_B=n_boot, seed=seed)
    )
    test_rows.append(
        _eval_method(df_test, "p_bow", bow_tau_bal, "quote_bow_balanced", boot_B=n_boot, seed=seed)
    )

    phase2_rows = [
        _eval_rewrite(df_test, "p_pm", probe_tau_agg, "line_a_probe_aggressive_rewrite"),
        _eval_rewrite(df_test, "p_pm", probe_tau_bal, "line_a_probe_balanced_rewrite"),
    ]

    baseline_audit = audit_gating(df_test, lambda _r: False).to_dict()

    probe_agg = test_rows[0]
    passed_min = bool(
        probe_agg.get("pm_reduction", 0) > 0
        and probe_agg.get("random_retention_matched", {}).get("pm_rate_mean", 1.0)
        > probe_agg.get("gated_pm_rate", 1.0)
    )

    frozen_scores = export_frozen_probe_scores(sub_instances, probe_map, bow_map)
    dev_thresholds = {
        "line_a_probe_aggressive": probe_tau_agg,
        "line_a_probe_balanced": probe_tau_bal,
        "quote_bow_aggressive": bow_tau_agg,
        "quote_bow_balanced": bow_tau_bal,
    }

    return {
        "schema_version": "ccer_line_k_v1",
        "experiment_status": "completed",
        "frozen_artifacts": {
            "probe_scores": str(LINE_K_PROBE_SCORES),
            "dev_thresholds": str(LINE_K_DEV_THRESHOLDS),
            "n_score_rows": len(frozen_scores),
        },
        "protocol": {
            "version": "line_k_v1",
            "probe_source": "line_a_v3",
            "position": LINE_A_PRIMARY_POSITION,
            "layer": layer,
            "layer_selected_on_dev": layer_pick,
            "intervention": "fixed_anchor_post_hoc_claim_deletion",
            "threshold_split": "dev",
            "eval_split": "test",
            "balanced_min_cb_retention": balanced_min_cb,
            "note": "Does not cite original paper MSPS numbers; Line A probe replaces BCP.",
        },
        "readiness": readiness,
        "dataset_manifest": ds.manifest,
        "dev_thresholds": dev_thresholds,
        "dev_pareto": {
            "line_a_probe": _val_pareto(df_dev, "p_pm", taus, "line_a_probe"),
            "quote_bow": _val_pareto(df_dev, "p_bow", taus, "quote_bow"),
        },
        "test_baseline": baseline_audit,
        "test_results": test_rows,
        "phase2_rewrite_stub": {
            "rewrite_template": REWRITE_STUB,
            "note": "Optimistic rewrite bound: flagged PM claims no longer count as PM; flagged clean claims count as CB lost. Full PGCS requires judge API.",
            "results": phase2_rows,
        },
        "success_criterion": {
            "min_pm_reduction_gt_zero": probe_agg.get("pm_reduction", 0) > 0,
            "beats_random_at_same_retention": passed_min,
            "passed": passed_min,
        },
        "_write_frozen_scores": frozen_scores,
    }
