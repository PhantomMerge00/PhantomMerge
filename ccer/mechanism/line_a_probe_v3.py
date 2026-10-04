"""Line A v3: expanded clean cohort + CEM/CAP/AH breakdown."""
from __future__ import annotations

from typing import Any

import numpy as np

from ccer.mechanism.line_a_cohort_v3 import VERDICT_SHORT, build_probe_dataset_v3
from ccer.mechanism.line_a_instance_activations import activation_matrix_for_instances
from ccer.mechanism.supervised_probe import (
    ProbeDataset,
    bootstrap_diff_ci,
    experiment_readiness,
    normalize_quote_for_bow,
    train_bow_baseline,
    train_layer_position_probe,
    _auroc,
    _holdout_scores_bow,
    _holdout_scores_probe,
)
from ccer.replay.hf_forward import probe_layer_indices

PM_TYPES = ("cross_object_merge", "constraint_projection", "anchored_hallucination")
LINE_A_PRIMARY_POSITION = "claim_onset"


def _available_layers(ds: ProbeDataset) -> list[int]:
    layers: set[int] = set()
    for layer_list in ds.activation_layers.values():
        layers.update(layer_list)
    return sorted(layers & set(probe_layer_indices(64)))


def _select_layer_on_dev(
    ds: ProbeDataset,
    *,
    layers: list[int],
    position: str = LINE_A_PRIMARY_POSITION,
) -> dict[str, Any]:
    instances = ds.instances
    y_all = np.array([r["y"] for r in instances], dtype=int)
    splits = [r["split"] for r in instances]
    tids = [r["trajectory_id"] for r in instances]
    quotes = [normalize_quote_for_bow(r["response_quote"]) for r in instances]

    best: dict[str, Any] | None = None
    for layer in layers:
        X, valid_idx = activation_matrix_for_instances(instances, position=position, layer=layer)
        if X is None or len(valid_idx) < 10:
            continue
        y = y_all[valid_idx]
        sub_splits = [splits[i] for i in valid_idx]
        sub_tids = [tids[i] for i in valid_idx]
        sub_quotes = [quotes[i] for i in valid_idx]
        dev_idx = [i for i, s in enumerate(sub_splits) if s == "dev"]
        if len(dev_idx) < 2 or len(np.unique(y[dev_idx])) < 2:
            continue
        probe_h = _holdout_scores_probe(X, y, sub_splits, sub_tids, {"dev"})
        bow_h = _holdout_scores_bow(sub_quotes, y, sub_splits, sub_tids, {"dev"}, normalize_quotes=False)
        y_d = y[dev_idx]
        pa = _auroc(y_d, probe_h)
        ba = _auroc(y_d, bow_h)
        if pa is None or ba is None:
            continue
        delta = float(pa - ba)
        row = {"layer": layer, "dev_probe_auroc": pa, "dev_bow_auroc": ba, "dev_delta_auroc": delta}
        if best is None or delta > best["dev_delta_auroc"]:
            best = row
    if best is None:
        mid = layers[len(layers) // 2] if layers else 32
        return {"layer": mid, "selection_fallback": True}
    return best


def _subset_metrics(
    instances: list[dict[str, Any]],
    X: np.ndarray,
    y_all: np.ndarray,
    *,
    position: str,
    layer: int,
    splits: list[str],
    tids: list[str],
    quotes: list[str],
    eval_splits: set[str],
    mask: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    idx = [i for i, m in enumerate(mask) if m]
    if len(idx) < 8 or len(np.unique(y_all[idx])) < 2:
        return {"n": len(idx), "skipped": True}
    Xi = X[idx]
    yi = y_all[idx]
    sp = [splits[i] for i in idx]
    tp = [tids[i] for i in idx]
    qp = [quotes[i] for i in idx]
    probe = train_layer_position_probe(Xi, yi, sp, tp, eval_splits=eval_splits, n_boot=n_boot, seed=seed)
    bow = train_bow_baseline(qp, yi, sp, tp, eval_splits=eval_splits, n_boot=n_boot, seed=seed, normalize_quotes=False)
    holdout = sorted(eval_splits)[0]
    hi = [i for i, s in enumerate(sp) if s == holdout]
    if len(hi) < 2:
        return {"n": len(idx), "skipped": True}
    y_h = yi[hi]
    clusters = np.array([tp[i] for i in hi], dtype=str)
    ph = _holdout_scores_probe(Xi, yi, sp, tp, eval_splits)
    bh = _holdout_scores_bow(qp, yi, sp, tp, eval_splits, normalize_quotes=False)
    delta = bootstrap_diff_ci(y_h, ph, bh, clusters, n_boot=n_boot, seed=seed)
    pt = (probe.get("eval") or {}).get(holdout) or {}
    bt = (bow.get("eval") or {}).get(holdout) or {}
    return {
        "n": len(idx),
        "n_eval": pt.get("n"),
        "probe_auroc": pt.get("auroc"),
        "bow_auroc": bt.get("auroc"),
        "delta_auroc": delta.get("diff"),
        "delta_ci_95": delta.get("diff_ci_95"),
    }


def run_line_a_v3_probe_experiment(
    *,
    n_boot: int = 2000,
    seed: int = 42,
    require_ready: bool = True,
) -> dict[str, Any]:
    ds = build_probe_dataset_v3(require_activation=True)
    readiness = experiment_readiness(ds)
    if require_ready and not readiness["ready"]:
        return {
            "schema_version": "ccer_line_a_v3_final_v1",
            "experiment_status": "blocked_pending_activations",
            "readiness": readiness,
            "message": "Extract clean expansion activations first.",
        }

    instances = ds.instances
    y_all = np.array([r["y"] for r in instances], dtype=int)
    splits = [r["split"] for r in instances]
    tids = [r["trajectory_id"] for r in instances]
    quotes = [normalize_quote_for_bow(r["response_quote"]) for r in instances]
    sources = [r.get("cohort_source") for r in instances]
    verdicts = [r.get("gold_verdict") for r in instances]

    layers = _available_layers(ds)
    layer_pick = _select_layer_on_dev(ds, layers=layers, position=LINE_A_PRIMARY_POSITION)
    layer = int(layer_pick["layer"])
    position = LINE_A_PRIMARY_POSITION

    X, valid_idx = activation_matrix_for_instances(instances, position=position, layer=layer)
    if X is None or len(valid_idx) < 20:
        return {"schema_version": "ccer_line_a_v3_final_v1", "experiment_status": "blocked", "message": "no vectors"}

    y = y_all[valid_idx]
    sub_splits = [splits[i] for i in valid_idx]
    sub_tids = [tids[i] for i in valid_idx]
    sub_quotes = [quotes[i] for i in valid_idx]
    sub_sources = [sources[i] for i in valid_idx]
    sub_verdicts = [verdicts[i] for i in valid_idx]

    # --- Primary: expanded balanced test ---
    probe = train_layer_position_probe(X, y, sub_splits, sub_tids, eval_splits={"test"}, n_boot=n_boot, seed=seed)
    bow = train_bow_baseline(sub_quotes, y, sub_splits, sub_tids, eval_splits={"test"}, n_boot=n_boot, seed=seed, normalize_quotes=False)
    hi = [i for i, s in enumerate(sub_splits) if s == "test"]
    y_h = y[hi]
    clusters = np.array([sub_tids[i] for i in hi], dtype=str)
    ph = _holdout_scores_probe(X, y, sub_splits, sub_tids, {"test"})
    bh = _holdout_scores_bow(sub_quotes, y, sub_splits, sub_tids, {"test"}, normalize_quotes=False)
    delta_test = bootstrap_diff_ci(y_h, ph, bh, clusters, n_boot=n_boot, seed=seed)
    probe_test = (probe.get("eval") or {}).get("test") or {}
    bow_test = (bow.get("eval") or {}).get("test") or {}

    passed = bool(
        delta_test.get("diff") is not None
        and delta_test["diff"] >= 0.03
        and delta_test.get("diff_ci_95")
        and delta_test["diff_ci_95"][0] > 0
    )

    # --- Adjudication-only test (historical diagnostic; often underpowered) ---
    adj_mask = np.array(
        [str(instances[valid_idx[i]]["instance_audit_key"]).startswith("shopping_incremental:") for i in range(len(valid_idx))],
        dtype=bool,
    )
    adjudication_only = _subset_metrics(
        [instances[i] for i in valid_idx],
        X,
        y,
        position=position,
        layer=layer,
        splits=sub_splits,
        tids=sub_tids,
        quotes=sub_quotes,
        eval_splits={"test"},
        mask=adj_mask,
        n_boot=n_boot,
        seed=seed,
    )

    # --- PM type breakdown: each type vs expanded clean (test only) ---
    pm_type_breakdown: dict[str, Any] = {}
    for v in PM_TYPES:
        type_mask = np.array(
            [
                sub_verdicts[i] == v
                or (y[i] == 0 and sub_sources[i] == "gold_clean_expansion")
                for i in range(len(valid_idx))
            ],
            dtype=bool,
        )
        pm_type_breakdown[VERDICT_SHORT[v]] = _subset_metrics(
            [instances[i] for i in valid_idx],
            X,
            y,
            position=position,
            layer=layer,
            splits=sub_splits,
            tids=sub_tids,
            quotes=sub_quotes,
            eval_splits={"test"},
            mask=type_mask,
            n_boot=n_boot,
            seed=seed + hash(v) % 1000,
        )
        pm_type_breakdown[VERDICT_SHORT[v]]["verdict"] = v
        pm_type_breakdown[VERDICT_SHORT[v]]["n_type_test"] = sum(
            1 for i in range(len(valid_idx)) if sub_splits[i] == "test" and sub_verdicts[i] == v
        )

    return {
        "schema_version": "ccer_line_a_v3_final_v1",
        "experiment_status": "completed",
        "readiness": readiness,
        "dataset_manifest": ds.manifest,
        "protocol": {
            "version": "line_a_v3",
            "pm_source": "incremental_adjudication_commitment_eligible",
            "clean_source": "2k trajectory_outcome=clean (zero overlap with inc adj), pseudo quote span",
            "dropped": "adjudication correct_binding (34) replaced by gold-clean expansion",
            "primary_position": position,
            "layer_selected_on_dev": layer_pick,
            "layer_reported": layer,
            "eval_split": "test",
        },
        "primary_comparison_expanded_test": {
            "position": position,
            "layer": layer,
            "probe_test_auroc": probe_test.get("auroc"),
            "bow_test_auroc": bow_test.get("auroc"),
            "delta_auroc": delta_test.get("diff"),
            "delta_ci_95": delta_test.get("diff_ci_95"),
            "n_test": probe_test.get("n"),
            "n_pm_test": probe_test.get("n_pm"),
            "n_clean_test": probe_test.get("n_clean"),
        },
        "diagnostic_adjudication_only_test": adjudication_only,
        "pm_type_breakdown_vs_expanded_clean_test": pm_type_breakdown,
        "success_criterion": {"min_delta_auroc": 0.03, "ci_excludes_zero": True, "passed": passed},
    }
