"""Export and load shopping train-split frozen probe for tau3 zero-shot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ccer.mechanism.activation_store import get_vector, load_activation_npz
from ccer.mechanism.agr.signals import AgrProbeModel
from ccer.mechanism.bind_surprise import try_auroc
from ccer.mechanism.line_a_cohort_v3 import build_probe_dataset_v3
from ccer.mechanism.line_a_instance_activations import instance_npz_path as shopping_npz_path
from ccer.paths import TAU3_SHOPPING_FROZEN_PROBE

LINE_A_LAYER = 49
LINE_A_POSITION = "claim_onset"


def fit_shopping_frozen_probe() -> AgrProbeModel:
    """Fit probe on shopping train split (same protocol as line_k_claim_filter)."""
    ds = build_probe_dataset_v3(require_activation=True)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for row in ds.instances:
        if str(row.get("split")) != "train":
            continue
        path = shopping_npz_path(str(row["instance_audit_key"]), str(row["trajectory_id"]))
        if not path.is_file():
            continue
        loaded = load_activation_npz(path)
        h = get_vector(loaded, position=LINE_A_POSITION, layer=LINE_A_LAYER)
        if h is None:
            continue
        X_list.append(h)
        y_list.append(int(row["y"]))
    if not X_list:
        raise RuntimeError("No shopping train activations for frozen probe")
    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(Xs, y)
    return AgrProbeModel(scaler=scaler, clf=clf, layer=LINE_A_LAYER, position=LINE_A_POSITION)


def export_shopping_frozen_probe(
    *,
    out_path: Path = TAU3_SHOPPING_FROZEN_PROBE,
) -> dict[str, Any]:
    probe = fit_shopping_frozen_probe()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(probe, out_path)

    ds = build_probe_dataset_v3(require_activation=True)
    n_train = sum(1 for row in ds.instances if str(row.get("split")) == "train")
    scores: list[dict[str, Any]] = []
    for row in ds.instances:
        if str(row.get("split")) != "test":
            continue
        path = shopping_npz_path(str(row["instance_audit_key"]), str(row["trajectory_id"]))
        if not path.is_file():
            continue
        loaded = load_activation_npz(path)
        h = get_vector(loaded, position=LINE_A_POSITION, layer=LINE_A_LAYER)
        if h is None:
            continue
        p = probe.prob(h)
        scores.append(
            {
                "instance_audit_key": row["instance_audit_key"],
                "split": row["split"],
                "y_pm": int(row["y"]),
                "p_pm": p,
            }
        )
    auroc = try_auroc(scores, "p_pm")
    summary = {
        "out_path": str(out_path),
        "n_train_fit": n_train,
        "shopping_test_auroc": auroc,
        "layer": LINE_A_LAYER,
        "position": LINE_A_POSITION,
    }
    return summary


def load_shopping_frozen_probe(
    *,
    path: Path = TAU3_SHOPPING_FROZEN_PROBE,
) -> AgrProbeModel:
    if not path.is_file():
        export_shopping_frozen_probe(out_path=path)
    probe = joblib.load(path)
    if not isinstance(probe, AgrProbeModel):
        raise TypeError(f"Expected AgrProbeModel, got {type(probe)}")
    return probe


if __name__ == "__main__":
    print(json.dumps(export_shopping_frozen_probe(), indent=2))
