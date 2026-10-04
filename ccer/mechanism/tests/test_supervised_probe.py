"""Unit tests for Line A supervised probe (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.mechanism.line_a_instance_activations import build_adjudication_instance_dataset
from ccer.mechanism.supervised_probe import (
    PM_VERDICTS,
    ProbeDataset,
    bootstrap_auroc_ci,
    experiment_readiness,
    line_a_trajectory_ids,
    train_bow_baseline,
    train_layer_position_probe,
)
from ccer.replay.hf_forward import probe_layer_indices


def test_probe_layer_indices_64():
    layers = probe_layer_indices(64)
    assert layers[0] >= 13
    assert layers[-1] <= 53
    assert 54 not in layers
    assert 63 not in layers
    assert 16 in layers
    assert 32 in layers


def test_probe_layer_indices_small():
    assert probe_layer_indices(0) == [0]
    assert probe_layer_indices(1) == [0]


def test_line_a_trajectory_ids_non_empty():
    tids = line_a_trajectory_ids()
    assert len(tids) >= 600


def test_build_probe_dataset_schema():
    ds = build_adjudication_instance_dataset(require_activation=False)
    assert ds.instances
    assert "y" in ds.instances[0]
    assert "split" in ds.instances[0]
    assert "trajectory_id" in ds.instances[0]
    splits = {r["split"] for r in ds.instances}
    assert "train" in splits
    assert "test" in splits


def test_trajectory_split_no_leakage():
    ds = build_adjudication_instance_dataset(require_activation=False)
    train_tids = {r["trajectory_id"] for r in ds.instances if r["split"] == "train"}
    test_tids = {r["trajectory_id"] for r in ds.instances if r["split"] == "test"}
    assert not train_tids & test_tids


def test_bootstrap_auroc_ci_smoke():
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    scores = y.astype(float) + rng.normal(0, 0.1, size=len(y))
    clusters = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    out = bootstrap_auroc_ci(y, scores, clusters, n_boot=200, seed=0)
    assert out["auroc"] is not None
    assert out["auroc_ci_95"] is not None
    assert out["auroc_ci_95"][0] <= out["auroc"] <= out["auroc_ci_95"][1]


def test_tfidf_fit_only_on_train():
    texts = ["alpha beta", "gamma delta", "alpha gamma", "delta epsilon"]
    y = np.array([0, 1, 0, 1])
    splits = ["train", "train", "test", "test"]
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    test_idx = [i for i, s in enumerate(splits) if s == "test"]
    vec = TfidfVectorizer()
    vec.fit([texts[i] for i in train_idx])
    X_test = vec.transform([texts[i] for i in test_idx])
    assert X_test.shape[0] == 2
    with pytest.raises(ValueError):
        vec.transform([]).toarray()


def test_train_probe_synthetic():
    rng = np.random.default_rng(42)
    n = 40
    X = rng.normal(size=(n, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    splits = ["train"] * 24 + ["test"] * 16
    tids = [f"t{i // 4}" for i in range(n)]
    result = train_layer_position_probe(
        X,
        y,
        splits,
        tids,
        eval_splits={"test"},
        n_boot=100,
        seed=0,
    )
    assert result["train"]["auroc"] is not None
    assert result["eval"]["test"]["auroc"] is not None


def test_normalize_quote_strips_prefix():
    from ccer.mechanism.supervised_probe import normalize_quote_for_bow

    assert normalize_quote_for_bow("Size: S") == "s"
    assert normalize_quote_for_bow("Color: Black") == "black"
    assert normalize_quote_for_bow("11*16*6cm") == "<num>*<num>*<num>cm"
    assert "size:" not in normalize_quote_for_bow("Size: Large (11cm)")


def test_train_bow_synthetic():
    texts = [f"claim text {i} with words" for i in range(40)]
    y = np.array([i % 2 for i in range(40)])
    splits = ["train"] * 24 + ["test"] * 16
    tids = [f"t{i // 4}" for i in range(40)]
    result = train_bow_baseline(
        texts,
        y,
        splits,
        tids,
        eval_splits={"test"},
        n_boot=100,
        seed=0,
    )
    assert result["eval"]["test"]["auroc"] is not None


def test_experiment_readiness_blocks_on_sparse_train():
    ds = build_adjudication_instance_dataset(require_activation=False)
    readiness = experiment_readiness(ds)
    assert readiness["ready"] is True
    sparse = ProbeDataset(instances=ds.instances[:5], manifest=ds.manifest)
    blocked = experiment_readiness(sparse)
    assert blocked["ready"] is False
    assert blocked["issues"]
