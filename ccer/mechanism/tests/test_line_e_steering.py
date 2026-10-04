"""Unit tests for Line E CAA steering (no GPU)."""
from __future__ import annotations

import numpy as np

from ccer.mechanism.steering_vector import (
    assign_build_eval_holdout,
    build_caa_vector,
    build_steering_controls,
    clean_pool_with_activations,
    pm_pool_with_activations,
)


def test_pm_and_clean_pools_non_empty():
    pm = pm_pool_with_activations()
    clean = clean_pool_with_activations()
    assert len(pm) >= 20
    assert len(clean) >= 20


def test_holdout_no_overlap():
    pm = pm_pool_with_activations()
    holdout = assign_build_eval_holdout(pm, holdout_frac=0.35, seed=42)
    build = set(holdout["build_pm_trajectory_ids"])
    eval_set = set(holdout["eval_pm_trajectory_ids"])
    assert not build & eval_set
    assert len(build) + len(eval_set) == len(pm)
    assert len(eval_set) >= 1


def test_caa_vector_and_controls():
    pm = pm_pool_with_activations()[:12]
    clean = clean_pool_with_activations()[:12]
    caa = build_caa_vector(pm, clean, layer=32, position="commitment")
    assert not isinstance(caa, dict)
    assert caa.vector.shape[0] == 5120
    assert caa.vector_norm > 0
    controls = build_steering_controls(caa.vector, seed=0)
    for key in ("caa_true", "caa_random", "caa_orthogonal"):
        assert key in controls
        assert abs(float(np.linalg.norm(controls[key])) - caa.vector_norm) < 1e-3
    orth = controls["caa_orthogonal"]
    cos = abs(float(np.dot(orth, caa.vector) / (np.linalg.norm(orth) * caa.vector_norm)))
    assert cos < 0.05
