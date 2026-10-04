"""Tests for AGR significance helpers."""
from __future__ import annotations

import numpy as np

from ccer.mechanism.agr.significance import delong_test


def test_delong_identical_scores_high_pvalue() -> None:
    y = np.array([0, 0, 1, 1, 1, 0])
    s = np.array([0.1, 0.2, 0.8, 0.7, 0.9, 0.3])
    out = delong_test(y, s, s)
    assert out["pvalue"] is not None
    assert out["pvalue"] > 0.5
