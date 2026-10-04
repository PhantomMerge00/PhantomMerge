"""Tests for Track K claim filter audit."""
from __future__ import annotations

import pandas as pd

from ccer.mechanism.claim_filter_audit import audit_gating, audit_rewrite


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"group_id": "t1", "claim_id": "c1", "y_pm": 1, "gold_verdict": "constraint_projection", "p_pm": 0.9},
            {"group_id": "t1", "claim_id": "c2", "y_pm": 0, "gold_verdict": "trajectory_clean", "p_pm": 0.1},
            {"group_id": "t2", "claim_id": "c3", "y_pm": 0, "gold_verdict": "trajectory_clean", "p_pm": 0.2},
        ]
    )


def test_deletion_drops_pm_trajectory() -> None:
    df = _df()
    audit = audit_gating(df, lambda r: float(r["p_pm"]) > 0.5)
    assert audit.baseline_pm_count == 1
    assert audit.gated_pm_count == 0
    assert audit.to_dict()["pm_reduction"] == 0.5
    assert audit.cb_retention_rate == 1.0


def test_rewrite_keeps_all_claims() -> None:
    df = _df()
    audit = audit_rewrite(df, lambda r: float(r["p_pm"]) > 0.5)
    assert audit.claims_retained_mean == 1.5
    assert audit.gated_pm_count == 0
    assert audit.pm_claims_removed == 1
