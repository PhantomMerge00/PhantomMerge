"""Tests for BindSurprise unified score."""
from __future__ import annotations

import math

from ccer.mechanism.bind_surprise import (
    compute_anchor_value_workspace_mass,
    compute_bind_surprise,
    compute_slot_workspace_mass,
    fit_tau_b,
    probe_logit,
)


def test_probe_logit_extremes() -> None:
    assert probe_logit(0.5) == 0.0
    assert probe_logit(0.9) > 0


def test_anchor_value_workspace_mass_hit() -> None:
    topk = [
        {"token": " Yellow", "logit": 4.0, "rank": 1},
        {"token": " green", "logit": 3.5, "rank": 2},
        {"token": " xyz", "logit": 0.5, "rank": 3},
    ]
    out = compute_anchor_value_workspace_mass(topk, "Yellow green")
    assert out["anchor_value_mass"] > 0.5
    assert out["log_anchor_value_mass"] is not None
    assert len(out["anchor_value_hits"]) >= 1


def test_anchor_value_workspace_mass_miss() -> None:
    topk = [{"token": " xyz", "logit": 1.0, "rank": 1}]
    out = compute_anchor_value_workspace_mass(topk, "anchor_val")
    assert out["anchor_value_mass"] == 0.0
    assert out["log_anchor_value_mass"] == float("-inf")


def test_slot_workspace_mass() -> None:
    topk = [
        {"token": " Price", "logit": 5.0, "rank": 1},
        {"token": " xyz", "logit": 1.0, "rank": 2},
    ]
    out = compute_slot_workspace_mass(topk, "price")
    assert out["slot_mass"] > 0.9
    assert out["log_slot_mass"] is not None


def test_bind_surprise_additive_in_log_space() -> None:
    bs = compute_bind_surprise(0.9, -1.0)
    assert math.isclose(bs["bind_surprise"], bs["probe_logit"] + bs["log_slot_mass"])


def test_fit_tau_b_improves_on_synthetic() -> None:
    rows = [
        {"bind_surprise": 5.0, "y_pm": 1},
        {"bind_surprise": 4.0, "y_pm": 1},
        {"bind_surprise": -1.0, "y_pm": 0},
        {"bind_surprise": -2.0, "y_pm": 0},
    ]
    fit = fit_tau_b(rows)
    assert fit["dev_f1"] == 1.0
