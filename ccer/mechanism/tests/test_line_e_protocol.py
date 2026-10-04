"""Unit tests for Line E v4 ultimate protocol gates."""
from __future__ import annotations

import numpy as np

from ccer.mechanism.activation_store import save_activation_npz
from ccer.mechanism.line_e_donor import bilateral_activation_gate, trajectory_commitment_stratum
from ccer.mechanism.line_e_protocol import build_trajectory_steering_bundle, line_e_v3_config
from ccer.replay.live_position import LINE_D_REFRESH_TAG


def _write_npz(path, *, tid: str, claim_idx: int = 100):
    save_activation_npz(
        path,
        trajectory_id=tid,
        condition_id="original",
        cohort="CEM",
        layer_indices=[49],
        positions={"claim_onset": claim_idx, "pre_value": claim_idx - 1},
        vectors={"claim_onset": {"layer_49": np.zeros(8, dtype=np.float16)}},
        metadata={LINE_D_REFRESH_TAG: True, "claim_anchor_mode": "symmetric"},
    )


def test_trajectory_commitment_stratum():
    assert trajectory_commitment_stratum({"commitment": {"commitment_relation": "mismatch"}}) == "commitment_mismatch"
    assert trajectory_commitment_stratum({"commitment": {"commitment_relation": "consistent"}}) == "commitment_consistent"


def test_bilateral_gate_requires_both_sides(tmp_path, monkeypatch):
    from ccer.mechanism.activation_store import activation_path

    def _fake_path(tid, condition_id="original"):
        return tmp_path / f"{tid}.npz"

    monkeypatch.setattr("ccer.mechanism.line_e_donor.activation_path", _fake_path)
    _write_npz(tmp_path / "pm.npz", tid="pm")
    _write_npz(tmp_path / "clean.npz", tid="clean")
    ok, reason = bilateral_activation_gate("pm", "clean", layer=49, position="claim_onset")
    assert ok
    assert reason == "ok"


def test_v4_config_no_legacy_fallback_by_default():
    cfg = line_e_v3_config()
    assert cfg.strict_bilateral_gate is True
    assert cfg.allow_legacy_fallback is False
    assert "slot_consistent_restore" in cfg.control_ids
