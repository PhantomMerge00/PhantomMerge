"""Unit tests for Task I joint patch helpers."""
from __future__ import annotations

import numpy as np

from ccer.mechanism.interchange import (
    InterchangeSpec,
    JointInterventionSlot,
    _joint_slot_group_key,
    _slot_active,
    _slot_patch_positions,
)
from ccer.mechanism.joint_patch import (
    JOINT_ARMS,
    _aggregate_arm_row,
    _infer_verdict,
    positions_for_arm,
)
from ccer.mechanism.round8_verify import _specificity_block


def _make_spec(*, layer: int, steering_apply: str = "anchor_once") -> InterchangeSpec:
    dim = 8
    return InterchangeSpec(
        control_id="target_interchange",
        layer=layer,
        position_token_idx=5,
        alpha=1.0,
        U=np.eye(dim, 4, dtype=np.float32),
        h_donor=np.ones(dim, dtype=np.float32),
        h_recipient=np.zeros(dim, dtype=np.float32),
        steering_apply=steering_apply,
    )


def test_joint_slot_grouping_splits_l32_l49():
    slots = [
        JointInterventionSlot("prompt_end", _make_spec(layer=32), anchor_pos=9),
        JointInterventionSlot("commitment", _make_spec(layer=32, steering_apply="generation_decode"), anchor_pos=100),
        JointInterventionSlot("claim_onset", _make_spec(layer=49, steering_apply="dynamic_answer_anchor"), anchor_pos=200),
        JointInterventionSlot("pre_value", _make_spec(layer=49, steering_apply="dynamic_answer_anchor"), anchor_pos=199),
    ]
    keys = {_joint_slot_group_key(s) for s in slots}
    assert keys == {(32, "residual"), (49, "residual")}


def test_positions_for_arm_minus_one():
    full = positions_for_arm("joint_target_interchange")
    assert full == ("prompt_end", "commitment", "claim_onset", "pre_value")
    minus = positions_for_arm("joint_minus_commitment")
    assert "commitment" not in minus
    assert len(minus) == 3


def test_slot_patch_positions_prompt_end_and_commitment():
    prompt = JointInterventionSlot(
        "prompt_end",
        _make_spec(layer=32, steering_apply="anchor_once"),
        anchor_pos=9,
    )
    seq = list(range(20))
    assert _slot_patch_positions(prompt, seq, prompt_len=10, decode_step=False) == [9]
    commit = JointInterventionSlot(
        "commitment",
        _make_spec(layer=32, steering_apply="generation_decode"),
        anchor_pos=100,
    )
    assert _slot_patch_positions(commit, seq, prompt_len=10, decode_step=True) == [-1]


def test_all_patched_gate_in_aggregate():
    details = [
        {
            "all_patched": True,
            "product_id_diff": True,
            "output_diff": True,
            "selected_product_block_diff": False,
            "patch_fallback_used": {"claim_onset": "decode_head"},
        },
        {
            "all_patched": False,
            "product_id_diff": True,
            "output_diff": True,
            "selected_product_block_diff": True,
            "patch_fallback_used": {},
        },
    ]
    row = _aggregate_arm_row(details)
    assert row["n_eligible"] == 2
    assert row["n_valid"] == 1
    assert row["n_invalid"] == 1
    assert row["n_product_id_diff"] == 1
    assert row["patch_fallback_counts"]["claim_onset:decode_head"] == 1


def test_specificity_block_on_valid_subset():
    ti = [
        {"pm_trajectory_id": "a", "product_id_diff": True, "selected_product_block_diff": False},
        {"pm_trajectory_id": "b", "product_id_diff": False, "selected_product_block_diff": False},
    ]
    wo = [
        {"pm_trajectory_id": "a", "product_id_diff": False, "selected_product_block_diff": False},
        {"pm_trajectory_id": "b", "product_id_diff": False, "selected_product_block_diff": False},
    ]
    spec = _specificity_block(ti, wo)
    assert spec["product_id_correct_only"] == 1
    assert spec["product_id_wrong_only"] == 0


def test_infer_verdict_null_with_valid_patch():
    target = {"n_valid": 19, "n_invalid": 0, "product_id_diff_rate": 0.0}
    wrong = {"n_valid": 19, "product_id_diff_rate": 0.0}
    spec = {"product_id_correct_only": 0}
    assert _infer_verdict(target_row=target, wrong_row=wrong, specificity=spec) == "null_joint_with_valid_patch"


def test_infer_verdict_invalid_engineering():
    target = {"n_valid": 0, "n_invalid": 5, "product_id_diff_rate": 0.0}
    wrong = {"n_valid": 0, "product_id_diff_rate": 0.0}
    spec = {"product_id_correct_only": 0}
    assert _infer_verdict(target_row=target, wrong_row=wrong, specificity=spec) == "invalid_patch_engineering"


def test_joint_arms_count():
    assert len(JOINT_ARMS) == 6


def test_inactive_slot_not_active():
    spec = _make_spec(layer=32)
    spec.control_id = "no_intervention"
    slot = JointInterventionSlot("prompt_end", spec, anchor_pos=9)
    assert not _slot_active(slot)
