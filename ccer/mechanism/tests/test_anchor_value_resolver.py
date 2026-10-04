"""Tests for anchor value resolver (Eq.2 s_{a_tau})."""
from __future__ import annotations

from ccer.mechanism.anchor_value_resolver import resolve_anchor_value


def test_clean_uses_obs_tau_dict() -> None:
    traj = {
        "commitment": {"action_anchor": "123"},
        "messages_final_call": [
            {"role": "user", "content": '{"product_id":"123","color":"Yellow green"}'},
        ],
        "evidence": [],
    }
    out = resolve_anchor_value(
        y_pm=0,
        response_quote="Color: Yellow green",
        adjudication_row=None,
        traj=traj,
    )
    assert out.provenance == "obs_tau_dict"
    assert out.v_anchor == "Yellow green"


def test_pm_obs_tau_dict_from_title() -> None:
    traj = {
        "commitment": {},
        "messages_final_call": [
            {
                "role": "user",
                "content": (
                    '{"product_id":"3010316109","title":"Product:X;Color: Red / Gray / Black / Gold;"}'
                ),
            }
        ],
        "evidence": [],
    }
    out = resolve_anchor_value(
        y_pm=1,
        response_quote="Color: Red",
        adjudication_row={"committed_anchor_pid": "3010316109", "slot_norm": "Color"},
        traj=traj,
    )
    assert out.provenance == "obs_tau_dict"
    assert "Red" in out.v_anchor


def test_pm_missing_when_obs_slot_absent() -> None:
    traj = {
        "messages_final_call": [
            {"role": "user", "content": '{"product_id":"999","title":"Product:Plain name only"}'},
        ],
        "evidence": [],
    }
    out = resolve_anchor_value(
        y_pm=1,
        response_quote="Color: Red",
        adjudication_row={"committed_anchor_pid": "999", "slot_norm": "Color"},
        traj=traj,
    )
    assert out.provenance == "missing"
    assert out.v_anchor is None


def test_pm_missing_without_adjudication() -> None:
    out = resolve_anchor_value(
        y_pm=1,
        response_quote="Color: Red",
        adjudication_row=None,
        traj={"evidence": []},
    )
    assert out.provenance == "missing"
    assert out.v_anchor is None
