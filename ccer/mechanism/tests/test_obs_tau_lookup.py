"""Tests for Obs_τ(a_τ) structured dictionary lookup."""
from __future__ import annotations

from ccer.mechanism.line_l_anchor_extract import build_obs_tau_dict, lookup_obs_tau_slot


def test_title_kv_lookup() -> None:
    traj = {
        "messages_final_call": [
            {
                "role": "user",
                "content": '{"product_id":"1","title":"Brand: GTY;Color: White/Black;Size: S,M,L;"}',
            }
        ],
        "evidence": [],
    }
    obs = build_obs_tau_dict(traj, "1")
    assert obs["color"] == "White/Black"
    hit = lookup_obs_tau_slot(traj, anchor_pid="1", slot_norm="Color")
    assert hit.branch == "extraction_hit"
    assert hit.v_anchor == "White/Black"


def test_structured_evidence_lookup() -> None:
    traj = {
        "messages_final_call": [],
        "evidence": [
            {
                "entity_ids": ["42"],
                "slot_norm": "material",
                "value_raw": "Cotton",
                "evidence_kind": "tool_observation",
            }
        ],
    }
    hit = lookup_obs_tau_slot(traj, anchor_pid="42", slot_norm="Material")
    assert hit.branch == "extraction_hit"
    assert hit.v_anchor == "Cotton"
