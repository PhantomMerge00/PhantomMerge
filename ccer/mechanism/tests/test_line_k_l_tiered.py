"""Tests for Track K+L tiered delete-or-rewrite routing."""
from __future__ import annotations

from ccer.mechanism.line_k_l_tiered import apply_tiered_mitigation, compute_tiered_routing_stats


def test_tiered_rewrite_on_extraction_hit() -> None:
    traj = {
        "evidence": [],
        "messages_final_call": [
            {
                "role": "user",
                "content": '{"product_id":"1","title":"Real Title","price":9.9}',
            }
        ],
    }
    row = {
        "p_pm": 0.9,
        "tau": 0.05,
        "slot_norm": "Price",
        "slot": "Price",
        "claim_value": "19.9",
        "response_quote": "Price: 19.9",
        "anchor_evidence_quote": "",
        "gold_verdict": "cross_object_merge",
    }
    out = apply_tiered_mitigation(row, traj, anchor_pid="1")
    assert out["mitigation_tier"] == "rewrite"
    assert out["action"] == "rewrite"


def test_tiered_fallback_rewrite_on_extraction_miss_v2() -> None:
    traj = {"evidence": [], "messages_final_call": [{"role": "user", "content": ""}]}
    row = {
        "p_pm": 0.9,
        "tau": 0.05,
        "slot_norm": "Price",
        "slot": "Price",
        "claim_value": "19.9",
        "response_quote": "Price: 19.9",
        "anchor_evidence_quote": "",
        "gold_verdict": "constraint_projection",
    }
    out = apply_tiered_mitigation(row, traj, anchor_pid="1")
    assert out["mitigation_tier"] == "rewrite"
    assert out["action"] == "rewrite"
    assert out["branch"] == "fallback_rewrite"


def test_tiered_delete_on_miss_v1_policy() -> None:
    traj = {"evidence": [], "messages_final_call": [{"role": "user", "content": ""}]}
    row = {
        "p_pm": 0.9,
        "tau": 0.05,
        "slot_norm": "Price",
        "response_quote": "Price: 19.9",
        "gold_verdict": "constraint_projection",
    }
    out = apply_tiered_mitigation(
        row, traj, anchor_pid="1", policy="delete_or_rewrite_v1", rewrite_fallback="none"
    )
    assert out["mitigation_tier"] == "delete"


def test_prism_gated_probe_only_deletes() -> None:
    traj = {"evidence": [], "messages_final_call": [{"role": "user", "content": ""}]}
    row = {
        "p_pm": 0.9,
        "tau": 0.05,
        "slot_norm": "price",
        "response_quote": "Price: 19.9",
        "prism_audit_state": "probe_only",
        "prism_slot_type_aligned": False,
    }
    out = apply_tiered_mitigation(row, traj, anchor_pid="1", policy="prism_gated_v1")
    assert out["mitigation_tier"] == "delete"
    assert out["prism_audit_state"] == "probe_only"


def test_prism_gated_confirmed_risk_rewrites() -> None:
    traj = {
        "evidence": [],
        "messages_final_call": [
            {"role": "user", "content": '{"product_id":"1","title":"Real Title","price":9.9}'}
        ],
    }
    row = {
        "p_pm": 0.9,
        "tau": 0.05,
        "slot_norm": "price",
        "claim_value": "19.9",
        "response_quote": "Price: 19.9",
        "prism_audit_state": "confirmed_risk",
        "prism_slot_type_aligned": True,
    }
    out = apply_tiered_mitigation(row, traj, anchor_pid="1", policy="prism_gated_v1")
    assert out["mitigation_tier"] == "rewrite"
    assert out["prism_audit_state"] == "confirmed_risk"


def test_tiered_routing_stats() -> None:
    rows = [
        {"p_pm": 0.9, "mitigation_tier": "rewrite", "gold_verdict": "cross_object_merge"},
        {"p_pm": 0.9, "mitigation_tier": "delete", "gold_verdict": "constraint_projection"},
        {"p_pm": 0.01, "mitigation_tier": "keep", "gold_verdict": "trajectory_clean"},
    ]
    stats = compute_tiered_routing_stats(rows, tau=0.05)
    assert stats["n_flagged"] == 2
    assert stats["n_rewrite"] == 1
    assert stats["n_delete"] == 1
