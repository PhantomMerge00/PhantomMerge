"""Tests for Line L+ quality, copy, and tiered v3."""
from __future__ import annotations

from ccer.mechanism.line_k_l_tiered import apply_selective_grounded_v3
from ccer.mechanism.line_l_anchor_extract import lookup_from_anchor_product_json, lookup_slot_value
from ccer.mechanism.line_l_baseline_copy import apply_copy_baseline_rewrite
from ccer.mechanism.line_l_rewrite_acceptance import acceptance_verify_bundle
from ccer.mechanism.line_l_rewrite_quality import build_evidence_corpus, format_valid, is_stub, verify_grounded
from ccer.mechanism.line_l_slot_nli import set_slot_entail_checker


def test_format_valid_rejects_slot_prefix() -> None:
    assert not format_valid("Slot: Blue", "Color")
    assert format_valid("Color: Blue", "Color")


def test_verify_grounded() -> None:
    corpus = '{"product_id":"1","color":"Red"}'
    assert verify_grounded("Red", corpus)
    assert not verify_grounded("Blue", corpus)


def test_nested_product_record_lookup() -> None:
    traj = {
        "evidence": [
            {
                "entity_ids": ["123"],
                "slot_norm": "product_record",
                "value_raw": '{"product_id":"123","material":"Cotton","price":9.9}',
            }
        ],
        "messages_final_call": [],
    }
    res = lookup_slot_value(traj, anchor_pid="123", slot_norm="Material", row=None)
    assert res.branch == "extraction_hit"
    assert res.v_anchor == "Cotton"


def test_blind_audit_macbook_color_blob_rejected() -> None:
    set_slot_entail_checker(None)
    traj = {
        "messages_final_call": [
            {
                "role": "user",
                "content": '{"product_id":"1","title":"MacBook Sleeve","color":"Gray"}',
            }
        ],
    }
    corpus = build_evidence_corpus(traj, "1")
    long_desc = (
        "MacBook Pro 16 inch sleeve premium leather fits 2023 models with magnetic closure"
    )
    bundle = acceptance_verify_bundle(
        long_desc,
        "Color",
        corpus,
        acceptance_pid="1",
        candidate_source="obs_tau_window",
    )
    assert not bundle["ok"]


def test_copy_baseline_hit() -> None:
    traj = {
        "evidence": [],
        "messages_final_call": [
            {
                "role": "user",
                "content": '{"product_id":"1","title":"T","price":19.9}',
            }
        ],
    }
    row = {
        "p_pm": 0.9,
        "tau": 0.05,
        "slot_norm": "Price",
        "slot": "Price",
        "claim_value": "99.9",
        "response_quote": "Price: 99.9",
        "gold_verdict": "constraint_projection",
    }
    out = apply_copy_baseline_rewrite(row, traj, anchor_pid="1")
    assert out["action"] == "rewrite"
    assert "19.9" in str(out.get("quote_after") or "")


def test_selective_v3_clean_skip() -> None:
    row = {
        "p_pm": 0.9,
        "tau": 0.05,
        "slot_norm": "",
        "response_quote": "Material: X",
        "gold_verdict": "trajectory_clean",
    }
    out = apply_selective_grounded_v3(row, {"evidence": []}, anchor_pid="1")
    assert out.get("branch") in ("skip_routed", "skip_unflagged")
