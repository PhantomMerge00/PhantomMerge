"""Tests for Line L anchor extractive rewrite."""
from __future__ import annotations

from ccer.mechanism.line_l_anchor_extract import (
    literal_replace_in_quote,
    lookup_from_anchor_product_json,
    lookup_slot_value,
    parse_anchor_evidence_quote,
)


def test_parse_anchor_evidence_quote() -> None:
    fields = parse_anchor_evidence_quote("Brand: GTY;Color: Red;Material: Cotton")
    assert fields["color"] == "Red"
    assert fields["brand"] == "GTY"


def test_literal_replace_in_quote() -> None:
    out = literal_replace_in_quote(
        "Color: Beige (as per user request)",
        slot="Color",
        old_value="Beige",
        v_anchor="White/Black/Khaki",
    )
    assert "White/Black/Khaki" in out
    assert "Beige" not in out


def test_lookup_from_adjudication_quote() -> None:
    traj = {"evidence": []}
    res = lookup_slot_value(
        traj,
        anchor_pid="123",
        slot_norm="Color",
        anchor_evidence_quote="Brand: GTY;Color: Red;",
    )
    assert res.branch == "extraction_hit"
    assert res.v_anchor == "Red"


def test_lookup_from_anchor_product_json() -> None:
    traj = {
        "evidence": [],
        "messages_final_call": [
            {
                "role": "user",
                "content": (
                    'results:[{"product_id":"3233041636","title":"Shell",'
                    '"price":139.0,"shop_id":"1589992"}]'
                ),
            }
        ],
    }
    res = lookup_from_anchor_product_json(
        traj, anchor_pid="3233041636", slot_norm="Price"
    )
    assert res.branch == "extraction_hit"
    assert res.v_anchor == "139.0"
    assert res.source_field == "product_json.price"
