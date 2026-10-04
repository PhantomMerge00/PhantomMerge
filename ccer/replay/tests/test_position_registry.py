"""Unit tests for claim_onset / pre_value position resolution."""
from __future__ import annotations

from ccer.replay.position_registry import (
    _answer_anchored_claim,
    _first_attribute_bullet,
    _resolve_claim_onset_pre_value,
    _selected_product_id_line,
    resolve_positions,
    resolve_symmetric_claim_pair,
)


def _mock_offsets(text: str) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(len(text))]


def test_answer_anchored_claim_prefers_constraint_projection():
    traj = {
        "claims": [
            {
                "legacy_label": "cross_object_merge",
                "response_span": "Price: 999.0 PHP",
                "value": "999.0 PHP",
            },
            {
                "legacy_label": "constraint_projection",
                "response_span": "Color: Pink",
                "value": "Pink",
            },
        ]
    }
    body = "### Selected product ID: 1\n\n### About the selected product\n- Color: Pink\n- Price: 159 pesos\n"
    pair = _answer_anchored_claim(traj, body)
    assert pair == ("Color: Pink", "Pink")


def test_first_attribute_bullet_parses_clean_style_answer():
    body = (
        "### Selected product ID: 2152663319\n\n"
        "### About the selected product\n"
        "- Price: 188.0 PHP\n"
        "- Service: Free shipping\n"
    )
    pair = _first_attribute_bullet(body)
    assert pair == ("Price: 188.0 PHP", "188.0 PHP")


def test_selected_product_id_line_fallback():
    body = "### Selected product ID: 2152663319\n"
    pair = _selected_product_id_line(body)
    assert pair is not None
    assert pair[1] == "2152663319"


def test_resolve_claim_onset_pre_value_on_attribute_bullet():
    body = (
        "### Selected product ID: 2152663319\n\n"
        "### About the selected product\n"
        "- Price: 188.0 PHP\n"
    )
    serialized = f"<response>\n{body}</response>"
    onset, pre_val, errors = _resolve_claim_onset_pre_value(
        answer_body=body,
        serialized_text=serialized,
        offset_mapping=_mock_offsets(serialized),
        response_span="Price: 188.0 PHP",
        value="188.0 PHP",
    )
    assert errors == []
    assert onset is not None
    assert pre_val is not None
    assert onset < pre_val


def test_symmetric_mode_prefers_about_bullet_over_metadata():
    traj = {
        "claims": [
            {
                "legacy_label": "constraint_projection",
                "response_span": "Color: Pink",
                "value": "Pink",
            }
        ],
        "commitment": {"action_anchor": "add_to_cart"},
        "metadata": {
            "final_answer": (
                "<response>\n"
                "### Selected product ID: 1\n\n"
                "### About the selected product\n"
                "- Price: 188.0 PHP\n"
                "</response>"
            )
        },
    }
    body = (
        "### Selected product ID: 1\n\n"
        "### About the selected product\n"
        "- Price: 188.0 PHP\n"
    )
    serialized = f"prompt\n<response>\n{body}</response>"
    res = resolve_positions(
        trajectory=traj,
        serialized_text=serialized,
        offset_mapping=_mock_offsets(serialized),
        prompt_token_count=10,
        full_token_count=len(serialized),
        claim_anchor_mode="symmetric",
    )
    assert res["claim_anchor_source"] == "about_first_bullet"
    pair = resolve_symmetric_claim_pair(body)
    assert pair is not None
    assert pair[2] == "about_first_bullet"


def test_clean_trajectory_without_claims_gets_positions():
    traj = {
        "commitment": {"action_anchor": "add_to_cart"},
        "metadata": {
            "final_answer": (
                "<response>\n"
                "### Selected product ID: 2152663319\n\n"
                "### About the selected product\n"
                "- Price: 188.0 PHP\n"
                "</response>"
            )
        },
    }
    body = (
        "### Selected product ID: 2152663319\n\n"
        "### About the selected product\n"
        "- Price: 188.0 PHP\n"
    )
    serialized = f"prompt text\n<response>\n{body}</response>"
    res = resolve_positions(
        trajectory=traj,
        serialized_text=serialized,
        offset_mapping=_mock_offsets(serialized),
        prompt_token_count=10,
        full_token_count=len(serialized),
    )
    assert res["positions"]["claim_onset"] is not None
    assert res["positions"]["pre_value"] is not None
