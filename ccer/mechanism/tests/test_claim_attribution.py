"""Unit tests for Task H claim attribution parser."""
from __future__ import annotations

from ccer.mechanism.claim_attribution import (
    evaluate_claim_attribution_pair,
    parse_claim_attribution_from_text,
)
from ccer.mechanism.pair_select import cem_valid_pair_ids, load_trajectory_index


def _wrap(body: str) -> str:
    return f"<response>\n{body}\n</response>"


def test_about_selected_claim_targets_selected_pid():
    text = _wrap(
        "### Selected product ID: 4641170502\n\n"
        "### About the selected product\n"
        "- Blades: 5\n"
        "- Material: Plastic\n\n"
        "### Compared (not selected): 4272344871\n"
        "- The compared product does not provide detailed information."
    )
    parsed = parse_claim_attribution_from_text(
        text,
        slot_norm="blades",
        claim_value="5",
        response_quote="Blades: 5",
    )
    assert parsed.scorable
    assert parsed.claim_target_entity == "4641170502"
    assert parsed.claim_section == "about_selected"
    assert parsed.claim_extraction_method == "adjudication_quote_v1"


def test_compared_section_claim_targets_compared_pid():
    text = _wrap(
        "### Selected product ID: 2926099811\n\n"
        "### About the selected product\n"
        "- Color: Red\n\n"
        "### Compared (not selected): 2046537506\n"
        "- Color: White"
    )
    parsed = parse_claim_attribution_from_text(
        text,
        slot_norm="color",
        claim_value="White",
        response_quote="Color: White",
    )
    assert parsed.scorable
    assert parsed.claim_section == "compared"
    assert parsed.claim_target_entity == "2046537506"
    assert parsed.claim_extraction_method == "adjudication_quote_v1"


def test_inline_pid_conflict_not_scorable():
    text = _wrap(
        "### Selected product ID: 1111111111\n\n"
        "### About the selected product\n"
        "- Price: 99 pesos for product 2222222222\n"
    )
    parsed = parse_claim_attribution_from_text(
        text,
        slot_norm="price",
        claim_value="99 pesos for product 2222222222",
        response_quote="Price: 99 pesos for product 2222222222",
    )
    assert not parsed.scorable
    assert parsed.parse_error == "inline_pid_conflicts_with_section"


def test_claim_not_found():
    text = _wrap(
        "### Selected product ID: 1234567890\n\n"
        "### About the selected product\n"
        "- Color: Red\n"
    )
    parsed = parse_claim_attribution_from_text(
        text,
        slot_norm="material",
        claim_value="Linen",
        response_quote="Material: Linen",
    )
    assert not parsed.scorable
    assert parsed.parse_error == "claim_sentence_not_found"


def test_evaluate_pair_detects_attribution_flip():
    ni = _wrap(
        "### Selected product ID: 4641170502\n\n"
        "### About the selected product\n"
        "- Blades: 5\n"
    )
    ti = _wrap(
        "### Selected product ID: 4641170502\n\n"
        "### About the selected product\n"
        "- Color: Pink\n\n"
        "### Compared (not selected): 4272344871\n"
        "- Blades: 5"
    )
    pm_traj = {
        "trajectory_id": "shop_D_4ff45f64d94874c9",
        "metadata": {"final_answer": ni},
        "messages_final_call": [],
        "claims": [{"legacy_label": "cross_object_merge", "slot_norm": "blades", "value": "5"}],
    }
    clean_traj = {
        "trajectory_id": "shop_I_clean",
        "metadata": {
            "final_answer": _wrap(
                "### Selected product ID: 9999999999\n\n"
                "### About the selected product\n"
                "- Blades: 5\n"
            )
        },
        "messages_final_call": [],
        "claims": [{"legacy_label": "cross_object_merge", "slot_norm": "blades", "value": "5"}],
    }
    out = evaluate_claim_attribution_pair(
        ni_text=ni,
        ti_text=ti,
        pm_trajectory=pm_traj,
        clean_trajectory=clean_traj,
        human_baseline={
            "claim_target_baseline": "4641170502",
            "clean_donor_entity": "9999999999",
            "method": "human_baseline_v1",
        },
    )
    assert out["claim_attribution_scorable"]
    assert out["claim_attribution_diff"] is True
    assert out["claim_target_baseline"] == "4641170502"
    assert out["claim_target_ti"] == "4272344871"


def test_clean_donor_entity_selected_pid_fallback():
    from ccer.mechanism.claim_attribution import resolve_clean_donor_entity

    rows = load_trajectory_index()
    cp = cem_valid_pair_ids(include_matched_clean=True)["pm_clean_cross_pairs"][1]
    pm_traj = rows[cp["pm_trajectory_id"]]
    clean_traj = rows[cp["clean_trajectory_id"]]
    entity, method = resolve_clean_donor_entity(
        pm_trajectory=pm_traj,
        clean_trajectory=clean_traj,
    )
    assert entity
    assert method in {
        "clean_parse_quote_v1",
        "clean_selected_pid_fallback_v1",
        "clean_compared_pid_fallback_v1",
    }


def test_real_cem_pairs_clean_donor_entity_coverage():
    from ccer.mechanism.claim_attribution import resolve_clean_donor_entity

    rows = load_trajectory_index()
    pairs = cem_valid_pair_ids(include_matched_clean=True)["pm_clean_cross_pairs"]
    filled = 0
    for cp in pairs:
        pm = rows[cp["pm_trajectory_id"]]
        clean = rows[cp["clean_trajectory_id"]]
        entity, _method = resolve_clean_donor_entity(pm_trajectory=pm, clean_trajectory=clean)
        if entity:
            filled += 1
    assert filled >= 20


def test_real_cem_pairs_baseline_scorable_rate():
    rows = load_trajectory_index()
    pairs = cem_valid_pair_ids(include_matched_clean=True)["pm_clean_cross_pairs"]
    scorable = 0
    for cp in pairs:
        pm = rows[cp["pm_trajectory_id"]]
        fa = str((pm.get("metadata") or {}).get("final_answer") or "")
        parsed = parse_claim_attribution_from_text(fa, pm_trajectory=pm)
        if parsed.scorable:
            scorable += 1
    assert scorable >= 15
