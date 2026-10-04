"""Tests for symmetric claim anchor and live answer-region token resolution."""
from __future__ import annotations

from ccer.replay.live_position import (
    LINE_D_REFRESH_TAG,
    activation_passes_line_d_v3_gate,
    intervention_steering_apply,
    line_d_layer_for_position,
    pair_has_symmetric_claim_anchor,
    resolve_answer_region_token_idx,
)
from ccer.replay.position_registry import resolve_positions, resolve_symmetric_claim_pair


def _mock_offsets(text: str) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(len(text))]


class _Tok:
    def __init__(self, text: str):
        self._text = text

    def decode(self, ids, skip_special_tokens=False):
        return self._text

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": _mock_offsets(text),
        }


def test_symmetric_pair_matches_pm_and_clean_style():
    pm_body = (
        "### Selected product ID: 1\n\n"
        "### About the selected product\n"
        "- Color: Pink\n"
    )
    clean_body = (
        "### Selected product ID: 2152663319\n\n"
        "### About the selected product\n"
        "- Price: 188.0 PHP\n"
    )
    pm = resolve_symmetric_claim_pair(pm_body)
    clean = resolve_symmetric_claim_pair(clean_body)
    assert pm is not None and clean is not None
    assert pm[2] == "about_first_bullet"
    assert clean[2] == "about_first_bullet"


def test_resolve_positions_symmetric_ignores_metadata_claim():
    traj = {
        "claims": [
            {
                "legacy_label": "constraint_projection",
                "response_span": "Color: Pink",
                "value": "Pink",
            }
        ],
        "commitment": {"action_anchor": "add"},
        "metadata": {
            "final_answer": (
                "<response>\n"
                "### About the selected product\n"
                "- Price: 188.0 PHP\n"
                "</response>"
            )
        },
    }
    body = "### About the selected product\n- Price: 188.0 PHP\n"
    serialized = f"<response>\n{body}</response>"
    res = resolve_positions(
        trajectory=traj,
        serialized_text=serialized,
        offset_mapping=_mock_offsets(serialized),
        prompt_token_count=5,
        full_token_count=len(serialized),
        claim_anchor_mode="symmetric",
    )
    assert res["claim_anchor_source"] == "about_first_bullet"


def test_live_resolve_finds_claim_onset_in_partial_seq():
    body = (
        "<response>\n"
        "### Selected product ID: 1\n\n"
        "### About the selected product\n"
        "- Color: Pink\n"
    )
    seq = list(range(len(body)))
    tok = _Tok(body)
    idx = resolve_answer_region_token_idx(
        tokenizer=tok,
        seq=seq,
        prompt_len=10,
        position="claim_onset",
    )
    assert idx >= 10
    pre = resolve_answer_region_token_idx(
        tokenizer=tok,
        seq=seq,
        prompt_len=10,
        position="pre_value",
    )
    assert pre >= 10


def test_intervention_steering_apply():
    assert intervention_steering_apply("commitment") == "generation_decode"
    assert intervention_steering_apply("prompt_end") == "anchor_once"
    assert intervention_steering_apply("claim_onset") == "dynamic_answer_anchor"


def test_pair_has_symmetric_claim_anchor():
    assert pair_has_symmetric_claim_anchor(
        {
            "metadata": {
                "final_answer": (
                    "<response>\n"
                    "### About the selected product\n"
                    "- Price: 1\n"
                    "</response>"
                )
            }
        }
    )
    assert not pair_has_symmetric_claim_anchor({"metadata": {"final_answer": ""}})


def test_activation_passes_line_d_v3_gate():
    good = {
        "metadata": {"refreshed_line_d_v3_symmetric": True, "claim_anchor_mode": "symmetric"},
        "positions": {"claim_onset": 100, "pre_value": 99},
    }
    assert activation_passes_line_d_v3_gate(good, position="claim_onset")
    bad = {"metadata": {"refreshed_line_d_v2": True}, "positions": {"claim_onset": 100}}
    assert not activation_passes_line_d_v3_gate(bad, position="claim_onset")


def test_line_d_layer_for_position():
    assert line_d_layer_for_position("claim_onset") == 49
    assert line_d_layer_for_position("prompt_end") == 32


def test_j_prime_compared_bullet_resolves_live():
    from ccer.replay.position_registry import resolve_live_claim_pair

    # No About bullets — J′ should fall through to Compared section.
    body = (
        "### Selected product ID: 2926099811\n\n"
        "### Compared (not selected): 2046537506\n"
        "- Color: White"
    )
    claim = resolve_live_claim_pair(body)
    assert claim is not None
    assert claim[2] == "compared_first_bullet"


def test_j_prime_env_flag():
    import os

    from ccer.replay.live_position import J_PRIME_ENV_VAR, j_prime_live_anchor_enabled

    old = os.environ.get(J_PRIME_ENV_VAR)
    try:
        os.environ[J_PRIME_ENV_VAR] = "0"
        assert not j_prime_live_anchor_enabled()
        os.environ[J_PRIME_ENV_VAR] = "1"
        assert j_prime_live_anchor_enabled()
    finally:
        if old is None:
            os.environ.pop(J_PRIME_ENV_VAR, None)
        else:
            os.environ[J_PRIME_ENV_VAR] = old


def test_dynamic_answer_anchor_patch_positions_priority():
    from ccer.replay.live_position import dynamic_answer_anchor_patch_positions

    pl, anchor = 100, 150
    # live wins when in bounds
    assert dynamic_answer_anchor_patch_positions(
        seq_len=160, prompt_len=pl, decode_step=True, anchor_pos=anchor, live_idx=155
    ) == [155]
    # stored npz fallback once sequence reaches anchor
    assert dynamic_answer_anchor_patch_positions(
        seq_len=160, prompt_len=pl, decode_step=True, anchor_pos=anchor, live_idx=-1
    ) == [150]
    # decode-head while answer is generating but anchor not yet in seq
    assert dynamic_answer_anchor_patch_positions(
        seq_len=120, prompt_len=pl, decode_step=True, anchor_pos=anchor, live_idx=-1
    ) == [-1]
    # prefill before any generated token: no patch
    assert dynamic_answer_anchor_patch_positions(
        seq_len=pl, prompt_len=pl, decode_step=False, anchor_pos=anchor, live_idx=-1
    ) == []
