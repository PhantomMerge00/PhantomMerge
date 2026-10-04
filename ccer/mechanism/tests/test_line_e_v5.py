"""Unit tests for Line E v5 extreme protocol."""
from __future__ import annotations

from ccer.mechanism.line_e_anchor_logit import build_anchor_logit_bias
from ccer.mechanism.line_e_v5_eval import behavioral_outcome_v5
from ccer.mechanism.line_e_v5_protocol import (
    LINE_E_V5_PROTOCOL_VERSION,
    line_e_v5_config,
    v5_control_ids,
)


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 1000 for c in text]


def test_v5_config_defaults():
    cfg = line_e_v5_config()
    assert cfg.protocol_version == LINE_E_V5_PROTOCOL_VERSION
    assert cfg.position == "claim_onset"
    assert cfg.layer == 49
    assert cfg.steering_apply == "dynamic_answer_anchor"
    assert "pca_target_interchange" in v5_control_ids(cfg)
    assert "probe_steer" in v5_control_ids(cfg)
    assert "anchor_logit_bias" in v5_control_ids(cfg)


def test_anchor_logit_bias_digit_tokens():
    tok = _FakeTokenizer()
    bias = build_anchor_logit_bias(tok, "SKU-12345", alpha=1.0, bias_per_token=3.0)
    assert bias
    assert all(v == 3.0 for ch, v in zip("12345", [bias[ord(c) % 1000] for c in "12345"]))


def test_anchor_logit_bias_zero_alpha():
    tok = _FakeTokenizer()
    assert build_anchor_logit_bias(tok, "123", alpha=0.0) == {}


def test_behavioral_outcome_v5_anchor_aligned():
    traj = {"commitment": {"action_anchor": "999"}}
    beh = behavioral_outcome_v5(
        "<response>Selected product ID: 999</response>",
        traj,
    )
    assert beh["anchor_aligned"] is True
    assert beh["product_id_correct"] is True


def test_wrong_owner_reduced_parseable_gate():
    from ccer.mechanism.line_e_v5_eval import _build_v5_row
    from ccer.mechanism.line_e_v5_protocol import line_e_v5_config

    cfg = line_e_v5_config()
    base = {
        "wrong_owner": True,
        "product_id_correct": False,
        "selected_product_id": "PID-111",
        "action_anchor": "PID-999",
    }
    beh_parseable_fix = {
        "wrong_owner": False,
        "product_id_correct": True,
        "selected_product_id": "PID-999",
        "anchor_aligned": True,
    }
    row_ok = _build_v5_row(
        tid="t1",
        track="CEM",
        control_id="pca_target_interchange",
        alpha=1.0,
        cfg=cfg,
        base=base,
        beh=beh_parseable_fix,
        out={"patched": True},
        paired_meta={},
    )
    assert row_ok["wrong_owner_reduced_parseable"] is True
    assert row_ok["anchor_hit"] is True

    beh_unparseable = {
        "wrong_owner": False,
        "product_id_correct": False,
        "selected_product_id": None,
        "anchor_aligned": False,
    }
    row_bad = _build_v5_row(
        tid="t1",
        track="CEM",
        control_id="caa_random",
        alpha=1.5,
        cfg=cfg,
        base=base,
        beh=beh_unparseable,
        out={"patched": True},
        paired_meta={},
    )
    assert row_bad["wrong_owner_reduced"] is True
    assert row_bad["wrong_owner_reduced_parseable"] is False
