"""Tests for PRISM audit layer."""
from __future__ import annotations

from ccer.mechanism.prism_audit import (
    classify_audit_state,
    score_slot_alignment,
)


def test_classify_audit_states() -> None:
    assert classify_audit_state(0.9, 0.05, True) == "confirmed_risk"
    assert classify_audit_state(0.9, 0.05, False) == "probe_only"
    assert classify_audit_state(0.01, 0.05, True) == "latent_slot"
    assert classify_audit_state(0.01, 0.05, False) == "clean"


def test_slot_alignment_price() -> None:
    topk = [
        {"token": " Price", "rank": 1},
        {"token": "19.9", "rank": 2},
    ]
    out = score_slot_alignment(topk, "price")
    assert out["slot_type_aligned"] is True
    assert out["slot_rank"] == 1
