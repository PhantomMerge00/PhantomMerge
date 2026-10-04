"""Tests for rewrite acceptance gates (rules; NLI mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ccer.mechanism.line_l_rewrite_acceptance import (
    acceptance_verify_bundle,
    rewrite_verify_policy,
)
from ccer.mechanism.line_l_rewrite_quality import classify_rewrite_value
from ccer.mechanism.line_l_slot_nli import set_slot_entail_checker


def test_classify_tool_trace() -> None:
    reasons = classify_rewrite_value('4908901270 ts:[{"product_id":"1"}]', "Season")
    assert "tool_trace" in reasons


def test_rules_reject_json_blob() -> None:
    val = '{"product_id": "1", "title": "MacBook case"}'
    reasons = classify_rewrite_value(val, "Color")
    assert "json_blob" in reasons


def test_acceptance_rules_only_price_ok() -> None:
    corpus = '{"product_id":"1","price":19.9}'
    b = acceptance_verify_bundle("19.9", "Price", corpus, policy="rules_only")
    assert b["rules_ok"] is True
    assert b["ok"] is True


def test_acceptance_nli_mock() -> None:
    mock = MagicMock()
    mock.slot_entails.return_value = True
    set_slot_entail_checker(mock)
    corpus = '{"product_id":"1","color":"Blue"}'
    b = acceptance_verify_bundle(
        "Blue",
        "Color",
        corpus,
        policy="rules_and_nli",
    )
    set_slot_entail_checker(None)
    assert b["ok"] is True
