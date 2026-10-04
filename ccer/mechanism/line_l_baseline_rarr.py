"""Line L+ Phase 1: RARR-style retrieve-revise-verify baseline (Gao et al., 2023).

DEPRECATED: use ``baseline_rarr_official`` (vendor RARR + Obs_τ) for PRISM-L+ experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccer.mechanism.line_l_anchor_extract import literal_replace_in_quote, lookup_slot_value
from ccer.mechanism.line_l_baseline_copy import apply_copy_baseline_rewrite
from ccer.mechanism.line_l_llm_common import (
    _cache_key,
    build_rewrite_messages,
    default_llm_base_url,
    llm_chat,
    parse_slot_value_line,
    retrieve_evidence_snippet,
)
from ccer.mechanism.line_l_rewrite_quality import build_evidence_corpus, verify_grounded

_CACHE_PATH = Path("${PHANTOM_MERGE_ROOT}/results/line_l_plus/cache/rarr_rewrite.jsonl")


def apply_rarr_baseline_rewrite(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    arm: str = "baseline_rarr",
    use_llm: bool = True,
) -> dict[str, Any]:
    """Retrieve → Revise (LLM) → Verify; fail verify → delete."""
    p_pm = float(row.get("p_pm", 0.0))
    tau = float(row.get("tau", 1.0))
    slot_norm = str(row.get("slot_norm") or "").strip()
    response_quote = str(row.get("response_quote") or "")
    claim_value = str(row.get("claim_value") or "")

    if p_pm <= tau:
        return {
            "arm": arm,
            "branch": "skip_unflagged",
            "action": "keep",
            "quote_before": response_quote,
            "quote_after": response_quote,
            "method": "b1_rarr",
        }

    if not slot_norm:
        return {
            "arm": arm,
            "branch": "skip_no_slot",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "no_slot_norm",
            "method": "b1_rarr",
        }

    # Step 1 Retrieve: extractive hit is already grounded
    extraction = lookup_slot_value(
        traj,
        anchor_pid=anchor_pid,
        slot_norm=slot_norm,
        anchor_evidence_quote=str(row.get("anchor_evidence_quote") or ""),
        row=row,
    )
    if extraction.branch == "extraction_hit" and extraction.v_anchor:
        corpus = build_evidence_corpus(traj, anchor_pid)
        if verify_grounded(extraction.v_anchor, corpus):
            new_quote = literal_replace_in_quote(
                response_quote,
                slot=str(row.get("slot") or slot_norm),
                old_value=claim_value,
                v_anchor=extraction.v_anchor,
            )
            return {
                "arm": arm,
                "branch": "rarr_extractive",
                "action": "rewrite",
                "quote_before": response_quote,
                "quote_after": new_quote,
                "v_anchor": extraction.v_anchor,
                "source_field": extraction.source_field,
                "evidence_kind": "rarr_retrieve_hit",
                "method": "b1_rarr",
            }

    if not use_llm:
        return apply_copy_baseline_rewrite(row, traj, anchor_pid=anchor_pid, arm=arm)

    # Step 2 Revise
    messages = build_rewrite_messages(row, traj, anchor_pid)
    key = _cache_key("rarr", row, anchor_pid)
    text, meta = llm_chat(messages, cache_key=key, cache_path=_CACHE_PATH, base_url=default_llm_base_url())
    if not text:
        return {
            "arm": arm,
            "branch": "rarr_llm_unavailable",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "llm_unavailable",
            "method": "b1_rarr",
        }

    _slot, value = parse_slot_value_line(text, slot_norm)
    if not value:
        return {
            "arm": arm,
            "branch": "rarr_parse_fail",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "llm_parse_fail",
            "method": "b1_rarr",
            "llm_raw": text,
        }

    # Step 3 Verify
    corpus = build_evidence_corpus(traj, anchor_pid)
    if not verify_grounded(value, corpus):
        return {
            "arm": arm,
            "branch": "rarr_verify_fail",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "verify_failed",
            "method": "b1_rarr",
            "llm_raw": text,
        }

    new_quote = literal_replace_in_quote(
        response_quote,
        slot=str(row.get("slot") or slot_norm),
        old_value=claim_value,
        v_anchor=value,
    )
    return {
        "arm": arm,
        "branch": "rarr_verify_pass",
        "action": "rewrite",
        "quote_before": response_quote,
        "quote_after": new_quote,
        "v_anchor": value,
        "source_field": "rarr_llm_revise",
        "evidence_kind": "rarr_verified",
        "method": "b1_rarr",
        "llm_meta": meta,
    }
