"""Line L+ Phase 2: CoVe-style draft + verification loop baseline (Dhuliawala et al., 2023).

DEPRECATED: use ``baseline_cove_official`` (prompt-faithful CoVe + Obs_τ) for PRISM-L+.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ccer.mechanism.line_l_anchor_extract import literal_replace_in_quote, lookup_slot_value
from ccer.mechanism.line_l_llm_common import (
    _cache_key,
    build_rewrite_messages,
    default_llm_base_url,
    llm_chat,
    parse_slot_value_line,
)
from ccer.mechanism.line_l_rewrite_quality import build_evidence_corpus, verify_grounded

_CACHE_DRAFT = Path("${PHANTOM_MERGE_ROOT}/results/line_l_plus/cache/cove_draft.jsonl")
_CACHE_PLAN = Path("${PHANTOM_MERGE_ROOT}/results/line_l_plus/cache/cove_plan.jsonl")


def _execute_verification_questions(
    value: str,
    slot_norm: str,
    corpus: str,
    questions: list[str],
) -> tuple[bool, list[dict[str, Any]]]:
    """Deterministic verification execution (substring checks)."""
    results: list[dict[str, Any]] = []
    all_pass = True
    for q in questions:
        q_lower = q.lower()
        passed = False
        if "appear" in q_lower or "verbatim" in q_lower or "evidence" in q_lower:
            passed = verify_grounded(value, corpus)
        elif "slot" in q_lower or slot_norm.lower() in q_lower:
            passed = bool(value.strip())
        else:
            passed = verify_grounded(value, corpus)
        results.append({"question": q, "passed": passed})
        if not passed:
            all_pass = False
    return all_pass, results


def apply_cove_baseline_rewrite(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    arm: str = "baseline_cove",
    use_llm: bool = True,
) -> dict[str, Any]:
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
            "method": "b2_cove",
        }

    if not slot_norm:
        return {
            "arm": arm,
            "branch": "skip_no_slot",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "no_slot_norm",
            "method": "b2_cove",
        }

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
                "branch": "cove_extractive",
                "action": "rewrite",
                "quote_before": response_quote,
                "quote_after": new_quote,
                "v_anchor": extraction.v_anchor,
                "method": "b2_cove",
            }

    if not use_llm:
        return {
            "arm": arm,
            "branch": "cove_no_llm",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "no_llm",
            "method": "b2_cove",
        }

    # Draft
    draft_messages = build_rewrite_messages(row, traj, anchor_pid)
    draft_key = _cache_key("cove_draft", row, anchor_pid)
    draft_text, draft_meta = llm_chat(
        draft_messages, cache_key=draft_key, cache_path=_CACHE_DRAFT, base_url=default_llm_base_url()
    )
    if not draft_text:
        return {
            "arm": arm,
            "branch": "cove_draft_fail",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "llm_unavailable",
            "method": "b2_cove",
        }

    _slot, value = parse_slot_value_line(draft_text, slot_norm)
    if not value:
        return {
            "arm": arm,
            "branch": "cove_parse_fail",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "draft_parse_fail",
            "method": "b2_cove",
        }

    corpus = build_evidence_corpus(traj, anchor_pid)
    slot_label = str(row.get("slot") or slot_norm)

    # Plan verifications
    plan_messages = [
        {
            "role": "system",
            "content": (
                "Generate 2-3 yes/no verification questions to check if a claim value "
                "is supported by the anchor evidence. Output JSON array of strings only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Slot: {slot_label}\n"
                f"Draft value: {value}\n"
                f"Evidence excerpt:\n{corpus[:800]}\n"
                'Example: ["Does the value appear in the evidence?", "Is this the correct slot?"]'
            ),
        },
    ]
    plan_key = _cache_key("cove_plan", row, anchor_pid)
    plan_text, plan_meta = llm_chat(
        plan_messages, cache_key=plan_key, cache_path=_CACHE_PLAN, base_url=default_llm_base_url(), max_tokens=384
    )
    questions: list[str] = [
        f'Does "{value}" appear verbatim in the anchor evidence for {slot_label}?',
        f"Is {slot_label} supported by the anchor product record?",
    ]
    if plan_text:
        try:
            m = re.search(r"\[.*\]", plan_text, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list) and parsed:
                    questions = [str(q) for q in parsed[:3]]
        except (json.JSONDecodeError, TypeError):
            pass

    passed, v_results = _execute_verification_questions(value, slot_norm, corpus, questions)
    if not passed:
        return {
            "arm": arm,
            "branch": "cove_verify_fail",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "verification_failed",
            "method": "b2_cove",
            "verification_results": v_results,
        }

    new_quote = literal_replace_in_quote(
        response_quote,
        slot=slot_label,
        old_value=claim_value,
        v_anchor=value,
    )
    return {
        "arm": arm,
        "branch": "cove_verify_pass",
        "action": "rewrite",
        "quote_before": response_quote,
        "quote_after": new_quote,
        "v_anchor": value,
        "source_field": "cove_llm_draft",
        "evidence_kind": "cove_verified",
        "method": "b2_cove",
        "verification_results": v_results,
        "llm_meta": draft_meta,
    }
