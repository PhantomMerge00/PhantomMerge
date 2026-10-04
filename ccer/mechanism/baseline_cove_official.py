"""Prompt-faithful CoVe baseline (plan → verify → revise) with Obs_τ evidence.

Distinct from RARR: uses verification-planning loop rather than agreement-gate editor.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ccer.mechanism.line_l_anchor_extract import ExtractionResult, literal_replace_in_quote, lookup_slot_value
from ccer.mechanism.line_l_llm_common import _cache_key, default_llm_base_url, llm_chat, parse_slot_value_line
from ccer.mechanism.line_l_rewrite_quality import build_evidence_corpus, strict_slot_verify
from ccer.mechanism.obs_tau_retriever import top_evidence_snippet
from ccer.paths import PRISM_L_PLUS_CACHE

_CACHE_DRAFT = PRISM_L_PLUS_CACHE / "cove_official_draft.jsonl"
_CACHE_PLAN = PRISM_L_PLUS_CACHE / "cove_official_plan.jsonl"
_CACHE_REVISE = PRISM_L_PLUS_CACHE / "cove_official_revise.jsonl"


def _execute_verifications(value: str, slot_norm: str, corpus: str, questions: list[str]) -> tuple[bool, list[dict]]:
    from ccer.mechanism.line_l_rewrite_quality import verify_grounded

    results: list[dict] = []
    all_pass = True
    for q in questions:
        q_l = q.lower()
        if "appear" in q_l or "verbatim" in q_l or "evidence" in q_l:
            passed = verify_grounded(value, corpus)
        elif "slot" in q_l or slot_norm.lower() in q_l:
            passed = bool(value.strip()) and not re.match(r"^\d{8,12}$", value.strip())
        else:
            passed = verify_grounded(value, corpus)
        results.append({"question": q, "passed": passed})
        if not passed:
            all_pass = False
    return all_pass, results


def apply_cove_official_rewrite(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    arm: str = "b2_cove_official",
    use_llm: bool = True,
    allow_extractive_shortcut: bool = False,
) -> dict[str, Any]:
    p_pm = float(row.get("p_pm", 0.0))
    tau = float(row.get("tau", 1.0))
    slot_norm = str(row.get("slot_norm") or "").strip()
    response_quote = str(row.get("response_quote") or "")
    claim_value = str(row.get("claim_value") or "")
    slot_label = str(row.get("slot") or slot_norm)

    if p_pm <= tau:
        return {
            "arm": arm,
            "branch": "skip_unflagged",
            "action": "keep",
            "quote_before": response_quote,
            "quote_after": response_quote,
            "method": "b2_cove_official",
        }

    if not slot_norm:
        return {
            "arm": arm,
            "branch": "skip_no_slot",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "no_slot_norm",
            "method": "b2_cove_official",
        }

    corpus = build_evidence_corpus(traj, anchor_pid)
    evidence = top_evidence_snippet(
        traj, anchor_pid, slot_norm, response_quote, row=row
    )

    if allow_extractive_shortcut:
        extraction = lookup_slot_value(
            traj,
            anchor_pid=anchor_pid,
            slot_norm=slot_norm,
            anchor_evidence_quote=str(row.get("anchor_evidence_quote") or ""),
            row=row,
        )
    else:
        extraction = ExtractionResult("extraction_miss")
    if allow_extractive_shortcut and extraction.branch == "extraction_hit" and extraction.v_anchor:
        ok, _ = strict_slot_verify(extraction.v_anchor, slot_norm, corpus)
        if ok:
            new_quote = literal_replace_in_quote(
                response_quote,
                slot=slot_label,
                old_value=claim_value,
                v_anchor=extraction.v_anchor,
            )
            return {
                "arm": arm,
                "branch": "cove_official_extractive",
                "action": "rewrite",
                "quote_before": response_quote,
                "quote_after": new_quote,
                "v_anchor": extraction.v_anchor,
                "method": "b2_cove_official",
            }

    if not use_llm:
        return {
            "arm": arm,
            "branch": "cove_official_no_llm",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "no_llm",
            "method": "b2_cove_official",
        }

    # CoVe Step 1: baseline draft
    draft_messages = [
        {
            "role": "system",
            "content": (
                "Draft a corrected product attribute claim using ONLY the evidence. "
                f"Output one line: {slot_label}: <value>"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original: {response_quote}\n"
                f"Slot: {slot_label}\n"
                f"Evidence:\n{evidence}\n"
                "Draft:"
            ),
        },
    ]
    draft_key = _cache_key("cove_off_draft", row, anchor_pid)
    draft_text, _ = llm_chat(
        draft_messages, cache_key=draft_key, cache_path=_CACHE_DRAFT, max_tokens=128
    )
    draft_branch = "cove_official_llm_draft"
    if not draft_text:
        retrieval = lookup_slot_value(
            traj,
            anchor_pid=anchor_pid,
            slot_norm=slot_norm,
            anchor_evidence_quote=str(row.get("anchor_evidence_quote") or ""),
            row=row,
        )
        if retrieval.branch == "extraction_hit" and retrieval.v_anchor:
            draft_value = retrieval.v_anchor
            draft_text = f"{slot_label}: {draft_value}"
            draft_branch = "cove_official_retrieval_draft"
        else:
            return {
                "arm": arm,
                "branch": "cove_official_draft_fail",
                "action": "delete",
                "quote_before": response_quote,
                "quote_after": None,
                "miss_reason": "llm_unavailable",
                "method": "b2_cove_official",
            }
    else:
        _s, draft_value = parse_slot_value_line(draft_text, slot_norm)
    if not draft_value:
        return {
            "arm": arm,
            "branch": "cove_official_parse_fail",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "draft_parse_fail",
            "method": "b2_cove_official",
        }
    # CoVe Step 2: plan verification questions
    plan_messages = [
        {
            "role": "system",
            "content": (
                "Generate 3 verification questions to fact-check the draft value against "
                "the evidence. Output JSON array of strings only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Slot: {slot_label}\nDraft: {draft_value}\n"
                f"Evidence:\n{evidence[:900]}\n"
                'Example: ["Is the value in evidence?", "Is slot correct?"]'
            ),
        },
    ]
    plan_key = _cache_key("cove_off_plan", row, anchor_pid)
    plan_text, _ = llm_chat(
        plan_messages, cache_key=plan_key, cache_path=_CACHE_PLAN, max_tokens=384
    )
    questions = [
        f'Is "{draft_value}" supported by the anchor evidence for {slot_label}?',
        f"Does the evidence specify a different {slot_label} than the claim?",
        f"Can {draft_value} be copied verbatim from the product record?",
    ]
    if plan_text:
        try:
            m = re.search(r"\[.*\]", plan_text, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list) and parsed:
                    questions = [str(q) for q in parsed[:4]]
        except (json.JSONDecodeError, TypeError):
            pass

    passed, v_results = _execute_verifications(draft_value, slot_norm, corpus, questions)

    # CoVe Step 3: revise conditioned on verification (even if some fail)
    revise_messages = [
        {
            "role": "system",
            "content": (
                "Revise the draft value using verification results and evidence. "
                f"Output one line: {slot_label}: <value>. Use verbatim evidence only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original claim: {response_quote}\n"
                f"Draft: {slot_label}: {draft_value}\n"
                f"Verification: {json.dumps(v_results)}\n"
                f"Evidence:\n{evidence}\n"
                "Revised value:"
            ),
        },
    ]
    revise_key = _cache_key("cove_off_revise", row, anchor_pid)
    revise_text, _ = llm_chat(
        revise_messages, cache_key=revise_key, cache_path=_CACHE_REVISE, max_tokens=128
    )
    value = draft_value
    if revise_text:
        _s2, rev_val = parse_slot_value_line(revise_text, slot_norm)
        if rev_val:
            value = rev_val

    ok, reason = strict_slot_verify(value, slot_norm, corpus)
    if not ok and passed:
        ok, reason = strict_slot_verify(draft_value, slot_norm, corpus)
        if ok:
            value = draft_value

    if not ok:
        return {
            "arm": arm,
            "branch": "cove_official_verify_fail",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": reason or "verification_failed",
            "method": "b2_cove_official",
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
        "branch": "cove_official_verify_pass",
        "action": "rewrite",
        "quote_before": response_quote,
        "quote_after": new_quote,
        "v_anchor": value,
        "source_field": "cove_official_revise",
        "evidence_kind": "cove_official_verified",
        "method": "b2_cove_official",
        "verification_results": v_results,
        "llm_backend": default_llm_base_url(),
        "cove_draft_branch": draft_branch,
    }
