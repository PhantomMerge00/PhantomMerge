"""Official RARR loop (vendor prompts) with Obs_τ retriever + vLLM shim.

Replaces Google Search with in-trajectory anchor observations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ccer.mechanism.line_l_anchor_extract import ExtractionResult, literal_replace_in_quote, lookup_slot_value
from ccer.mechanism.line_l_baseline_copy import apply_copy_baseline_rewrite
from ccer.mechanism.line_l_llm_common import _cache_key, default_llm_base_url
from ccer.mechanism.line_l_rewrite_quality import build_evidence_corpus, strict_slot_verify
from ccer.mechanism.obs_tau_retriever import retrieve_for_rarr
from ccer.mechanism.rarr_vllm_shim import (
    completion_create,
    parse_agreement_gate,
    parse_editor_response,
    parse_rarr_questions,
    parse_slot_line_from_edit,
)
from ccer.paths import PRISM_L_PLUS_CACHE

_RARR_ROOT = Path("${PHANTOM_MERGE_ROOT}/third_party/RARR")
if str(_RARR_ROOT) not in sys.path:
    sys.path.insert(0, str(_RARR_ROOT))

from prompts import rarr_prompts  # noqa: E402

_CACHE_PATH = PRISM_L_PLUS_CACHE / "rarr_official.jsonl"

_MAX_RARR_CLAIM_CHARS = 480
_MAX_RARR_PASSAGE_CHARS = 1600


def _truncate_rarr_field(text: str, max_chars: int) -> str:
    s = str(text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 24] + "\n…[truncated for context]"


def _run_qgen(claim: str, *, cache_prefix: str) -> list[str]:
    claim = _truncate_rarr_field(claim, _MAX_RARR_CLAIM_CHARS)
    prompt = rarr_prompts.QGEN_PROMPT.format(claim=claim).strip()
    resp = completion_create(
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        cache_key=f"{cache_prefix}|qgen",
        cache_path=_CACHE_PATH,
    )
    return parse_rarr_questions(resp["choices"][0]["text"])


def _run_agreement_gate(claim: str, query: str, evidence: str, *, cache_prefix: str) -> bool:
    claim = _truncate_rarr_field(claim, _MAX_RARR_CLAIM_CHARS)
    evidence = _truncate_rarr_field(evidence, _MAX_RARR_PASSAGE_CHARS)
    prompt = rarr_prompts.AGREEMENT_GATE_PROMPT.format(
        claim=claim, query=query, evidence=evidence
    ).strip()
    resp = completion_create(
        prompt=prompt,
        temperature=0.0,
        max_tokens=256,
        stop=["\n\n"],
        cache_key=f"{cache_prefix}|gate|{query[:40]}",
        cache_path=_CACHE_PATH,
    )
    is_open, _, _ = parse_agreement_gate(resp["choices"][0]["text"])
    return is_open


def _run_editor(claim: str, query: str, evidence: str, *, cache_prefix: str) -> str | None:
    claim = _truncate_rarr_field(claim, _MAX_RARR_CLAIM_CHARS)
    evidence = _truncate_rarr_field(evidence, _MAX_RARR_PASSAGE_CHARS)
    prompt = rarr_prompts.EDITOR_PROMPT.format(claim=claim, query=query, evidence=evidence).strip()
    resp = completion_create(
        prompt=prompt,
        temperature=0.0,
        max_tokens=512,
        stop=["\n\n"],
        cache_key=f"{cache_prefix}|edit|{query[:40]}",
        cache_path=_CACHE_PATH,
    )
    return parse_editor_response(resp["choices"][0]["text"])


def apply_rarr_official_rewrite(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    arm: str = "b1_rarr_official",
    use_llm: bool = True,
    allow_extractive_shortcut: bool = True,
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
            "method": "b1_rarr_official",
        }

    if not slot_norm:
        return {
            "arm": arm,
            "branch": "skip_no_slot",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "no_slot_norm",
            "method": "b1_rarr_official",
        }

    if allow_extractive_shortcut:
        extraction = lookup_slot_value(
            traj,
            anchor_pid=anchor_pid,
            slot_norm=slot_norm,
            anchor_evidence_quote=str(row.get("anchor_evidence_quote") or ""),
            row=row if not row.get("e2e_mode") else {**row, "donor_owners": row.get("donor_owners")},
        )
    else:
        extraction = ExtractionResult("extraction_miss")
    if allow_extractive_shortcut and extraction.branch == "extraction_hit" and extraction.v_anchor:
        corpus = build_evidence_corpus(traj, anchor_pid)
        ok, reason = strict_slot_verify(extraction.v_anchor, slot_norm, corpus)
        if ok:
            new_quote = literal_replace_in_quote(
                response_quote,
                slot=str(row.get("slot") or slot_norm),
                old_value=claim_value,
                v_anchor=extraction.v_anchor,
            )
            return {
                "arm": arm,
                "branch": "rarr_official_extractive",
                "action": "rewrite",
                "quote_before": response_quote,
                "quote_after": new_quote,
                "v_anchor": extraction.v_anchor,
                "method": "b1_rarr_official",
            }

    if not use_llm:
        out = apply_copy_baseline_rewrite(row, traj, anchor_pid=anchor_pid, arm=arm)
        return {**out, "method": "b1_rarr_official"}

    claim = response_quote
    cache_prefix = _cache_key("rarr_off", row, anchor_pid)
    passages = retrieve_for_rarr(
        traj,
        anchor_pid=anchor_pid,
        claim=claim,
        slot_norm=slot_norm,
        row=row,
    )
    if not passages:
        return {
            "arm": arm,
            "branch": "rarr_official_no_evidence",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "no_obs_tau_passages",
            "method": "b1_rarr_official",
        }

    questions = _run_qgen(claim, cache_prefix=cache_prefix)
    if not questions:
        questions = [p.query for p in passages[:3]]

    edited_value: str | None = None
    for qp in passages:
        query = qp.query
        if query not in questions and questions:
            query = questions[0]
        if not _run_agreement_gate(claim, query, qp.passage, cache_prefix=cache_prefix):
            continue
        edit = _run_editor(claim, query, qp.passage, cache_prefix=cache_prefix)
        if not edit:
            continue
        value = parse_slot_line_from_edit(edit, slot_norm) or edit
        corpus = build_evidence_corpus(traj, anchor_pid)
        ok, reason = strict_slot_verify(value, slot_norm, corpus)
        if ok:
            edited_value = value
            break

    if not edited_value:
        return {
            "arm": arm,
            "branch": "rarr_official_verify_fail",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "strict_verify_failed",
            "method": "b1_rarr_official",
        }

    new_quote = literal_replace_in_quote(
        response_quote,
        slot=str(row.get("slot") or slot_norm),
        old_value=claim_value,
        v_anchor=edited_value,
    )
    return {
        "arm": arm,
        "branch": "rarr_official_verify_pass",
        "action": "rewrite",
        "quote_before": response_quote,
        "quote_after": new_quote,
        "v_anchor": edited_value,
        "source_field": "rarr_official_editor",
        "evidence_kind": "rarr_official_verified",
        "method": "b1_rarr_official",
        "llm_backend": default_llm_base_url(),
    }
