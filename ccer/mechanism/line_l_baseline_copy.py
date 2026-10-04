"""Line L+ Phase 3: Constrained copy baseline (FEVER-style span selection)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ccer.mechanism.line_l_anchor_extract import (
    _SLOT_TO_PRODUCT_FIELDS,
    _anchor_product_record,
    _norm_slot,
    _norm_value_compare,
    literal_replace_in_quote,
    lookup_slot_value,
)
from ccer.mechanism.line_l_anchor_extract import lookup_obs_tau_slot, parse_anchor_evidence_quote
from ccer.mechanism.line_l_rewrite_quality import build_evidence_corpus, verify_grounded
from ccer.mechanism.obs_tau_retriever import retrieve_for_rarr

MethodId = str


@dataclass(frozen=True)
class CopyCandidate:
    value: str
    source: str
    score: float


def _flatten_record_values(record: dict[str, Any], slot_key: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for field in _SLOT_TO_PRODUCT_FIELDS.get(slot_key, (slot_key,)):
        val = record.get(field)
        if val is not None and str(val).strip():
            out.append((f"product_json.{field}", str(val).strip()))
    for key, val in record.items():
        if key in ("product_id",):
            continue
        if isinstance(val, (str, int, float)) and str(val).strip():
            kn = _norm_slot(str(key))
            if kn == slot_key or slot_key in kn:
                out.append((f"product_json.{key}", str(val).strip()))
    return out


def enumerate_copy_candidates_snippet_fallback(
    traj: dict[str, Any],
    anchor_pid: str,
    slot_norm: str,
    *,
    row: dict[str, Any] | None = None,
) -> list[CopyCandidate]:
    """Optional obs_tau snippets — only when aligned to structured slot values."""
    slot_key = _norm_slot(slot_norm)
    candidates: list[CopyCandidate] = []
    corpus = build_evidence_corpus(traj, anchor_pid)
    obs_hit = lookup_obs_tau_slot(traj, anchor_pid=anchor_pid, slot_norm=slot_norm)
    allowed_values: set[str] = set()
    if obs_hit.branch == "extraction_hit" and obs_hit.v_anchor:
        allowed_values.add(_norm_value_compare(obs_hit.v_anchor))
    if row is not None:
        for val in parse_anchor_evidence_quote(str(row.get("anchor_evidence_quote") or "")).values():
            allowed_values.add(_norm_value_compare(val))
    for hit in retrieve_for_rarr(
        traj,
        anchor_pid=anchor_pid,
        claim=slot_norm,
        slot_norm=slot_norm,
        max_evidences=6,
    ):
        snippet = hit.passage.strip()
        if not snippet or not verify_grounded(snippet, corpus):
            continue
        short = snippet[:120]
        if allowed_values and _norm_value_compare(short) not in allowed_values:
            if not any(v in _norm_value_compare(snippet) for v in allowed_values if v):
                continue
        candidates.append(CopyCandidate(value=short, source="obs_tau_window", score=0.7))
    return candidates


def enumerate_copy_candidates(
    traj: dict[str, Any],
    anchor_pid: str,
    slot_norm: str,
    *,
    allow_snippet_fallback: bool = False,
    row: dict[str, Any] | None = None,
) -> list[CopyCandidate]:
    slot_key = _norm_slot(slot_norm)
    candidates: list[CopyCandidate] = []
    record = _anchor_product_record(traj, anchor_pid)
    if record:
        for src, val in _flatten_record_values(record, slot_key):
            candidates.append(CopyCandidate(value=val, source=src, score=1.0))
    corpus = build_evidence_corpus(traj, anchor_pid)
    if allow_snippet_fallback:
        candidates.extend(
            enumerate_copy_candidates_snippet_fallback(traj, anchor_pid, slot_norm, row=row)
        )
    for field in _SLOT_TO_PRODUCT_FIELDS.get(slot_key, (slot_key,)):
        m = re.search(rf'"{field}"\s*:\s*"([^"]+)"', corpus, re.I)
        if m:
            candidates.append(CopyCandidate(value=m.group(1), source=f"corpus.{field}", score=0.9))
        m2 = re.search(rf'"{field}"\s*:\s*([\d.]+)', corpus, re.I)
        if m2:
            candidates.append(CopyCandidate(value=m2.group(1), source=f"corpus.{field}", score=0.9))
    # dedupe by normalized value
    seen: set[str] = set()
    unique: list[CopyCandidate] = []
    for c in candidates:
        nk = _norm_value_compare(c.value)
        if nk and nk not in seen:
            seen.add(nk)
            unique.append(c)
    return unique


def score_copy_candidate(
    candidate: CopyCandidate,
    *,
    claim_value: str,
    gold_verdict: str = "",
    corpus: str,
) -> float:
    score = candidate.score
    if not verify_grounded(candidate.value, corpus):
        return -1.0
    cv = _norm_value_compare(claim_value)
    cand_n = _norm_value_compare(candidate.value)
    if cv and cand_n == cv:
        score -= 0.5
    if candidate.source.startswith("product_json"):
        score += 0.5
    if len(candidate.value) <= 40:
        score += 0.2
    elif len(candidate.value) > 80:
        score -= 0.5
    return score


def apply_copy_baseline_rewrite(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    arm: str = "baseline_copy",
    use_gold_cem_hint: bool = False,
    allow_snippet_fallback: bool = False,
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
            "method": "b3_copy",
        }

    if not slot_norm:
        return {
            "arm": arm,
            "branch": "skip_no_slot",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "no_slot_norm",
            "method": "b3_copy",
        }

    # Try exact extractive first
    extraction = lookup_slot_value(
        traj,
        anchor_pid=anchor_pid,
        slot_norm=slot_norm,
        anchor_evidence_quote=str(row.get("anchor_evidence_quote") or ""),
        row=row,
    )
    if extraction.branch == "extraction_hit" and extraction.v_anchor:
        new_quote = literal_replace_in_quote(
            response_quote,
            slot=str(row.get("slot") or slot_norm),
            old_value=claim_value,
            v_anchor=extraction.v_anchor,
        )
        return {
            "arm": arm,
            "branch": "extraction_hit",
            "action": "rewrite",
            "quote_before": response_quote,
            "quote_after": new_quote,
            "v_anchor": extraction.v_anchor,
            "source_field": extraction.source_field,
            "evidence_kind": extraction.evidence_kind,
            "method": "b3_copy",
        }

    corpus = build_evidence_corpus(traj, anchor_pid)
    cands = enumerate_copy_candidates(
        traj,
        anchor_pid,
        slot_norm,
        allow_snippet_fallback=allow_snippet_fallback,
        row=row,
    )
    best: CopyCandidate | None = None
    best_score = -1.0
    for c in cands:
        s = score_copy_candidate(
            c,
            claim_value=claim_value,
            gold_verdict=str(row.get("gold_verdict") or "") if use_gold_cem_hint else "",
            corpus=corpus,
        )
        if s > best_score:
            best_score = s
            best = c

    if best is None or best_score < 0:
        return {
            "arm": arm,
            "branch": "copy_miss",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "no_copy_candidate",
            "method": "b3_copy",
        }

    new_quote = literal_replace_in_quote(
        response_quote,
        slot=str(row.get("slot") or slot_norm),
        old_value=claim_value,
        v_anchor=best.value,
    )
    return {
        "arm": arm,
        "branch": "copy_hit",
        "action": "rewrite",
        "quote_before": response_quote,
        "quote_after": new_quote,
        "v_anchor": best.value,
        "source_field": best.source,
        "candidate_source": best.source,
        "evidence_kind": "copy_constrained",
        "method": "b3_copy",
    }
