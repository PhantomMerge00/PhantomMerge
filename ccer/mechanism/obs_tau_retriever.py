"""Obs_τ retriever: in-trajectory anchor evidence for RARR/CoVe (replaces web search)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ccer.mechanism.line_l_anchor_extract import _donor_pids_from_row, _norm_slot
from ccer.mechanism.line_l_rewrite_quality import build_evidence_corpus

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class RARRQueryPassage:
    query: str
    passage: str
    source_pid: str
    rank: float = 0.0


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(str(text or "")) if p.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])


def _sliding_windows(
    sentences: list[str],
    *,
    max_sentences: int = 4,
    stride: int = 1,
) -> list[str]:
    if not sentences:
        return []
    if len(sentences) <= max_sentences:
        return [" ".join(sentences)]
    out: list[str] = []
    for i in range(0, len(sentences) - max_sentences + 1, max(1, stride)):
        out.append(" ".join(sentences[i : i + max_sentences]))
    return out


def _pool_pids(
    traj: dict[str, Any],
    anchor_pid: str,
    row: dict[str, Any] | None,
) -> list[str]:
    pids = [str(anchor_pid)]
    if row is not None:
        for d in _donor_pids_from_row(row):
            if d and d not in pids:
                pids.append(d)
    return pids


def _slot_queries(slot_norm: str, claim: str) -> list[str]:
    slot = _norm_slot(slot_norm)
    label = slot.replace("_", " ")
    return [
        f"What is the {label} of the anchor product?",
        f"Does the evidence support the {label} in: {claim[:120]}?",
        f"What {label} appears in the anchor product record?",
    ]


def _score_passage(passage: str, slot_norm: str, claim: str) -> float:
    slot = _norm_slot(slot_norm)
    p_low = passage.lower()
    score = 0.0
    if slot and slot.replace("_", " ") in p_low:
        score += 2.0
    for tok in claim.lower().split()[:12]:
        if len(tok) > 3 and tok in p_low:
            score += 0.2
    score += min(len(passage) / 500.0, 0.5)
    return score


def retrieve_passages_for_pid(
    traj: dict[str, Any],
    pid: str,
    slot_norm: str,
    claim: str,
    *,
    max_passages: int = 8,
) -> list[RARRQueryPassage]:
    corpus = build_evidence_corpus(traj, pid)
    if not corpus.strip():
        return []
    windows = _sliding_windows(_split_sentences(corpus), max_sentences=4, stride=1)
    scored: list[tuple[float, str]] = []
    for w in windows:
        scored.append((_score_passage(w, slot_norm, claim), w))
    scored.sort(key=lambda x: -x[0])
    queries = _slot_queries(slot_norm, claim)
    out: list[RARRQueryPassage] = []
    for rank, (sc, passage) in enumerate(scored[:max_passages]):
        q = queries[rank % len(queries)]
        out.append(RARRQueryPassage(query=q, passage=passage, source_pid=pid, rank=sc))
    return out


def retrieve_for_rarr(
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    claim: str,
    slot_norm: str,
    row: dict[str, Any] | None = None,
    max_evidences: int = 5,
) -> list[RARRQueryPassage]:
    """Return ranked (query, passage) pairs from Obs_τ corpus (anchor + CEM donors)."""
    all_passages: list[RARRQueryPassage] = []
    for pid in _pool_pids(traj, anchor_pid, row):
        all_passages.extend(retrieve_passages_for_pid(traj, pid, slot_norm, claim))
    all_passages.sort(key=lambda p: -p.rank)
    seen: set[str] = set()
    unique: list[RARRQueryPassage] = []
    for p in all_passages:
        key = p.passage[:200]
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique[:max_evidences]


def top_evidence_snippet(
    traj: dict[str, Any],
    anchor_pid: str,
    slot_norm: str,
    claim: str,
    *,
    row: dict[str, Any] | None = None,
    limit: int = 1500,
) -> str:
    hits = retrieve_for_rarr(
        traj,
        anchor_pid=anchor_pid,
        claim=claim,
        slot_norm=slot_norm,
        row=row,
        max_evidences=3,
    )
    if not hits:
        return build_evidence_corpus(traj, anchor_pid)[:limit]
    return "\n\n".join(h.passage for h in hits)[:limit]
