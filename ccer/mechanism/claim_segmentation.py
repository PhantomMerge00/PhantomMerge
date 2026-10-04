"""E2E claim segmentation from trajectory final response (no adjudication oracle)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from ccer.replay.answer_utils import extract_answer, extract_selected_product_id, normalize_final_synthesis_text
from ccer.replay.position_registry import _ATTRIBUTE_BULLET_RE

Section = Literal["about_selected", "compared", "other"]


def _norm_slot(slot: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(slot or "").strip().lower()).strip("_")


@dataclass(frozen=True)
class SegmentedClaim:
    response_quote: str
    slot: str
    slot_norm: str
    claim_value: str
    section: Section


def _final_answer_text(traj: dict[str, Any]) -> str:
    for m in reversed(traj.get("messages_final_call") or []):
        if m.get("role") == "assistant":
            return normalize_final_synthesis_text(str(m.get("content") or ""))
    return ""


def _about_section(answer_body: str) -> str:
    m = re.search(
        r"(###\s*About\s+the\s+selected\s+product\s*)(.*)",
        answer_body,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return ""
    tail = m.group(2)
    compared = re.search(r"###\s*Compared", tail, re.IGNORECASE)
    if compared:
        tail = tail[: compared.start()]
    return tail.strip()


def _compared_section(answer_body: str) -> str:
    m = re.search(
        r"###\s*Compared\s*(?:\(\s*not\s+selected\s*\))?\s*:?\s*\d*",
        answer_body,
        re.IGNORECASE,
    )
    if not m:
        return ""
    tail = answer_body[m.end() :]
    next_hdr = re.search(r"\n###\s+", tail)
    body = tail[: next_hdr.start()] if next_hdr else tail
    return body.strip()


def _bullets_from_section(section_text: str, section: Section) -> list[SegmentedClaim]:
    out: list[SegmentedClaim] = []
    for match in _ATTRIBUTE_BULLET_RE.finditer(section_text):
        key = match.group(1).strip()
        val = match.group(2).strip()
        if not key or not val:
            continue
        quote = f"{key}: {val}"
        out.append(
            SegmentedClaim(
                response_quote=quote,
                slot=key,
                slot_norm=_norm_slot(key),
                claim_value=val,
                section=section,
            )
        )
    return out


def segment_attribute_claims(traj: dict[str, Any]) -> list[SegmentedClaim]:
    """Parse all attribute bullets from final response (About + Compared)."""
    answer = extract_answer(_final_answer_text(traj))
    if not answer:
        return []
    claims: list[SegmentedClaim] = []
    about = _about_section(answer)
    if about:
        claims.extend(_bullets_from_section(about, "about_selected"))
    compared = _compared_section(answer)
    if compared:
        claims.extend(_bullets_from_section(compared, "compared"))
    return claims


def parse_bullet_quote(response_quote: str) -> tuple[str, str, str] | None:
    """Parse slot, slot_norm, claim_value from a bullet line."""
    text = str(response_quote or "").strip()
    m = re.match(r"^([^:]+)\s*:\s*(.+)$", text)
    if not m:
        return None
    slot = m.group(1).strip()
    val = m.group(2).strip()
    if not slot or not val:
        return None
    return slot, _norm_slot(slot), val


def match_segmented_to_quote(
    segments: list[SegmentedClaim],
    response_quote: str,
) -> SegmentedClaim | None:
    q = str(response_quote or "").strip()
    for s in segments:
        if s.response_quote.strip() == q:
            return s
    q_low = q.lower()
    for s in segments:
        if s.response_quote.strip().lower() == q_low:
            return s
    return None
