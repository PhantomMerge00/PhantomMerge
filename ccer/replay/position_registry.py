"""Token position registry for P3 activation extraction (§8 P3a)."""
from __future__ import annotations

import re
from typing import Any, Literal

ClaimAnchorMode = Literal["metadata_first", "symmetric"]

from ccer.replay.answer_utils import extract_answer, normalize_final_synthesis_text

POSITION_NAMES = (
    "prompt_end",
    "commitment",
    "query_owner",
    "query_value",
    "claim_onset",
    "pre_value",
)

_CLAIM_LABEL_PRIORITY = {
    "constraint_projection": 0,
    "anchored_hallucination": 1,
    "cross_object_merge": 2,
}

_ATTRIBUTE_BULLET_RE = re.compile(r"^-\s*([^:\n]+?):\s*(.+)$", re.MULTILINE)
_SELECTED_PID_RE = re.compile(
    r"(###\s*Selected\s+product\s+ID:\s*)(\d+)",
    re.IGNORECASE,
)


def char_span_to_token_indices(
    offset_mapping: list[tuple[int, int]],
    start: int,
    end: int,
) -> list[int]:
    idx: list[int] = []
    for i, (a, b) in enumerate(offset_mapping):
        if b <= start or a >= end:
            continue
        if a == 0 and b == 0:
            continue
        idx.append(i)
    return idx


def _query_evidence(trajectory: dict[str, Any]) -> dict[str, Any] | None:
    for ev in trajectory.get("evidence") or []:
        if ev.get("scope") == "query_constraint":
            return ev
    return None


def _find_char_span(text: str, needle: str, *, after: int = 0) -> tuple[int, int]:
    start = text.find(needle, after)
    if start < 0:
        return -1, -1
    return start, start + len(needle)


def _claim_candidates(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    labels = set(_CLAIM_LABEL_PRIORITY)
    return [c for c in (trajectory.get("claims") or []) if c.get("legacy_label") in labels]


def _span_in_text(span: str, text: str) -> tuple[int, int] | None:
    if not span:
        return None
    start = text.find(span)
    if start >= 0:
        return start, start + len(span)
    lower_span = span.lower()
    start = text.lower().find(lower_span)
    if start >= 0:
        return start, start + len(span)
    return None


def resolve_symmetric_claim_pair(answer_body: str) -> tuple[str, str, str] | None:
    """Cross-pair symmetric anchor: same resolution order for PM and clean."""
    pair = _first_attribute_bullet(answer_body)
    if pair is not None:
        return pair[0], pair[1], "about_first_bullet"
    pair = _selected_product_id_line(answer_body)
    if pair is not None:
        return pair[0], pair[1], "selected_product_id_line"
    return None


def _resolve_claim_pair(
    trajectory: dict[str, Any],
    answer_body: str,
    *,
    claim_anchor_mode: ClaimAnchorMode,
) -> tuple[str, str, str] | None:
    if claim_anchor_mode == "symmetric":
        return resolve_symmetric_claim_pair(answer_body)
    claim_pair = _answer_anchored_claim(trajectory, answer_body)
    if claim_pair is not None:
        return claim_pair[0], claim_pair[1], "claim_metadata"
    pair = _first_attribute_bullet(answer_body)
    if pair is not None:
        return pair[0], pair[1], "about_first_bullet"
    pair = _selected_product_id_line(answer_body)
    if pair is not None:
        return pair[0], pair[1], "selected_product_id_line"
    return None


def _answer_anchored_claim(
    trajectory: dict[str, Any],
    answer_body: str,
) -> tuple[str, str] | None:
    """Pick claim whose response_span is present in the final answer (not metadata order)."""
    ranked: list[tuple[int, int, str, str]] = []
    for claim in _claim_candidates(trajectory):
        span = str(claim.get("response_span") or "").strip()
        value = str(claim.get("value") or "").strip()
        if not span or not value:
            continue
        loc = _span_in_text(span, answer_body)
        if loc is None:
            continue
        label = str(claim.get("legacy_label") or "")
        ranked.append(
            (
                _CLAIM_LABEL_PRIORITY.get(label, 99),
                loc[0],
                span,
                value,
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    _, _, span, value = ranked[0]
    return span, value


def _about_section_span(answer_body: str) -> str | None:
    m = re.search(
        r"(###\s*About\s+the\s+selected\s+product\s*)(.*)",
        answer_body,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    tail = m.group(2)
    compared = re.search(r"###\s*Compared", tail, re.IGNORECASE)
    if compared:
        tail = tail[: compared.start()]
    return tail.strip() or None


def _first_attribute_bullet(answer_body: str) -> tuple[str, str] | None:
    section = _about_section_span(answer_body)
    if not section:
        return None
    for match in _ATTRIBUTE_BULLET_RE.finditer(section):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if key and value:
            return f"{key}: {value}", value
    return None


def _compared_section_body(answer_body: str) -> str | None:
    """Body of ### Compared (not selected): PID section."""
    m = re.search(
        r"###\s*Compared\s*(?:\(\s*not\s+selected\s*\))?\s*:?\s*\d*",
        answer_body,
        re.IGNORECASE,
    )
    if not m:
        return None
    tail = answer_body[m.end() :]
    next_hdr = re.search(r"\n###\s+", tail)
    body = tail[: next_hdr.start()] if next_hdr else tail
    return body.strip() or None


def _first_compared_attribute_bullet(answer_body: str) -> tuple[str, str] | None:
    section = _compared_section_body(answer_body)
    if not section:
        return None
    for match in _ATTRIBUTE_BULLET_RE.finditer(section):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if key and value:
            return f"{key}: {value}", value
    return None


def resolve_live_claim_pair(answer_body: str) -> tuple[str, str, str] | None:
    """Task J′: greedy live claim anchor — About bullet, Compared bullet, Selected PID."""
    pair = _first_attribute_bullet(answer_body)
    if pair is not None:
        return pair[0], pair[1], "about_first_bullet"
    pair = _first_compared_attribute_bullet(answer_body)
    if pair is not None:
        return pair[0], pair[1], "compared_first_bullet"
    pair = _selected_product_id_line(answer_body)
    if pair is not None:
        return pair[0], pair[1], "selected_product_id_line"
    return None


def _selected_product_id_line(answer_body: str) -> tuple[str, str] | None:
    m = _SELECTED_PID_RE.search(answer_body)
    if not m:
        return None
    full_line = m.group(0).strip()
    pid = m.group(2)
    return full_line, pid


def _resolve_claim_onset_pre_value(
    *,
    answer_body: str,
    serialized_text: str,
    offset_mapping: list[tuple[int, int]],
    response_span: str,
    value: str,
) -> tuple[int | None, int | None, list[str]]:
    errors: list[str] = []
    claim_start = _span_in_text(response_span, answer_body)
    if claim_start is None:
        return None, None, ["response_span_not_in_answer"]

    abs_start = serialized_text.find(answer_body)
    if abs_start < 0:
        abs_start = serialized_text.lower().find(answer_body.lower())
    if abs_start < 0:
        return None, None, ["answer_body_not_in_serialized_text"]

    claim_char_start, claim_char_end = abs_start + claim_start[0], abs_start + claim_start[1]
    claim_idx = char_span_to_token_indices(offset_mapping, claim_char_start, claim_char_end)
    if not claim_idx:
        return None, None, ["claim_onset_token_not_found"]

    value_pos = response_span.lower().find(value.lower())
    if value_pos < 0 and ":" in response_span:
        derived_value = response_span.split(":", 1)[1].strip()
        if derived_value:
            derived_pos = response_span.lower().find(derived_value.lower())
            if derived_pos >= 0:
                value = derived_value
                value_pos = derived_pos
    if value_pos < 0:
        return claim_idx[0], None, ["claim_value_in_span_not_found"]

    val_start = claim_char_start + value_pos
    val_end = val_start + len(value)
    if val_start <= claim_char_start:
        pre_idx = claim_idx[:1]
        if len(claim_idx) > 1:
            pre_idx = claim_idx[:-1]
    else:
        pre_idx = char_span_to_token_indices(offset_mapping, claim_char_start, val_start)
    if not pre_idx:
        return claim_idx[0], None, ["pre_value_token_not_found"]
    return claim_idx[0], pre_idx[-1], errors


def resolve_positions(
    *,
    trajectory: dict[str, Any],
    serialized_text: str,
    offset_mapping: list[tuple[int, int]],
    prompt_token_count: int,
    full_token_count: int,
    claim_anchor_mode: ClaimAnchorMode = "metadata_first",
) -> dict[str, Any]:
    """Map semantic positions to token indices in the full teacher-forced sequence."""
    positions: dict[str, int | None] = {name: None for name in POSITION_NAMES}
    errors: list[str] = []

    if prompt_token_count > 0:
        positions["prompt_end"] = prompt_token_count - 1

    anchor = str((trajectory.get("commitment") or {}).get("action_anchor") or "")
    if anchor:
        s, e = _find_char_span(serialized_text, anchor)
        if s >= 0:
            idx = char_span_to_token_indices(offset_mapping, s, e)
            if idx:
                positions["commitment"] = idx[-1]
            else:
                errors.append("commitment_token_not_found")
        else:
            errors.append("commitment_char_span_not_found")
    else:
        errors.append("missing_action_anchor")

    qev = _query_evidence(trajectory)
    if qev and qev.get("char_span"):
        qs, qe = qev["char_span"]
        qidx = char_span_to_token_indices(offset_mapping, int(qs), int(qe))
        if qidx:
            positions["query_owner"] = qidx[0]
            positions["query_value"] = qidx[-1]
        else:
            errors.append("query_constraint_token_not_found")
    else:
        errors.append("missing_query_evidence")

    final_answer = str((trajectory.get("metadata") or {}).get("final_answer") or "")
    answer_body = extract_answer(normalize_final_synthesis_text(final_answer))

    resolved = _resolve_claim_pair(
        trajectory,
        answer_body,
        claim_anchor_mode=claim_anchor_mode,
    )
    source = resolved[2] if resolved else "none"
    claim_pair = (resolved[0], resolved[1]) if resolved else None

    if claim_pair and answer_body:
        response_span, value = claim_pair
        onset, pre_val, claim_errors = _resolve_claim_onset_pre_value(
            answer_body=answer_body,
            serialized_text=serialized_text,
            offset_mapping=offset_mapping,
            response_span=response_span,
            value=value,
        )
        positions["claim_onset"] = onset
        positions["pre_value"] = pre_val
        errors.extend(claim_errors)
        if onset is None and source != "claim_metadata":
            errors.append(f"missing_claim_or_answer_via_{source}")
    else:
        errors.append("missing_claim_or_answer")

    for name in POSITION_NAMES:
        idx = positions.get(name)
        if idx is not None and (idx < 0 or idx >= full_token_count):
            errors.append(f"{name}_out_of_range")
            positions[name] = None

    return {
        "positions": positions,
        "position_errors": errors,
        "claim_anchor_mode": claim_anchor_mode,
        "claim_anchor_source": source if claim_pair else None,
        "position_ok": not any(
            positions.get(n) is None for n in ("prompt_end", "commitment")
        ),
    }
