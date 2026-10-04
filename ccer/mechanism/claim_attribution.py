"""Task H: strict claim attribution target extraction for CEM cross pairs."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ccer.adjudication.loader import (
    load_adjudication_index,
    resolve_cem_swap_target,
)
from ccer.mechanism.pair_select import load_trajectory_index
from ccer.paths import P3_DIR
from ccer.replay.answer_utils import extract_answer, extract_selected_product_id, normalize_final_synthesis_text
from ccer.replay.position_registry import _ATTRIBUTE_BULLET_RE, _about_section_span

TASK_H_DIR = P3_DIR / "task_h"
BASELINE_ANNOTATIONS_PATH = TASK_H_DIR / "claim_target_baseline_annotations.jsonl"

ClaimSection = Literal["about_selected", "compared", "other"]
ExtractionMethod = Literal[
    "adjudication_quote_v1",
    "cem_slot_bullet_v1",
    "human_baseline_v1",
    "clean_parse_quote_v1",
    "clean_selected_pid_fallback_v1",
]

_PID_IN_SENTENCE_RE = re.compile(r"\b(\d{8,12})\b")
_COMPARED_HEADER_RE = re.compile(
    r"###\s*Compared\s*(?:\(\s*not\s+selected\s*\))?\s*:?\s*(\d+)?",
    re.IGNORECASE,
)


def _norm_slot(slot: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(slot or "").strip().lower()).strip("_")


def _norm_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _compared_section_span(answer_body: str) -> tuple[str | None, str | None]:
    """Return (compared_body, compared_pid_from_header)."""
    m = _COMPARED_HEADER_RE.search(answer_body)
    if not m:
        return None, None
    header_end = m.end()
    compared_pid = str(m.group(1) or "").strip() or None
    tail = answer_body[header_end:]
    next_hdr = re.search(r"\n###\s+", tail)
    body = tail[: next_hdr.start()] if next_hdr else tail
    return body.strip() or None, compared_pid


def _bullet_matches_slot_value(
    *,
    key: str,
    value: str,
    slot_norm: str,
    claim_value: str,
    response_quote: str,
) -> bool:
    if _norm_slot(key) != _norm_slot(slot_norm):
        return False
    nv = _norm_value(value)
    cv = _norm_value(claim_value)
    if nv == cv:
        return True
    rq = str(response_quote or "").strip()
    if rq and _norm_value(rq.split(":", 1)[-1]) == nv:
        return True
    return False


def _find_claim_in_section(
    section_text: str,
    *,
    slot_norm: str,
    claim_value: str,
    response_quote: str,
    slot_only: bool = False,
) -> str | None:
    for match in _ATTRIBUTE_BULLET_RE.finditer(section_text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if _norm_slot(key) != _norm_slot(slot_norm):
            continue
        if slot_only or _bullet_matches_slot_value(
            key=key,
            value=value,
            slot_norm=slot_norm,
            claim_value=claim_value,
            response_quote=response_quote,
        ):
            return f"{key}: {value}"
    rq = str(response_quote or "").strip()
    if not slot_only and rq and rq in section_text:
        return rq
    return None


@dataclass
class ClaimAttributionParse:
    claim_extraction_method: ExtractionMethod | str
    claim_sentence: str | None
    claim_section: ClaimSection | str
    claim_target_entity: str | None
    scorable: bool
    parse_error: str | None
    clean_donor_entity: str | None = None
    claim_attribution_diff: bool | None = None
    claim_attribution_corrected: bool | None = None
    inline_pid_conflict: bool = False
    auto_vs_human_mismatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_entity_for_claim(
    *,
    claim_section: ClaimSection,
    claim_sentence: str,
    selected_pid: str | None,
    compared_pid: str | None,
) -> tuple[str | None, str | None]:
    """Return (entity_id, parse_error)."""
    inline_pids = _PID_IN_SENTENCE_RE.findall(claim_sentence)
    section_entity: str | None
    if claim_section == "about_selected":
        section_entity = selected_pid
    elif claim_section == "compared":
        section_entity = compared_pid
    else:
        section_entity = None

    if not section_entity:
        if len(inline_pids) == 1:
            return inline_pids[0], None
        return None, "missing_section_entity"

    if inline_pids:
        unique_inline = set(inline_pids)
        if unique_inline != {section_entity}:
            return None, "inline_pid_conflicts_with_section"
    return section_entity, None


def parse_claim_attribution_from_text(
    text: str,
    *,
    pm_trajectory: dict[str, Any] | None = None,
    slot_norm: str | None = None,
    claim_value: str | None = None,
    response_quote: str | None = None,
    method_hint: ExtractionMethod | str = "adjudication_quote_v1",
) -> ClaimAttributionParse:
    """Parse which entity owns the CEM claim sentence in ``text``."""
    if pm_trajectory is not None:
        target = resolve_cem_swap_target(pm_trajectory)
        if target is None:
            return ClaimAttributionParse(
                claim_extraction_method=method_hint,
                claim_sentence=None,
                claim_section="other",
                claim_target_entity=None,
                scorable=False,
                parse_error="no_cem_swap_target",
            )
        slot_norm = slot_norm or target.slot_norm
        claim_value = claim_value or target.claim_value
        response_quote = response_quote or target.response_quote

    if not slot_norm or not claim_value:
        return ClaimAttributionParse(
            claim_extraction_method=method_hint,
            claim_sentence=None,
            claim_section="other",
            claim_target_entity=None,
            scorable=False,
            parse_error="missing_slot_or_value",
        )

    answer_body = extract_answer(normalize_final_synthesis_text(text))
    if not answer_body:
        return ClaimAttributionParse(
            claim_extraction_method=method_hint,
            claim_sentence=None,
            claim_section="other",
            claim_target_entity=None,
            scorable=False,
            parse_error="empty_answer_body",
        )

    selected_pid = extract_selected_product_id(text)
    about = _about_section_span(answer_body) or ""
    compared_body, compared_pid = _compared_section_span(answer_body)

    claim_sentence: str | None = None
    claim_section: ClaimSection = "other"
    extraction_method: ExtractionMethod | str = method_hint

    if about:
        hit = _find_claim_in_section(
            about,
            slot_norm=slot_norm,
            claim_value=claim_value,
            response_quote=str(response_quote or ""),
        )
        if hit:
            claim_sentence = hit
            claim_section = "about_selected"
            extraction_method = "adjudication_quote_v1"

    if claim_sentence is None and compared_body:
        hit = _find_claim_in_section(
            compared_body,
            slot_norm=slot_norm,
            claim_value=claim_value,
            response_quote=str(response_quote or ""),
        )
        if hit:
            claim_sentence = hit
            claim_section = "compared"
            extraction_method = "adjudication_quote_v1"

    # CEM PM answers may write a wrong slot value vs adjudication quote; still section-aware.
    if claim_sentence is None and pm_trajectory is not None and about:
        hit = _find_claim_in_section(
            about,
            slot_norm=slot_norm,
            claim_value=claim_value,
            response_quote=str(response_quote or ""),
            slot_only=True,
        )
        if hit:
            claim_sentence = hit
            claim_section = "about_selected"
            extraction_method = "cem_slot_bullet_v1"

    if claim_sentence is None and pm_trajectory is not None and compared_body:
        hit = _find_claim_in_section(
            compared_body,
            slot_norm=slot_norm,
            claim_value=claim_value,
            response_quote=str(response_quote or ""),
            slot_only=True,
        )
        if hit:
            claim_sentence = hit
            claim_section = "compared"
            extraction_method = "cem_slot_bullet_v1"

    if claim_sentence is None:
        return ClaimAttributionParse(
            claim_extraction_method=extraction_method,
            claim_sentence=None,
            claim_section="other",
            claim_target_entity=None,
            scorable=False,
            parse_error="claim_sentence_not_found",
        )

    entity, err = _resolve_entity_for_claim(
        claim_section=claim_section,
        claim_sentence=claim_sentence,
        selected_pid=selected_pid,
        compared_pid=compared_pid,
    )
    inline_conflict = err == "inline_pid_conflicts_with_section"
    return ClaimAttributionParse(
        claim_extraction_method=extraction_method,
        claim_sentence=claim_sentence,
        claim_section=claim_section,
        claim_target_entity=entity,
        scorable=entity is not None and not inline_conflict,
        parse_error=err,
        inline_pid_conflict=inline_conflict,
    )


def load_baseline_annotations(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = path or BASELINE_ANNOTATIONS_PATH
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tid = str(row.get("pm_trajectory_id") or "")
        if tid:
            out[tid] = row
    return out


def resolve_clean_donor_entity(
    *,
    pm_trajectory: dict[str, Any],
    clean_trajectory: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Entity the clean donor attributes the CEM slot claim to (for corrected metric)."""
    target = resolve_cem_swap_target(pm_trajectory)
    clean_fa = str((clean_trajectory.get("metadata") or {}).get("final_answer") or "")
    if not clean_fa.strip():
        return None, "empty_clean_final_answer"

    pm_parsed = parse_claim_attribution_from_text(
        str((pm_trajectory.get("metadata") or {}).get("final_answer") or ""),
        pm_trajectory=pm_trajectory,
    )

    if target is not None:
        parsed = parse_claim_attribution_from_text(
            clean_fa,
            slot_norm=target.slot_norm,
            claim_value=target.claim_value,
            response_quote=target.response_quote,
        )
        if parsed.claim_target_entity:
            return parsed.claim_target_entity, "clean_parse_quote_v1"

    if pm_parsed.claim_section == "about_selected":
        selected = extract_selected_product_id(clean_fa)
        if selected:
            return selected, "clean_selected_pid_fallback_v1"

    if pm_parsed.claim_section == "compared":
        answer_body = extract_answer(normalize_final_synthesis_text(clean_fa))
        if answer_body:
            _body, compared_pid = _compared_section_span(answer_body)
            if compared_pid:
                return compared_pid, "clean_compared_pid_fallback_v1"

    return None, "clean_donor_entity_unresolved"


def parse_clean_donor_entity(
    clean_trajectory: dict[str, Any],
    *,
    pm_trajectory: dict[str, Any] | None = None,
) -> str | None:
    if pm_trajectory is None:
        return None
    entity, _method = resolve_clean_donor_entity(
        pm_trajectory=pm_trajectory,
        clean_trajectory=clean_trajectory,
    )
    return entity


def evaluate_claim_attribution_pair(
    *,
    ni_text: str,
    ti_text: str,
    pm_trajectory: dict[str, Any],
    clean_trajectory: dict[str, Any],
    human_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Enrich one sweep pair row with Task H attribution fields."""
    target = resolve_cem_swap_target(pm_trajectory)
    clean_entity, clean_entity_method = resolve_clean_donor_entity(
        pm_trajectory=pm_trajectory,
        clean_trajectory=clean_trajectory,
    )
    pm_tid = str(
        pm_trajectory.get("trajectory_id")
        or pm_trajectory.get("pm_trajectory_id")
        or ""
    )
    if human_baseline is None and pm_tid:
        human_baseline = load_baseline_annotations().get(pm_tid)

    baseline_parse = parse_claim_attribution_from_text(ni_text, pm_trajectory=pm_trajectory)
    ti_parse = parse_claim_attribution_from_text(ti_text, pm_trajectory=pm_trajectory)

    baseline_entity = baseline_parse.claim_target_entity
    extraction_method = baseline_parse.claim_extraction_method
    if human_baseline and human_baseline.get("claim_target_baseline"):
        baseline_entity = str(human_baseline["claim_target_baseline"])
        extraction_method = str(human_baseline.get("method") or "human_baseline_v1")
        auto_mismatch = (
            baseline_parse.claim_target_entity is not None
            and str(baseline_parse.claim_target_entity) != baseline_entity
        )
    else:
        auto_mismatch = False

    if human_baseline and human_baseline.get("clean_donor_entity"):
        clean_entity = str(human_baseline["clean_donor_entity"])

    ti_entity = ti_parse.claim_target_entity
    scorable = bool(
        baseline_entity
        and ti_entity
        and baseline_parse.scorable
        and ti_parse.scorable
        and not baseline_parse.inline_pid_conflict
        and not ti_parse.inline_pid_conflict
    )
    attr_diff: bool | None = None
    attr_corrected: bool | None = None
    if scorable:
        attr_diff = str(ti_entity) != str(baseline_entity)
        attr_corrected = clean_entity is not None and str(ti_entity) == str(clean_entity)

    return {
        "claim_extraction_method": extraction_method,
        "claim_sentence_baseline": baseline_parse.claim_sentence,
        "claim_sentence_ti": ti_parse.claim_sentence,
        "claim_section_baseline": baseline_parse.claim_section,
        "claim_section_ti": ti_parse.claim_section,
        "claim_target_baseline": baseline_entity,
        "claim_target_ti": ti_entity,
        "clean_donor_entity": clean_entity,
        "clean_donor_entity_method": clean_entity_method,
        "claim_attribution_diff": attr_diff,
        "claim_attribution_corrected": attr_corrected,
        "claim_attribution_scorable": scorable,
        "claim_parse_error_baseline": baseline_parse.parse_error,
        "claim_parse_error_ti": ti_parse.parse_error,
        "auto_vs_human_mismatch": auto_mismatch,
    }


def aggregate_claim_attribution_metrics(pair_details: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize attribution metrics over scorable, patched rows."""
    from ccer.mechanism.stats import wilson_ci

    eligible = [
        d
        for d in pair_details
        if d.get("patched") and d.get("claim_attribution_scorable")
    ]
    n = len(eligible)
    diff_k = sum(1 for d in eligible if d.get("claim_attribution_diff"))
    corrected_k = sum(1 for d in eligible if d.get("claim_attribution_corrected"))
    return {
        "n_claim_attribution_scorable": n,
        "n_claim_attribution_diff": diff_k,
        "claim_attribution_diff_rate": diff_k / n if n else 0.0,
        "claim_attribution_diff_ci95": wilson_ci(diff_k, n),
        "n_claim_attribution_corrected": corrected_k,
        "claim_attribution_corrected_rate": corrected_k / n if n else 0.0,
        "claim_attribution_corrected_ci95": wilson_ci(corrected_k, n),
    }


def generate_baseline_annotation_rows() -> list[dict[str, Any]]:
    """Semi-auto baseline annotations for all CEM cross pairs."""
    from ccer.mechanism.pair_select import cem_valid_pair_ids

    rows_index = load_trajectory_index()
    pairs = cem_valid_pair_ids(include_matched_clean=True)["pm_clean_cross_pairs"]
    out: list[dict[str, Any]] = []
    for cp in pairs:
        pm_tid = str(cp["pm_trajectory_id"])
        clean_tid = str(cp["clean_trajectory_id"])
        pm_traj = rows_index.get(pm_tid) or {}
        clean_traj = rows_index.get(clean_tid) or {}
        final_answer = str((pm_traj.get("metadata") or {}).get("final_answer") or "")
        parsed = parse_claim_attribution_from_text(final_answer, pm_trajectory=pm_traj)
        clean_entity, clean_entity_method = resolve_clean_donor_entity(
            pm_trajectory=pm_traj,
            clean_trajectory=clean_traj,
        )
        target = resolve_cem_swap_target(pm_traj)
        out.append(
            {
                "pm_trajectory_id": pm_tid,
                "clean_trajectory_id": clean_tid,
                "claim_target_baseline": parsed.claim_target_entity,
                "clean_donor_entity": clean_entity,
                "clean_donor_entity_method": clean_entity_method,
                "slot_norm": target.slot_norm if target else None,
                "claim_value": target.claim_value if target else None,
                "response_quote": target.response_quote if target else None,
                "claim_sentence": parsed.claim_sentence,
                "claim_section": parsed.claim_section,
                "method": "human_baseline_v1",
                "annotator": "task_h_semi_auto_v1",
                "notes": "semi-auto from stored final_answer; parser adjudication_quote_v1",
                "scorable": parsed.scorable,
                "parse_error": parsed.parse_error,
            }
        )
    return out


def write_baseline_annotations(path: Path | None = None) -> Path:
    path = path or BASELINE_ANNOTATIONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_baseline_annotation_rows()
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
