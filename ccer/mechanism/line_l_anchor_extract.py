"""Line L: anchor-evidence-grounded extractive rewrite (deterministic, no LLM)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from ccer.counterfactual.base import find_product_record_snippets

Branch = Literal["extraction_hit", "extraction_miss", "skip_unflagged", "skip_no_slot"]

_PRODUCT_JSON_RE = re.compile(
    r'\{[^{}]*"product_id"\s*:\s*"?(\d+)"?[^{}]*\}'
)

# adjudication slot_norm -> tool_observation JSON field names (exact keys only)
_SLOT_TO_PRODUCT_FIELDS: dict[str, tuple[str, ...]] = {
    "price": ("price",),
    "title": ("title",),
    "shop_id": ("shop_id",),
    "material": ("material",),
    "formulation": ("formulation",),
    "brand": ("brand",),
    "color": ("color", "colour"),
    "size": ("size",),
    "type": ("type",),
    "capacity": ("capacity",),
    "quantity": ("quantity",),
    "description": ("description",),
    "compatibility": ("compatibility",),
    "interface_port": ("interface", "port", "interface_port"),
    "cut_style": ("cut_style",),
    "length": ("length",),
    "tip_size": ("tip_size",),
    "surge_protection": ("surge_protection",),
    "product_configuration": ("product_configuration", "configuration"),
}


def _norm_slot(slot: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(slot or "").strip().lower()).strip("_")


def _norm_value_compare(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    text = re.sub(r"\s*(php|pesos|usd)\s*$", "", text, flags=re.I)
    return text


_SLOT_ALIASES: dict[str, str] = {
    "material": "material",
    "price": "price",
    "title": "title",
    "color": "color",
    "colour": "color",
    "size": "size",
    "brand": "brand",
    "type": "type",
}


@dataclass(frozen=True)
class ExtractionResult:
    branch: Branch
    v_anchor: str | None = None
    source_field: str | None = None
    evidence_kind: str | None = None
    miss_reason: str | None = None


def parse_anchor_evidence_quote(anchor_evidence_quote: str) -> dict[str, str]:
    """Parse semicolon-delimited slot:value fields from adjudication quote."""
    out: dict[str, str] = {}
    text = str(anchor_evidence_quote or "").strip()
    if not text:
        return out
    if text.startswith("{"):
        return out
    for part in re.split(r";+", text):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, val = part.split(":", 1)
        key_n = _norm_slot(key)
        val_s = val.strip()
        if key_n and val_s:
            out[key_n] = val_s
    return out


def _entity_ids(ev: dict[str, Any]) -> list[str]:
    raw = ev.get("entity_ids") or ev.get("source_id") or []
    if isinstance(raw, str):
        out = [raw]
    else:
        out = [str(x) for x in raw]
    hint = str(ev.get("entity_hint") or "").strip()
    if hint and hint not in out:
        out.append(hint)
    return out


def _trajectory_evidence(traj: dict[str, Any]) -> list[dict[str, Any]]:
    from ccer.mechanism.anchor_resolve_heuristic import _evidence_records

    return _evidence_records(traj)


def _final_call_user_content(traj: dict[str, Any]) -> str:
    for msg in traj.get("messages_final_call") or []:
        if str(msg.get("role") or "") == "user":
            return str(msg.get("content") or "")
    return ""


def _merge_product_records(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    merged: dict[str, Any] = {}
    for rec in sorted(records, key=lambda r: len(r)):
        merged.update(rec)
    return merged


def parse_product_title_attributes(title: str) -> dict[str, str]:
    """Parse semicolon-delimited ``Key: Value`` pairs from tool product title strings."""
    out: dict[str, str] = {}
    for part in re.split(r";+", str(title or "")):
        part = part.strip()
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        key_n = _norm_slot(key)
        val_s = val.strip()
        if key_n and val_s:
            out[key_n] = val_s
    return out


# claim slot_norm -> Obs_τ(a_τ) attribute keys (paper: Obs_τ(a_τ)[slot(c)])
_SLOT_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "fit": ("fit", "fitment", "type", "style"),
    "length": ("length", "pants_length", "item_length", "size_length"),
    "size": ("size",),
    "color": ("color", "colour"),
    "material": ("material", "fabric", "main_fabric_composition"),
    "price": ("price",),
    "brand": ("brand",),
    "type": ("type", "style", "item_type"),
    "interface_port": ("interface", "port", "interface_port", "interface_material"),
    "formulation": ("formulation",),
    "design": ("design", "style", "pattern"),
    "compatibility": ("compatibility", "fitment"),
    "plug_compatibility": ("plug_compatibility", "compatibility", "interface"),
    "size_class": ("size", "size_class", "type"),
}


def build_obs_tau_dict(traj: dict[str, Any], anchor_pid: str) -> dict[str, str]:
    """
    Build Obs_τ(a_τ) as attribute→value map from structured tool observations.

    Sources (merged): per-slot trajectory evidence, product JSON fields, title KV pairs.
    """
    obs: dict[str, str] = {}
    if not anchor_pid:
        return obs

    for ev in _trajectory_evidence(traj):
        ents = _entity_ids(ev)
        if anchor_pid not in ents:
            continue
        ev_slot = _norm_slot(str(ev.get("slot_norm") or ev.get("slot_raw") or ""))
        if ev_slot in ("", "product_record", "user_utterance", "service_filter"):
            continue
        val = str(ev.get("value_raw") or ev.get("value_norm") or "").strip()
        if val:
            obs[ev_slot] = val

    record = _anchor_product_record(traj, anchor_pid)
    if not record:
        return obs

    for field, val in record.items():
        if field in ("product_id", "title"):
            continue
        if val is not None and str(val).strip():
            obs[_norm_slot(str(field))] = str(val).strip()
    obs.update(parse_product_title_attributes(str(record.get("title") or "")))
    return obs


def lookup_obs_tau_slot(
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    slot_norm: str,
) -> ExtractionResult:
    """Obs_τ(a_τ)[slot(c)] — direct structured lookup, no free-text extraction."""
    slot_key = _norm_slot(slot_norm)
    if not slot_key or not anchor_pid:
        return ExtractionResult("extraction_miss", miss_reason="missing_slot_or_anchor")

    obs = build_obs_tau_dict(traj, anchor_pid)
    if not obs:
        return ExtractionResult("extraction_miss", miss_reason="empty_obs_tau")

    canonical = _SLOT_ALIASES.get(slot_key, slot_key)
    query_keys = _SLOT_QUERY_ALIASES.get(canonical, (canonical, slot_key))
    for key in query_keys:
        if key in obs and str(obs[key]).strip():
            return ExtractionResult(
                "extraction_hit",
                v_anchor=str(obs[key]).strip(),
                source_field=f"obs_tau[{key}]",
                evidence_kind="obs_tau_dict",
            )
    for key in query_keys:
        for obs_key, val in obs.items():
            if key in obs_key or obs_key in key:
                if str(val).strip():
                    return ExtractionResult(
                        "extraction_hit",
                        v_anchor=str(val).strip(),
                        source_field=f"obs_tau[{obs_key}]",
                        evidence_kind="obs_tau_dict",
                    )
    return ExtractionResult(
        "extraction_miss",
        miss_reason="obs_tau_slot_missing",
    )


def _records_from_evidence_product_record(
    traj: dict[str, Any], anchor_pid: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in _trajectory_evidence(traj):
        ents = _entity_ids(ev)
        if anchor_pid not in ents:
            continue
        slot = _norm_slot(str(ev.get("slot_norm") or ev.get("slot_raw") or ""))
        if slot != "product_record":
            continue
        raw = str(ev.get("value_raw") or ev.get("value_norm") or "")
        if not raw:
            continue
        try:
            rec = json.loads(raw)
            if isinstance(rec, dict):
                if str(rec.get("product_id") or anchor_pid) == str(anchor_pid):
                    out.append(rec)
        except json.JSONDecodeError:
            continue
    return out


def _anchor_product_record(traj: dict[str, Any], anchor_pid: str) -> dict[str, Any] | None:
    """Richest product JSON object for anchor_pid from final-call user Obs_τ(a_τ)."""
    if not anchor_pid:
        return None
    user = _final_call_user_content(traj)
    candidates: list[dict[str, Any]] = []
    for match in _PRODUCT_JSON_RE.finditer(user):
        try:
            record = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if str(record.get("product_id")) != str(anchor_pid):
            continue
        candidates.append(record)
    for snippet in find_product_record_snippets(user, str(anchor_pid)):
        try:
            record = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if str(record.get("product_id")) == str(anchor_pid):
            candidates.append(record)
    candidates.extend(_records_from_evidence_product_record(traj, anchor_pid))
    return _merge_product_records(candidates)


def lookup_from_anchor_product_json(
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    slot_norm: str,
) -> ExtractionResult:
    """Lookup slot value from Obs_τ(a_τ) structured product observations."""
    return lookup_obs_tau_slot(traj, anchor_pid=anchor_pid, slot_norm=slot_norm)


def lookup_from_trajectory_evidence(
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    slot_norm: str,
) -> ExtractionResult:
    slot_key = _norm_slot(slot_norm)
    canonical = _SLOT_ALIASES.get(slot_key, slot_key)
    if not slot_key or not anchor_pid:
        return ExtractionResult("extraction_miss", miss_reason="missing_slot_or_anchor")

    for ev in _trajectory_evidence(traj):
        ents = _entity_ids(ev)
        if anchor_pid not in ents:
            continue
        ev_slot = _norm_slot(str(ev.get("slot_norm") or ev.get("slot_raw") or ""))
        if ev_slot == "product_record":
            raw = str(ev.get("value_raw") or ev.get("value_norm") or "")
            try:
                nested = json.loads(raw)
                if isinstance(nested, dict):
                    for field in _SLOT_TO_PRODUCT_FIELDS.get(canonical, (canonical,)):
                        val = nested.get(field)
                        if val is not None and str(val).strip():
                            return ExtractionResult(
                                "extraction_hit",
                                v_anchor=str(val).strip(),
                                source_field=f"evidence.product_record.{field}",
                                evidence_kind="nested_product_record",
                            )
            except json.JSONDecodeError:
                pass
        if not ev_slot:
            continue
        if ev_slot != canonical and canonical not in ev_slot and ev_slot not in canonical:
            continue
        val = str(ev.get("value_raw") or ev.get("value_norm") or "").strip()
        if val:
            return ExtractionResult(
                "extraction_hit",
                v_anchor=val,
                source_field=f"evidence.{ev.get('evidence_id')}.{ev_slot}",
                evidence_kind=str(ev.get("evidence_kind") or "structured_evidence"),
            )
    return ExtractionResult("extraction_miss", miss_reason="no_trajectory_evidence_match")


def _donor_pids_from_row(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for d in row.get("donor_owners") or []:
        if isinstance(d, dict):
            pid = str(d.get("pid") or "").strip()
        else:
            pid = str(d).strip()
        if pid:
            out.append(pid)
    return out


def detect_donor_pollution(
    row: dict[str, Any],
    traj: dict[str, Any],
    acceptance_pid: str,
    slot_norm: str,
) -> bool:
    """True when claim_value is grounded on donor corpus but not anchor structured field."""
    from ccer.mechanism.line_l_rewrite_quality import build_evidence_corpus, verify_grounded
    from ccer.mechanism.line_l_wrong_anchor import resolve_wrong_anchor_pid

    claim_value = _norm_value_compare(str(row.get("claim_value") or ""))
    if not claim_value or not acceptance_pid:
        return False
    anchor_hit = lookup_from_anchor_product_json(
        traj, anchor_pid=acceptance_pid, slot_norm=slot_norm
    )
    anchor_val = ""
    if anchor_hit.branch == "extraction_hit" and anchor_hit.v_anchor:
        anchor_val = _norm_value_compare(anchor_hit.v_anchor)
    if anchor_val and anchor_val == claim_value:
        return False
    donor_pids = list(_donor_pids_from_row(row))
    wrong = resolve_wrong_anchor_pid(row, traj)
    if wrong and wrong not in donor_pids:
        donor_pids.append(wrong)
    acceptance_corpus = build_evidence_corpus(traj, acceptance_pid)
    for donor_pid in donor_pids:
        if donor_pid == acceptance_pid:
            continue
        donor_corpus = build_evidence_corpus(traj, donor_pid)
        if not verify_grounded(str(row.get("claim_value") or ""), donor_corpus):
            continue
        if anchor_val and anchor_val != claim_value:
            return True
        donor_hit = lookup_from_anchor_product_json(traj, anchor_pid=donor_pid, slot_norm=slot_norm)
        if donor_hit.branch == "extraction_hit" and donor_hit.v_anchor:
            if _norm_value_compare(donor_hit.v_anchor) == claim_value:
                return True
    return False


def lookup_cem_corrective_value(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    slot_norm: str,
) -> ExtractionResult:
    """CEM: if claim_value matches donor field, use anchor's conflicting field."""
    if not detect_donor_pollution(row, traj, anchor_pid, slot_norm):
        return ExtractionResult("extraction_miss", miss_reason="not_cem")
    claim_value = _norm_value_compare(str(row.get("claim_value") or ""))
    if not claim_value:
        return ExtractionResult("extraction_miss", miss_reason="no_claim_value")
    slot_key = _norm_slot(slot_norm)
    anchor_hit = lookup_from_anchor_product_json(traj, anchor_pid=anchor_pid, slot_norm=slot_norm)
    if anchor_hit.branch != "extraction_hit" or not anchor_hit.v_anchor:
        return ExtractionResult("extraction_miss", miss_reason="anchor_no_field")
    anchor_val = _norm_value_compare(anchor_hit.v_anchor)
    if anchor_val == claim_value:
        return ExtractionResult("extraction_miss", miss_reason="anchor_matches_claim")
    for donor_pid in _donor_pids_from_row(row):
        if donor_pid == anchor_pid:
            continue
        donor_hit = lookup_from_anchor_product_json(traj, anchor_pid=donor_pid, slot_norm=slot_norm)
        if donor_hit.branch == "extraction_hit" and donor_hit.v_anchor:
            if _norm_value_compare(donor_hit.v_anchor) == claim_value:
                return ExtractionResult(
                    "extraction_hit",
                    v_anchor=anchor_hit.v_anchor,
                    source_field=f"cem_corrective.{anchor_hit.source_field}",
                    evidence_kind="cem_donor_pollution_corrected",
                )
    return ExtractionResult("extraction_miss", miss_reason="no_donor_match")


def lookup_slot_value(
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    slot_norm: str,
    anchor_evidence_quote: str | None = None,
    row: dict[str, Any] | None = None,
) -> ExtractionResult:
    hit = lookup_from_trajectory_evidence(traj, anchor_pid=anchor_pid, slot_norm=slot_norm)
    if hit.branch == "extraction_hit":
        return hit

    hit = lookup_from_anchor_product_json(traj, anchor_pid=anchor_pid, slot_norm=slot_norm)
    if hit.branch == "extraction_hit":
        return hit

    if row is not None:
        cem = lookup_cem_corrective_value(row, traj, anchor_pid=anchor_pid, slot_norm=slot_norm)
        if cem.branch == "extraction_hit":
            return cem

    fields = parse_anchor_evidence_quote(anchor_evidence_quote or "")
    slot_key = _norm_slot(slot_norm)
    canonical = _SLOT_ALIASES.get(slot_key, slot_key)
    for key in (canonical, slot_key):
        if key in fields and fields[key].strip():
            return ExtractionResult(
                "extraction_hit",
                v_anchor=fields[key].strip(),
                source_field=f"anchor_evidence_quote.{key}",
                evidence_kind="adjudication_quote",
            )
    return ExtractionResult("extraction_miss", miss_reason="no_exact_field_match")


def literal_replace_in_quote(
    response_quote: str,
    *,
    slot: str,
    old_value: str,
    v_anchor: str,
) -> str:
    """Replace claim value span with anchor literal (no generation)."""
    quote = str(response_quote or "")
    slot_s = str(slot or "").strip()
    old_s = str(old_value or "").strip()
    new_s = str(v_anchor or "").strip()
    if not quote or not new_s:
        return quote

    prefix = f"{slot_s}:"
    if prefix.lower() in quote.lower():
        m = re.match(rf"(?i)({re.escape(slot_s)}\s*:\s*)(.+)", quote)
        if m:
            rest = m.group(2)
            if old_s and old_s in rest:
                rest = rest.replace(old_s, new_s, 1)
            else:
                paren = rest.find("(")
                if paren > 0:
                    rest = new_s + rest[paren:]
                else:
                    rest = new_s
            return f"{slot_s}: {rest.lstrip()}"
    if old_s and old_s in quote:
        return quote.replace(old_s, new_s, 1)
    return f"{slot_s}: {new_s}" if slot_s else quote


def apply_extractive_rewrite(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
    arm: str = "target_extraction",
) -> dict[str, Any]:
    """Return intervention outcome for one claim row."""
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
        }

    if not slot_norm:
        return {
            "arm": arm,
            "branch": "skip_no_slot",
            "action": "delete",
            "quote_before": response_quote,
            "quote_after": None,
            "miss_reason": "no_slot_norm",
        }

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
            "slot_norm": slot_norm,
            "anchor_pid": anchor_pid,
        }

    return {
        "arm": arm,
        "branch": "extraction_miss",
        "action": "delete",
        "quote_before": response_quote,
        "quote_after": None,
        "miss_reason": extraction.miss_reason,
        "slot_norm": slot_norm,
        "anchor_pid": anchor_pid,
    }
