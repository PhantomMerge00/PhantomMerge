"""Line L+: rewrite quality metrics and evidence grounding verification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ccer.mechanism.line_k_claim_filter import REWRITE_STUB
from ccer.mechanism.line_l_anchor_extract import (
    _anchor_product_record,
    _final_call_user_content,
    _norm_slot,
    parse_anchor_evidence_quote,
)

_PID_TITLE_RE = re.compile(r"^title\s*:\s*\d{8,12}\s*$", re.I)
_PID_VALUE_RE = re.compile(r"^\d{8,12}$")
_SLOT_PREFIX_RE = re.compile(r"^slot\s*:", re.I)
_BULLET_RE = re.compile(r"^([^:]+)\s*:\s*(.+)$")
_PRICE_LIKE_RE = re.compile(r"^[\d.,]+\s*(php|usd|eur|pesos)?$", re.I)
_TOOL_TRACE_RE = re.compile(r"\bts\s*:\s*\[", re.I)
_JSON_BLOB_RE = re.compile(r'"product_id"\s*:|product_id.*\{', re.I)
_VALUE_MAX_LEN = 80
_LONG_VALUE_SLOTS = frozenset(
    {"description", "title", "product_style", "surge_protection", "compatibility", "length"}
)


def norm_value(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    text = re.sub(r"\s*(php|pesos|usd)\s*$", "", text, flags=re.I)
    return text


def is_stub(quote_after: str | None) -> bool:
    text = str(quote_after or "")
    return REWRITE_STUB in text or "evidence boundary" in text.lower()


def format_valid(quote_after: str | None, slot_norm: str = "") -> bool:
    text = str(quote_after or "").strip()
    if not text:
        return False
    if is_stub(text):
        return False
    if _SLOT_PREFIX_RE.match(text):
        return False
    if _PID_TITLE_RE.match(text):
        return False
    m = _BULLET_RE.match(text)
    if not m:
        return False
    key = _norm_slot(m.group(1))
    val = m.group(2).strip()
    if not val:
        return False
    sk = _norm_slot(slot_norm)
    if sk and key != sk and sk not in key and key not in sk:
        return False
    return True


def _flatten_json_values(obj: Any, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_flatten_json_values(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_flatten_json_values(v, f"{prefix}[{i}]"))
    elif obj is not None:
        s = str(obj).strip()
        if s:
            out.append(s)
    return out


def build_evidence_corpus(traj: dict[str, Any], anchor_pid: str) -> str:
    """Union of anchor product JSON, trajectory evidence, and user snippet."""
    parts: list[str] = []
    record = _anchor_product_record(traj, anchor_pid)
    if record:
        parts.append(json.dumps(record, ensure_ascii=False))
        parts.extend(_flatten_json_values(record))
    from ccer.mechanism.line_l_anchor_extract import _entity_ids, _trajectory_evidence

    for ev in _trajectory_evidence(traj):
        ents = _entity_ids(ev)
        if anchor_pid not in ents:
            continue
        raw = str(ev.get("value_raw") or ev.get("value_norm") or ev.get("snippet") or "")
        if raw:
            parts.append(raw)
            if str(ev.get("slot_norm") or "") == "product_record":
                try:
                    nested = json.loads(raw)
                    parts.append(json.dumps(nested, ensure_ascii=False))
                    parts.extend(_flatten_json_values(nested))
                except (json.JSONDecodeError, TypeError):
                    pass
    user = _final_call_user_content(traj)
    idx = user.find(str(anchor_pid))
    if idx >= 0:
        parts.append(user[max(0, idx - 120) : idx + 600])
    return "\n".join(parts)


def extract_value_from_quote(quote_after: str | None) -> str | None:
    text = str(quote_after or "").strip()
    m = _BULLET_RE.match(text)
    if m:
        return m.group(2).strip()
    return None


def _is_pid_as_value(value: str, slot_norm: str) -> bool:
    """Reject product IDs masquerading as slot values (e.g. Color: 4908901270)."""
    v = str(value or "").strip()
    slot = _norm_slot(slot_norm)
    if not _PID_VALUE_RE.match(v):
        return False
    if slot in ("shop_id", "product_id"):
        return False
    if slot == "price" and _PRICE_LIKE_RE.match(v):
        return False
    return True


_VALUE_LEN_REJECT = 80
_LONG_VALUE_OK_SLOTS = frozenset({"price", "title", "description"})


def classify_rewrite_value(value: str | None, slot_norm: str) -> list[str]:
    """Return rejection reason codes for rewrite value form (no corpus check)."""
    reasons: list[str] = []
    v = str(value or "").strip()
    if not v:
        return ["empty_value"]
    if _SLOT_PREFIX_RE.match(v):
        reasons.append("slot_prefix_value")
    if re.search(r"\bts\s*:\s*\[", v, re.I):
        reasons.append("tool_trace")
    if "product_id" in v.lower() and "{" in v:
        reasons.append("json_blob")
    if _is_pid_as_value(v, slot_norm):
        reasons.append("pid_as_value")
    slot = _norm_slot(slot_norm)
    if slot == "title" and _PID_VALUE_RE.match(v):
        reasons.append("pid_as_title")
    if len(v) > _VALUE_LEN_REJECT and slot not in _LONG_VALUE_OK_SLOTS:
        reasons.append("obs_window_oversize")
    if slot in ("color", "colour", "material", "brand", "title") and _PID_VALUE_RE.match(v):
        reasons.append("pid_as_semantic_slot")
    return reasons


def strict_slot_verify_v2(
    value: str | None,
    slot_norm: str,
    evidence_corpus: str,
    *,
    candidate_source: str = "",
    allow_snippet_fallback: bool = False,
) -> tuple[bool, str | None]:
    """Strict verify v2: form checks + grounding + snippet-source policy."""
    v = str(value or "").strip()
    form_reasons = classify_rewrite_value(v, slot_norm)
    if form_reasons:
        return False, form_reasons[0]
    src = str(candidate_source or "")
    if src == "obs_tau_window" and not allow_snippet_fallback:
        return False, "obs_tau_window_rejected"
    if not verify_grounded(v, evidence_corpus):
        return False, "not_grounded"
    slot = _norm_slot(slot_norm)
    if slot == "price" and not (_PRICE_LIKE_RE.match(v) or re.search(r"\d", v)):
        return False, "price_semantic_mismatch"
    return True, None


def strict_slot_verify(
    value: str | None,
    slot_norm: str,
    evidence_corpus: str,
) -> tuple[bool, str | None]:
    """Strict verify: grounded + no PID-as-value + slot semantic plausibility."""
    return strict_slot_verify_v2(value, slot_norm, evidence_corpus)


def verify_grounded(value: str | None, evidence_corpus: str) -> bool:
    """Check value appears in evidence corpus (normalized substring)."""
    if not value or not evidence_corpus:
        return False
    val_n = norm_value(value)
    if not val_n or len(val_n) < 2:
        return False
    corp_n = norm_value(evidence_corpus)
    if val_n in corp_n:
        return True
    # Also try raw substring for long titles
    raw = str(value).strip()
    return len(raw) >= 3 and raw in evidence_corpus


@dataclass(frozen=True)
class RewriteQuality:
    format_valid: bool
    is_stub: bool
    verified_grounded: bool
    v_anchor: str | None
    evidence_corpus_len: int
    acceptance_ok: bool = False
    acceptance_rules_ok: bool = False
    acceptance_nli_ok: bool = False
    acceptance_policy: str = "rules_and_nli"
    acceptance_pid_textual: str = ""
    acceptance_pid_committed: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_valid": self.format_valid,
            "is_stub": self.is_stub,
            "verified_grounded": self.verified_grounded,
            "v_anchor": self.v_anchor,
            "evidence_corpus_len": self.evidence_corpus_len,
            "acceptance_ok": self.acceptance_ok,
            "acceptance_rules_ok": self.acceptance_rules_ok,
            "acceptance_nli_ok": self.acceptance_nli_ok,
            "acceptance_policy": self.acceptance_policy,
            "acceptance_pid_textual": self.acceptance_pid_textual,
            "acceptance_pid_committed": self.acceptance_pid_committed,
        }


def assess_rewrite_quality(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
) -> RewriteQuality:
    from ccer.mechanism.line_l_rewrite_acceptance import acceptance_verify_bundle
    from ccer.mechanism.line_l_wrong_anchor import (
        resolve_acceptance_pid_committed,
        resolve_acceptance_pid_textual,
    )

    quote_after = str(row.get("quote_after") or row.get("_quote_after") or "")
    slot_norm = str(row.get("slot_norm") or "")
    pid_textual = resolve_acceptance_pid_textual(row, traj) or anchor_pid
    pid_committed = resolve_acceptance_pid_committed(row, traj) or anchor_pid
    corpus = build_evidence_corpus(traj, pid_textual)
    v = row.get("v_anchor") or extract_value_from_quote(quote_after)
    v_s = str(v).strip() if v else None
    fmt_ok = format_valid(quote_after, slot_norm)
    stub = is_stub(quote_after)
    candidate_source = str(row.get("source_field") or row.get("candidate_source") or "")
    if "obs_tau" in candidate_source:
        candidate_source = "obs_tau_window"
    bundle = acceptance_verify_bundle(
        v_s or "",
        slot_norm,
        corpus,
        row=row,
        traj=traj,
        acceptance_pid=pid_textual,
        candidate_source=candidate_source,
    )
    strict_ok = bool(bundle.get("rules_ok"))
    grounded = strict_ok
    verified = fmt_ok and not stub and grounded and bool(bundle.get("ok"))
    return RewriteQuality(
        format_valid=fmt_ok,
        is_stub=stub,
        verified_grounded=verified,
        v_anchor=v_s,
        evidence_corpus_len=len(corpus),
        acceptance_ok=bool(bundle.get("ok")),
        acceptance_rules_ok=bool(bundle.get("rules_ok")),
        acceptance_nli_ok=bool(bundle.get("nli_ok")),
        acceptance_policy=str(bundle.get("policy") or ""),
        acceptance_pid_textual=pid_textual,
        acceptance_pid_committed=pid_committed,
    )


def aggregate_quality_metrics(
    rows: list[dict[str, Any]],
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    anchor_pid_fn: Any,
) -> dict[str, Any]:
    """Aggregate quality stats over flagged rewrites."""
    flagged = [r for r in rows if float(r.get("p_pm", 0)) > float(tau)]
    rewrites = [r for r in flagged if str(r.get("action") or "") == "rewrite"]
    if not rewrites:
        return {
            "n_flagged": len(flagged),
            "n_rewrite": 0,
            "grounded_rate": None,
            "format_valid_rate": None,
            "stub_rate": None,
        }
    n_grounded = n_fmt = n_stub = 0
    for r in rewrites:
        tid = str(r.get("trajectory_id") or "")
        traj = traj_index.get(tid, {})
        pid = anchor_pid_fn(r, traj) if anchor_pid_fn else str(r.get("anchor_pid") or "")
        q = assess_rewrite_quality(r, traj, anchor_pid=pid)
        n_grounded += int(q.verified_grounded)
        n_fmt += int(q.format_valid)
        n_stub += int(q.is_stub)
    n = len(rewrites)
    return {
        "n_flagged": len(flagged),
        "n_rewrite": n,
        "grounded_rate": n_grounded / n,
        "format_valid_rate": n_fmt / n,
        "stub_rate": n_stub / n,
        "n_verified_grounded": n_grounded,
        "n_format_valid": n_fmt,
        "n_stub": n_stub,
    }
