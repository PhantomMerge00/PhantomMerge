"""Rewrite acceptance: rules v2 + optional NLI on acceptance-PID evidence corpus."""

from __future__ import annotations

import os
import re
from typing import Any, Literal

from ccer.mechanism.line_l_anchor_extract import (
    _SLOT_TO_PRODUCT_FIELDS,
    _anchor_product_record,
    _norm_slot,
    _norm_value_compare,
    lookup_obs_tau_slot,
    parse_anchor_evidence_quote,
)
from ccer.mechanism.line_l_rewrite_quality import (
    build_evidence_corpus,
    classify_rewrite_value,
    extract_value_from_quote,
    strict_slot_verify_v2,
)
from ccer.mechanism.line_l_slot_nli import nli_shadow_mode, slot_entails

RewriteVerifyPolicy = Literal["rules_only", "nli_only", "rules_and_nli", "rules_or_nli"]
VerifyPolicy = RewriteVerifyPolicy


def rewrite_verify_policy() -> RewriteVerifyPolicy:
    raw = os.environ.get("REWRITE_VERIFY_POLICY", "rules_and_nli").strip().lower()
    if raw in ("rules_only", "nli_only", "rules_and_nli", "rules_or_nli"):
        return raw  # type: ignore[return-value]
    return "rules_and_nli"


def get_rewrite_verify_policy() -> RewriteVerifyPolicy:
    return rewrite_verify_policy()


def nli_enabled() -> bool:
    return os.environ.get("REWRITE_SLOT_NLI", "1").strip() not in ("0", "false", "no")


def _slot_aligned_in_obs(
    value: str,
    slot_norm: str,
    traj: dict[str, Any],
    acceptance_pid: str,
    row: dict[str, Any],
) -> bool:
    slot_key = _norm_slot(slot_norm)
    val_n = _norm_value_compare(value)
    hit = lookup_obs_tau_slot(traj, anchor_pid=acceptance_pid, slot_norm=slot_norm)
    if hit.branch == "extraction_hit" and hit.v_anchor:
        if _norm_value_compare(hit.v_anchor) == val_n:
            return True
    record = _anchor_product_record(traj, acceptance_pid) or {}
    for field in _SLOT_TO_PRODUCT_FIELDS.get(slot_key, (slot_key,)):
        rv = record.get(field)
        if rv is not None and _norm_value_compare(str(rv)) == val_n:
            return True
    fields = parse_anchor_evidence_quote(str(row.get("anchor_evidence_quote") or ""))
    for key in (slot_key, _norm_slot(str(row.get("slot") or ""))):
        if key in fields and _norm_value_compare(fields[key]) == val_n:
            return True
    return False


def acceptance_verify_rules(
    value: str | None,
    slot_norm: str,
    evidence_corpus: str,
    *,
    candidate_source: str = "",
    allow_snippet_fallback: bool = False,
    row: dict[str, Any] | None = None,
    traj: dict[str, Any] | None = None,
    acceptance_pid: str = "",
) -> tuple[bool, list[str]]:
    reasons = list(classify_rewrite_value(value, slot_norm))
    if reasons:
        return False, reasons
    src = str(candidate_source or "")
    allow_snip = allow_snippet_fallback
    if src == "obs_tau_window" and not allow_snip and row is not None and traj is not None:
        allow_snip = _slot_aligned_in_obs(str(value or ""), slot_norm, traj, acceptance_pid, row)
    ok, reason = strict_slot_verify_v2(
        value,
        slot_norm,
        evidence_corpus,
        candidate_source=src,
        allow_snippet_fallback=allow_snip,
    )
    if not ok and reason:
        return False, [reason]
    return ok, []


def _combine_policy(rules_ok: bool, nli_ok: bool, policy: RewriteVerifyPolicy) -> bool:
    if policy == "rules_only":
        return rules_ok
    if policy == "nli_only":
        return nli_ok
    if policy == "rules_or_nli":
        return rules_ok or nli_ok
    return rules_ok and nli_ok


def acceptance_verify_bundle(
    value: str,
    slot_norm: str,
    evidence_corpus: str,
    *,
    row: dict[str, Any] | None = None,
    traj: dict[str, Any] | None = None,
    acceptance_pid: str = "",
    candidate_source: str = "",
    allow_snippet_fallback: bool = False,
    policy: RewriteVerifyPolicy | None = None,
) -> dict[str, Any]:
    pol: RewriteVerifyPolicy = policy or rewrite_verify_policy()
    rules_ok, reasons = acceptance_verify_rules(
        value,
        slot_norm,
        evidence_corpus,
        candidate_source=candidate_source,
        allow_snippet_fallback=allow_snippet_fallback,
        row=row,
        traj=traj,
        acceptance_pid=acceptance_pid,
    )
    slot = str(slot_norm or (row.get("slot_norm") if row else "") or "").strip()
    slot_label = str((row or {}).get("slot") or slot)
    if slot_label and re.match(r"^[a-z]", slot_label):
        slot_label = slot_label[0].upper() + slot_label[1:]
    hyp = f"{slot_label}: {str(value or '').strip()}"
    premise = str(evidence_corpus or "")[:4000]
    nli_ok = False
    nli_skipped = False
    if not nli_enabled():
        nli_skipped = True
        nli_ok = rules_ok
    else:
        try:
            nli_ok = slot_entails(premise, hyp)
        except Exception:
            nli_skipped = True
            nli_ok = rules_ok

    if nli_shadow_mode():
        ok = _combine_policy(rules_ok, rules_ok, pol if pol != "nli_only" else "rules_only")
    elif nli_skipped and pol in ("nli_only", "rules_and_nli"):
        ok = _combine_policy(rules_ok, rules_ok, "rules_only")
    else:
        ok = _combine_policy(rules_ok, nli_ok, pol)

    return {
        "rules_ok": rules_ok,
        "nli_ok": nli_ok,
        "nli_skipped": nli_skipped,
        "ok": ok,
        "reasons": reasons,
        "policy": pol,
        "acceptance_pid": acceptance_pid,
        "candidate_source": candidate_source,
        "hypothesis": hyp[:300],
    }


def rewrite_passes_acceptance(
    outcome: dict[str, Any],
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    acceptance_pid: str,
    policy: RewriteVerifyPolicy | None = None,
) -> dict[str, Any]:
    quote_after = str(outcome.get("quote_after") or "")
    slot_norm = str(row.get("slot_norm") or "")
    val = outcome.get("v_anchor") or extract_value_from_quote(quote_after) or ""
    corpus = build_evidence_corpus(traj, acceptance_pid)
    src = str(outcome.get("source_field") or outcome.get("candidate_source") or "")
    if "obs_tau" in src:
        src = "obs_tau_window"
    return acceptance_verify_bundle(
        str(val),
        slot_norm,
        corpus,
        row=row,
        traj=traj,
        acceptance_pid=acceptance_pid,
        candidate_source=src,
        policy=policy,
    )


def verify_rewrite_row_acceptance(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    acceptance_pid: str,
    policy: RewriteVerifyPolicy | None = None,
) -> dict[str, Any]:
    outcome = {
        "quote_after": row.get("quote_after"),
        "v_anchor": row.get("v_anchor"),
        "source_field": row.get("source_field"),
        "candidate_source": row.get("candidate_source"),
    }
    return rewrite_passes_acceptance(outcome, row, traj, acceptance_pid=acceptance_pid, policy=policy)
