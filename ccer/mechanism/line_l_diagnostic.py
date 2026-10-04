"""Line L pre-null diagnostics: stratified hit rates + CEM miss forensics."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from ccer.io_utils import write_json, write_jsonl
from ccer.mechanism.line_k_claim_filter import load_frozen_dev_thresholds, load_frozen_probe_scores
from ccer.mechanism.line_l_anchor_extract import (
    _anchor_product_record,
    _norm_slot,
    lookup_from_anchor_product_json,
    lookup_from_trajectory_evidence,
    lookup_slot_value,
    parse_anchor_evidence_quote,
)
from ccer.mechanism.line_l_claim_filter import _enrich_score_rows
from ccer.mechanism.line_l_wrong_anchor import resolve_committed_anchor_pid
from ccer.mechanism.pair_select import load_trajectory_index

PM_LABELS: dict[str, str] = {
    "cross_object_merge": "CEM",
    "constraint_projection": "CAP",
    "anchored_hallucination": "AH",
    "trajectory_clean": "clean_expansion",
}


@dataclass(frozen=True)
class MissForensics:
    quote_kind: str
    root_cause: str
    n_ev_total: int
    n_ev_anchor: int
    anchor_slots: list[str]
    parsed_quote_keys: list[str]
    traj_miss_reason: str | None
    anchor_evidence_dict: dict[str, Any]


def _classify_quote(quote: str, slot_norm: str) -> str:
    text = str(quote or "").strip()
    sk = _norm_slot(slot_norm)
    fields = parse_anchor_evidence_quote(text)
    if not text:
        return "empty_quote"
    if re.fullmatch(r"\d+", text):
        return "pid_only"
    if sk in fields:
        return "parseable_slot_match"
    if fields:
        return "parseable_other_keys"
    if ":" in text:
        return "colon_text_no_norm_match"
    return "prose_no_colon_slots"


def _classify_root_cause(
    *,
    slot_norm: str,
    quote_kind: str,
    traj_hit_branch: str,
    json_hit_branch: str,
    json_miss_reason: str | None,
    ev_for_anchor: list[dict[str, Any]],
    n_ev_total: int,
    has_anchor_product_json: bool,
) -> str:
    sk = _norm_slot(slot_norm)
    anchor_slots = [_norm_slot(str(e.get("slot_norm") or "")) for e in ev_for_anchor]
    has_product_record = "product_record" in anchor_slots
    has_exact_slot_ev = sk in anchor_slots

    if traj_hit_branch == "extraction_hit":
        return "would_hit_traj"
    if json_hit_branch == "extraction_hit":
        return "would_hit_product_json"
    if has_exact_slot_ev:
        return "anchor_has_slot_ev_but_lookup_failed"
    if has_anchor_product_json and json_miss_reason == "no_product_json_field_match":
        return "anchor_product_json_missing_field"
    if has_product_record and not has_exact_slot_ev:
        return "product_record_only_no_per_slot_evidence"
    if ev_for_anchor:
        return "anchor_evidence_wrong_slots"
    if n_ev_total > 0:
        return "traj_has_evidence_but_none_for_anchor_pid"
    if quote_kind == "parseable_slot_match":
        return "quote_should_hit_bug"
    return "traj_evidence_empty_or_quote_miss"


def diagnose_extraction(
    row: dict[str, Any],
    traj: dict[str, Any],
) -> dict[str, Any]:
    """Return per-claim extraction diagnostics (no rewrite side effects)."""
    anchor = resolve_committed_anchor_pid(row, traj)
    slot_norm = str(row.get("slot_norm") or "")
    quote = str(row.get("anchor_evidence_quote") or "")
    ev_list = traj.get("evidence") or []
    ev_for_anchor = [
        e
        for e in ev_list
        if anchor in [str(x) for x in (e.get("entity_ids") or [])]
    ]

    traj_res = lookup_from_trajectory_evidence(
        traj, anchor_pid=anchor, slot_norm=slot_norm
    )
    json_res = lookup_from_anchor_product_json(
        traj, anchor_pid=anchor, slot_norm=slot_norm
    )
    full_res = lookup_slot_value(
        traj,
        anchor_pid=anchor,
        slot_norm=slot_norm,
        anchor_evidence_quote=quote,
    )
    fields = parse_anchor_evidence_quote(quote)
    quote_kind = _classify_quote(quote, slot_norm)
    anchor_product_json = _anchor_product_record(traj, anchor)
    root_cause = _classify_root_cause(
        slot_norm=slot_norm,
        quote_kind=quote_kind,
        traj_hit_branch=traj_res.branch,
        json_hit_branch=json_res.branch,
        json_miss_reason=json_res.miss_reason,
        ev_for_anchor=ev_for_anchor,
        n_ev_total=len(ev_list),
        has_anchor_product_json=anchor_product_json is not None,
    )

    anchor_evidence_dict = {
        "trajectory_id": row.get("trajectory_id"),
        "anchor_pid": anchor,
        "slot_norm": slot_norm,
        "n_evidence_total": len(ev_list),
        "n_evidence_for_anchor": len(ev_for_anchor),
        "anchor_evidence_slots": [
            {
                "slot_norm": e.get("slot_norm"),
                "value_raw_preview": str(e.get("value_raw") or "")[:200],
                "entity_ids": e.get("entity_ids"),
            }
            for e in ev_for_anchor[:6]
        ],
        "parsed_anchor_evidence_quote": fields,
        "anchor_product_json": anchor_product_json,
        "lookup_traj_branch": traj_res.branch,
        "lookup_traj_miss_reason": traj_res.miss_reason,
        "lookup_json_branch": json_res.branch,
        "lookup_json_miss_reason": json_res.miss_reason,
        "lookup_full_branch": full_res.branch,
        "lookup_full_miss_reason": full_res.miss_reason,
    }

    forensics = MissForensics(
        quote_kind=quote_kind,
        root_cause=root_cause,
        n_ev_total=len(ev_list),
        n_ev_anchor=len(ev_for_anchor),
        anchor_slots=[str(e.get("slot_norm") or "") for e in ev_for_anchor[:8]],
        parsed_quote_keys=list(fields.keys())[:12],
        traj_miss_reason=traj_res.miss_reason,
        anchor_evidence_dict=anchor_evidence_dict,
    )

    return {
        "claim_id": row.get("claim_id"),
        "instance_audit_key": row.get("instance_audit_key"),
        "gold_verdict": row.get("gold_verdict"),
        "pm_label": PM_LABELS.get(str(row.get("gold_verdict") or ""), "other"),
        "p_pm": float(row.get("p_pm", 0.0)),
        "split": row.get("split"),
        "extraction_branch": full_res.branch,
        "miss_reason": full_res.miss_reason,
        "quote_kind": forensics.quote_kind,
        "root_cause": forensics.root_cause,
        "n_ev_total": forensics.n_ev_total,
        "n_ev_anchor": forensics.n_ev_anchor,
        "anchor_slots": forensics.anchor_slots,
        "parsed_quote_keys": forensics.parsed_quote_keys,
        "anchor_evidence_dict": forensics.anchor_evidence_dict,
        "anchor_evidence_quote_preview": quote[:240],
        "response_quote_preview": str(row.get("response_quote") or "")[:160],
        "claim_value": row.get("claim_value"),
    }


def _stratified_hit_rates(
    rows: list[dict[str, Any]],
    traj_index: dict[str, Any],
    *,
    tau: float,
    split: str | None = None,
) -> dict[str, Any]:
    if split:
        rows = [r for r in rows if r.get("split") == split]

    out: dict[str, Any] = {"tau": tau, "split": split or "all", "by_pm_type": {}}
    for verdict, label in PM_LABELS.items():
        sub = [r for r in rows if r.get("gold_verdict") == verdict]
        flagged = [r for r in sub if float(r.get("p_pm", 0.0)) > tau]
        hits = 0
        for r in flagged:
            traj = traj_index.get(str(r.get("trajectory_id") or ""), {})
            rec = dict(r)
            rec["tau"] = tau
            anchor = resolve_committed_anchor_pid(rec, traj)
            ex = lookup_slot_value(
                traj,
                anchor_pid=anchor,
                slot_norm=str(r.get("slot_norm") or ""),
                anchor_evidence_quote=str(r.get("anchor_evidence_quote") or ""),
            )
            if ex.branch == "extraction_hit":
                hits += 1
        out["by_pm_type"][label] = {
            "n_instances": len(sub),
            "n_flagged": len(flagged),
            "n_hit": hits,
            "hit_rate_flagged": (hits / len(flagged)) if flagged else None,
        }

    all_flagged = [r for r in rows if float(r.get("p_pm", 0.0)) > tau]
    all_hits = sum(
        1
        for r in all_flagged
        if diagnose_extraction(r, traj_index.get(str(r.get("trajectory_id") or ""), {}))[
            "extraction_branch"
        ]
        == "extraction_hit"
    )
    out["all_types"] = {
        "n_instances": len(rows),
        "n_flagged": len(all_flagged),
        "n_hit": all_hits,
        "hit_rate_flagged": (all_hits / len(all_flagged)) if all_flagged else None,
    }
    return out


def _interpretation_gate(
    step1: dict[str, Any],
    cem_root_causes: Counter[str],
) -> dict[str, Any]:
    cem_hr = step1["aggressive"]["by_pm_type"]["CEM"]["hit_rate_flagged"]
    all_hr = step1["aggressive"]["all_types"]["hit_rate_flagged"]
    blockers: list[str] = []
    if cem_hr is None or cem_hr == 0:
        blockers.append(
            "CEM subset hit_rate still 0% after product_json fix — check remaining misses"
        )
    elif cem_hr < 0.1:
        blockers.append(
            f"CEM hit_rate={cem_hr:.1%} remains low; many shell-anchor JSON lack price/attribute fields"
        )
    if all_hr is not None and all_hr < 0.05:
        blockers.append(
            "Overall flagged hit_rate <5%; target vs wrong_anchor comparison remains low-power"
        )
    may_null = cem_hr is not None and cem_hr < 0.1 and bool(blockers)
    if not blockers:
        blockers.append("Re-run Line L main experiment and re-check target vs wrong_anchor p-value")
    return {
        "cem_hit_rate_test_aggressive": cem_hr,
        "all_types_hit_rate_test_aggressive": all_hr,
        "cem_root_cause_counts": dict(cem_root_causes),
        "may_claim_honest_null": may_null,
        "blockers": blockers,
    }


def run_line_l_diagnostics(
    *,
    max_cem_miss_samples: int = 15,
) -> dict[str, Any]:
    """Run three-step diagnostic bundle; no experiment re-run."""
    enriched = _enrich_score_rows(load_frozen_probe_scores())
    thresholds = load_frozen_dev_thresholds()
    traj_index = load_trajectory_index()

    step1: dict[str, Any] = {}
    for tau_name, tau in [
        ("aggressive", float(thresholds["line_a_probe_aggressive"])),
        ("balanced", float(thresholds["line_a_probe_balanced"])),
    ]:
        step1[tau_name] = _stratified_hit_rates(
            enriched, traj_index, tau=tau, split="test"
        )

    tau_aggressive = float(thresholds["line_a_probe_aggressive"])
    cem_flagged_test = [
        r
        for r in enriched
        if r.get("split") == "test"
        and r.get("gold_verdict") == "cross_object_merge"
        and float(r.get("p_pm", 0.0)) > tau_aggressive
    ]
    cem_miss_samples: list[dict[str, Any]] = []
    root_causes: Counter[str] = Counter()
    quote_kinds: Counter[str] = Counter()
    for r in cem_flagged_test:
        traj = traj_index.get(str(r.get("trajectory_id") or ""), {})
        diag = diagnose_extraction(r, traj)
        if diag["extraction_branch"] == "extraction_miss":
            cem_miss_samples.append(diag)
            root_causes[diag["root_cause"]] += 1
            quote_kinds[diag["quote_kind"]] += 1

    cem_miss_samples = cem_miss_samples[:max_cem_miss_samples]

    step3_samples = [
        diagnose_extraction(r, traj_index.get(str(r.get("trajectory_id") or ""), {}))[
            "anchor_evidence_dict"
        ]
        for r in cem_flagged_test[:5]
    ]

    cem_hit = step1["aggressive"]["by_pm_type"]["CEM"]["hit_rate_flagged"]
    pipeline_fixed = cem_hit is not None and cem_hit > 0
    summary = {
        "schema_version": "line_l_diagnostic_v2",
        "diagnostic_status": (
            "post_product_json_fix"
            if pipeline_fixed
            else "pipeline_mismatch_or_sparse_evidence"
        ),
        "step1_stratified_hit_rate_test": step1,
        "step2_cem_miss_forensics": {
            "tau": tau_aggressive,
            "n_cem_flagged_test": len(cem_flagged_test),
            "n_cem_miss": len(
                [d for d in cem_miss_samples if d.get("extraction_branch") == "extraction_miss"]
            ),
            "root_cause_counts": dict(root_causes),
            "quote_kind_counts": dict(quote_kinds),
        },
        "step3_anchor_evidence_load_check": {
            "n_cem_samples": len(step3_samples),
            "samples": step3_samples,
            "note": (
                "traj['evidence'] loads non-empty for CEM flagged cases; "
                "misses are lookup/slot-namespace mismatch, not missing trajectories."
            ),
        },
        "interpretation_gate": _interpretation_gate(step1, root_causes),
    }
    return {
        "summary": summary,
        "cem_miss_samples": cem_miss_samples,
    }


def render_diagnostic_report(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# Line L Diagnostic Report (pre-null gate)",
        "",
        f"**Status**: `{s.get('diagnostic_status')}`",
        "",
        "## Step 1: extraction_hit_rate by PM type (test, flagged only)",
        "",
    ]
    for tau_name, block in (s.get("step1_stratified_hit_rate_test") or {}).items():
        lines.append(f"### τ = {block.get('tau')} ({tau_name})")
        lines.append("")
        lines.append("| PM type | n_instances | n_flagged | n_hit | hit_rate |")
        lines.append("|---------|-------------|-----------|-------|----------|")
        for label, row in (block.get("by_pm_type") or {}).items():
            hr = row.get("hit_rate_flagged")
            hr_s = f"{hr:.1%}" if hr is not None else "—"
            lines.append(
                f"| {label} | {row.get('n_instances')} | {row.get('n_flagged')} | "
                f"{row.get('n_hit')} | {hr_s} |"
            )
        all_row = block.get("all_types") or {}
        hr = all_row.get("hit_rate_flagged")
        lines.append(
            f"| **ALL** | {all_row.get('n_instances')} | {all_row.get('n_flagged')} | "
            f"{all_row.get('n_hit')} | {hr:.1%} |"
        )
        lines.append("")

    s2 = s.get("step2_cem_miss_forensics") or {}
    lines.extend(
        [
            "## Step 2: CEM extraction_miss forensics (test, τ=aggressive)",
            "",
            f"- CEM flagged: **{s2.get('n_cem_flagged_test')}**",
            f"- Root causes: `{json.dumps(s2.get('root_cause_counts'), ensure_ascii=False)}`",
            f"- Quote kinds: `{json.dumps(s2.get('quote_kind_counts'), ensure_ascii=False)}`",
            "",
            f"Sample cases: [`cem_miss_samples.jsonl`](cem_miss_samples.jsonl) (n={len(payload.get('cem_miss_samples') or [])})",
            "",
        ]
    )

    lines.extend(
        [
            "## Step 3: Obs_τ(a_τ) load check (CEM flagged samples)",
            "",
            s.get("step3_anchor_evidence_load_check", {}).get("note", ""),
            "",
        ]
    )
    for i, sample in enumerate(
        (s.get("step3_anchor_evidence_load_check") or {}).get("samples") or [], 1
    ):
        lines.append(f"### Sample {i}: `{sample.get('trajectory_id')}` anchor={sample.get('anchor_pid')}")
        lines.append(f"- n_evidence_total={sample.get('n_evidence_total')}, n_for_anchor={sample.get('n_evidence_for_anchor')}")
        slots = sample.get("anchor_evidence_slots") or []
        if slots:
            lines.append(f"- anchor slots: {[x.get('slot_norm') for x in slots]}")
        else:
            lines.append("- anchor slots: **(none matched committed_anchor_pid)**")
        lines.append("")

    gate = s.get("interpretation_gate") or {}
    lines.extend(
        [
            "## Interpretation gate",
            "",
            f"- CEM hit_rate (test, aggressive): **{gate.get('cem_hit_rate_test_aggressive')}**",
            f"- May claim honest null: **{gate.get('may_claim_honest_null')}**",
            "- Blockers:",
        ]
    )
    for b in gate.get("blockers") or []:
        lines.append(f"  - {b}")
    lines.append("")
    return "\n".join(lines)
