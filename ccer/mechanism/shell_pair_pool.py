"""Shell-anchor subclass pool for Line F (evidence mask + correct donor expansion)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from ccer.adjudication.loader import load_adjudication_index, resolve_cem_swap_target
from ccer.counterfactual.base import find_product_record_snippets
from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.mechanism_pool import (
    MECHANISM_RESEARCH_SPLITS,
    PoolName,
    build_cem_mechanism_cross_pairs,
)
from ccer.mechanism.pair_select import load_trajectory_index, messages_for_condition
from ccer.replay.evidence_mask import classify_anchor_title_quality, mask_cem_evidence_spans

SHELL_PAIR_AUDIT_JSONL = Path(
    "${PHANTOM_MERGE_ROOT}/results/reports/manual_verification/round12_shell_expanded_audit.jsonl"
)
ROUND10_SHELL_AUDIT_JSONL = Path(
    "${PHANTOM_MERGE_ROOT}/results/reports/manual_verification/round10_cem_cross_pair_audit.jsonl"
)
SHELL_POOL_MANIFEST = Path(
    "${PHANTOM_MERGE_ROOT}/results/p3/round12_line_f/shell_pool_manifest.json"
)


def _anchor_quality(pm_traj: dict[str, Any] | None) -> dict[str, Any]:
    if not pm_traj:
        return {}
    target = resolve_cem_swap_target(pm_traj)
    anchor_pid = str(
        getattr(target, "committed_anchor_pid", "")
        or getattr(target, "textual_selected_pid", "")
        or ""
    )
    if not anchor_pid:
        return {}
    return classify_anchor_title_quality(pm_traj, anchor_pid)


def _load_r11_legacy_shell_pairs() -> list[dict[str, Any]]:
    """Frozen R11 shell cohort: exact pm×clean pairing from round10 audit (never re-pair)."""
    if not ROUND10_SHELL_AUDIT_JSONL.is_file():
        return []
    rows_index = load_trajectory_index()
    legacy: list[dict[str, Any]] = []
    for line in ROUND10_SHELL_AUDIT_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("anchor_title_quality") != "shell_pid_only":
            continue
        pm_tid = str(row["pm_trajectory_id"])
        traj = rows_index.get(pm_tid) or {}
        legacy.append(
            {
                "pm_trajectory_id": pm_tid,
                "clean_trajectory_id": str(row["clean_trajectory_id"]),
                "slot_norm": row.get("slot_norm"),
                "pairing_method": row.get("pairing_method") or "r11_legacy_frozen",
                "pairing_source": "r11_legacy",
                "r11_pair_index": int(row.get("pair_index") or 0),
                "track": "CEM",
                "pool": "mechanism_research",
                "anchor_title_quality": "shell_pid_only",
                "split": traj.get("split"),
            }
        )
    legacy.sort(key=lambda r: int(r.get("r11_pair_index") or 0))
    return legacy


def validate_r11_legacy_pairing(shell_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Runtime gate: R11 legacy donors must match round10 audit exactly."""
    legacy_audit = {
        str(json.loads(line)["pm_trajectory_id"]): str(json.loads(line)["clean_trajectory_id"])
        for line in ROUND10_SHELL_AUDIT_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("anchor_title_quality") == "shell_pid_only"
    }
    mismatches: list[dict[str, str]] = []
    for cp in shell_pairs:
        if cp.get("pairing_source") != "r11_legacy":
            continue
        pm = str(cp["pm_trajectory_id"])
        expected = legacy_audit.get(pm)
        actual = str(cp.get("clean_trajectory_id") or "")
        if expected and expected != actual:
            mismatches.append({"pm_trajectory_id": pm, "expected": expected, "actual": actual})
    return {
        "n_r11_legacy": len(legacy_audit),
        "n_mismatches": len(mismatches),
        "passed": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def build_shell_cross_pairs(
    *,
    pool: PoolName = "mechanism_research",
) -> dict[str, Any]:
    """Shell subclass pool: R11 legacy pairs frozen + mechanism_expansion for new shell only."""
    base = build_cem_mechanism_cross_pairs(pool=pool)
    rows = load_trajectory_index()
    legacy_pairs = _load_r11_legacy_shell_pairs()
    legacy_pm_ids = {str(p["pm_trajectory_id"]) for p in legacy_pairs}

    expansion_pairs: list[dict[str, Any]] = []
    for cp in base["pm_clean_cross_pairs"]:
        pm_tid = str(cp["pm_trajectory_id"])
        if pm_tid in legacy_pm_ids:
            continue
        traj = rows.get(pm_tid)
        q = _anchor_quality(traj)
        if q.get("anchor_title_quality") != "shell_pid_only":
            continue
        expansion_pairs.append(
            {
                **cp,
                "anchor_title_quality": "shell_pid_only",
                "anchor_title_preview": q.get("anchor_title_preview"),
                "split": (traj or {}).get("split"),
                "pairing_source": "mechanism_expansion",
            }
        )

    shell_pairs = legacy_pairs + expansion_pairs
    pairing_validation = validate_r11_legacy_pairing(shell_pairs)
    n_legacy = sum(1 for p in shell_pairs if p.get("pairing_source") == "r11_legacy")
    n_expansion = sum(1 for p in shell_pairs if p.get("pairing_source") == "mechanism_expansion")
    return {
        "pool": pool,
        "anchor_filter": "shell_pid_only",
        "splits_included": list(MECHANISM_RESEARCH_SPLITS) if pool == "mechanism_research" else ["dev"],
        "n_full_cross_pairs": len(base["pm_clean_cross_pairs"]),
        "n_shell_cross_pairs": len(shell_pairs),
        "n_r11_legacy_pairs": n_legacy,
        "n_mechanism_expansion_pairs": n_expansion,
        "r11_legacy_pairing_validation": pairing_validation,
        "pm_clean_cross_pairs": shell_pairs,
        "methodology_note": (
            "Shell subclass: anchor title equals committed PID. "
            "R11 legacy 15 pairs keep round10 audit pm×clean pairing (frozen). "
            "New shell pairs use mechanism_research expansion pairing only. "
            "No multi_clean_expansion."
        ),
    }


def activation_eligible_shell_pairs(
    shell_pairs: list[dict[str, Any]],
    *,
    layer: int = 32,
    position: str = "commitment",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (eligible, ineligible) shell pairs based on L32 commitment activations."""
    eligible: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    for cp in shell_pairs:
        pm_tid = str(cp["pm_trajectory_id"])
        clean_tid = str(cp["clean_trajectory_id"])
        pm_path = activation_path(pm_tid, "original")
        clean_path = activation_path(clean_tid, "original")
        pm_npz = load_activation_npz(pm_path) if pm_path.is_file() else None
        clean_npz = load_activation_npz(clean_path) if clean_path.is_file() else None
        pm_vec = get_vector(pm_npz, position=position, layer=layer) if pm_npz else None
        clean_vec = get_vector(clean_npz, position=position, layer=layer) if clean_npz else None
        row = dict(cp)
        row["pm_activation_ok"] = pm_vec is not None
        row["clean_activation_ok"] = clean_vec is not None
        row["eval_eligible"] = pm_vec is not None and clean_vec is not None
        if row["eval_eligible"]:
            eligible.append(row)
        else:
            ineligible.append(row)
    return eligible, ineligible


def activation_coverage_for_shell_pool(
    *,
    pool: PoolName = "mechanism_research",
    layer: int = 32,
    position: str = "commitment",
) -> dict[str, Any]:
    shell = build_shell_cross_pairs(pool=pool)
    pairs = shell["pm_clean_cross_pairs"]
    eligible, ineligible = activation_eligible_shell_pairs(pairs, layer=layer, position=position)
    by_split: dict[str, dict[str, int]] = {}
    for cp in pairs:
        sp = str(cp.get("split") or "unknown")
        bucket = by_split.setdefault(sp, {"total": 0, "eligible": 0})
        bucket["total"] += 1
    for cp in eligible:
        sp = str(cp.get("split") or "unknown")
        by_split.setdefault(sp, {"total": 0, "eligible": 0})["eligible"] += 1
    return {
        "schema_version": "ccer_line_f_shell_activation_coverage_v1",
        "pool": pool,
        "layer": layer,
        "position": position,
        "n_shell_cross_pairs": len(pairs),
        "n_eval_eligible": len(eligible),
        "n_missing_activation": len(ineligible),
        "coverage_rate": len(eligible) / len(pairs) if pairs else 0.0,
        "by_split": by_split,
        "eligible_pairs": [
            {"pm_trajectory_id": cp["pm_trajectory_id"], "clean_trajectory_id": cp["clean_trajectory_id"]}
            for cp in eligible
        ],
        "ineligible_pairs": [
            {
                "pm_trajectory_id": cp["pm_trajectory_id"],
                "clean_trajectory_id": cp["clean_trajectory_id"],
                "pm_activation_ok": cp.get("pm_activation_ok"),
                "clean_activation_ok": cp.get("clean_activation_ok"),
                "split": cp.get("split"),
            }
            for cp in ineligible
        ],
    }


def build_shell_pool_manifest(*, pool: PoolName = "mechanism_research") -> dict[str, Any]:
    shell = build_shell_cross_pairs(pool=pool)
    coverage = activation_coverage_for_shell_pool(pool=pool)
    return {
        "schema_version": "ccer_shell_pool_manifest_v1",
        "line": "F",
        "pool": pool,
        "splits_included": shell["splits_included"],
        "methodology_note": shell["methodology_note"],
        "n_full_cross_pairs": shell["n_full_cross_pairs"],
        "n_shell_cross_pairs": shell["n_shell_cross_pairs"],
        "n_eval_eligible": coverage["n_eval_eligible"],
        "n_missing_activation": coverage["n_missing_activation"],
        "coverage_rate": coverage["coverage_rate"],
        "by_split": coverage["by_split"],
        "shell_cross_pairs": shell["pm_clean_cross_pairs"],
        "baseline_round10_shell_n": 15,
        "baseline_round11_shell_n": 15,
        "n_r11_legacy_pairs": shell.get("n_r11_legacy_pairs"),
        "n_mechanism_expansion_pairs": shell.get("n_mechanism_expansion_pairs"),
        "r11_legacy_pairing_validation": shell.get("r11_legacy_pairing_validation"),
    }


def export_shell_pair_audit(
    *,
    pool: PoolName = "mechanism_research",
    layer: int = 32,
    position: str = "commitment",
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export shell-only pair audit with leak_audit v2 (round 10/11 QA standard)."""
    shell = build_shell_cross_pairs(pool=pool)
    cross_pairs = shell["pm_clean_cross_pairs"]
    rows_index = load_trajectory_index()
    adj = load_adjudication_index()
    path = Path(out_path or SHELL_PAIR_AUDIT_JSONL)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    n_eligible = 0
    n_leak_free = 0
    n_pid_extractable = 0

    for i, cp in enumerate(cross_pairs):
        pair_index = i + 1
        pm_tid = str(cp["pm_trajectory_id"])
        clean_tid = str(cp["clean_trajectory_id"])
        pm_traj = rows_index.get(pm_tid)
        target = resolve_cem_swap_target(pm_traj) if pm_traj else None
        anchor_pid = str(
            (getattr(target, "committed_anchor_pid", None) or getattr(target, "textual_selected_pid", None) or "")
            if target
            else ""
        )
        anchor_q = _anchor_quality(pm_traj)

        mask_meta: dict[str, Any] = {}
        masked_user_excerpt = ""
        if pm_traj:
            msgs = messages_for_condition(pm_traj, "original", track="CEM")
            if msgs:
                masked_msgs, mask_meta = mask_cem_evidence_spans(msgs, pm_traj)
                masked_user = next(m["content"] for m in masked_msgs if m.get("role") == "user")
                masked_user_excerpt = masked_user[:1200] + ("…" if len(masked_user) > 1200 else "")

        pm_path = activation_path(pm_tid, "original")
        clean_path = activation_path(clean_tid, "original")
        pm_npz = load_activation_npz(pm_path) if pm_traj and pm_path.is_file() else None
        clean_npz = load_activation_npz(clean_path) if pm_traj and clean_path.is_file() else None
        pm_vec = get_vector(pm_npz, position=position, layer=layer) if pm_npz else None
        clean_vec = get_vector(clean_npz, position=position, layer=layer) if clean_npz else None
        eligible = pm_vec is not None and clean_vec is not None and bool(mask_meta.get("masked"))
        if eligible:
            n_eligible += 1
        if mask_meta.get("leak_free"):
            n_leak_free += 1

        from ccer.replay.answer_utils import extract_selected_product_id

        baseline_answer = str(((pm_traj or {}).get("metadata") or {}).get("final_answer") or "")
        if not baseline_answer and pm_traj:
            for m in reversed(pm_traj.get("messages_final_call") or []):
                if m.get("role") == "assistant":
                    baseline_answer = str(m.get("content") or "")
                    break
        baseline_pid = extract_selected_product_id(baseline_answer)
        if baseline_pid:
            n_pid_extractable += 1

        adj_row = adj.pick_primary_cem_instance(pm_tid) if pm_traj else None
        row = {
            "pair_index": pair_index,
            "pm_trajectory_id": pm_tid,
            "clean_trajectory_id": clean_tid,
            "split": cp.get("split"),
            "pairing_source": cp.get("pairing_source"),
            "r11_pair_index": cp.get("r11_pair_index"),
            "pairing_method": cp.get("pairing_method"),
            "slot_norm": cp.get("slot_norm"),
            "donor_pid": getattr(target, "donor_pid", None) if target else None,
            "anchor_pid": anchor_pid or None,
            "claim_value": getattr(target, "claim_value", None) if target else None,
            "anchor_title_quality": anchor_q.get("anchor_title_quality"),
            "anchor_title_preview": anchor_q.get("anchor_title_preview"),
            "has_pins_not_viewed": anchor_q.get("has_pins_not_viewed", False),
            "strong_mask": mask_meta,
            "masked_user_excerpt": masked_user_excerpt if pair_index <= 7 else None,
            "interchange_eligible": eligible,
            "pm_has_activation": pm_vec is not None,
            "clean_has_activation": clean_vec is not None,
            "baseline_pid_extractable": baseline_pid is not None,
            "baseline_pid": baseline_pid,
            "adjudication_status": (adj_row or {}).get("adjudication_status") if adj_row else None,
            "expert_review_prompt": (
                "Shell subclass only. Confirm leak_free, donor=rival PID, baseline PID extractable."
            ),
        }
        rows.append(row)

    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_legacy = sum(1 for r in rows if r.get("pairing_source") == "r11_legacy")
    n_expansion = sum(1 for r in rows if r.get("pairing_source") == "mechanism_expansion")
    return {
        "schema_version": "ccer_shell_cross_pair_audit_v2",
        "line": "F",
        "pool": pool,
        "n_pairs": len(rows),
        "n_r11_legacy": n_legacy,
        "n_mechanism_expansion": n_expansion,
        "r11_legacy_pairing_validation": shell.get("r11_legacy_pairing_validation"),
        "n_interchange_eligible": n_eligible,
        "n_strong_mask_leak_free": n_leak_free,
        "n_baseline_pid_extractable": n_pid_extractable,
        "n_baseline_pid_missing": len(rows) - n_pid_extractable,
        "out_path": str(path),
        "pairing_note": shell["methodology_note"],
    }


def shell_trajectory_ids_for_activation_extract(
    *,
    pool: PoolName = "mechanism_research",
) -> set[str]:
    """All PM + clean trajectory IDs in shell pool (for P3-0 activation extraction)."""
    shell = build_shell_cross_pairs(pool=pool)
    tids: set[str] = set()
    for cp in shell["pm_clean_cross_pairs"]:
        tids.add(str(cp["pm_trajectory_id"]))
        tids.add(str(cp["clean_trajectory_id"]))
    return tids
