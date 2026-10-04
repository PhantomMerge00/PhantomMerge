"""Airline (tau2) trajectory normalization — mirrors shopping build_claims schema."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ccer.io_utils import sha256_text, write_jsonl
from ccer.normalize.shopping import PM_CORE_VERDICTS, VERDICT_TO_LEGACY

ROOT = Path("${PHANTOM_MERGE_ROOT}")
PACK = ROOT / "data/rollouts/tau3_exports/airline_pm_label_pack_v2"
MERGED = PACK / "merged_gold_v1"
NORMALIZED_AIRLINE = ROOT / "results/normalized/airline_trajectories.jsonl"


def _norm_slot(slot: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(slot or "").strip().lower()).strip("_")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _find_trajectory(tid: str) -> dict[str, Any] | None:
    for d in sorted(PACK.glob("batch_*")) + [PACK / "calibration"]:
        for row in _load_jsonl(d / "trajectories.jsonl"):
            if str(row.get("trajectory_id")) == tid:
                return row
    return None


def build_claims(instances: list[dict[str, Any]], anchor_ids: list[str]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    anchor = anchor_ids[0] if anchor_ids else None
    for inst in instances:
        legacy = str(inst.get("gold_verdict") or "")
        claims.append(
            {
                "claim_id": f"{inst.get('trajectory_id')}:{inst.get('instance_index')}",
                "instance_audit_key": inst.get("instance_audit_key"),
                "response_span": inst.get("response_quote"),
                "referent_ids": [anchor] if anchor else [],
                "slot_norm": _norm_slot(inst.get("slot") or ""),
                "value": inst.get("value"),
                "scope": inst.get("claim_scope") or "anchor",
                "mode": "assertion",
                "legacy_label": legacy,
                "normalized_axes": {
                    "pm_binary": inst.get("pm_binary"),
                    "legacy_short": VERDICT_TO_LEGACY.get(legacy, legacy),
                },
                "human_status": "annotated",
                "y_pm": 1 if legacy in PM_CORE_VERDICTS else 0,
            }
        )
    return claims


def normalize_airline_trajectory(
    tid: str,
    traj: dict[str, Any],
    sheet: dict[str, Any] | None,
    instances: list[dict[str, Any]],
) -> dict[str, Any]:
    digest = traj.get("evidence_digest") or {}
    anchor_ids = list(digest.get("anchor_ids") or [])
    messages = traj.get("full_trajectory") or []
    return {
        "root_id": f"airline_{tid}",
        "trajectory_id": tid,
        "trajectory_audit_key": traj.get("trajectory_audit_key"),
        "domain": "airline",
        "model_manifest_id": "qwen3-32b_tau2_airline_v1",
        "trajectory_outcome": (sheet or {}).get("trajectory_outcome"),
        "pm_core_count": (sheet or {}).get("pm_core_count"),
        "commitment": {"action_anchor": anchor_ids[0] if anchor_ids else None, "anchor_ids": anchor_ids},
        "evidence": {"E_seen": digest.get("E_seen") or [], "tool_digest": traj.get("tool_digest") or []},
        "claims": build_claims(instances, anchor_ids),
        "messages": messages,
        "metadata": {
            "official_reward": digest.get("official_reward"),
            "termination_reason": digest.get("termination_reason"),
            "message_count": digest.get("message_count"),
            "final_agent_text_hash": sha256_text(str(digest.get("final_agent_text") or "")),
        },
        "eligible": {"native_replay": False, "reason": "tau2_message_format"},
    }


def run_normalize_airline(
    *,
    out_path: Path = NORMALIZED_AIRLINE,
) -> dict[str, Any]:
    sheets = {r["trajectory_id"]: r for r in _load_jsonl(MERGED / "trajectory_sheet.jsonl")}
    inst_by_tid: dict[str, list[dict[str, Any]]] = {}
    for inst in _load_jsonl(MERGED / "expert_instances.jsonl"):
        inst_by_tid.setdefault(str(inst["trajectory_id"]), []).append(inst)

    rows: list[dict[str, Any]] = []
    missing = 0
    for tid in sorted(sheets.keys(), key=int):
        traj = _find_trajectory(str(tid))
        if not traj:
            missing += 1
            continue
        rows.append(
            normalize_airline_trajectory(
                str(tid),
                traj,
                sheets.get(str(tid)),
                inst_by_tid.get(str(tid), []),
            )
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, rows)
    n_claims = sum(len(r.get("claims") or []) for r in rows)
    pm_trajs = sum(1 for r in rows if r.get("trajectory_outcome") == "has_pm_core")
    return {
        "n_normalized": len(rows),
        "n_claims": n_claims,
        "pm_trajectories": pm_trajs,
        "missing_trajectory": missing,
        "out_path": str(out_path),
    }


if __name__ == "__main__":
    print(json.dumps(run_normalize_airline(), indent=2))
