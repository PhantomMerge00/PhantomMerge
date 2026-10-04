"""Telecom (tau3) trajectory normalization — mirrors airline/shopping schema."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ccer.io_utils import load_jsonl, sha256_text, write_jsonl
from ccer.normalize.shopping import PM_CORE_VERDICTS, VERDICT_TO_LEGACY
from ccer.paths import NORMALIZED_TELECOM, TELECOM_GOLD_EXPORT

ROOT = Path("${PHANTOM_MERGE_ROOT}")


def _norm_slot(slot: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(slot or "").strip().lower()).strip("_")


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


def normalize_telecom_row(row: dict[str, Any]) -> dict[str, Any]:
    tid = str(row["trajectory_id"])
    traj = row.get("trajectory") or {}
    sheet = (row.get("pm_label") or {}).get("trajectory_sheet") or {}
    instances = (row.get("pm_label") or {}).get("expert_instances") or []
    digest = traj.get("evidence_digest") or {}
    anchor_ids = list(digest.get("anchor_ids") or [])
    messages = traj.get("full_trajectory") or []
    final_answer = str(digest.get("final_agent_text") or "")

    return {
        "root_id": f"telecom_{tid}",
        "trajectory_id": tid,
        "trajectory_audit_key": traj.get("trajectory_audit_key"),
        "domain": "telecom",
        "pack_source": row.get("pack_source"),
        "model_manifest_id": "qwen3-32b_shopping_v1",
        "trajectory_outcome": sheet.get("trajectory_outcome"),
        "pm_core_count": sheet.get("pm_core_count"),
        "commitment": {
            "action_anchor": anchor_ids[0] if anchor_ids else None,
            "anchor_ids": anchor_ids,
        },
        "evidence": {
            "E_seen": digest.get("E_seen") or [],
            "tool_digest": traj.get("tool_digest") or [],
        },
        "claims": build_claims(instances, anchor_ids),
        "messages": messages,
        "messages_final_call": messages,
        "metadata": {
            "official_reward": digest.get("official_reward"),
            "termination_reason": digest.get("termination_reason"),
            "message_count": digest.get("message_count"),
            "final_answer": final_answer,
            "final_agent_text_hash": sha256_text(final_answer),
            "corpus_source": row.get("corpus_source"),
        },
        "eligible": {"native_replay": True, "reason": "tau2_full_trajectory_replay"},
    }


def run_normalize_telecom(
    *,
    in_path: Path = TELECOM_GOLD_EXPORT,
    out_path: Path = NORMALIZED_TELECOM,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(in_path):
        rows.append(normalize_telecom_row(row))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, rows)
    n_claims = sum(len(r.get("claims") or []) for r in rows)
    pm_trajs = sum(1 for r in rows if r.get("trajectory_outcome") == "has_pm_core")
    clean_trajs = sum(1 for r in rows if r.get("trajectory_outcome") == "clean")
    return {
        "n_normalized": len(rows),
        "n_claims": n_claims,
        "pm_trajectories": pm_trajs,
        "clean_trajectories": clean_trajs,
        "out_path": str(out_path),
    }


if __name__ == "__main__":
    print(json.dumps(run_normalize_telecom(), indent=2))
