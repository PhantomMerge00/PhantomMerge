"""Retail (tau3) trajectory normalization from 1k gold export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ccer.io_utils import load_jsonl, sha256_text, write_jsonl
from ccer.normalize.telecom import build_claims
from ccer.paths import NORMALIZED_RETAIL_QWEN_1K, RETAIL_GOLD_EXPORT


def normalize_retail_row(row: dict[str, Any]) -> dict[str, Any]:
    tid = str(row["trajectory_id"])
    traj = row.get("trajectory") or {}
    sheet = (row.get("pm_label") or {}).get("trajectory_sheet") or {}
    instances = (row.get("pm_label") or {}).get("expert_instances") or []
    digest = traj.get("evidence_digest") or {}
    anchor_ids = list(digest.get("anchor_ids") or [])
    messages = traj.get("full_trajectory") or []
    final_answer = str(digest.get("final_agent_text") or "")

    return {
        "root_id": f"retail_{tid}",
        "trajectory_id": tid,
        "trajectory_audit_key": traj.get("trajectory_audit_key"),
        "domain": "retail",
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
        "eligible": {"native_replay": bool(messages), "reason": "tau2_full_trajectory_replay"},
    }


def run_normalize_retail_qwen_1k(
    *,
    in_path: Path = RETAIL_GOLD_EXPORT,
    out_path: Path = NORMALIZED_RETAIL_QWEN_1K,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(in_path):
        rows.append(normalize_retail_row(row))
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
    print(json.dumps(run_normalize_retail_qwen_1k(), indent=2))
