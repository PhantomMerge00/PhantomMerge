"""Tau3 instance cohort for Line A activation extraction."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ccer.io_utils import load_jsonl
from ccer.mechanism.supervised_probe import CLEAN_VERDICT, PM_VERDICTS
from ccer.mechanism.tau3.domain import Tau3DomainConfig, get_tau3_domain


@dataclass
class Tau3InstanceRow:
    instance_audit_key: str
    trajectory_id: str
    split: str
    y: int
    gold_verdict: str
    response_quote: str
    final_answer: str
    quote_start: int
    quote_end: int
    slot_norm: str
    claim_value: str
    pack_source: str | None = None


def instance_npz_path(
    instance_audit_key: str,
    trajectory_id: str,
    *,
    cfg: Tau3DomainConfig | None = None,
) -> Path:
    cfg = cfg or get_tau3_domain("telecom")
    safe = hashlib.sha1(instance_audit_key.encode("utf-8")).hexdigest()[:16]
    tid = trajectory_id.replace("/", "_").replace("[", "_").replace("]", "_")
    return cfg.activations_dir / tid / f"{safe}.npz"


def _load_split_manifest(cfg: Tau3DomainConfig) -> dict[str, str]:
    if not cfg.split_manifest.is_file():
        return {}
    payload = json.loads(cfg.split_manifest.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in (payload.get("trajectory_split") or {}).items()}


def tau3_instance_rows(
    *,
    require_activation: bool = False,
    cfg: Tau3DomainConfig | None = None,
    adjudication_path: Path | None = None,
) -> list[Tau3InstanceRow]:
    cfg = cfg or get_tau3_domain("telecom")
    adjudication_path = adjudication_path or cfg.adjudication
    traj_meta = {r["trajectory_id"]: r for r in load_jsonl(cfg.normalized)}
    traj_split = _load_split_manifest(cfg)
    rows: list[Tau3InstanceRow] = []

    for row in load_jsonl(adjudication_path):
        if not row.get("commitment_eligible"):
            continue
        verdict = str(row.get("gold_verdict") or "")
        if verdict not in PM_VERDICTS and verdict != CLEAN_VERDICT:
            continue
        tid = str(row["trajectory_id"])
        traj = traj_meta.get(tid, {})
        if not (traj.get("eligible") or {}).get("native_replay", True):
            continue
        final_answer = str((traj.get("metadata") or {}).get("final_answer") or "")
        if not final_answer.strip():
            continue
        qs = row.get("quote_start")
        qe = row.get("quote_end")
        if qs is None or qe is None:
            continue
        iak = str(row["instance_audit_key"])
        path = instance_npz_path(iak, tid, cfg=cfg)
        if require_activation and not path.is_file():
            continue
        split = traj_split.get(tid, "unassigned")
        rows.append(
            Tau3InstanceRow(
                instance_audit_key=iak,
                trajectory_id=tid,
                split=split,
                y=int(row.get("y_pm") or 0),
                gold_verdict=verdict,
                response_quote=str(row.get("response_quote") or ""),
                final_answer=final_answer,
                quote_start=int(qs),
                quote_end=int(qe),
                slot_norm=str(row.get("slot_norm") or ""),
                claim_value=str(row.get("value") or ""),
                pack_source=row.get("pack_source"),
            )
        )
    return rows


def telecom_instance_rows(
    *,
    require_activation: bool = False,
    adjudication_path: Path | None = None,
) -> list[Tau3InstanceRow]:
    return tau3_instance_rows(
        require_activation=require_activation,
        cfg=get_tau3_domain("telecom"),
        adjudication_path=adjudication_path,
    )


def airline_instance_rows(
    *,
    require_activation: bool = False,
    adjudication_path: Path | None = None,
) -> list[Tau3InstanceRow]:
    return tau3_instance_rows(
        require_activation=require_activation,
        cfg=get_tau3_domain("airline"),
        adjudication_path=adjudication_path,
    )


def group_instances_by_trajectory(
    rows: list[Tau3InstanceRow],
) -> dict[str, list[Tau3InstanceRow]]:
    grouped: dict[str, list[Tau3InstanceRow]] = {}
    for r in rows:
        grouped.setdefault(r.trajectory_id, []).append(r)
    return grouped
