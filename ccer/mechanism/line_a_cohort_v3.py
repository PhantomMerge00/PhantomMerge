"""Line A v3: expand clean negatives from 2k gold + PM-type breakdown."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from ccer.io_utils import load_jsonl, write_json
from ccer.mechanism.line_a_instance_activations import (
    LineAInstanceRow,
    instance_npz_path,
    line_a_adjudication_instance_rows,
)
from ccer.mechanism.supervised_probe import (
    CLEAN_VERDICT,
    PM_VERDICTS,
    ProbeDataset,
    line_a_skip_trajectory_ids,
    normalize_quote_for_bow,
)
from ccer.paths import (
    INCREMENTAL_ADJUDICATION_JSONL,
    LINE_A_V3_COHORT_MANIFEST,
    NORMALIZED_SHOPPING,
    SPLIT_MANIFEST_JSON,
    TRAJECTORY_SHEET,
)
from ccer.replay.answer_utils import extract_answer, normalize_final_synthesis_text
from ccer.replay.position_registry import _first_attribute_bullet, _selected_product_id_line

VERDICT_SHORT = {
    "cross_object_merge": "CEM",
    "constraint_projection": "CAP",
    "anchored_hallucination": "AH",
    "correct_binding": "clean",
    "trajectory_clean": "clean",
}

DEFAULT_CLEAN_TARGETS = {"train": 350, "dev": 50, "test": 125}


def _load_splits() -> dict[str, str]:
    payload = json.loads(SPLIT_MANIFEST_JSON.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in (payload.get("splits") or {}).items()}


def incremental_trajectory_ids() -> set[str]:
    return {str(r["trajectory_id"]) for r in load_jsonl(INCREMENTAL_ADJUDICATION_JSONL)}


def clean_trajectory_pool(*, exclude_incremental: bool = True) -> list[dict[str, Any]]:
    """2k trajectory_outcome=clean, native_replay, optional exclude any inc-adj trajectory."""
    inc = incremental_trajectory_ids() if exclude_incremental else set()
    skips = line_a_skip_trajectory_ids()
    norm = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}
    splits = _load_splits()
    pool: list[dict[str, Any]] = []
    for row in load_jsonl(TRAJECTORY_SHEET):
        if row.get("trajectory_outcome") != "clean":
            continue
        tid = str(row["trajectory_id"])
        if tid in inc or tid in skips:
            continue
        traj = norm.get(tid)
        if not traj or not traj.get("eligible", {}).get("native_replay"):
            continue
        if not (traj.get("messages_final_call") and str((traj.get("metadata") or {}).get("final_answer") or "").strip()):
            continue
        sp = splits.get(tid)
        if sp not in ("train", "dev", "test"):
            continue
        pool.append({"trajectory_id": tid, "split": sp, "trajectory": traj})
    return pool


def pseudo_clean_quote(traj: dict[str, Any]) -> tuple[str, int, int] | None:
    """First About bullet or Selected PID line → (response_quote, start, end) in answer_body."""
    final_answer = str((traj.get("metadata") or {}).get("final_answer") or "")
    answer_body = extract_answer(normalize_final_synthesis_text(final_answer))
    pair = _first_attribute_bullet(answer_body) or _selected_product_id_line(answer_body)
    if not pair:
        return None
    response_quote, _ = pair
    start = answer_body.find(response_quote)
    if start < 0:
        start = answer_body.lower().find(response_quote.lower())
    if start < 0:
        return None
    end = start + len(response_quote)
    return response_quote, start, end


def build_clean_expansion_rows(
    selected: list[dict[str, Any]],
) -> list[LineAInstanceRow]:
    rows: list[LineAInstanceRow] = []
    for item in selected:
        tid = str(item["trajectory_id"])
        split = str(item["split"])
        traj = item["trajectory"]
        parsed = pseudo_clean_quote(traj)
        if parsed is None:
            continue
        quote, qs, qe = parsed
        iak = f"gold_clean:{tid}"
        rows.append(
            LineAInstanceRow(
                instance_audit_key=iak,
                trajectory_id=tid,
                split=split,
                y=0,
                gold_verdict="trajectory_clean",
                response_quote=quote,
                final_answer=str((traj.get("metadata") or {}).get("final_answer") or ""),
                quote_start=qs,
                quote_end=qe,
            )
        )
    return rows


def sample_clean_expansion_manifest(
    *,
    targets: dict[str, int] | None = None,
    seed: int = 42,
    exclude_incremental: bool = True,
) -> dict[str, Any]:
    targets = targets or DEFAULT_CLEAN_TARGETS
    pool = clean_trajectory_pool(exclude_incremental=exclude_incremental)
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for item in pool:
        by_split[item["split"]].append(item)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    per_split: dict[str, Any] = {}
    for sp in ("train", "dev", "test"):
        avail = by_split[sp]
        rng.shuffle(avail)
        n = min(int(targets.get(sp, 0)), len(avail))
        picked = avail[:n]
        selected.extend(picked)
        per_split[sp] = {"available": len(avail), "target": targets.get(sp, 0), "sampled": n}
    rows = build_clean_expansion_rows(selected)
    return {
        "schema": "line_a_v3_clean_expansion_v1",
        "seed": seed,
        "exclude_incremental_trajectories": exclude_incremental,
        "targets": targets,
        "per_split": per_split,
        "n_sampled_trajectories": len(selected),
        "n_pseudo_instances": len(rows),
        "selected_trajectory_ids": [r.trajectory_id for r in rows],
        "instances": [
            {
                "instance_audit_key": r.instance_audit_key,
                "trajectory_id": r.trajectory_id,
                "split": r.split,
                "response_quote": r.response_quote,
                "quote_start": r.quote_start,
                "quote_end": r.quote_end,
            }
            for r in rows
        ],
    }


def write_v3_cohort_manifest(
    *,
    targets: dict[str, int] | None = None,
    seed: int = 42,
    path=LINE_A_V3_COHORT_MANIFEST,
) -> dict[str, Any]:
    manifest = sample_clean_expansion_manifest(targets=targets, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    return manifest


def line_a_v3_all_rows(*, require_activation: bool = False) -> list[LineAInstanceRow]:
    """Adjudication PM/CB instances + expanded gold-clean pseudo-instances."""
    if not LINE_A_V3_COHORT_MANIFEST.is_file():
        write_v3_cohort_manifest()
    payload = json.loads(LINE_A_V3_COHORT_MANIFEST.read_text(encoding="utf-8"))
    inc_rows = line_a_adjudication_instance_rows(require_activation=False)
    # Drop adjudication correct_binding (34) — replaced by expanded gold-clean pool.
    pm_rows = [r for r in inc_rows if r.gold_verdict in PM_VERDICTS]
    clean_rows: list[LineAInstanceRow] = []
    norm = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}
    for inst in payload.get("instances") or []:
        tid = str(inst["trajectory_id"])
        traj = norm.get(tid, {})
        row = LineAInstanceRow(
            instance_audit_key=str(inst["instance_audit_key"]),
            trajectory_id=tid,
            split=str(inst["split"]),
            y=0,
            gold_verdict="trajectory_clean",
            response_quote=str(inst["response_quote"]),
            final_answer=str((traj.get("metadata") or {}).get("final_answer") or ""),
            quote_start=int(inst["quote_start"]),
            quote_end=int(inst["quote_end"]),
        )
        path = instance_npz_path(row.instance_audit_key, tid)
        if require_activation and not path.is_file():
            continue
        clean_rows.append(row)
    return pm_rows + clean_rows


def build_probe_dataset_v3(*, require_activation: bool = True) -> ProbeDataset:
    rows = line_a_v3_all_rows(require_activation=require_activation)
    activation_layers: dict[str, list[int]] = {}
    instances: list[dict[str, Any]] = []
    for r in rows:
        path = instance_npz_path(r.instance_audit_key, r.trajectory_id)
        loaded = None
        if path.is_file():
            from ccer.mechanism.activation_store import load_activation_npz

            loaded = load_activation_npz(path)
            activation_layers[r.instance_audit_key] = list(loaded["layer_indices"])
        if require_activation and loaded is None:
            continue
        instances.append(
            {
                "instance_audit_key": r.instance_audit_key,
                "trajectory_id": r.trajectory_id,
                "split": r.split,
                "y": r.y,
                "gold_verdict": r.gold_verdict,
                "response_quote": r.response_quote,
                "final_answer": r.final_answer,
                "cohort_source": "adjudication_pm" if r.gold_verdict in PM_VERDICTS else "gold_clean_expansion",
                "verdict_short": VERDICT_SHORT.get(r.gold_verdict, r.gold_verdict),
                "has_activation": loaded is not None,
            }
        )
    manifest = {
        "schema": "line_a_v3_expanded_clean",
        "n_instances": len(instances),
        "n_trajectories": len({r["trajectory_id"] for r in instances}),
        "split_counts": {sp: sum(1 for r in instances if r["split"] == sp) for sp in ("train", "dev", "test")},
        "label_counts": {
            "pm": sum(1 for r in instances if r["y"] == 1),
            "clean": sum(1 for r in instances if r["y"] == 0),
        },
        "verdict_counts": {},
        "cohort_sources": {},
    }
    from collections import Counter

    manifest["verdict_counts"] = dict(Counter(r["gold_verdict"] for r in instances))
    manifest["cohort_sources"] = dict(Counter(r["cohort_source"] for r in instances))
    for sp in ("train", "dev", "test"):
        sub = [r for r in instances if r["split"] == sp]
        manifest[f"split_{sp}_verdicts"] = dict(Counter(r["gold_verdict"] for r in instances if r["split"] == sp))
        manifest[f"split_{sp}_y"] = {"pm": sum(1 for r in sub if r["y"] == 1), "clean": sum(1 for r in sub if r["y"] == 0)}
    return ProbeDataset(instances=instances, activation_layers=activation_layers, manifest=manifest)


def rows_needing_extraction() -> list[LineAInstanceRow]:
    """Clean expansion rows without npz yet."""
    if not LINE_A_V3_COHORT_MANIFEST.is_file():
        write_v3_cohort_manifest()
    payload = json.loads(LINE_A_V3_COHORT_MANIFEST.read_text(encoding="utf-8"))
    norm = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}
    out: list[LineAInstanceRow] = []
    for inst in payload.get("instances") or []:
        tid = str(inst["trajectory_id"])
        iak = str(inst["instance_audit_key"])
        if instance_npz_path(iak, tid).is_file():
            continue
        traj = norm.get(tid)
        if not traj:
            continue
        out.append(
            LineAInstanceRow(
                instance_audit_key=iak,
                trajectory_id=tid,
                split=str(inst["split"]),
                y=0,
                gold_verdict="trajectory_clean",
                response_quote=str(inst["response_quote"]),
                final_answer=str((traj.get("metadata") or {}).get("final_answer") or ""),
                quote_start=int(inst["quote_start"]),
                quote_end=int(inst["quote_end"]),
            )
        )
    return out
