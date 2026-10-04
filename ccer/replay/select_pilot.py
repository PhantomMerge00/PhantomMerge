"""Select 8-16 length-stratified pilot trajectories."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ccer.io_utils import load_jsonl, write_jsonl
from ccer.paths import ARTIFACTS, NORMALIZED_SHOPPING


def length_bucket(row: dict[str, Any]) -> str:
    n = int((row.get("metadata") or {}).get("prompt_char_len") or 0)
    if n < 8000:
        return "short"
    if n < 20000:
        return "medium"
    return "long"


def select_pilot_cohort(
    rows: list[dict[str, Any]] | None = None,
    *,
    min_n: int = 8,
    max_n: int = 16,
    normalized_path: Path = NORMALIZED_SHOPPING,
) -> list[dict[str, Any]]:
    if rows is None:
        rows = load_jsonl(normalized_path)
    eligible = [r for r in rows if r.get("eligible", {}).get("native_replay")]
    by_bucket: dict[str, list[dict[str, Any]]] = {"short": [], "medium": [], "long": []}
    for r in eligible:
        by_bucket[length_bucket(r)].append(r)
    for k in by_bucket:
        by_bucket[k].sort(key=lambda x: x["trajectory_id"])

    selected: list[dict[str, Any]] = []
    # Round-robin across buckets
    while len(selected) < max_n and any(by_bucket[k] for k in by_bucket):
        for b in ("short", "medium", "long"):
            if by_bucket[b] and len(selected) < max_n:
                selected.append(by_bucket[b].pop(0))
    if len(selected) < min_n:
        for r in eligible:
            if r not in selected:
                selected.append(r)
            if len(selected) >= min_n:
                break
    return selected[:max_n]


def write_pilot_manifest(selected: list[dict[str, Any]], out_path: Path | None = None) -> Path:
    out_path = out_path or ARTIFACTS / "replay_pilot_cohort.jsonl"
    slim = [
        {
            "trajectory_id": r["trajectory_id"],
            "length_bucket": length_bucket(r),
            "prompt_char_len": (r.get("metadata") or {}).get("prompt_char_len"),
            "input_hash": r.get("input_hash"),
            "replay_fidelity_level": (r.get("metadata") or {}).get("replay_fidelity_level"),
        }
        for r in selected
    ]
    write_jsonl(out_path, slim)
    return out_path


if __name__ == "__main__":
    sel = select_pilot_cohort()
    p = write_pilot_manifest(sel)
    print(json.dumps({"n": len(sel), "path": str(p)}, indent=2))
