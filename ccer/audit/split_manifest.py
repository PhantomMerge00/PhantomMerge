"""Split and cohort manifest (§4)."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ccer.io_utils import load_jsonl, write_json
from ccer.paths import COHORT_MANIFEST_JSON, NORMALIZED_SHOPPING, SPLIT_MANIFEST_JSON

SPLIT_SEED = 42
DEV_TARGETS = {"CEM": 64, "CAP": 64, "matched_clean": 64}


def _group_key(row: dict[str, Any]) -> str:
    anchor = (row.get("commitment") or {}).get("action_anchor") or ""
    query = (row.get("metadata") or {}).get("query") or ""
    q_norm = " ".join(query.lower().split())[:80]
    soil = ""
    for step in (row.get("messages_raw") or {}).get("steps") or []:
        ei = (step or {}).get("extra_info") or {}
        if ei.get("soil_id"):
            soil = str(ei["soil_id"])
            break
    base = f"{anchor}|{q_norm}|{soil}" if anchor or q_norm else row["trajectory_id"]
    return hashlib.sha256(base.encode()).hexdigest()[:16]


def assign_splits(rows: list[dict[str, Any]], *, seed: int = SPLIT_SEED) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r.get("is_calibration"):
            continue
        gid = _group_key(r)
        groups[gid].append(r["trajectory_id"])

    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    n = len(group_ids)
    n_train = int(n * 0.6)
    n_dev = int(n * 0.2)
    split_map: dict[str, str] = {}
    for i, gid in enumerate(group_ids):
        if i < n_train:
            split = "train"
        elif i < n_train + n_dev:
            split = "dev"
        else:
            split = "test"
        for tid in groups[gid]:
            split_map[tid] = split
    for r in rows:
        if r.get("is_calibration"):
            split_map[r["trajectory_id"]] = "calibration"
    return split_map


def _primary_pm_type(instances: list[dict[str, Any]]) -> str | None:
    verdicts = [i.get("legacy_label") or i.get("gold_verdict") for i in instances]
    for v in ("cross_object_merge", "constraint_projection", "anchored_hallucination"):
        if v in verdicts:
            return {"cross_object_merge": "CEM", "constraint_projection": "CAP", "anchored_hallucination": "AH"}.get(v, v)
    return None


def _length_bucket(row: dict[str, Any]) -> str:
    n = int((row.get("metadata") or {}).get("prompt_char_len") or 0)
    if n < 8000:
        return "short"
    if n < 20000:
        return "medium"
    return "long"


def select_dev_cohort(
    rows: list[dict[str, Any]],
    split_map: dict[str, str],
    *,
    targets: dict[str, int] | None = None,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    targets = targets or DEV_TARGETS
    rng = random.Random(seed)
    dev_rows = [r for r in rows if split_map.get(r["trajectory_id"]) == "dev" and r["eligible"].get("counterfactual")]

    cem_roots: list[str] = []
    cap_roots: list[str] = []
    clean_roots: list[str] = []
    for r in dev_rows:
        outcome = r.get("trajectory_outcome")
        claims = r.get("claims") or []
        pm_type = _primary_pm_type(claims)
        if pm_type == "CEM":
            cem_roots.append(r["trajectory_id"])
        elif pm_type == "CAP":
            cap_roots.append(r["trajectory_id"])
        elif outcome == "clean":
            clean_roots.append(r["trajectory_id"])

    rng.shuffle(cem_roots)
    rng.shuffle(cap_roots)
    rng.shuffle(clean_roots)

    def _pick(ids: list[str], k: int) -> tuple[list[str], int]:
        return ids[:k], len(ids)

    cem_sel, cem_avail = _pick(cem_roots, targets["CEM"])
    cap_sel, cap_avail = _pick(cap_roots, targets["CAP"])

    # matched clean: match slot distribution from PM roots
    pm_slots = Counter()
    for r in dev_rows:
        if r["trajectory_id"] in cem_sel or r["trajectory_id"] in cap_sel:
            for c in r.get("claims") or []:
                pm_slots[c.get("slot_norm") or "unknown"] += 1
    top_slots = {s for s, _ in pm_slots.most_common(5)}

    clean_candidates = [r for r in dev_rows if r["trajectory_id"] in clean_roots]
    scored = []
    for r in clean_candidates:
        slots = {c.get("slot_norm") for c in r.get("claims") or []}
        score = len(slots & top_slots)
        scored.append((score, r))
    scored.sort(key=lambda x: (-x[0], x[1]["trajectory_id"]))
    clean_sel = [r["trajectory_id"] for _, r in scored[: targets["matched_clean"]]]
    clean_avail = len(clean_candidates)

    return {
        "dev_targets": targets,
        "actual_n": {
            "CEM": len(cem_sel),
            "CAP": len(cap_sel),
            "matched_clean": len(clean_sel),
        },
        "available_n": {"CEM": cem_avail, "CAP": cap_avail, "matched_clean": clean_avail},
        "cohorts": {
            "CEM": cem_sel,
            "CAP": cap_sel,
            "matched_clean": clean_sel,
        },
        "selection_rules": {
            "split": "dev_only",
            "eligible": "counterfactual=true",
            "matched_clean": "slot_overlap_with_pm_dev + length_bucket metadata",
        },
    }


def run_split_manifest(
    normalized_path: Path = NORMALIZED_SHOPPING,
) -> dict[str, Any]:
    rows = load_jsonl(normalized_path)
    split_map = assign_splits(rows)
    for r in rows:
        r["split"] = split_map.get(r["trajectory_id"], "unassigned")
        r["group_id"] = _group_key(r)

    cohort = select_dev_cohort(rows, split_map)

    split_manifest = {
        "schema_version": "ccer_split_v1",
        "seed": SPLIT_SEED,
        "ratio": "60/20/20_group_level",
        "splits": split_map,
        "group_count": len({split_map[t] for t in split_map}),
        "split_counts": Counter(split_map.values()),
        "calibration_policy": "calibration split excluded from test",
        "leakage_rule": "counterfactual descendants inherit root split",
    }
    write_json(SPLIT_MANIFEST_JSON, split_manifest)

    cohort_manifest = {
        "schema_version": "ccer_cohort_v1",
        "primary_endpoints_frozen": [
            "paired_source_contrast_CEM",
            "paired_source_contrast_CAP",
            "pm_reduction_with_known_slot_coverage",
        ],
        **cohort,
    }
    write_json(COHORT_MANIFEST_JSON, cohort_manifest)

    # Update normalized file with splits
    from ccer.io_utils import write_jsonl

    write_jsonl(normalized_path, rows)
    return {"split_manifest": split_manifest, "cohort_manifest": cohort_manifest}


if __name__ == "__main__":
    print(json.dumps(run_split_manifest(), indent=2, default=str))
