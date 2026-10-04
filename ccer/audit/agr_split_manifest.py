"""AGR §0: four-way trajectory-level split (D_p, D_c, D_f, test)."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from typing import Any

from ccer.audit.split_manifest import _group_key
from ccer.io_utils import load_jsonl, write_json
from ccer.mechanism.line_a_cohort_v3 import build_probe_dataset_v3
from ccer.paths import AGR_SPLIT_MANIFEST, NORMALIZED_SHOPPING, SPLIT_MANIFEST_JSON

AGR_SEED = 42
AGR_RATIOS = {"D_p": 0.40, "D_c": 0.30, "D_f": 0.30}


def _locked_test_trajectory_ids() -> set[str]:
    ds = build_probe_dataset_v3(require_activation=True)
    return {str(r["trajectory_id"]) for r in ds.instances if r["split"] == "test"}


def build_agr_split_manifest(*, seed: int = AGR_SEED) -> dict[str, Any]:
    ds = build_probe_dataset_v3(require_activation=True)
    all_tids = sorted({str(r["trajectory_id"]) for r in ds.instances})
    test_tids = _locked_test_trajectory_ids()
    fit_pool = [t for t in all_tids if t not in test_tids]

    norm = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}
    groups: dict[str, list[str]] = defaultdict(list)
    for tid in fit_pool:
        traj = norm.get(tid) or {"trajectory_id": tid}
        groups[_group_key(traj)].append(tid)

    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n = len(group_ids)
    n_dp = int(n * AGR_RATIOS["D_p"])
    n_dc = int(n * AGR_RATIOS["D_c"])
    traj_split: dict[str, str] = {tid: "test" for tid in test_tids}

    for i, gid in enumerate(group_ids):
        if i < n_dp:
            sp = "D_p"
        elif i < n_dp + n_dc:
            sp = "D_c"
        else:
            sp = "D_f"
        for tid in groups[gid]:
            traj_split[tid] = sp

    instance_splits: dict[str, str] = {}
    verdict_by_split: dict[str, Counter] = defaultdict(Counter)
    for row in ds.instances:
        tid = str(row["trajectory_id"])
        iak = str(row["instance_audit_key"])
        sp = traj_split.get(tid, "test")
        instance_splits[iak] = sp
        verdict_by_split[sp][str(row.get("gold_verdict") or "")] += 1

    split_counts = Counter(instance_splits.values())
    traj_counts = Counter(traj_split.values())

    payload = {
        "schema": "agr_split_manifest_v1",
        "seed": seed,
        "ratios_fit_pool": AGR_RATIOS,
        "locked_test_trajectories": len(test_tids),
        "n_trajectories_total": len(all_tids),
        "n_instances_total": len(ds.instances),
        "trajectory_split": traj_split,
        "instance_split": instance_splits,
        "trajectory_counts": dict(traj_counts),
        "instance_counts": dict(split_counts),
        "verdict_counts_by_split": {k: dict(v) for k, v in verdict_by_split.items()},
        "parent_split_manifest": str(SPLIT_MANIFEST_JSON),
    }
    return payload


def write_agr_split_manifest(*, seed: int = AGR_SEED) -> dict[str, Any]:
    payload = build_agr_split_manifest(seed=seed)
    AGR_SPLIT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    write_json(AGR_SPLIT_MANIFEST, payload)
    return payload


if __name__ == "__main__":
    p = write_agr_split_manifest()
    print(json.dumps(p["trajectory_counts"], indent=2))
    print(json.dumps(p["instance_counts"], indent=2))
