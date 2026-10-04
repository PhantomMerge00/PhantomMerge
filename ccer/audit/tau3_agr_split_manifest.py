"""Tau3 AGR four-way trajectory-level split for cross-domain experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from typing import Any

from ccer.io_utils import load_jsonl, write_json
from ccer.mechanism.tau3.cohort import tau3_instance_rows
from ccer.mechanism.tau3.domain import Tau3DomainConfig, get_tau3_domain

AGR_SEED = 42
AGR_RATIOS = {"D_p": 0.40, "D_c": 0.30, "D_f": 0.30}
TEST_FRACTION = 0.20


def _group_key(traj: dict[str, Any]) -> str:
    pack = str(traj.get("pack_source") or "")
    tid = str(traj.get("trajectory_id") or "")
    base = f"{pack}|{tid}"
    return hashlib.sha256(base.encode()).hexdigest()[:16]


def build_tau3_agr_split_manifest(
    cfg: Tau3DomainConfig | None = None,
    *,
    seed: int = AGR_SEED,
) -> dict[str, Any]:
    cfg = cfg or get_tau3_domain("telecom")
    norm = {r["trajectory_id"]: r for r in load_jsonl(cfg.normalized)}
    all_tids = sorted(norm.keys())

    groups: dict[str, list[str]] = defaultdict(list)
    for tid in all_tids:
        groups[_group_key(norm[tid])].append(tid)

    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_test = max(1, int(round(len(group_ids) * TEST_FRACTION)))
    test_group_ids = set(group_ids[:n_test])
    fit_group_ids = group_ids[n_test:]

    traj_split: dict[str, str] = {}
    for gid in test_group_ids:
        for tid in groups[gid]:
            traj_split[tid] = "test"

    n_fit = len(fit_group_ids)
    n_dp = int(n_fit * AGR_RATIOS["D_p"])
    n_dc = int(n_fit * AGR_RATIOS["D_c"])
    for i, gid in enumerate(fit_group_ids):
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
    rows = tau3_instance_rows(require_activation=False, cfg=cfg)
    for row in rows:
        tid = row.trajectory_id
        iak = row.instance_audit_key
        sp = traj_split.get(tid, "test")
        instance_splits[iak] = sp
        verdict_by_split[sp][row.gold_verdict] += 1

    test_tids = {tid for tid, sp in traj_split.items() if sp == "test"}
    fit_tids = {tid for tid, sp in traj_split.items() if sp in ("D_p", "D_c", "D_f")}
    leak = test_tids & fit_tids

    payload = {
        "schema": "tau3_agr_split_manifest_v1",
        "domain": cfg.domain,
        "seed": seed,
        "ratios_fit_pool": AGR_RATIOS,
        "test_fraction": TEST_FRACTION,
        "n_trajectories_total": len(all_tids),
        "n_instances_total": len(rows),
        "trajectory_split": traj_split,
        "instance_split": instance_splits,
        "trajectory_counts": dict(Counter(traj_split.values())),
        "instance_counts": dict(Counter(instance_splits.values())),
        "verdict_counts_by_split": {k: dict(v) for k, v in verdict_by_split.items()},
        "leak_check": {
            "test_intersect_fit_pool": len(leak),
            "leak_trajectory_ids": sorted(leak),
        },
    }
    return payload


def write_tau3_agr_split_manifest(
    *,
    domain: str = "telecom",
    seed: int = AGR_SEED,
) -> dict[str, Any]:
    cfg = get_tau3_domain(domain)
    payload = build_tau3_agr_split_manifest(cfg, seed=seed)
    cfg.split_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(cfg.split_manifest, payload)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="telecom", choices=["telecom", "airline"])
    ap.add_argument("--seed", type=int, default=AGR_SEED)
    args = ap.parse_args()
    p = write_tau3_agr_split_manifest(domain=args.domain, seed=args.seed)
    print(json.dumps(p["trajectory_counts"], indent=2))
    print(json.dumps(p["leak_check"], indent=2))
