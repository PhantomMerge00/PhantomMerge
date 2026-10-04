"""Retail tau3: all instances in test — zeroshot-only (31 PM traj too sparse for D_p)."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from ccer.io_utils import load_jsonl, write_json
from ccer.mechanism.tau3.cohort import tau3_instance_rows
from ccer.mechanism.tau3.domain import get_tau3_domain


def build_retail_zeroshot_test_manifest(cfg=None) -> dict:
    cfg = cfg or get_tau3_domain("retail")
    norm = {r["trajectory_id"]: r for r in load_jsonl(cfg.normalized)}
    traj_split = {tid: "test" for tid in norm}
    instance_split: dict[str, str] = {}
    verdict_by_split: dict[str, Counter] = defaultdict(Counter)
    for row in tau3_instance_rows(require_activation=False, cfg=cfg):
        instance_split[row.instance_audit_key] = "test"
        verdict_by_split["test"][row.gold_verdict] += 1
    pm_traj = sum(1 for r in norm.values() if r.get("trajectory_outcome") == "has_pm_core")
    return {
        "schema": "tau3_retail_zeroshot_test_manifest_v1",
        "domain": "retail",
        "mode": "zeroshot_test_only",
        "indomain_eligible": False,
        "indomain_skip_reason": f"only {pm_traj} PM trajectories — insufficient for D_p probe training",
        "n_trajectories_total": len(norm),
        "n_instances_total": len(instance_split),
        "trajectory_split": traj_split,
        "instance_split": instance_split,
        "trajectory_counts": dict(Counter(traj_split.values())),
        "instance_counts": dict(Counter(instance_split.values())),
        "verdict_counts_by_split": {k: dict(v) for k, v in verdict_by_split.items()},
        "leak_check": {"test_intersect_fit_pool": 0, "leak_trajectory_ids": []},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cfg = get_tau3_domain("retail")
    payload = build_retail_zeroshot_test_manifest(cfg)
    if args.write:
        cfg.split_manifest.parent.mkdir(parents=True, exist_ok=True)
        write_json(cfg.split_manifest, payload)
    print(json.dumps(payload["trajectory_counts"], indent=2))
    print(json.dumps({"pm_trajectories": sum(1 for t, s in payload["trajectory_split"].items() if s == "test")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
