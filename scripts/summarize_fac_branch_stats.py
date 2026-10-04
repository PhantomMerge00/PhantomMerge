"""FAC (prism_l_plus_a) branch stats from method_comparison_summary + rewrite logs."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json
from ccer.paths import ARTIFACTS


def _load_summary(domain: str) -> dict | None:
    if domain == "shopping":
        p = ARTIFACTS / "prism_l_plus" / "method_comparison_summary.json"
    else:
        p = ARTIFACTS / "tau3" / "mitigation" / "probe_gated" / domain / "method_comparison_summary.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _fac_block(summary: dict) -> dict | None:
    rbt = summary.get("results_by_tau") or {}
    for key in ("aggressive", "balanced", "probe_gated"):
        methods = (rbt.get(key) or {}).get("methods") or {}
        if methods.get("prism_l_plus_a"):
            return methods["prism_l_plus_a"]
    for bucket in rbt.values():
        methods = bucket.get("methods") or {}
        if methods.get("prism_l_plus_a"):
            return methods["prism_l_plus_a"]
    return None


def _rewrite_stats(domain: str) -> dict:
    if domain == "shopping":
        log = ARTIFACTS / "prism_l_plus" / "rewrite_log.jsonl"
        arm_match = "PRISM_L_plus_A_delete_always"
    else:
        log = ARTIFACTS / "tau3" / "mitigation" / "probe_gated" / domain / "rewrite_log.jsonl"
        arm_match = "PRISM_L_plus_A_delete_always"
    if not log.is_file():
        return {"error": f"missing log {log}"}
    branches: Counter = Counter()
    actions: Counter = Counter()
    flagged = 0
    rewrite_ok = 0
    pm_after = Counter()
    cb_harm = 0
    for line in log.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("arm") != arm_match:
            continue
        if r.get("split") != "test":
            continue
        p_pm = float(r.get("p_pm") or 0)
        tau = float(r.get("tau") or 1.0)
        if p_pm <= tau:
            continue
        flagged += 1
        branches[r.get("branch") or "?"] += 1
        actions[r.get("action") or "?"] += 1
        act = str(r.get("action") or "")
        if act == "rewrite" and r.get("quote_after"):
            rewrite_ok += 1
        # post-hoc PM: delete -> no PM in output; keep -> same y_pm; rewrite unknown without re-judge
        y_pm = int(r.get("y_pm") or 0)
        if act == "delete":
            pm_after["deleted"] += 0 if y_pm else 0
        elif act == "keep":
            pm_after["kept_pm"] += y_pm
            pm_after["kept_total"] += 1
        if y_pm == 0 and act == "delete":
            cb_harm += 1
    return {
        "n_flagged_test": flagged,
        "by_branch": dict(branches),
        "by_action": dict(actions),
        "n_rewrite_with_quote_after": rewrite_ok,
        "cb_false_delete_among_flagged_clean": cb_harm,
    }


def main() -> int:
    domains = ["shopping", "telecom", "airline", "retail"]
    out: dict = {"schema": "fac_branch_stats_v1", "domains": {}}
    for d in domains:
        summary = _load_summary(d)
        entry: dict = {"domain": d}
        if summary:
            block = _fac_block(summary)
            if block:
                cons = block.get("audit_conservative") or {}
                tier = block.get("tiered_routing") or {}
                entry["gated_pm_rate_conservative"] = cons.get("gated_pm_rate")
                entry["cb_retention_conservative"] = cons.get("cb_retention_rate")
                entry["claims_retained_mean"] = cons.get("claims_retained_mean")
                entry["tiered_routing"] = tier
                entry["quality_metrics"] = block.get("quality_metrics")
                entry["anchor_evidence_attribution"] = block.get("anchor_evidence_attribution")
        entry["rewrite_log"] = _rewrite_stats(d)
        out["domains"][d] = entry

    path = ARTIFACTS / "rq3" / "FAC_BRANCH_STATS.json"
    write_json(path, out)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
