"""Plan B orchestrator: CAP interchange evidence + action-text mismatch stats."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.plan_b.cap_evidence import aggregate_cap_evidence
from ccer.plan_b.expert_brief import render_expert_brief
from ccer.plan_b.mismatch_stats import run_mismatch_stats
from ccer.plan_b.round10_audits import run_round10_audits
from ccer.paths import PLAN_B_DIR, PLAN_B_EXPERT_BRIEF, REPORTS


def run_plan_b(*, n_boot: int = 2000) -> dict[str, Any]:
    PLAN_B_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    mismatch = run_mismatch_stats(n_boot=n_boot)
    cap = aggregate_cap_evidence()
    round10 = run_round10_audits()

    PLAN_B_EXPERT_BRIEF.write_text(
        render_expert_brief(cap, mismatch, round10), encoding="utf-8"
    )

    summary = {
        "schema_version": "plan_b_run_v1",
        "cap": {"n_rows": cap["n_cap_rows"], "p1_follow": cap["p1_cap_follow"]["cap_cohort"]},
        "mismatch": {
            "consistent_n": mismatch["population"]["crosstab"].get("consistent", {}).get("n"),
            "mismatch_n": mismatch["population"]["crosstab"].get("mismatch", {}).get("n"),
            "odds_ratio": mismatch["population"]["odds_ratio_unadjusted"].get("odds_ratio"),
        },
        "expert_brief": str(PLAN_B_EXPERT_BRIEF),
    }
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run Plan B parallel track 2")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    print(json.dumps(run_plan_b(n_boot=args.n_boot), indent=2))
