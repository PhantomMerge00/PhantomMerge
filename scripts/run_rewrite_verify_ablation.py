#!/usr/bin/env python3
"""Replay fac_rewrite_audit_queue under verify policies (rules / NLI)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccer.mechanism.line_l_pm_elimination_eval import (
    render_fac_rewrite_metrics_markdown,
    replay_audit_queue_verify_policies,
)
from ccer.mechanism.pair_select import load_trajectory_index
from ccer.paths import ARTIFACTS


def main() -> int:
    queue = ARTIFACTS / "rq3" / "fac_rewrite_audit_queue.jsonl"
    traj_index = load_trajectory_index()
    ablation = replay_audit_queue_verify_policies(queue, traj_index)
    out_md = ARTIFACTS / "rq3" / "FAC_REWRITE_METRICS_TABLE.md"
    body = render_fac_rewrite_metrics_markdown([], policy_ablation=ablation)
    out_md.write_text(body, encoding="utf-8")
    print(json.dumps(ablation, indent=2))
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
