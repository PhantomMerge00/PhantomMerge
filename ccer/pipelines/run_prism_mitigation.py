"""PRISM Phase 2: dual-channel gated mitigation ablation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json, write_jsonl
from ccer.mechanism.prism_mitigation import render_prism_task_report, run_prism_mitigation_experiment
from ccer.paths import PRISM_DIR, PRISM_MITIGATION_SUMMARY, PRISM_REWRITE_LOG, PRISM_TASK_REPORT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    PRISM_DIR.mkdir(parents=True, exist_ok=True)
    summary = run_prism_mitigation_experiment(n_boot=args.n_boot, seed=args.seed)
    logs = summary.pop("_rewrite_logs", [])
    write_json(PRISM_MITIGATION_SUMMARY, summary)
    write_jsonl(PRISM_REWRITE_LOG, logs)
    PRISM_TASK_REPORT.write_text(render_prism_task_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
