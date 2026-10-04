"""Line A v3: expanded clean + PM-type breakdown."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json
from ccer.mechanism.line_a_cohort_v3 import write_v3_cohort_manifest
from ccer.mechanism.line_a_probe_v3 import run_line_a_v3_probe_experiment
from ccer.paths import LINE_A_V3_PROBE_SUMMARY


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    write_v3_cohort_manifest()
    summary = run_line_a_v3_probe_experiment(n_boot=args.n_boot, seed=args.seed, require_ready=not args.force)
    write_json(LINE_A_V3_PROBE_SUMMARY, summary)
    print(json.dumps(summary.get("primary_comparison_expanded_test"), indent=2))
    print(f"WROTE {LINE_A_V3_PROBE_SUMMARY}")
    return 0 if summary.get("experiment_status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
