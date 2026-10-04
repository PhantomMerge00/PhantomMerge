"""Line A sensitivity suite: label/BoW variants + attribution report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.mechanism.line_a_sensitivity import (
    run_sensitivity_suite,
    write_sensitivity_artifacts,
)
from ccer.paths import LINE_A_DIR


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Line A sensitivity & attribution experiments")
    ap.add_argument("--out-dir", type=Path, default=LINE_A_DIR)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Only evaluate layer 47 (smoke / fast)",
    )
    args = ap.parse_args(argv)

    summary = run_sensitivity_suite(n_boot=args.n_boot, seed=args.seed, quick=args.quick)
    write_sensitivity_artifacts(summary, args.out_dir)
    print(json.dumps(summary.get("paper_recommendation") or {}, indent=2, ensure_ascii=False))
    print(f"WROTE {args.out_dir / 'line_a_sensitivity_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
