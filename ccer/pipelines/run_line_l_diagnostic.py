"""Run Line L pre-null diagnostics (no full experiment re-run)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json, write_jsonl
from ccer.mechanism.line_l_diagnostic import render_diagnostic_report, run_line_l_diagnostics
from ccer.paths import LINE_L_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Line L diagnostic gate")
    parser.add_argument(
        "--max-cem-samples",
        type=int,
        default=15,
        help="Max CEM extraction_miss cases to export",
    )
    args = parser.parse_args()

    LINE_L_DIR.mkdir(parents=True, exist_ok=True)
    payload = run_line_l_diagnostics(max_cem_miss_samples=args.max_cem_samples)

    summary_path = LINE_L_DIR / "diagnostic_summary.json"
    samples_path = LINE_L_DIR / "cem_miss_samples.jsonl"
    report_path = LINE_L_DIR / "LINE_L_DIAGNOSTIC.md"

    write_json(summary_path, payload["summary"])
    write_jsonl(samples_path, payload["cem_miss_samples"])
    report_path.write_text(render_diagnostic_report(payload), encoding="utf-8")

    print(json.dumps(payload["summary"]["interpretation_gate"], indent=2, ensure_ascii=False))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
