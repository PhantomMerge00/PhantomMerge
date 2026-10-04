"""Track K: probe-guided claim filtering pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json, write_jsonl
from ccer.mechanism.line_k_claim_filter import (
    export_frozen_probe_scores,
    run_line_k_claim_filter_experiment,
)
from ccer.paths import (
    LINE_K_DEV_THRESHOLDS,
    LINE_K_DIR,
    LINE_K_PROBE_SCORES,
    LINE_K_REPORT,
    LINE_K_SUMMARY,
)


def _render_report(summary: dict) -> str:
    lines = [
        "# Task K: Probe-Guided Claim Filtering (Line A v3)",
        "",
        f"**Schema**: `{summary.get('schema_version')}`",
        f"**Status**: `{summary.get('experiment_status')}`",
        "",
        "## Protocol",
        "",
        f"- Position: L{summary.get('protocol', {}).get('layer')} `{summary.get('protocol', {}).get('position')}`",
        f"- Threshold selection: **{summary.get('protocol', {}).get('threshold_split')}** only",
        f"- Evaluation: **{summary.get('protocol', {}).get('eval_split')}** (single report)",
        "",
        "## Test baseline",
        "",
    ]
    base = summary.get("test_baseline") or {}
    lines.append(
        f"- Trajectory PM rate: **{base.get('baseline_pm_rate', 0):.3f}** "
        f"({base.get('baseline_pm_count')}/{base.get('n_trajectories')} trajectories)"
    )
    lines.append("")
    lines.append("## Test results (deletion)")
    lines.append("")
    lines.append("| Method | τ | PM rate | PM reduction | CB retention | boot CI95 |")
    lines.append("|--------|---|---------|--------------|--------------|-----------|")
    for row in summary.get("test_results") or []:
        ci = f"[{row.get('boot_ci95_lo', 0):.3f}, {row.get('boot_ci95_hi', 0):.3f}]"
        lines.append(
            f"| {row.get('method')} | {row.get('tau', 0):.2f} | "
            f"{row.get('gated_pm_rate', 0):.3f} | {row.get('pm_reduction', 0):.3f} | "
            f"{row.get('cb_retention_rate', 0):.3f} | {ci} |"
        )
    lines.append("")
    rr = (summary.get("test_results") or [{}])[0].get("random_retention_matched") or {}
    if rr:
        lines.append("## Random retention-matched control (probe aggressive)")
        lines.append("")
        lines.append(
            f"- Matched PM rate mean: **{rr.get('pm_rate_mean', 0):.3f}** "
            f"(p05–p95: [{rr.get('pm_rate_p05', 0):.3f}, {rr.get('pm_rate_p95', 0):.3f}])"
        )
        lines.append("")
    lines.append("## Phase 2 stub rewrite (vs deletion)")
    lines.append("")
    lines.append("| Method | τ | PM rate | CB retention | claims retained |")
    lines.append("|--------|---|---------|----------------|-----------------|")
    for row in (summary.get("phase2_rewrite_stub") or {}).get("results") or []:
        lines.append(
            f"| {row.get('method')} | {row.get('tau', 0):.2f} | "
            f"{row.get('gated_pm_rate', 0):.3f} | {row.get('cb_retention_rate', 0):.3f} | "
            f"{row.get('claims_retained_mean', 0):.2f} |"
        )
    lines.append("")
    sc = summary.get("success_criterion") or {}
    lines.append(f"**Success criterion passed**: {sc.get('passed')}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track K: Line A probe claim filtering")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balanced-min-cb", type=float, default=0.85)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    summary = run_line_k_claim_filter_experiment(
        n_boot=args.n_boot,
        seed=args.seed,
        balanced_min_cb=args.balanced_min_cb,
        require_ready=not args.force,
    )
    LINE_K_DIR.mkdir(parents=True, exist_ok=True)
    frozen = summary.pop("_write_frozen_scores", None)
    if frozen:
        write_jsonl(LINE_K_PROBE_SCORES, frozen)
    if summary.get("dev_thresholds"):
        write_json(LINE_K_DEV_THRESHOLDS, summary["dev_thresholds"])
    write_json(LINE_K_SUMMARY, summary)
    LINE_K_REPORT.write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(summary.get("success_criterion"), indent=2))
    print(f"WROTE {LINE_K_SUMMARY}")
    print(f"WROTE {LINE_K_REPORT}")
    return 0 if summary.get("experiment_status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
