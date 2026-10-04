"""Track L: anchor-evidence-grounded extractive rewrite pipeline."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json, write_jsonl
from ccer.mechanism.line_l_claim_filter import run_line_l_claim_filter_experiment
from ccer.paths import (
    LINE_L_DIR,
    LINE_L_REPORT,
    LINE_L_REWRITE_LOG,
    LINE_L_REWRITE_SAMPLE,
    LINE_L_SUMMARY,
)


def _render_report(summary: dict) -> str:
    lines = [
        "# Task L: Anchor-Evidence-Grounded Extractive Rewrite",
        "",
        f"**Schema**: `{summary.get('schema_version')}`",
        f"**Status**: `{summary.get('experiment_status')}`",
        "",
    ]
    base = summary.get("test_baseline") or {}
    lines.append(
        f"## Test baseline: trajectory PM **{base.get('baseline_pm_rate', 0):.3f}** "
        f"({base.get('baseline_pm_count')}/{base.get('n_trajectories')})"
    )
    lines.append("")
    for tau_name, block in (summary.get("results_by_tau") or {}).items():
        lines.append(f"## τ = {block.get('tau')} ({tau_name})")
        lines.append("")
        lines.append("| Method | PM rate | PM reduction | CB retention | claims retained | rewrite / delete | boot CI95 |")
        lines.append("|--------|---------|--------------|--------------|-----------------|------------------|-----------|")
        for key in (
            "track_k_deletion",
            "rewrite_all_flagged",
            "tiered_delete_or_rewrite_v1",
            "wrong_anchor_extraction",
            "random_edit",
        ):
            row = block.get(key) or {}
            tier = row.get("tiered_routing") or {}
            if tier:
                route_s = f"{tier.get('n_rewrite', 0)} / {tier.get('n_delete', 0)}"
            else:
                route_s = "—"
            ci = f"[{row.get('boot_ci95_lo', 0):.3f}, {row.get('boot_ci95_hi', 0):.3f}]"
            lines.append(
                f"| {row.get('method', key)} | {row.get('gated_pm_rate', 0):.3f} | "
                f"{row.get('pm_reduction', 0):.3f} | {row.get('cb_retention_rate', 0):.3f} | "
                f"{row.get('claims_retained_mean', 0):.2f} | {route_s} | {ci} |"
            )
        perm = block.get("target_vs_wrong_permutation") or {}
        tiered = block.get("rewrite_all_flagged") or {}
        tr = tiered.get("tiered_routing") or {}
        v1 = block.get("tiered_delete_or_rewrite_v1") or {}
        v1r = v1.get("tiered_routing") or {}
        lines.append("")
        lines.append(
            f"- **rewrite_all_flagged (v2)**: extractive **{tr.get('n_extractive_rewrite')}** + "
            f"fallback **{tr.get('n_fallback_rewrite')}** / delete **{tr.get('n_delete')}** "
            f"(retention {tiered.get('claims_retained_mean', 0):.2f})"
        )
        lines.append(
            f"- **tiered v1 ablation**: rewrite **{v1r.get('n_rewrite')}** / delete **{v1r.get('n_delete')}** "
            f"(retention {v1.get('claims_retained_mean', 0):.2f})"
        )
        lines.append(
            f"- Target vs wrong PM diff: **{perm.get('observed_diff', 0):.3f}** "
            f"(p={perm.get('p_value_two_sided', 1):.3f})"
        )
        cmp = block.get("comparison") or {}
        lines.append(
            f"- Claim retention gain vs Track K deletion: **{cmp.get('retention_gain', 0):.3f}**"
        )
        lines.append("")
    sc = summary.get("success_criterion") or {}
    lines.append(f"**Success criterion passed**: {sc.get('passed')}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track L: extractive rewrite")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--rewrite-fallback",
        choices=("stub", "llm", "none"),
        default="stub",
        help="Fallback when extractive miss (v2 policy): stub, llm, or none",
    )
    args = ap.parse_args(argv)

    summary = run_line_l_claim_filter_experiment(
        n_boot=args.n_boot,
        seed=args.seed,
        rewrite_fallback=args.rewrite_fallback,
    )
    LINE_L_DIR.mkdir(parents=True, exist_ok=True)

    logs = summary.pop("_rewrite_logs", [])
    write_jsonl(LINE_L_REWRITE_LOG, logs)
    if logs:
        rng = random.Random(args.seed)
        sample = rng.sample(logs, min(20, len(logs)))
        write_jsonl(LINE_L_REWRITE_SAMPLE, sample)

    write_json(LINE_L_SUMMARY, summary)
    LINE_L_REPORT.write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(summary.get("success_criterion"), indent=2))
    print(f"WROTE {LINE_L_SUMMARY}")
    print(f"WROTE {LINE_L_REPORT}")
    print(f"WROTE {LINE_L_REWRITE_LOG}")
    return 0 if summary.get("experiment_status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
