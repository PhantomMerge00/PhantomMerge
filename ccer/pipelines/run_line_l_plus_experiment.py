"""Line L+ full experiment pipeline (Phase 0–6)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json, write_jsonl
from ccer.mechanism.line_l_plus import METHOD_SPECS, run_line_l_plus_experiment
from ccer.paths import (
    LINE_L_PLUS_DIR,
    LINE_L_PLUS_EVAL_REPORT,
    LINE_L_PLUS_REPORT,
    LINE_L_PLUS_REWRITE_LOG,
    LINE_L_PLUS_SUMMARY,
)


def _render_method_table(summary: dict) -> str:
    lines = [
        "# Line L+ Method Comparison",
        "",
        f"**Schema**: `{summary.get('schema_version')}`",
        f"**Status**: `{summary.get('experiment_status')}`",
        "",
    ]
    base = summary.get("test_baseline") or {}
    lines.append(
        f"Test baseline PM: **{base.get('baseline_pm_rate', 0):.3f}** "
        f"({base.get('baseline_pm_count')}/{base.get('n_trajectories')})"
    )
    lines.append("")
    for tau_name, block in (summary.get("results_by_tau") or {}).items():
        lines.append(f"## τ = {block.get('tau')} ({tau_name})")
        lines.append("")
        lines.append(
            "| Method | PM (conservative) | PM (optimistic) | Retention | "
            "Grounded | Format valid | Attr. corrected |"
        )
        lines.append(
            "|--------|-------------------|-----------------|-----------|"
            "----------|--------------|-----------------|"
        )
        for key, row in (block.get("methods") or {}).items():
            con = row.get("audit_conservative") or {}
            opt = row.get("audit_optimistic") or {}
            qm = row.get("quality_metrics") or {}
            am = row.get("attribution_metrics") or {}
            gr = qm.get("grounded_rate")
            fv = qm.get("format_valid_rate")
            ar = am.get("attribution_corrected_rate")
            lines.append(
                f"| {row.get('label', key)} | {con.get('gated_pm_rate', 0):.3f} | "
                f"{opt.get('gated_pm_rate', 0):.3f} | {con.get('claims_retained_mean', 0):.2f} | "
                f"{gr if gr is not None else '—'} | {fv if fv is not None else '—'} | "
                f"{ar if ar is not None else '—'} |"
            )
        lines.append("")
        ours = (block.get("methods") or {}).get("ours_v3") or {}
        perm = ours.get("target_vs_wrong_attribution_permutation") or {}
        if perm:
            lines.append(
                f"- Ours vs wrong attribution diff: **{perm.get('observed_diff', 0):.3f}** "
                f"(p={perm.get('p_value_two_sided', 1):.3f})"
            )
        lines.append("")
    sc = summary.get("success_criterion") or {}
    lines.append(f"**Success criterion passed**: {sc.get('passed')}")
    lines.append("")
    return "\n".join(lines)


def _render_eval_baselines(summary: dict) -> str:
    lines = [
        "# Line L+ Evaluation Baselines (E1/E2/E3)",
        "",
        "## E1 — Optimistic audit (v2 upper bound)",
        "Any rewrite clears PM label regardless of grounding.",
        "",
        "## E2 — Conservative grounded audit (primary PM)",
        "Only verified_grounded rewrites clear PM; ungrounded rewrites retain text but keep gold PM.",
        "",
        "## E3 — Attribution specificity",
        "wrong_anchor uses extractive-only (no LLM/stub); target vs wrong tested on attribution_corrected_rate.",
        "",
    ]
    agg = (summary.get("results_by_tau") or {}).get("aggressive", {}).get("methods", {})
    for key in ("eval_e1", "eval_e2", "eval_e3"):
        row = agg.get(key) or {}
        if not row:
            continue
        lines.append(f"### {row.get('label', key)}")
        opt = row.get("audit_optimistic") or {}
        con = row.get("audit_conservative") or {}
        lines.append(f"- Optimistic PM: {opt.get('gated_pm_rate', 0):.3f}")
        lines.append(f"- Conservative PM: {con.get('gated_pm_rate', 0):.3f}")
        am = row.get("attribution_metrics") or {}
        lines.append(f"- Attribution corrected rate: {am.get('attribution_corrected_rate')}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Line L+ experiment")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--methods",
        default="track_k,eval_e1,eval_e2,eval_e3,b3_copy,b1_rarr,b2_cove,ours_v3,wrong_anchor_extractive",
        help="Comma-separated method keys",
    )
    ap.add_argument("--no-llm", action="store_true", help="Skip LLM calls (use cache or delete)")
    args = ap.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHOD_SPECS and m != "track_k":
            print(f"Warning: unknown method {m}, skipping validation")

    LINE_L_PLUS_DIR.mkdir(parents=True, exist_ok=True)
    summary = run_line_l_plus_experiment(
        methods=methods,
        n_boot=args.n_boot,
        seed=args.seed,
        use_llm=not args.no_llm,
    )
    logs = summary.pop("_rewrite_logs", [])
    write_json(LINE_L_PLUS_SUMMARY, summary)
    write_jsonl(LINE_L_PLUS_REWRITE_LOG, logs)
    LINE_L_PLUS_REPORT.write_text(_render_method_table(summary), encoding="utf-8")
    LINE_L_PLUS_EVAL_REPORT.write_text(_render_eval_baselines(summary), encoding="utf-8")
    print(f"Wrote {LINE_L_PLUS_SUMMARY}")
    print(f"Wrote {LINE_L_PLUS_REPORT}")
    print(f"Success: {summary.get('success_criterion')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
