"""PRISM-L+ unified experiment pipeline."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json, write_jsonl
from ccer.mechanism.prism_l_plus_experiment import METHOD_SPECS, run_prism_l_plus_experiment
from ccer.paths import (
    PRISM_L_PLUS_DIR,
    PRISM_L_PLUS_REPORT,
    PRISM_L_PLUS_REWRITE_LOG,
    PRISM_L_PLUS_SUMMARY,
)


def _render_report(summary: dict) -> str:
    lines = [
        "# PRISM-L+ Method Comparison",
        "",
        f"**Schema**: `{summary.get('schema_version')}`",
        "",
    ]
    base = summary.get("test_baseline") or {}
    lines.append(
        f"Test baseline PM: **{base.get('baseline_pm_rate', 0):.3f}** "
        f"({base.get('baseline_pm_count')}/{base.get('n_trajectories')})"
    )
    lines.append("")
    gates = summary.get("quality_gates") or {}
    for tau_name, block in (summary.get("results_by_tau") or {}).items():
        g = gates.get(tau_name) or {}
        lines.append(f"## τ = {block.get('tau')} ({tau_name})")
        lines.append("")
        if g:
            diff_rate = g.get("b1_b2_diff_rate")
            diff_str = f"{diff_rate:.1%}" if diff_rate is not None else "—"
            denom = (g.get("b1_b2_identical_count") or 0) + (g.get("b1_b2_diff_count") or 0)
            lines.append(
                f"Gates: B2≠B1 diff rate **{diff_str}** "
                f"({g.get('b1_b2_diff_count')}/{denom}), "
                f"PID-as-value verified **{g.get('pid_as_value_verified_count')}**"
            )
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

    sc = summary.get("success_criterion") or {}
    lines.append("## Quality gates")
    lines.append("")
    lines.append(f"- B2 ≠ B1 (>5% diff): **{sc.get('b2_ne_b1')}**")
    lines.append(f"- PID-as-value = 0: **{sc.get('pid_as_value_zero')}**")
    lines.append(f"- PRISM-L+ A PM ≤ B1 official: **{sc.get('prism_a_pm_lte_b1')}**")
    lines.append(f"- **Overall passed**: {sc.get('passed')}")
    lines.append("")
    return "\n".join(lines)


def _ensure_mitigation_llm_env() -> None:
    os.environ.setdefault("LINE_L_PLUS_VLLM_BASE_URL", "http://127.0.0.1:8012/v1")
    os.environ.setdefault("LINE_L_PLUS_LLM_MODEL", "Qwen3-8B")


def main(argv: list[str] | None = None) -> int:
    _ensure_mitigation_llm_env()
    ap = argparse.ArgumentParser(description="PRISM-L+ experiment")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--methods",
        default="track_k,b3_copy,b1_rarr_official,b2_cove_official,prism_l_plus_a,prism_l_plus_b,prism_audit_only",
    )
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    PRISM_L_PLUS_DIR.mkdir(parents=True, exist_ok=True)

    summary = run_prism_l_plus_experiment(
        methods=methods,
        n_boot=args.n_boot,
        seed=args.seed,
        use_llm=not args.no_llm,
    )
    logs = summary.pop("_rewrite_logs", [])
    write_json(PRISM_L_PLUS_SUMMARY, summary)
    write_jsonl(PRISM_L_PLUS_REWRITE_LOG, logs)
    PRISM_L_PLUS_REPORT.write_text(_render_report(summary), encoding="utf-8")
    print(f"Wrote {PRISM_L_PLUS_SUMMARY}")
    print(f"Wrote {PRISM_L_PLUS_REPORT}")
    print(f"Gates: {json.dumps(summary.get('success_criterion'), indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
