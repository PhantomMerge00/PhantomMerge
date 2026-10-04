"""E2E mitigation experiment pipeline (main table)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json, write_jsonl
from ccer.mechanism.e2e_mitigation_experiment import E2E_METHOD_SPECS, run_e2e_mitigation_experiment
from ccer.paths import (
    E2E_MITIGATION_DIR,
    E2E_MITIGATION_REPORT,
    E2E_MITIGATION_REWRITE_LOG,
    E2E_MITIGATION_SUMMARY,
    PRISM_L_PLUS_APPENDIX_REPORT,
    PRISM_L_PLUS_REPORT,
)


def _render_report(summary: dict) -> str:
    proto = summary.get("protocol") or {}
    lines = [
        "# E2E Mitigation Method Comparison (Main Table)",
        "",
        "**Mode**: end-to-end — each system uses its **own detection**; gold labels eval-only.",
        "",
        f"- Inference: {proto.get('inference')}",
        f"- Probe used for: {proto.get('probe_for')}",
        "",
    ]
    base = summary.get("test_baseline") or {}
    lines.append(
        f"Test baseline PM: **{base.get('baseline_pm_rate', 0):.3f}** "
        f"({base.get('baseline_pm_count')}/{base.get('n_trajectories')})"
    )
    lines.append("")
    lines.append(
        "> Probe-gated oracle cohort ablation: see "
        "`results/prism_l_plus/APPENDIX_PROBE_GATED.md` (prior `METHOD_COMPARISON.md`)."
    )
    lines.append("")

    for tau_name, block in (summary.get("results_by_tau") or {}).items():
        lines.append(f"## τ = {block.get('tau')} ({tau_name})")
        lines.append("")
        lines.append(
            "| Method | Gate | PM (cons) | PM (opt) | Retention | Grounded | Rewrites |"
        )
        lines.append("|--------|------|-----------|----------|-----------|----------|----------|")
        for key, row in (block.get("methods") or {}).items():
            con = row.get("audit_conservative") or {}
            opt = row.get("audit_optimistic") or {}
            qm = row.get("quality_metrics") or {}
            tier = row.get("tiered_routing") or {}
            gr = qm.get("grounded_rate")
            n_rw = tier.get("n_rewrite", qm.get("n_rewrite", "—"))
            gate = row.get("gate_description", "")
            lines.append(
                f"| {row.get('label', key)} | {gate[:40]} | "
                f"{con.get('gated_pm_rate', 0):.3f} | {opt.get('gated_pm_rate', 0):.3f} | "
                f"{con.get('claims_retained_mean', 0):.2f} | "
                f"{gr if gr is not None else '—'} | {n_rw} |"
            )
        lines.append("")

    return "\n".join(lines)


def _write_appendix_pointer() -> None:
    text = PRISM_L_PLUS_REPORT.read_text(encoding="utf-8") if PRISM_L_PLUS_REPORT.is_file() else ""
    header = (
        "# Appendix: Probe-Gated Cohort Ablation\n\n"
        "This appendix reports the **controlled** experiment where all methods share "
        "Line A probe τ-gating and adjudication-derived slots (not end-to-end).\n\n"
        "Use the **E2E main table** (`results/e2e_mitigation/METHOD_COMPARISON_E2E.md`) "
        "for primary comparisons.\n\n---\n\n"
    )
    PRISM_L_PLUS_APPENDIX_REPORT.write_text(header + text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E2E mitigation experiment")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--methods",
        default="track_k_e2e,b3_copy_e2e,b1_rarr_e2e,b2_cove_e2e,prism_l_plus_a_e2e,prism_l_plus_b_e2e",
    )
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    E2E_MITIGATION_DIR.mkdir(parents=True, exist_ok=True)

    summary = run_e2e_mitigation_experiment(
        methods=methods,
        n_boot=args.n_boot,
        seed=args.seed,
        use_llm=not args.no_llm,
    )
    logs = summary.pop("_rewrite_logs", [])
    write_json(E2E_MITIGATION_SUMMARY, summary)
    write_jsonl(E2E_MITIGATION_REWRITE_LOG, logs)
    E2E_MITIGATION_REPORT.write_text(_render_report(summary), encoding="utf-8")
    _write_appendix_pointer()
    print(f"Wrote {E2E_MITIGATION_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
