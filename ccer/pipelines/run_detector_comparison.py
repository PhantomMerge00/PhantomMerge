"""Evaluate AGR ρ(c) vs BindSurprise on detection + gating + E2E correction (test cohort)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json
from ccer.mechanism.agr.detector_comparison import (
    build_unified_detector_rows,
    run_detector_gating_comparison,
    run_e2e_rewrite_comparison,
)
from ccer.paths import AGR_DIR


def _render_report(summary: dict) -> str:
    lines = [
        "# AGR vs BindSurprise Detector Evaluation",
        "",
        "Same test cohort (agr_split=test, n=274). Thresholds fit on D_f (F1-optimal).",
        "**Two formulations**: AGR ρ(c) (theory-led calibrated fusion) vs BindSurprise "
        "(coarse ℓ_pm + T_slot). BindSurprise typically wins AUROC empirically.",
        "",
        "## Delete-gating (PM-rate / CB-retention)",
        "",
        "| Detector | AUROC | F1 | PM-rate | CB-ret | τ (D_f) |",
        "|----------|-------|-----|---------|--------|---------|",
    ]
    for d in summary.get("gating_comparison", {}).get("detectors", []):
        det = d.get("detection_test") or {}
        g = d.get("gating_test_delete_policy") or {}
        lines.append(
            f"| {d.get('label', d.get('detector_id'))} | "
            f"{det.get('auroc', '—')} | {det.get('f1', '—')} | "
            f"{g.get('gated_pm_rate', '—')} | {g.get('cb_retention_rate', '—')} | "
            f"{d.get('tau_D_f', '—')} |"
        )
    sig = summary.get("gating_comparison", {}).get("significance_bind_surprise_vs_agr_rho") or {}
    boot = sig.get("bootstrap_diff") or {}
    delong = sig.get("delong") or {}
    lines.extend(
        [
            "",
            f"BindSurprise vs AGR ρ(c) AUROC diff (B−A): **{-float(boot.get('diff') or 0):.4f}** "
            f"(AGR−B bootstrap diff {boot.get('diff')}; CI [{boot.get('diff_ci_low')}, {boot.get('diff_ci_hi')}], "
            f"DeLong p={delong.get('pvalue')})",
            "",
            "## E2E fixed-anchor correction (use_llm="
            f"{summary.get('e2e_rewrite', {}).get('arms', {}).get('bindsurprise_prism_l_plus_a', {}).get('use_llm', False)})",
            "",
            "| Arm | PM-rate (opt) | CB-ret | τ |",
            "|-----|---------------|--------|---|",
        ]
    )
    for arm_id, arm in (summary.get("e2e_rewrite") or {}).get("arms", {}).items():
        opt = arm.get("audit_optimistic") or {}
        lines.append(
            f"| {arm.get('label', arm_id)} | {opt.get('gated_pm_rate', '—')} | "
            f"{opt.get('cb_retention_rate', '—')} | {arm.get('tau_score', '—')} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-llm", action="store_true", help="Use LLM in E2E rewrite (slow)")
    ap.add_argument("--skip-e2e", action="store_true", help="Only run gating comparison")
    args = ap.parse_args()

    rows, meta = build_unified_detector_rows()
    gating = run_detector_gating_comparison(rows)
    summary: dict = {
        "schema": "detector_comparison_v1",
        "calibration": meta.get("calibration"),
        "fusion_weights": meta.get("fusion_weights"),
        "gating_comparison": gating,
    }
    if not args.skip_e2e:
        summary["e2e_rewrite"] = run_e2e_rewrite_comparison(rows, use_llm=args.use_llm)

    out_json = AGR_DIR / "DETECTOR_COMPARISON_SUMMARY.json"
    out_md = AGR_DIR / "DETECTOR_COMPARISON_REPORT.md"
    write_json(out_json, summary)
    out_md.write_text(_render_report(summary), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")

    best_auroc = max(
        gating["detectors"],
        key=lambda d: float((d.get("detection_test") or {}).get("auroc") or 0),
    )
    print(
        f"Best AUROC (gating): {best_auroc.get('detector_id')} "
        f"= {(best_auroc.get('detection_test') or {}).get('auroc')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
