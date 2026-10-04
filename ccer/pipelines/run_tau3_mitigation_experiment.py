"""τ³ multi-domain mitigation (Phase 1: E2E axis, mirrors Shopping e2e_mitigation)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json, write_jsonl
from ccer.mechanism.e2e_mitigation_experiment import E2E_METHOD_SPECS
from ccer.mechanism.tau3.domain import SUPPORTED_DOMAINS
from ccer.mechanism.tau3_mitigation_experiment import (
    _methods_done,
    run_tau3_e2e_mitigation_experiment,
)
from ccer.paths import (
    TAU3_E2E_MITIGATION_POOLED_REPORT,
    TAU3_E2E_MITIGATION_POOLED_SUMMARY,
    tau3_e2e_mitigation_paths,
)


def _render_domain_report(summary: dict) -> str:
    proto = summary.get("protocol") or {}
    base = summary.get("test_baseline") or {}
    domain = summary.get("tau3_domain", "?")
    lines = [
        f"# τ³ E2E Mitigation — {domain}",
        "",
        f"**Protocol**: {proto.get('aligned_with', '§0.4')}",
        f"**Cohort**: {proto.get('cohort')}",
        f"**Claims (test)**: {base.get('n_claims', '—')} | **Trajectories**: {base.get('n_trajectories', '—')}",
        "",
        "| τ | Method | PM (cons) | PM (opt) | Retention |",
        "|---|--------|-----------|----------|-----------|",
    ]
    proto = summary.get("protocol") or {}
    lines.insert(4, f"**Detection (probe-gated)**: {proto.get('detection_probe_gated')}")
    lines.insert(5, f"**τ_B / τ_probe**: {proto.get('tau_b')} / {proto.get('tau_probe')} (D_f F1)")
    lines.insert(6, "")
    for tau_name, block in (summary.get("results_by_tau") or {}).items():
        tau_b = block.get("tau")
        tau_p = block.get("tau_probe")
        for _key, row in (block.get("methods") or {}).items():
            con = row.get("audit_conservative") or {}
            opt = row.get("audit_optimistic") or {}
            lines.append(
                f"| {tau_name} (τ_B={tau_b}, τ_p={tau_p}) | {row.get('label', _key)} | "
                f"{con.get('gated_pm_rate', 0):.3f} | {opt.get('gated_pm_rate', 0):.3f} | "
                f"{con.get('claims_retained_mean', 0):.2f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _render_pooled_report(pooled: dict) -> str:
    lines = [
        "# τ³ E2E Mitigation — Pooled index",
        "",
        "Per-domain reports: `results/tau3/mitigation/e2e/<domain>/METHOD_COMPARISON_E2E.md`",
        "",
    ]
    for d, block in (pooled.get("per_domain") or {}).items():
        base = block.get("test_baseline") or {}
        lines.append(f"- **{d}**: {base.get('n_claims', '?')} claims, status={block.get('experiment_status')}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="τ³ E2E mitigation (telecom / airline / retail)")
    ap.add_argument(
        "--domains",
        default=",".join(sorted(SUPPORTED_DOMAINS)),
        help="Comma-separated domains",
    )
    ap.add_argument(
        "--methods",
        default="track_k_e2e,b3_copy_e2e,b1_rarr_e2e,b2_cove_e2e,prism_l_plus_a_e2e,prism_l_plus_b_e2e",
    )
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-llm", action="store_true", help="Skip RARR/CoVe/PRISM-L+ LLM calls")
    ap.add_argument("--pooled-only", action="store_true", help="Skip per-domain files (not recommended)")
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing per-domain method_comparison_summary.json",
    )
    args = ap.parse_args(argv)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    per_domain: dict[str, dict] = {}
    for domain in domains:
        paths = tau3_e2e_mitigation_paths(domain)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        prior: dict[str, Any] | None = None
        skip_methods: set[str] = set()
        if not args.no_resume and paths["summary"].is_file():
            prior = json.loads(paths["summary"].read_text(encoding="utf-8"))
            skip_methods = _methods_done(prior)
            if skip_methods:
                print(f"[tau3] {domain}: resume skip {sorted(skip_methods)}", flush=True)

        def _checkpoint(payload: dict[str, Any], logs: list[dict[str, Any]]) -> None:
            if args.pooled_only:
                return
            write_json(paths["summary"], payload)
            write_jsonl(paths["rewrite_log"], logs)
            paths["report"].write_text(_render_domain_report(payload), encoding="utf-8")

        summary = run_tau3_e2e_mitigation_experiment(
            domain,
            methods=methods,
            n_boot=args.n_boot,
            seed=args.seed,
            use_llm=not args.no_llm,
            skip_methods=skip_methods,
            prior_summary=prior,
            on_checkpoint=_checkpoint,
        )
        logs = summary.pop("_rewrite_logs", [])
        per_domain[domain] = summary
        if not args.pooled_only:
            write_json(paths["summary"], summary)
            write_jsonl(paths["rewrite_log"], logs)
            paths["report"].write_text(_render_domain_report(summary), encoding="utf-8")
            print(f"Wrote {paths['report']} (status={summary.get('experiment_status')})")

    pooled = {
        "schema_version": "ccer_tau3_e2e_mitigation_pooled_v1",
        "domains": domains,
        "per_domain": per_domain,
    }
    TAU3_E2E_MITIGATION_POOLED_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    write_json(TAU3_E2E_MITIGATION_POOLED_SUMMARY, pooled)
    TAU3_E2E_MITIGATION_POOLED_REPORT.write_text(_render_pooled_report(pooled), encoding="utf-8")
    print(f"Wrote {TAU3_E2E_MITIGATION_POOLED_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
