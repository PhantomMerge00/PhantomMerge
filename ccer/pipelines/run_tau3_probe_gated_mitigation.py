"""τ³ probe-gated mitigation (Shopping §4.2 seven arms per domain)."""
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
from ccer.mechanism.prism_l_plus_experiment import METHOD_SPECS
from ccer.mechanism.tau3.domain import SUPPORTED_DOMAINS
from ccer.mechanism.tau3_mitigation_experiment import (
    _methods_done_probe_gated,
    run_tau3_probe_gated_mitigation_experiment,
)
from ccer.paths import (
    TAU3_PROBE_GATED_MITIGATION_POOLED_SUMMARY,
    tau3_probe_gated_mitigation_paths,
)


def _render_domain_report(summary: dict) -> str:
    proto = summary.get("protocol") or {}
    base = summary.get("test_baseline") or {}
    domain = summary.get("tau3_domain", "?")
    lines = [
        f"# τ³ Probe-Gated Mitigation — {domain}",
        "",
        f"**Protocol**: {proto.get('aligned_with', 'Shopping §4.2')}",
        f"**Cohort**: {proto.get('cohort')}",
        f"**Detection**: {proto.get('detection_probe_gated')}",
        f"**τ_probe**: {proto.get('tau_probe')} ({proto.get('tau_probe_source')})",
        f"**Track-K gate**: {proto.get('track_k_gate')}",
        f"**Claims (test)**: {base.get('n_claims', '—')} | **Trajectories**: {base.get('n_trajectories', '—')}",
        "",
        "| Block | Method | PM (cons) | PM (opt) | Retention |",
        "|-------|--------|-----------|----------|-----------|",
    ]
    for tau_name, block in (summary.get("results_by_tau") or {}).items():
        tau_p = block.get("tau")
        for _key, row in (block.get("methods") or {}).items():
            con = row.get("audit_conservative") or {}
            opt = row.get("audit_optimistic") or {}
            lines.append(
                f"| {tau_name} (τ={tau_p}) | {row.get('label', _key)} | "
                f"{con.get('gated_pm_rate', 0):.3f} | {opt.get('gated_pm_rate', 0):.3f} | "
                f"{con.get('claims_retained_mean', 0):.2f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _build_pooled_summary(per_domain: dict[str, dict]) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for d, summary in per_domain.items():
        block = (summary.get("results_by_tau") or {}).get("probe_gated") or {}
        methods_out: dict[str, Any] = {}
        for mk, row in (block.get("methods") or {}).items():
            con = row.get("audit_conservative") or {}
            methods_out[mk] = {
                "label": row.get("label", mk),
                "gated_pm_rate": con.get("gated_pm_rate"),
                "claims_retained_mean": con.get("claims_retained_mean"),
            }
        index[d] = {
            "tau_probe": block.get("tau"),
            "baseline_pm_rate": (summary.get("test_baseline") or {}).get("baseline_pm_rate"),
            "methods": methods_out,
            "experiment_status": summary.get("experiment_status"),
        }
    return {
        "schema_version": "ccer_tau3_probe_gated_pooled_v1",
        "domains": list(per_domain.keys()),
        "per_domain": index,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="τ³ probe-gated mitigation (telecom / airline / retail)")
    ap.add_argument(
        "--domains",
        default=",".join(sorted(SUPPORTED_DOMAINS)),
        help="Comma-separated domains",
    )
    ap.add_argument(
        "--methods",
        default=",".join(METHOD_SPECS.keys()),
        help="Comma-separated method IDs (Shopping §4.2)",
    )
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-llm", action="store_true", help="Skip RARR/CoVe/PRISM-L+ LLM calls")
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
        paths = tau3_probe_gated_mitigation_paths(domain)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        prior: dict[str, Any] | None = None
        skip_methods: set[str] = set()
        if not args.no_resume and paths["summary"].is_file():
            prior = json.loads(paths["summary"].read_text(encoding="utf-8"))
            skip_methods = _methods_done_probe_gated(prior)
            if skip_methods:
                print(f"[tau3 probe-gated] {domain}: resume skip {sorted(skip_methods)}", flush=True)

        def _checkpoint(payload: dict[str, Any], logs: list[dict[str, Any]]) -> None:
            write_json(paths["summary"], payload)
            write_jsonl(paths["rewrite_log"], logs)
            paths["report"].write_text(_render_domain_report(payload), encoding="utf-8")

        summary = run_tau3_probe_gated_mitigation_experiment(
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
        write_json(paths["summary"], summary)
        write_jsonl(paths["rewrite_log"], logs)
        paths["report"].write_text(_render_domain_report(summary), encoding="utf-8")
        print(f"Wrote {paths['report']} (status={summary.get('experiment_status')})")

    pooled = _build_pooled_summary(per_domain)
    TAU3_PROBE_GATED_MITIGATION_POOLED_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    write_json(TAU3_PROBE_GATED_MITIGATION_POOLED_SUMMARY, pooled)
    print(f"Wrote {TAU3_PROBE_GATED_MITIGATION_POOLED_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
