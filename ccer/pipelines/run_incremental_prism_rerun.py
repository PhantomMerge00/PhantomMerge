"""Incremental PRISM-L+ rerun after top-k=50 audit; merge into existing summaries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import load_jsonl, write_json, write_jsonl
from ccer.mechanism.e2e_mitigation_experiment import run_e2e_mitigation_experiment
from ccer.mechanism.prism_l_plus_experiment import run_prism_l_plus_experiment
from ccer.paths import (
    E2E_MITIGATION_REPORT,
    E2E_MITIGATION_REWRITE_LOG,
    E2E_MITIGATION_SUMMARY,
    PRISM_L_PLUS_REWRITE_LOG,
    PRISM_L_PLUS_SUMMARY,
)


def _merge_summaries(base: dict, incr: dict, method_keys: list[str]) -> dict:
    out = dict(base)
    out["protocol"] = {
        **(base.get("protocol") or {}),
        "incremental_rerun": True,
        "incremental_methods": method_keys,
        "audit_note": "top-k=50 BindSurprise audit",
    }
    for tau_name, block in (incr.get("results_by_tau") or {}).items():
        base_block = (out.get("results_by_tau") or {}).setdefault(tau_name, {"tau": block.get("tau"), "methods": {}})
        base_methods = base_block.setdefault("methods", {})
        for key in method_keys:
            if key in (block.get("methods") or {}):
                base_methods[key] = block["methods"][key]
    if incr.get("quality_gates"):
        out["quality_gates"] = {**(out.get("quality_gates") or {}), **incr["quality_gates"]}
    if incr.get("success_criterion"):
        out["success_criterion"] = incr["success_criterion"]
    return out


def _merge_logs(base_path: Path, incr_logs: list[dict], arms: set[str]) -> list[dict]:
    if not base_path.is_file():
        return incr_logs
    kept = [r for r in load_jsonl(base_path) if str(r.get("arm") or "") not in arms]
    return kept + incr_logs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    probe_methods = ["prism_l_plus_a", "prism_l_plus_b", "prism_audit_only"]
    e2e_methods = ["prism_l_plus_a_e2e", "prism_l_plus_b_e2e"]
    probe_arms = {
        "PRISM_L_plus_A_delete_always",
        "PRISM_L_plus_B_y0del_y1rarr",
        "PRISM_audit_only_no_gate",
    }
    e2e_arms = {"PRISM_L_plus_A_E2E", "PRISM_L_plus_B_E2E"}

    use_llm = not args.no_llm

    # Axis B: probe-gated PRISM arms
    base_probe = json.loads(PRISM_L_PLUS_SUMMARY.read_text(encoding="utf-8")) if PRISM_L_PLUS_SUMMARY.is_file() else {}
    incr_probe = run_prism_l_plus_experiment(
        methods=probe_methods,
        n_boot=args.n_boot,
        seed=args.seed,
        use_llm=use_llm,
    )
    probe_logs = incr_probe.pop("_rewrite_logs", [])
    merged_probe = _merge_summaries(base_probe, incr_probe, probe_methods)
    write_json(PRISM_L_PLUS_SUMMARY, merged_probe)
    merged_probe_logs = _merge_logs(PRISM_L_PLUS_REWRITE_LOG, probe_logs, probe_arms)
    write_jsonl(PRISM_L_PLUS_REWRITE_LOG, merged_probe_logs)

    # Axis A: E2E PRISM arms
    base_e2e = json.loads(E2E_MITIGATION_SUMMARY.read_text(encoding="utf-8")) if E2E_MITIGATION_SUMMARY.is_file() else {}
    incr_e2e = run_e2e_mitigation_experiment(
        methods=e2e_methods,
        n_boot=args.n_boot,
        seed=args.seed,
        use_llm=use_llm,
    )
    e2e_logs = incr_e2e.pop("_rewrite_logs", [])
    merged_e2e = _merge_summaries(base_e2e, incr_e2e, e2e_methods)
    write_json(E2E_MITIGATION_SUMMARY, merged_e2e)
    merged_e2e_logs = _merge_logs(E2E_MITIGATION_REWRITE_LOG, e2e_logs, e2e_arms)
    write_jsonl(E2E_MITIGATION_REWRITE_LOG, merged_e2e_logs)

    # Re-render reports via existing pipelines
    from ccer.pipelines.run_e2e_mitigation_experiment import _render_report as render_e2e
    from ccer.pipelines.run_prism_l_plus_experiment import _render_report as render_probe

    PRISM_L_PLUS_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    E2E_MITIGATION_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    from ccer.paths import PRISM_L_PLUS_REPORT

    PRISM_L_PLUS_REPORT.write_text(render_probe(merged_probe), encoding="utf-8")
    E2E_MITIGATION_REPORT.write_text(render_e2e(merged_e2e), encoding="utf-8")

    agg_b = merged_probe.get("results_by_tau", {}).get("aggressive", {}).get("methods", {})
    agg_e = merged_e2e.get("results_by_tau", {}).get("aggressive", {}).get("methods", {})
    a_b = (agg_b.get("prism_l_plus_a") or {}).get("audit_conservative") or {}
    a_e = (agg_e.get("prism_l_plus_a_e2e") or {}).get("audit_conservative") or {}
    print(
        "PRISM-L+ A probe-gated @tau=0.05:",
        a_b.get("gated_pm_rate"),
        "retention",
        a_b.get("claims_retained_mean"),
    )
    print(
        "PRISM-L+ A E2E @tau=0.05:",
        a_e.get("gated_pm_rate"),
        "retention",
        a_e.get("claims_retained_mean"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
