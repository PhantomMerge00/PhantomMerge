"""PRISM-L+ unified experiment orchestration."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ccer.mechanism.claim_filter_audit import (
    audit_gating,
    audit_line_l,
    audit_line_l_conservative,
    bootstrap_traj_pm,
    bootstrap_traj_pm_line_l,
    bootstrap_traj_pm_line_l_conservative,
    instances_to_dataframe,
)
from ccer.mechanism.line_k_claim_filter import load_frozen_dev_thresholds, load_frozen_probe_scores
from ccer.mechanism.line_k_l_tiered import MethodId, apply_method_rewrite, compute_tiered_routing_stats
from ccer.mechanism.line_l_attribution_eval import (
    anchor_evidence_attribution_rate,
    evaluate_attribution_batch,
)
from ccer.mechanism.line_l_claim_filter import _enrich_score_rows
from ccer.mechanism.claim_filter_audit import compute_extraction_hit_rate
from ccer.mechanism.line_l_plus import _apply_method_arm
from ccer.mechanism.line_l_rewrite_quality import aggregate_quality_metrics
from ccer.mechanism.line_l_pm_elimination_eval import (
    aggregate_pm_elimination_metrics,
    aggregate_rewrite_quality_v2,
    render_fac_rewrite_metrics_markdown,
    replay_audit_queue_verify_policies,
)
from ccer.mechanism.line_l_wrong_anchor import resolve_committed_anchor_pid
from ccer.mechanism.pair_select import load_trajectory_index
from ccer.paths import ROOT
from ccer.mechanism.prism_audit import build_audit_index, merge_prism_audit_fields
from ccer.io_utils import load_jsonl
from ccer.paths import LINE_K_SUMMARY, PRISM_AUDIT_JSONL

METHOD_SPECS: dict[str, dict[str, Any]] = {
    "track_k": {"method": None, "audit": "optimistic", "label": "Track_K_deletion"},
    "b1_rarr_official": {"method": "b1_rarr_official", "audit": "both", "label": "B1_RARR_official"},
    "b2_cove_official": {"method": "b2_cove_official", "audit": "both", "label": "B2_CoVe_official"},
    "b3_copy": {"method": "b3_copy", "audit": "both", "label": "B3_Copy_constrained"},
    "prism_l_plus_a": {
        "method": "prism_l_plus_a",
        "audit": "both",
        "label": "PRISM_L_plus_A_delete_always",
    },
    "prism_l_plus_b": {
        "method": "prism_l_plus_b",
        "audit": "both",
        "label": "PRISM_L_plus_B_y0del_y1rarr",
    },
    "prism_audit_only": {
        "method": "prism_audit_only",
        "audit": "both",
        "label": "PRISM_audit_only_no_gate",
    },
}


def _scores_to_dataframe(score_rows: list[dict[str, Any]]) -> pd.DataFrame:
    audit_index = (
        build_audit_index(list(load_jsonl(PRISM_AUDIT_JSONL))) if PRISM_AUDIT_JSONL.is_file() else {}
    )
    probe_map = {str(r["instance_audit_key"]): float(r["p_pm"]) for r in score_rows}
    enriched = _enrich_score_rows(score_rows)
    enriched = [merge_prism_audit_fields(r, audit_index) for r in enriched]
    instances = [
        {
            "instance_audit_key": r["instance_audit_key"],
            "trajectory_id": r["trajectory_id"],
            "split": r["split"],
            "y": r["y"],
            "y_pm": r.get("y_pm", r["y"]),
            "gold_verdict": r["gold_verdict"],
            "response_quote": r["response_quote"],
        }
        for r in enriched
    ]
    df = instances_to_dataframe(instances, probe_map)
    for col in (
        "slot",
        "slot_norm",
        "claim_value",
        "committed_anchor_pid",
        "anchor_evidence_quote",
        "donor_owners",
        "textual_selected_pid",
        "prism_audit_state",
        "prism_slot_type_aligned",
        "slot_type_aligned",
    ):
        df[col] = [r.get(col) for r in enriched]
    return df


def _evaluate_method_block(
    df_target: pd.DataFrame,
    logs: list[dict[str, Any]],
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    method_key: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    from ccer.mechanism.line_l_plus import _verified_fn_factory

    spec = METHOD_SPECS[method_key]
    verified_fn = _verified_fn_factory(df_target, traj_index, anchor_mode="target")
    opt_audit = audit_line_l(df_target)
    con_audit = audit_line_l_conservative(df_target, verified_fn)
    quality = aggregate_quality_metrics(
        logs,
        traj_index,
        tau=tau,
        anchor_pid_fn=lambda r, t: resolve_committed_anchor_pid(r, t),
    )
    attr = evaluate_attribution_batch(logs, traj_index, tau=tau)
    attr_anchor = anchor_evidence_attribution_rate(
        logs,
        traj_index,
        tau=tau,
        anchor_pid_fn=lambda r, t: resolve_committed_anchor_pid(r, t),
    )
    hit = compute_extraction_hit_rate(df_target, tau)
    tier = compute_tiered_routing_stats(logs, tau=tau)
    pm_elim = aggregate_pm_elimination_metrics(logs, traj_index, tau=tau)
    rewrite_v2 = aggregate_rewrite_quality_v2(logs, traj_index, tau=tau)
    return {
        "method_key": method_key,
        "label": spec["label"],
        "tau": tau,
        "pm_elimination_metrics": pm_elim,
        "rewrite_quality_v2": rewrite_v2,
        "attribution_rate_committed": attr_anchor.get("rate"),
        "audit_optimistic": {
            **opt_audit.to_dict(),
            **bootstrap_traj_pm_line_l(df_target, B=n_boot, seed=seed),
        },
        "audit_conservative": {
            **con_audit.to_dict(),
            **bootstrap_traj_pm_line_l_conservative(df_target, verified_fn, B=n_boot, seed=seed),
        },
        "quality_metrics": quality,
        "attribution_metrics": attr,
        "anchor_evidence_attribution": attr_anchor,
        "extraction_hit_rate": hit,
        "tiered_routing": tier,
    }


def _gate_checks(logs: list[dict[str, Any]], tau: float) -> dict[str, Any]:
    """Quality gates: B2≠B1, PID-as-value=0 after strict verify."""
    import re

    by_arm: dict[str, dict[str, dict]] = {}
    for row in logs:
        if abs(float(row.get("tau", 0)) - tau) > 1e-9:
            continue
        arm = str(row.get("arm") or "")
        iak = str(row.get("instance_audit_key") or "")
        by_arm.setdefault(arm, {})[iak] = row

    b1 = by_arm.get("B1_RARR_official", {})
    b2 = by_arm.get("B2_CoVe_official", {})
    same = diff = 0
    for iak, r1 in b1.items():
        r2 = b2.get(iak)
        if not r2:
            continue
        if r1.get("action") == r2.get("action") and r1.get("quote_after") == r2.get("quote_after"):
            same += 1
        else:
            diff += 1
    total = same + diff
    pid_bad = 0
    pid_re = re.compile(r"^\d{8,12}$")
    for row in logs:
        if str(row.get("action") or "") != "rewrite":
            continue
        v = str(row.get("v_anchor") or "")
        slot = str(row.get("slot_norm") or "")
        if pid_re.match(v.strip()) and slot not in ("shop_id", "product_id", "price"):
            q = row.get("quality") or {}
            if q.get("verified_grounded"):
                pid_bad += 1

    return {
        "b1_b2_identical_count": same,
        "b1_b2_diff_count": diff,
        "b1_b2_diff_rate": (diff / total) if total else None,
        "b1_b2_gate_pass": (diff / total) > 0.05 if total else False,
        "pid_as_value_verified_count": pid_bad,
        "pid_as_value_gate_pass": pid_bad == 0,
    }


def run_prism_l_plus_experiment(
    *,
    methods: list[str] | None = None,
    n_boot: int = 500,
    seed: int = 42,
    use_llm: bool = True,
) -> dict[str, Any]:
    methods = methods or [
        "track_k",
        "b3_copy",
        "b1_rarr_official",
        "b2_cove_official",
        "prism_l_plus_a",
        "prism_l_plus_b",
        "prism_audit_only",
    ]

    score_rows = load_frozen_probe_scores()
    thresholds = load_frozen_dev_thresholds()
    traj_index = load_trajectory_index()
    df_all = _scores_to_dataframe(score_rows)
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)

    tau_specs = [
        ("aggressive", float(thresholds["line_a_probe_aggressive"])),
        ("balanced", float(thresholds["line_a_probe_balanced"])),
    ]

    all_logs: list[dict[str, Any]] = []
    results_by_tau: dict[str, Any] = {}
    gates_by_tau: dict[str, Any] = {}

    llm_methods = {
        "b1_rarr_official",
        "b2_cove_official",
        "prism_l_plus_a",
        "prism_l_plus_b",
        "prism_audit_only",
    }

    for tau_name, tau in tau_specs:
        tau_results: dict[str, Any] = {"tau": tau, "methods": {}}

        if "track_k" in methods:
            from ccer.mechanism.line_k_claim_filter import _score_drop_factory

            drop_fn = _score_drop_factory("p_pm", tau)
            tk = audit_gating(df_test, drop_fn)
            tau_results["methods"]["track_k"] = {
                "label": "Track_K_deletion",
                "audit_optimistic": {
                    **tk.to_dict(),
                    **bootstrap_traj_pm(df_test, drop_fn, B=n_boot, seed=seed),
                },
                "audit_conservative": {
                    **tk.to_dict(),
                    **bootstrap_traj_pm(df_test, drop_fn, B=n_boot, seed=seed),
                },
            }

        for key in methods:
            if key == "track_k":
                continue
            spec = METHOD_SPECS.get(key)
            if not spec or spec.get("method") is None:
                continue
            method_id: MethodId = spec["method"]
            arm = str(spec.get("label") or key)
            df_arm, logs = _apply_method_arm(
                df_test,
                traj_index,
                tau=tau,
                arm=arm,
                method=method_id,
                anchor_mode="target",
                use_llm=use_llm and method_id in llm_methods,
            )
            all_logs.extend(logs)
            tau_results["methods"][key] = _evaluate_method_block(
                df_arm,
                logs,
                traj_index,
                tau=tau,
                method_key=key,
                n_boot=n_boot,
                seed=seed,
            )
            # Patch method spec label for report
            tau_results["methods"][key]["label"] = spec["label"]

        gates_by_tau[tau_name] = _gate_checks(all_logs, tau)
        results_by_tau[tau_name] = tau_results

    baseline = audit_gating(df_test, lambda _r: False).to_dict()
    agg = results_by_tau.get("aggressive", {}).get("methods", {})
    gates = gates_by_tau.get("aggressive", {})
    b1_con = (agg.get("b1_rarr_official") or {}).get("audit_conservative") or {}
    prism_a = (agg.get("prism_l_plus_a") or {}).get("audit_conservative") or {}

    track_k_summary = {}
    if LINE_K_SUMMARY.is_file():
        track_k_summary = json.loads(LINE_K_SUMMARY.read_text(encoding="utf-8"))

    fac_blocks = []
    prism_a_block = (agg.get("prism_l_plus_a") or {})
    if prism_a_block:
        fac_blocks.append(
            {
                "method_key": "prism_l_plus_a",
                "label": prism_a_block.get("label"),
                "pm_elimination_metrics": prism_a_block.get("pm_elimination_metrics"),
                "attribution_rate_committed": prism_a_block.get("attribution_rate_committed"),
                "n_rewrite": (prism_a_block.get("rewrite_quality_v2") or {}).get("n_rewrite"),
            }
        )
    b3 = agg.get("b3_copy") or {}
    if b3:
        fac_blocks.append(
            {
                "method_key": "b3_copy",
                "label": b3.get("label"),
                "pm_elimination_metrics": b3.get("pm_elimination_metrics"),
                "attribution_rate_committed": b3.get("attribution_rate_committed"),
                "n_rewrite": (b3.get("rewrite_quality_v2") or {}).get("n_rewrite"),
            }
        )
    queue_path = ROOT / "results/rq3/fac_rewrite_audit_queue.jsonl"
    policy_ablation = replay_audit_queue_verify_policies(queue_path, traj_index)
    fac_md_path = ROOT / "results/rq3/FAC_REWRITE_METRICS_TABLE.md"
    fac_md_path.parent.mkdir(parents=True, exist_ok=True)
    fac_md_path.write_text(
        render_fac_rewrite_metrics_markdown(fac_blocks, policy_ablation=policy_ablation),
        encoding="utf-8",
    )

    return {
        "schema_version": "ccer_prism_l_plus_v2",
        "experiment_status": "completed",
        "protocol": {
            "version": "prism_l_plus_unified",
            "llm_backend": "Qwen3-8B@8012",
            "vendor_rarr": "third_party/RARR",
            "retriever": "obs_tau",
            "verify": "acceptance_verify_bundle",
            "methods_run": methods,
        },
        "policy_ablation_57": policy_ablation,
        "test_baseline": baseline,
        "results_by_tau": results_by_tau,
        "quality_gates": gates_by_tau,
        "track_k_reference": track_k_summary.get("test_results"),
        "success_criterion": {
            "b2_ne_b1": gates.get("b1_b2_gate_pass"),
            "pid_as_value_zero": gates.get("pid_as_value_gate_pass"),
            "prism_a_pm_lte_b1": prism_a.get("gated_pm_rate", 1.0) <= b1_con.get("gated_pm_rate", 1.0),
            "passed": bool(gates.get("b1_b2_gate_pass") and gates.get("pid_as_value_gate_pass")),
        },
        "_rewrite_logs": all_logs,
    }
