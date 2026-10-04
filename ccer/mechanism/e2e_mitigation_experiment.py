"""E2E mitigation experiment: native detection per system, gold labels eval-only."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ccer.mechanism.anchor_resolve_heuristic import resolve_heuristic_anchor_pid
from ccer.mechanism.claim_filter_audit import (
    audit_gating,
    audit_line_l,
    audit_line_l_conservative,
    bootstrap_traj_pm,
    bootstrap_traj_pm_line_l,
    bootstrap_traj_pm_line_l_conservative,
    compute_extraction_hit_rate,
    instances_to_dataframe,
)
from ccer.mechanism.e2e_mitigation import E2EMethodId, apply_e2e_mitigation
from ccer.mechanism.line_k_claim_filter import _score_drop_factory, load_frozen_dev_thresholds, load_frozen_probe_scores
from ccer.mechanism.line_k_l_tiered import compute_tiered_routing_stats
from ccer.mechanism.line_l_attribution_eval import (
    anchor_evidence_attribution_rate,
    evaluate_attribution_batch,
)
from ccer.mechanism.line_l_claim_filter import _enrich_score_rows
from ccer.mechanism.line_l_plus import _verified_fn_factory
from ccer.mechanism.line_l_rewrite_quality import aggregate_quality_metrics, assess_rewrite_quality
from ccer.mechanism.pair_select import load_trajectory_index
from ccer.mechanism.prism_audit import build_audit_index, merge_prism_audit_fields
from ccer.io_utils import load_jsonl
from ccer.paths import LINE_K_SUMMARY, PRISM_AUDIT_JSONL

E2E_METHOD_SPECS: dict[str, dict[str, Any]] = {
    "track_k_e2e": {
        "method": "track_k_e2e",
        "label": "Track_K_E2E_probe_delete",
        "gate": "Line A probe → delete (no oracle slot)",
    },
    "b1_rarr_e2e": {
        "method": "b1_rarr_e2e",
        "label": "B1_RARR_E2E_native",
        "gate": "RARR agreement gate (no probe pre-gate)",
    },
    "b2_cove_e2e": {
        "method": "b2_cove_e2e",
        "label": "B2_CoVe_E2E_native",
        "gate": "CoVe verify loop (no probe pre-gate)",
    },
    "b3_copy_e2e": {
        "method": "b3_copy_e2e",
        "label": "B3_Copy_E2E_all_claims",
        "gate": "Scan all parsed claims (no probe, no gold CEM)",
    },
    "prism_l_plus_a_e2e": {
        "method": "prism_l_plus_a_e2e",
        "label": "PRISM_L_plus_A_E2E",
        "gate": "Probe + J-lens gate → RewriteStack",
    },
    "prism_l_plus_b_e2e": {
        "method": "prism_l_plus_b_e2e",
        "label": "PRISM_L_plus_B_E2E",
        "gate": "Probe + J-lens; probe_only y0 delete / y1 stack",
    },
}


def _eval_dataframe(score_rows: list[dict[str, Any]]) -> pd.DataFrame:
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
        "prism_slot_type_aligned",
        "prism_audit_state",
    ):
        df[col] = [r.get(col) for r in enriched]
    return df


def _apply_e2e_arm(
    df: pd.DataFrame,
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    arm: str,
    method: E2EMethodId,
    use_llm: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    logs: list[dict[str, Any]] = []
    actions: list[str] = []
    branches: list[str] = []
    quotes_after: list[str | None] = []
    verified_flags: list[bool] = []

    for _, row in df.iterrows():
        rec = row.to_dict()
        rec["tau"] = tau
        tid = str(rec["trajectory_id"])
        traj = traj_index.get(tid, {})
        anchor_pid = resolve_heuristic_anchor_pid(traj)
        if not anchor_pid:
            outcome = {
                "arm": arm,
                "branch": "e2e_no_anchor",
                "action": "delete",
                "quote_before": rec.get("response_quote"),
                "quote_after": None,
                "miss_reason": "heuristic_anchor_miss",
                "method": method,
            }
        else:
            outcome = apply_e2e_mitigation(
                rec,
                traj,
                method=method,
                arm=arm,
                tau=tau,
                anchor_pid=anchor_pid,
                use_llm=use_llm,
            )
        q = assess_rewrite_quality({**outcome, **rec}, traj, anchor_pid=anchor_pid)
        enriched = {
            **rec,
            **outcome,
            "trajectory_id": tid,
            "anchor_pid": anchor_pid,
            "_verified_grounded": q.verified_grounded,
            "quality": q.to_dict(),
        }
        logs.append(enriched)
        actions.append(str(outcome.get("action") or "keep"))
        branches.append(str(outcome.get("branch") or "skip"))
        quotes_after.append(outcome.get("quote_after"))
        verified_flags.append(q.verified_grounded)

    out = df.copy()
    out["_line_l_action"] = actions
    out["_line_l_branch"] = branches
    out["quote_after"] = quotes_after
    out["_verified_grounded"] = verified_flags
    return out, logs


def _evaluate_block(
    df_target: pd.DataFrame,
    logs: list[dict[str, Any]],
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    method_key: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    spec = E2E_METHOD_SPECS[method_key]
    verified_fn = _verified_fn_factory(df_target, traj_index, anchor_mode="target")
    opt = audit_line_l(df_target)
    con = audit_line_l_conservative(df_target, verified_fn)
    quality = aggregate_quality_metrics(
        logs,
        traj_index,
        tau=tau,
        anchor_pid_fn=lambda r, t: str(r.get("anchor_pid") or resolve_heuristic_anchor_pid(t)),
    )
    attr = evaluate_attribution_batch(logs, traj_index, tau=tau)
    attr_anchor = anchor_evidence_attribution_rate(
        logs,
        traj_index,
        tau=tau,
        anchor_pid_fn=lambda r, t: str(r.get("anchor_pid") or resolve_heuristic_anchor_pid(t)),
    )
    tier = compute_tiered_routing_stats(logs, tau=tau)
    return {
        "method_key": method_key,
        "label": spec["label"],
        "gate_description": spec["gate"],
        "tau": tau,
        "audit_optimistic": {
            **opt.to_dict(),
            **bootstrap_traj_pm_line_l(df_target, B=n_boot, seed=seed),
        },
        "audit_conservative": {
            **con.to_dict(),
            **bootstrap_traj_pm_line_l_conservative(df_target, verified_fn, B=n_boot, seed=seed),
        },
        "quality_metrics": quality,
        "attribution_metrics": attr,
        "anchor_evidence_attribution": attr_anchor,
        "extraction_hit_rate": compute_extraction_hit_rate(df_target, tau),
        "tiered_routing": tier,
    }


def run_e2e_mitigation_experiment(
    *,
    methods: list[str] | None = None,
    n_boot: int = 500,
    seed: int = 42,
    use_llm: bool = True,
) -> dict[str, Any]:
    methods = methods or list(E2E_METHOD_SPECS.keys())
    score_rows = load_frozen_probe_scores()
    thresholds = load_frozen_dev_thresholds()
    traj_index = load_trajectory_index()
    df_all = _eval_dataframe(score_rows)
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)

    tau_specs = [
        ("aggressive", float(thresholds["line_a_probe_aggressive"])),
        ("balanced", float(thresholds["line_a_probe_balanced"])),
    ]
    llm_methods = {"b1_rarr_e2e", "b2_cove_e2e", "prism_l_plus_a_e2e", "prism_l_plus_b_e2e"}

    all_logs: list[dict[str, Any]] = []
    results_by_tau: dict[str, Any] = {}

    for tau_name, tau in tau_specs:
        tau_results: dict[str, Any] = {"tau": tau, "methods": {}}

        if "track_k_e2e" in methods:
            drop_fn = _score_drop_factory("p_pm", tau)
            tk = audit_gating(df_test, drop_fn)
            tau_results["methods"]["track_k_e2e"] = {
                "label": E2E_METHOD_SPECS["track_k_e2e"]["label"],
                "gate_description": E2E_METHOD_SPECS["track_k_e2e"]["gate"],
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
            if key == "track_k_e2e":
                continue
            spec = E2E_METHOD_SPECS.get(key)
            if not spec:
                continue
            method_id: E2EMethodId = spec["method"]
            arm = str(spec["label"])
            df_arm, logs = _apply_e2e_arm(
                df_test,
                traj_index,
                tau=tau,
                arm=arm,
                method=method_id,
                use_llm=use_llm and key in llm_methods,
            )
            all_logs.extend(logs)
            tau_results["methods"][key] = _evaluate_block(
                df_arm,
                logs,
                traj_index,
                tau=tau,
                method_key=key,
                n_boot=n_boot,
                seed=seed,
            )

        results_by_tau[tau_name] = tau_results

    baseline = audit_gating(df_test, lambda _r: False).to_dict()
    track_k_summary = {}
    if LINE_K_SUMMARY.is_file():
        track_k_summary = json.loads(LINE_K_SUMMARY.read_text(encoding="utf-8"))

    return {
        "schema_version": "ccer_e2e_mitigation_v1",
        "experiment_status": "completed",
        "protocol": {
            "mode": "end_to_end",
            "inference": "heuristic anchor + parsed slot; no gold_verdict / committed_anchor at decision",
            "evaluation": "gold PM labels eval-only; same test cohort 243 traj",
            "probe_for": "Track_K + PRISM-L+ only; RARR/CoVe/Copy use native gates",
            "methods_run": methods,
        },
        "test_baseline": baseline,
        "results_by_tau": results_by_tau,
        "track_k_reference": track_k_summary.get("test_results"),
        "_rewrite_logs": all_logs,
    }
