"""PRISM Phase 2: dual-channel gated mitigation ablation."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ccer.io_utils import load_jsonl
from ccer.mechanism.claim_filter_audit import (
    audit_gating,
    audit_line_l,
    bootstrap_traj_pm,
    bootstrap_traj_pm_line_l,
    compute_extraction_hit_rate,
    instances_to_dataframe,
)
from ccer.mechanism.line_k_claim_filter import (
    _score_drop_factory,
    load_frozen_dev_thresholds,
    load_frozen_probe_scores,
)
from ccer.mechanism.line_k_l_tiered import MitigationPolicy, apply_tiered_mitigation, compute_tiered_routing_stats
from ccer.mechanism.line_l_claim_filter import _enrich_score_rows
from ccer.mechanism.line_l_generative_rewrite import RewriteFallback
from ccer.mechanism.line_l_wrong_anchor import resolve_committed_anchor_pid
from ccer.mechanism.pair_select import load_trajectory_index
from ccer.mechanism.prism_audit import build_audit_index, merge_prism_audit_fields
from ccer.paths import LINE_K_SUMMARY, PRISM_AUDIT_JSONL


def _scores_to_dataframe(score_rows: list[dict[str, Any]]) -> pd.DataFrame:
    probe_map = {str(r["instance_audit_key"]): float(r["p_pm"]) for r in score_rows}
    enriched = _enrich_score_rows(score_rows)
    instances = [
        {
            "instance_audit_key": r["instance_audit_key"],
            "trajectory_id": r["trajectory_id"],
            "split": r["split"],
            "y": r["y"],
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
    ):
        df[col] = [r.get(col) for r in enriched]
    return df


def load_prism_audit_index() -> dict[str, dict[str, Any]]:
    if not PRISM_AUDIT_JSONL.is_file():
        raise FileNotFoundError(
            f"Missing PRISM audit file: {PRISM_AUDIT_JSONL}. Run run_prism_audit.py first."
        )
    return build_audit_index(list(load_jsonl(PRISM_AUDIT_JSONL)))


def _attach_prism_fields(rec: dict[str, Any], audit_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    iak = str(rec.get("instance_audit_key") or "")
    audit = audit_index.get(iak)
    if not audit:
        return rec
    return {
        **rec,
        "prism_audit_state": audit.get("audit_state"),
        "prism_slot_type_aligned": audit.get("slot_type_aligned"),
        "slot_type_aligned": audit.get("slot_type_aligned"),
        "prism_jlens_top5": audit.get("jlens_top5"),
        "prism_audit_rationale": audit.get("prism_audit_rationale"),
    }


def _apply_prism_arm(
    df: pd.DataFrame,
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    arm: str,
    mitigation_policy: MitigationPolicy,
    audit_index: dict[str, dict[str, Any]],
    rewrite_fallback: RewriteFallback = "stub",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    logs: list[dict[str, Any]] = []
    actions: list[str] = []
    branches: list[str] = []
    tiers: list[str] = []
    quotes_after: list[str | None] = []

    for _, row in df.iterrows():
        rec = _attach_prism_fields(row.to_dict(), audit_index)
        rec["tau"] = tau
        tid = str(rec["trajectory_id"])
        traj = traj_index.get(tid, {})
        anchor_pid = resolve_committed_anchor_pid(rec, traj)
        outcome = apply_tiered_mitigation(
            rec,
            traj,
            anchor_pid=anchor_pid,
            arm=arm,
            policy=mitigation_policy,
            rewrite_fallback=rewrite_fallback,
        )
        enriched = merge_prism_audit_fields(
            {
                **rec,
                "claim_id": rec.get("claim_id"),
                "trajectory_id": tid,
                "gold_verdict": rec.get("gold_verdict"),
                "arm": arm,
                "branch": str(outcome.get("branch") or ""),
                "action": str(outcome.get("action") or "keep"),
                "mitigation_tier": str(outcome.get("mitigation_tier") or "keep"),
                "mitigation_policy": outcome.get("mitigation_policy"),
                "quote_before": outcome.get("quote_before"),
                "quote_after": outcome.get("quote_after"),
                "tau": tau,
                "p_pm": rec.get("p_pm"),
                "miss_reason": outcome.get("miss_reason"),
                "prism_audit_state": outcome.get("prism_audit_state") or rec.get("prism_audit_state"),
            },
            audit_index,
        )
        logs.append(enriched)
        actions.append(str(enriched.get("action") or "keep"))
        branches.append(str(enriched.get("branch") or ""))
        tiers.append(str(enriched.get("mitigation_tier") or "keep"))
        quotes_after.append(enriched.get("quote_after"))

    out = df.copy()
    out["_line_l_action"] = actions
    out["_line_l_branch"] = branches
    out["_mitigation_tier"] = tiers
    out["quote_after"] = quotes_after
    return out, logs


def _probe_fp_rate(logs: list[dict[str, Any]], tau: float) -> dict[str, Any]:
    y0 = [r for r in logs if int(r.get("y") or 0) == 0]
    flagged_y0 = [r for r in y0 if float(r.get("p_pm") or 0) > float(tau)]
    intervened = [
        r
        for r in flagged_y0
        if str(r.get("action") or "") in ("delete", "rewrite")
    ]
    return {
        "n_y0": len(y0),
        "n_flagged_y0": len(flagged_y0),
        "n_intervened_y0": len(intervened),
        "probe_fp_intervention_rate": (len(intervened) / len(flagged_y0)) if flagged_y0 else None,
    }


def run_prism_mitigation_experiment(
    *,
    n_boot: int = 2000,
    seed: int = 42,
    rewrite_fallback: RewriteFallback = "stub",
) -> dict[str, Any]:
    score_rows = load_frozen_probe_scores()
    thresholds = load_frozen_dev_thresholds()
    traj_index = load_trajectory_index()
    audit_index = load_prism_audit_index()
    df_all = _scores_to_dataframe(score_rows)
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)

    tau_specs = [
        ("aggressive", float(thresholds["line_a_probe_aggressive"])),
        ("balanced", float(thresholds["line_a_probe_balanced"])),
    ]

    all_logs: list[dict[str, Any]] = []
    results_by_tau: dict[str, Any] = {}

    for tau_name, tau in tau_specs:
        df_line_l, logs_line_l = _apply_prism_arm(
            df_test,
            traj_index,
            tau=tau,
            arm="line_l_rewrite_v2",
            mitigation_policy="rewrite_all_flagged_v2",
            audit_index=audit_index,
            rewrite_fallback=rewrite_fallback,
        )
        df_prism, logs_prism = _apply_prism_arm(
            df_test,
            traj_index,
            tau=tau,
            arm="prism_gated_v1",
            mitigation_policy="prism_gated_v1",
            audit_index=audit_index,
            rewrite_fallback=rewrite_fallback,
        )
        df_audit_only, logs_audit_only = _apply_prism_arm(
            df_test,
            traj_index,
            tau=tau,
            arm="prism_audit_only",
            mitigation_policy="prism_audit_only",
            audit_index=audit_index,
            rewrite_fallback=rewrite_fallback,
        )

        all_logs.extend(logs_prism)

        track_k_row = {
            "method": f"track_k_deletion_{tau_name}",
            "tau": tau,
            **audit_gating(df_test, _score_drop_factory("p_pm", tau)).to_dict(),
            **bootstrap_traj_pm(df_test, _score_drop_factory("p_pm", tau), B=n_boot, seed=seed),
        }

        line_l_audit = audit_line_l(df_line_l)
        prism_audit = audit_line_l(df_prism)
        audit_only_audit = audit_line_l(df_audit_only)

        line_l_row = line_l_audit.to_dict()
        line_l_row.update(
            {
                "method": f"line_l_rewrite_v2_{tau_name}",
                "tau": tau,
                **bootstrap_traj_pm_line_l(df_line_l, B=n_boot, seed=seed),
                "extraction_hit_rate": compute_extraction_hit_rate(df_line_l, tau),
                "tiered_routing": compute_tiered_routing_stats(logs_line_l, tau=tau),
                "probe_fp": _probe_fp_rate(logs_line_l, tau),
            }
        )
        prism_row = prism_audit.to_dict()
        prism_row.update(
            {
                "method": f"prism_gated_v1_{tau_name}",
                "tau": tau,
                **bootstrap_traj_pm_line_l(df_prism, B=n_boot, seed=seed),
                "extraction_hit_rate": compute_extraction_hit_rate(df_prism, tau),
                "tiered_routing": compute_tiered_routing_stats(logs_prism, tau=tau),
                "probe_fp": _probe_fp_rate(logs_prism, tau),
            }
        )
        audit_only_row = audit_only_audit.to_dict()
        audit_only_row.update(
            {
                "method": f"prism_audit_only_{tau_name}",
                "tau": tau,
                **bootstrap_traj_pm_line_l(df_audit_only, B=n_boot, seed=seed),
                "tiered_routing": compute_tiered_routing_stats(logs_audit_only, tau=tau),
                "probe_fp": _probe_fp_rate(logs_audit_only, tau),
            }
        )

        results_by_tau[tau_name] = {
            "tau": tau,
            "track_k_deletion": track_k_row,
            "line_l_rewrite_v2": line_l_row,
            "prism_gated_v1": prism_row,
            "prism_audit_only": audit_only_row,
            "audit_only_matches_line_l": {
                "pm_rate_equal": abs(
                    line_l_row.get("gated_pm_rate", 0) - audit_only_row.get("gated_pm_rate", 0)
                )
                < 1e-9,
                "cb_retention_equal": abs(
                    line_l_row.get("cb_retention_rate", 0) - audit_only_row.get("cb_retention_rate", 0)
                )
                < 1e-9,
            },
        }

    baseline = audit_gating(df_test, lambda _r: False).to_dict()
    track_k_summary = {}
    if LINE_K_SUMMARY.is_file():
        track_k_summary = json.loads(LINE_K_SUMMARY.read_text(encoding="utf-8"))

    return {
        "schema_version": "ccer_prism_mitigation_v1",
        "experiment_status": "completed",
        "protocol": {
            "version": "prism_gated_v1",
            "eval_split": "test",
            "prism_audit_source": str(PRISM_AUDIT_JSONL),
            "thresholds_source": "results/line_k/dev_thresholds.json",
            "gate_rules": {
                "confirmed_risk": "rewrite_all_flagged_v2",
                "probe_only": "track_k_deletion",
                "latent_slot": "keep",
                "clean": "keep",
            },
        },
        "test_baseline": baseline,
        "results_by_tau": results_by_tau,
        "track_k_reference": track_k_summary.get("test_results"),
        "_rewrite_logs": all_logs,
    }


def render_prism_task_report(summary: dict[str, Any]) -> str:
    lines = [
        "# TASK PRISM Report — Probe-Readout Integrated Slot Mitigation",
        "",
        "## Diagnosis claim",
        "",
        "At claim_onset L49, the supervised binding probe (Line A) and unsupervised",
        "J-lens slot-type readout form **complementary dual channels**. We independently",
        "verify that J-lens reliably surfaces slot-type language (Price/Color/Size) but not",
        "concrete values or owner attribution — a granularity mismatch, not an execution failure.",
        "",
        "## Mitigation ablation (test split)",
        "",
    ]
    base = summary.get("test_baseline") or {}
    lines.append(
        f"Baseline trajectory PM: **{base.get('baseline_pm_rate', 0):.3f}** "
        f"({base.get('baseline_pm_count')}/{base.get('n_trajectories')})"
    )
    lines.append("")
    for tau_name, block in (summary.get("results_by_tau") or {}).items():
        lines.append(f"### τ = {block.get('tau')} ({tau_name})")
        lines.append("")
        lines.append(
            "| Method | PM rate | PM reduction | CB retention | claims retained | probe FP intervened (y=0) |"
        )
        lines.append("|--------|---------|--------------|--------------|-----------------|---------------------------|")
        for key in ("track_k_deletion", "line_l_rewrite_v2", "prism_gated_v1", "prism_audit_only"):
            row = block.get(key) or {}
            fp = row.get("probe_fp") or {}
            fp_s = fp.get("n_intervened_y0", "—")
            lines.append(
                f"| {row.get('method', key)} | {row.get('gated_pm_rate', 0):.3f} | "
                f"{row.get('pm_reduction', 0):.3f} | {row.get('cb_retention_rate', 0):.3f} | "
                f"{row.get('claims_retained_mean', 0):.2f} | {fp_s} |"
            )
        match = block.get("audit_only_matches_line_l") or {}
        lines.append("")
        lines.append(
            f"- prism_audit_only matches line_l numerically: "
            f"PM={match.get('pm_rate_equal')} retention={match.get('cb_retention_equal')}"
        )
        prism = block.get("prism_gated_v1") or {}
        line_l = block.get("line_l_rewrite_v2") or {}
        lines.append(
            f"- PRISM vs Line L CB retention delta: "
            f"{(prism.get('cb_retention_rate', 0) - line_l.get('cb_retention_rate', 0)):+.3f}"
        )
        lines.append("")
    lines.extend(
        [
            "## Rewrite log",
            "",
            "Each rewrite record includes `prism_audit_state`, `prism_slot_type_aligned`,",
            "`prism_jlens_top5`, and `prism_audit_rationale` for auditable mitigation decisions.",
        ]
    )
    return "\n".join(lines)
