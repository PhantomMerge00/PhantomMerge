"""Track L: anchor-evidence-grounded extractive rewrite."""

from __future__ import annotations

import json
import random
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
    permutation_pm_rate_diff,
)
from ccer.mechanism.line_k_claim_filter import (
    _score_drop_factory,
    load_frozen_dev_thresholds,
    load_frozen_probe_scores,
)
from ccer.mechanism.line_k_l_tiered import MitigationPolicy, apply_tiered_mitigation, compute_tiered_routing_stats
from ccer.mechanism.line_l_generative_rewrite import RewriteFallback
from ccer.mechanism.line_l_wrong_anchor import (
    resolve_committed_anchor_pid,
    resolve_wrong_anchor_pid,
)
from ccer.mechanism.pair_select import load_trajectory_index
from ccer.paths import INCREMENTAL_ADJUDICATION_JSONL, LINE_K_SUMMARY


def _load_adjudication_by_iak() -> dict[str, dict[str, Any]]:
    return {str(r["instance_audit_key"]): r for r in load_jsonl(INCREMENTAL_ADJUDICATION_JSONL)}


def enrich_rewrite_logs_with_prism(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach PRISM audit fields to rewrite log rows when audit artifact exists."""
    from ccer.mechanism.prism_audit import build_audit_index, merge_prism_audit_fields
    from ccer.paths import PRISM_AUDIT_JSONL

    if not PRISM_AUDIT_JSONL.is_file():
        return logs
    audit_index = build_audit_index(list(load_jsonl(PRISM_AUDIT_JSONL)))
    return [merge_prism_audit_fields(row, audit_index) for row in logs]


def _enrich_score_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adj = _load_adjudication_by_iak()
    out: list[dict[str, Any]] = []
    for row in score_rows:
        iak = str(row["instance_audit_key"])
        adj_row = adj.get(iak) or {}
        out.append(
            {
                **row,
                "slot": str(adj_row.get("slot") or ""),
                "slot_norm": str(adj_row.get("slot_norm") or ""),
                "claim_value": str(adj_row.get("value") or ""),
                "committed_anchor_pid": str(adj_row.get("committed_anchor_pid") or ""),
                "anchor_evidence_quote": str(adj_row.get("anchor_evidence_quote") or ""),
                "donor_owners": adj_row.get("donor_owners") or [],
                "textual_selected_pid": str(adj_row.get("textual_selected_pid") or ""),
            }
        )
    return out


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


def _apply_arm(
    df: pd.DataFrame,
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    arm: str,
    anchor_mode: str,
    mitigation_policy: MitigationPolicy = "rewrite_all_flagged_v2",
    rewrite_fallback: RewriteFallback = "stub",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    logs: list[dict[str, Any]] = []
    actions: list[str] = []
    branches: list[str] = []
    tiers: list[str] = []
    quotes_after: list[str | None] = []
    row_records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        rec = row.to_dict()
        rec["tau"] = tau
        tid = str(rec["trajectory_id"])
        traj = traj_index.get(tid, {})

        if anchor_mode == "target":
            anchor_pid = resolve_committed_anchor_pid(rec, traj)
            outcome = apply_tiered_mitigation(
                rec,
                traj,
                anchor_pid=anchor_pid,
                arm=arm,
                policy=mitigation_policy,
                rewrite_fallback=rewrite_fallback,
            )
        elif anchor_mode == "wrong":
            anchor_pid = resolve_wrong_anchor_pid(rec, traj) or ""
            if not anchor_pid:
                outcome = {
                    "arm": arm,
                    "branch": "extraction_miss",
                    "action": "delete",
                    "mitigation_tier": "delete",
                    "mitigation_policy": mitigation_policy,
                    "quote_before": rec.get("response_quote"),
                    "quote_after": None,
                    "miss_reason": "no_wrong_anchor_pid",
                }
            else:
                outcome = apply_tiered_mitigation(
                    rec,
                    traj,
                    anchor_pid=anchor_pid,
                    arm=arm,
                    policy=mitigation_policy,
                    rewrite_fallback=rewrite_fallback,
                )
        else:
            raise ValueError(anchor_mode)

        action = str(outcome.get("action") or "keep")
        branch = str(outcome.get("branch") or "skip_unflagged")
        tier = str(outcome.get("mitigation_tier") or "keep")
        enriched = {
            **rec,
            "claim_id": rec.get("claim_id"),
            "trajectory_id": tid,
            "gold_verdict": rec.get("gold_verdict"),
            "arm": arm,
            "branch": branch,
            "action": action,
            "mitigation_tier": tier,
            "mitigation_policy": outcome.get("mitigation_policy"),
            "slot_norm": outcome.get("slot_norm") or rec.get("slot_norm"),
            "v_anchor": outcome.get("v_anchor"),
            "source_field": outcome.get("source_field"),
            "wrong_anchor_pid": anchor_pid if anchor_mode == "wrong" else None,
            "quote_before": outcome.get("quote_before"),
            "quote_after": outcome.get("quote_after"),
            "tau": tau,
            "p_pm": rec.get("p_pm"),
            "miss_reason": outcome.get("miss_reason"),
        }
        logs.append(enriched)
        row_records.append(enriched)
        actions.append(action)
        branches.append(branch)
        tiers.append(tier)
        quotes_after.append(outcome.get("quote_after"))

    out = df.copy()
    out["_line_l_action"] = actions
    out["_line_l_branch"] = branches
    out["_mitigation_tier"] = tiers
    out["quote_after"] = quotes_after
    return out, logs


def _apply_random_edit(df: pd.DataFrame, tau: float, *, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    actions: list[str] = []
    branches: list[str] = []
    quotes_after: list[str | None] = []
    for _, row in df.iterrows():
        if float(row["p_pm"]) <= tau:
            actions.append("keep")
            branches.append("skip_unflagged")
            quotes_after.append(str(row.get("response_quote") or ""))
            continue
        quote = str(row.get("response_quote") or "")
        if quote:
            actions.append("random_edit")
            branches.append("random_edit")
            quotes_after.append(quote + f" [rand:{rng.randint(1000, 9999)}]")
        else:
            actions.append("delete")
            branches.append("extraction_miss")
            quotes_after.append(None)
    out = df.copy()
    out["_line_l_action"] = actions
    out["_line_l_branch"] = branches
    out["quote_after"] = quotes_after
    return out


def _track_k_deletion_row(df_test: pd.DataFrame, tau: float, name: str, *, boot_B: int, seed: int) -> dict[str, Any]:
    drop_fn = _score_drop_factory("p_pm", tau)
    audit = audit_gating(df_test, drop_fn)
    row = audit.to_dict()
    row["method"] = name
    row["tau"] = tau
    row.update(bootstrap_traj_pm(df_test, drop_fn, B=boot_B, seed=seed))
    return row


def run_line_l_claim_filter_experiment(
    *,
    n_boot: int = 2000,
    seed: int = 42,
    mitigation_policy: MitigationPolicy = "rewrite_all_flagged_v2",
    rewrite_fallback: RewriteFallback = "stub",
) -> dict[str, Any]:
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

    for tau_name, tau in tau_specs:
        df_target, logs_t = _apply_arm(
            df_test,
            traj_index,
            tau=tau,
            arm="target_extraction",
            anchor_mode="target",
            mitigation_policy=mitigation_policy,
            rewrite_fallback=rewrite_fallback,
        )
        df_wrong, logs_w = _apply_arm(
            df_test,
            traj_index,
            tau=tau,
            arm="wrong_anchor_extraction",
            anchor_mode="wrong",
            mitigation_policy=mitigation_policy,
            rewrite_fallback=rewrite_fallback,
        )
        df_v1, logs_v1 = _apply_arm(
            df_test,
            traj_index,
            tau=tau,
            arm="tiered_delete_or_rewrite_v1",
            anchor_mode="target",
            mitigation_policy="delete_or_rewrite_v1",
            rewrite_fallback="none",
        )
        df_random = _apply_random_edit(df_test, tau, seed=seed)

        all_logs.extend(logs_t)
        all_logs.extend(logs_w)

        target_audit = audit_line_l(df_target)
        wrong_audit = audit_line_l(df_wrong)
        v1_audit = audit_line_l(df_v1)
        random_audit = audit_line_l(df_random)
        hit_stats = compute_extraction_hit_rate(df_target, tau)
        tier_stats = compute_tiered_routing_stats(logs_t, tau=tau)
        v1_stats = compute_tiered_routing_stats(logs_v1, tau=tau)
        perm = permutation_pm_rate_diff(df_target, df_wrong, n_perm=1000, seed=seed)

        track_k_row = _track_k_deletion_row(
            df_test,
            tau,
            f"track_k_deletion_{tau_name}",
            boot_B=n_boot,
            seed=seed,
        )

        tiered_row = target_audit.to_dict()
        tiered_row.update(
            {
                "method": f"rewrite_all_flagged_{tau_name}",
                "tau": tau,
                **bootstrap_traj_pm_line_l(df_target, B=n_boot, seed=seed),
                "extraction_hit_rate": hit_stats,
                "tiered_routing": tier_stats,
            }
        )
        v1_row = v1_audit.to_dict()
        v1_row.update(
            {
                "method": f"tiered_delete_or_rewrite_v1_{tau_name}",
                "tau": tau,
                **bootstrap_traj_pm_line_l(df_v1, B=n_boot, seed=seed),
                "tiered_routing": v1_stats,
            }
        )
        target_row = dict(tiered_row)
        target_row["method"] = f"target_extraction_{tau_name}"
        wrong_row = wrong_audit.to_dict()
        wrong_row.update(
            {
                "method": f"wrong_anchor_extraction_{tau_name}",
                "tau": tau,
                **bootstrap_traj_pm_line_l(df_wrong, B=n_boot, seed=seed),
            }
        )
        random_row = random_audit.to_dict()
        random_row.update(
            {
                "method": f"retention_matched_random_edit_{tau_name}",
                "tau": tau,
                **bootstrap_traj_pm_line_l(df_random, B=n_boot, seed=seed),
            }
        )

        results_by_tau[tau_name] = {
            "tau": tau,
            "track_k_deletion": track_k_row,
            "rewrite_all_flagged": tiered_row,
            "tiered_delete_or_rewrite_v1": v1_row,
            "target_extraction": target_row,
            "wrong_anchor_extraction": wrong_row,
            "random_edit": random_row,
            "target_vs_wrong_permutation": perm,
            "comparison": {
                "claim_retention_target": target_row["claims_retained_mean"],
                "claim_retention_track_k": track_k_row["claims_retained_mean"],
                "retention_gain": target_row["claims_retained_mean"] - track_k_row["claims_retained_mean"],
                "tiered_vs_track_k": {
                    "n_rewrite": tier_stats.get("n_rewrite"),
                    "n_delete": tier_stats.get("n_delete"),
                    "rewrite_rate_flagged": tier_stats.get("rewrite_rate_among_flagged"),
                },
            },
        }

    baseline = audit_gating(df_test, lambda _r: False).to_dict()
    agg = results_by_tau["aggressive"]
    success = {
        "pm_below_baseline": agg["target_extraction"]["gated_pm_rate"] < baseline["baseline_pm_rate"],
        "baseline_ci_non_overlap": agg["target_extraction"]["boot_ci95_hi"] < baseline["baseline_pm_rate"],
        "target_beats_wrong": agg["target_vs_wrong_permutation"]["observed_diff"] < 0,
        "target_beats_wrong_significant": agg["target_vs_wrong_permutation"]["p_value_two_sided"] < 0.05,
        "retention_gt_track_k": agg["comparison"]["retention_gain"] > 0,
    }
    success["passed"] = all(
        [
            success["pm_below_baseline"],
            success["target_beats_wrong"],
            success["retention_gt_track_k"],
        ]
    )

    track_k_summary = {}
    if LINE_K_SUMMARY.is_file():
        track_k_summary = json.loads(LINE_K_SUMMARY.read_text(encoding="utf-8"))

    return {
        "schema_version": "ccer_line_l_v1",
        "experiment_status": "completed",
        "protocol": {
            "version": "line_k_l_tiered_v2",
            "intervention": "rewrite_all_flagged",
            "mitigation_policy": mitigation_policy,
            "rewrite_fallback": rewrite_fallback,
            "routing": {
                "unflagged": "keep",
                "extraction_hit": "extractive_rewrite",
                "extraction_miss_or_no_slot": "fallback_rewrite_stub_or_llm",
                "track_k_baseline": "delete_all_flagged",
            },
            "probe_scores_source": "results/line_k/probe_scores.jsonl",
            "thresholds_source": "results/line_k/dev_thresholds.json",
            "eval_split": "test",
            "note": "Unified Track K deletion fallback + Line L evidence rewrite; all PM types same routing.",
        },
        "test_baseline": baseline,
        "results_by_tau": results_by_tau,
        "track_k_reference": track_k_summary.get("test_results"),
        "success_criterion": success,
        "_rewrite_logs": all_logs,
    }
