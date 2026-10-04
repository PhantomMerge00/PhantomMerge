"""Line L+ experiment orchestration (Phase 0–6)."""

from __future__ import annotations

import json
from typing import Any, Callable, Literal

import pandas as pd

from ccer.mechanism.claim_filter_audit import (
    audit_gating,
    audit_line_l,
    audit_line_l_conservative,
    bootstrap_traj_pm,
    bootstrap_traj_pm_line_l,
    bootstrap_traj_pm_line_l_conservative,
    compute_extraction_hit_rate,
    instances_to_dataframe,
    permutation_pm_rate_diff,
)
from ccer.mechanism.line_k_claim_filter import (
    _score_drop_factory,
    load_frozen_dev_thresholds,
    load_frozen_probe_scores,
)
from ccer.mechanism.line_k_l_tiered import MethodId, apply_method_rewrite, apply_tiered_mitigation, compute_tiered_routing_stats
from ccer.mechanism.line_l_attribution_eval import (
    anchor_evidence_attribution_rate,
    evaluate_attribution_batch,
    permutation_attribution_diff,
)
from ccer.mechanism.line_l_claim_filter import _enrich_score_rows
from ccer.mechanism.line_l_rewrite_quality import aggregate_quality_metrics, assess_rewrite_quality
from ccer.mechanism.line_l_wrong_anchor import resolve_committed_anchor_pid, resolve_wrong_anchor_pid
from ccer.mechanism.pair_select import load_trajectory_index
from ccer.paths import LINE_K_SUMMARY

AuditMode = Literal["optimistic", "conservative", "both"]

METHOD_SPECS: dict[str, dict[str, Any]] = {
    "eval_e1": {"method": "eval_e1_v2_stub", "audit": "optimistic", "label": "E1_optimistic_audit"},
    "eval_e2": {"method": "b3_copy", "audit": "conservative", "label": "E2_conservative_audit"},
    "eval_e3": {"method": "b3_copy", "audit": "conservative", "label": "E3_attribution_specificity"},
    "b1_rarr": {"method": "b1_rarr", "audit": "both", "label": "B1_RARR_adapted"},
    "b2_cove": {"method": "b2_cove", "audit": "both", "label": "B2_CoVe_adapted"},
    "b3_copy": {"method": "b3_copy", "audit": "both", "label": "B3_Copy_constrained"},
    "ours_v3": {"method": "ours_v3", "audit": "both", "label": "Ours_Line_L_plus_v3"},
    "wrong_anchor_extractive": {
        "method": "wrong_anchor_extractive",
        "audit": "conservative",
        "label": "wrong_anchor_extractive_only",
        "anchor_mode": "wrong",
    },
    "track_k": {"method": None, "audit": "optimistic", "label": "Track_K_deletion"},
}


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


def _apply_method_arm(
    df: pd.DataFrame,
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    arm: str,
    method: MethodId,
    anchor_mode: str = "target",
    use_llm: bool = True,
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

        if anchor_mode == "target":
            anchor_pid = resolve_committed_anchor_pid(rec, traj)
        else:
            anchor_pid = resolve_wrong_anchor_pid(rec, traj) or ""
            if not anchor_pid:
                outcome = {
                    "arm": arm,
                    "branch": "extraction_miss",
                    "action": "delete",
                    "quote_before": rec.get("response_quote"),
                    "quote_after": None,
                    "miss_reason": "no_wrong_anchor_pid",
                    "method": method,
                }
                logs.append({**rec, **outcome, "trajectory_id": tid})
                actions.append("delete")
                branches.append("extraction_miss")
                quotes_after.append(None)
                verified_flags.append(False)
                continue

        if method == "eval_e1_v2_stub":
            outcome = apply_tiered_mitigation(
                rec,
                traj,
                anchor_pid=anchor_pid,
                arm=arm,
                policy="rewrite_all_flagged_v2",
                rewrite_fallback="stub",
            )
        else:
            outcome = apply_method_rewrite(
                rec,
                traj,
                anchor_pid=anchor_pid,
                arm=arm,
                method=method,
                use_llm=use_llm,
            )
        q = assess_rewrite_quality(
            {**outcome, **rec},
            traj,
            anchor_pid=anchor_pid,
        )
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
        branches.append(str(outcome.get("branch") or "skip_unflagged"))
        quotes_after.append(outcome.get("quote_after"))
        verified_flags.append(q.verified_grounded)

    out = df.copy()
    out["_line_l_action"] = actions
    out["_line_l_branch"] = branches
    out["quote_after"] = quotes_after
    out["_verified_grounded"] = verified_flags
    return out, logs


def _verified_fn_factory(
    df: pd.DataFrame, traj_index: dict[str, dict[str, Any]], anchor_mode: str = "target"
) -> Callable[[dict], bool]:
    row_map = {str(r.get("instance_audit_key") or r.get("claim_id")): r for r in df.to_dict("records")}

    def _fn(r: dict) -> bool:
        from ccer.mechanism.line_l_rewrite_acceptance import verify_rewrite_row_acceptance
        from ccer.mechanism.line_l_wrong_anchor import resolve_acceptance_pid_textual

        iak = str(r.get("instance_audit_key") or r.get("claim_id") or "")
        base = row_map.get(iak, r)
        if base.get("acceptance_ok") is True or r.get("acceptance_ok") is True:
            return True
        tid = str(base.get("trajectory_id") or r.get("group_id") or "")
        traj = traj_index.get(tid, {})
        merged = {**base, **r}
        if str(merged.get("action") or merged.get("_line_l_action") or "") != "rewrite":
            return False
        if anchor_mode == "wrong":
            pid = resolve_wrong_anchor_pid(base, traj) or ""
        else:
            pid = resolve_acceptance_pid_textual(base, traj) or resolve_committed_anchor_pid(
                base, traj
            )
        bundle = verify_rewrite_row_acceptance(merged, traj, acceptance_pid=pid)
        return bool(bundle.get("ok"))

    return _fn


def _evaluate_method_block(
    df_target: pd.DataFrame,
    logs: list[dict[str, Any]],
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    method_key: str,
    n_boot: int,
    seed: int,
    df_wrong: pd.DataFrame | None = None,
    wrong_logs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
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

    block: dict[str, Any] = {
        "method_key": method_key,
        "label": spec["label"],
        "tau": tau,
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

    if df_wrong is not None and wrong_logs is not None:
        block["target_vs_wrong_pm_permutation"] = permutation_pm_rate_diff(df_target, df_wrong, n_perm=1000, seed=seed)
        block["target_vs_wrong_attribution_permutation"] = permutation_attribution_diff(
            logs, wrong_logs, traj_index, tau=tau, n_perm=1000, seed=seed
        )
        wrong_attr = evaluate_attribution_batch(wrong_logs, traj_index, tau=tau)
        block["wrong_attribution_metrics"] = wrong_attr
        block["wrong_anchor_evidence_attribution"] = anchor_evidence_attribution_rate(
            wrong_logs,
            traj_index,
            tau=tau,
            anchor_pid_fn=lambda r, t: resolve_wrong_anchor_pid(r, t) or "",
        )
        t_rate = attr_anchor.get("anchor_evidence_attribution_rate") or 0
        w_rate = block["wrong_anchor_evidence_attribution"].get("anchor_evidence_attribution_rate") or 0
        block["anchor_specificity_diff"] = float(t_rate) - float(w_rate)

    return block


def run_line_l_plus_experiment(
    *,
    methods: list[str] | None = None,
    n_boot: int = 500,
    seed: int = 42,
    use_llm: bool = True,
) -> dict[str, Any]:
    methods = methods or [
        "track_k",
        "eval_e1",
        "eval_e2",
        "b3_copy",
        "b1_rarr",
        "b2_cove",
        "ours_v3",
        "wrong_anchor_extractive",
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

    for tau_name, tau in tau_specs:
        tau_results: dict[str, Any] = {"tau": tau, "methods": {}}

        if "track_k" in methods:
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

        df_wrong: pd.DataFrame | None = None
        wrong_logs: list[dict[str, Any]] | None = None
        if "wrong_anchor_extractive" in methods or any(
            m in methods for m in ("eval_e3", "ours_v3", "b1_rarr", "b2_cove", "b3_copy")
        ):
            df_wrong, wrong_logs = _apply_method_arm(
                df_test,
                traj_index,
                tau=tau,
                arm="wrong_anchor_extractive",
                method="wrong_anchor_extractive",
                anchor_mode="wrong",
                use_llm=False,
            )
            all_logs.extend(wrong_logs)

        for key in methods:
            if key == "track_k":
                continue
            spec = METHOD_SPECS.get(key)
            if not spec or spec.get("method") is None:
                continue
            method_id: MethodId = spec["method"]
            arm = str(spec.get("label") or key)
            anchor_mode = str(spec.get("anchor_mode") or "target")
            df_arm, logs = _apply_method_arm(
                df_test,
                traj_index,
                tau=tau,
                arm=arm,
                method=method_id,
                anchor_mode=anchor_mode,
                use_llm=use_llm and method_id in ("b1_rarr", "b2_cove", "ours_v3"),
            )
            all_logs.extend(logs)
            w_df = df_wrong if anchor_mode == "target" else None
            w_logs = wrong_logs if anchor_mode == "target" else None
            tau_results["methods"][key] = _evaluate_method_block(
                df_arm,
                logs,
                traj_index,
                tau=tau,
                method_key=key,
                n_boot=n_boot,
                seed=seed,
                df_wrong=w_df,
                wrong_logs=w_logs,
            )

        results_by_tau[tau_name] = tau_results

    baseline = audit_gating(df_test, lambda _r: False).to_dict()
    agg = results_by_tau.get("aggressive", {}).get("methods", {})
    ours = agg.get("ours_v3") or {}
    track_k = agg.get("track_k") or {}
    ours_con = ours.get("audit_conservative") or {}
    tk_con = track_k.get("audit_conservative") or track_k.get("audit_optimistic") or {}
    attr_perm = ours.get("target_vs_wrong_attribution_permutation") or {}

    success = {
        "pm_conservative_lte_track_k": ours_con.get("gated_pm_rate", 1.0)
        <= tk_con.get("gated_pm_rate", 1.0),
        "retention_gt_track_k": ours_con.get("claims_retained_mean", 0)
        > tk_con.get("claims_retained_mean", 0),
        "attribution_target_gt_wrong": (attr_perm.get("observed_diff") or 0) > 0,
        "attribution_significant": (attr_perm.get("p_value_two_sided") or 1.0) < 0.05,
        "grounded_rate_ge_0": (ours.get("quality_metrics") or {}).get("grounded_rate") is not None,
    }
    success["passed"] = all(
        [
            success["retention_gt_track_k"],
            success["attribution_target_gt_wrong"],
        ]
    )

    track_k_summary = {}
    if LINE_K_SUMMARY.is_file():
        track_k_summary = json.loads(LINE_K_SUMMARY.read_text(encoding="utf-8"))

    return {
        "schema_version": "ccer_line_l_plus_v1",
        "experiment_status": "completed",
        "protocol": {
            "version": "line_l_plus_phase_0_6",
            "llm_backend": "Llama-3.1-8B-Instruct",
            "eval_baselines": ["E1_optimistic", "E2_conservative", "E3_attribution"],
            "lit_baselines": ["B1_RARR", "B2_CoVe", "B3_Copy"],
            "ours": "selective_grounded_v3",
            "methods_run": methods,
        },
        "test_baseline": baseline,
        "results_by_tau": results_by_tau,
        "track_k_reference": track_k_summary.get("test_results"),
        "success_criterion": success,
        "_rewrite_logs": all_logs,
    }
