"""Detector evaluation: AGR ρ(c) vs BindSurprise vs ablations vs frozen probe."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ccer.mechanism.agr.calibration import fit_threshold_f1, sigmoid
from ccer.mechanism.agr.fusion import (
    enrich_calibrated,
    fit_fusion_calibrations,
    fit_mle_fusion,
    fusion_score,
)
from ccer.mechanism.agr.signals import apply_z2_variant, build_agr_signal_rows, train_agr_probe
from ccer.mechanism.anchor_resolve_heuristic import resolve_heuristic_anchor_pid
from ccer.mechanism.claim_filter_audit import audit_gating, audit_line_l, audit_line_l_conservative
from ccer.mechanism.claim_filter_audit import instances_to_dataframe
from ccer.mechanism.agr.significance import compare_scores_on_rows
from ccer.mechanism.bind_surprise import detection_metrics, try_auroc
from ccer.mechanism.line_k_claim_filter import _score_drop_factory
from ccer.mechanism.line_l_rewrite_quality import assess_rewrite_quality
from ccer.mechanism.pair_select import load_trajectory_index
from ccer.mechanism.prism_audit import build_audit_index
from ccer.mechanism.prism_l_plus import apply_prism_l_plus
from ccer.io_utils import load_jsonl
from ccer.paths import PRISM_AUDIT_JSONL


DETECTOR_SPECS: dict[str, dict[str, Any]] = {
    "frozen_probe_p_pm": {
        "label": "Frozen Line-A probe (prism_audit)",
        "score_key": "frozen_p_pm",
        "higher_is_riskier": True,
    },
    "bind_surprise": {
        "label": "BindSurprise B=ℓ_pm+log π_J(T_slot)",
        "score_key": "bind_surprise",
        "higher_is_riskier": True,
    },
    "agr_probe_only": {
        "label": "AGR z_rep only (ablation)",
        "score_key": "agr_probe_logit",
        "higher_is_riskier": True,
    },
    "agr_rho": {
        "label": "AGR ρ(c) calibrated fusion",
        "score_key": "agr_rho_logit",
        "higher_is_riskier": True,
    },
}


def build_unified_detector_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """AGR signal rows enriched with prism_audit legacy scores / frozen probe."""
    probe = train_agr_probe(train_split="D_p")
    rows = build_agr_signal_rows(probe, z2_source="anchor_value")
    apply_z2_variant(rows, variant="odds", z2_source="anchor_value")
    cal = fit_fusion_calibrations(rows, cal_split="D_c")
    enrich_calibrated(rows, cal)
    weights = fit_mle_fusion(rows, fit_split="D_f", use_calibrated=True, interaction=False)

    audit_index = (
        build_audit_index(list(load_jsonl(PRISM_AUDIT_JSONL))) if PRISM_AUDIT_JSONL.is_file() else {}
    )
    for r in rows:
        iak = str(r["instance_audit_key"])
        pa = audit_index.get(iak) or {}
        r["frozen_p_pm"] = float(pa.get("p_pm") or 0.0)
        r["bind_surprise"] = float(pa.get("bind_surprise") or 0.0)
        r["prism_split"] = str(pa.get("split") or r.get("agr_split") or "")
        r["slot_type_aligned"] = bool(pa.get("slot_type_aligned"))
        r["prism_audit_state"] = pa.get("audit_state")
        r["agr_probe_logit"] = float(fusion_score(r, mode="probe_only", cal=cal, weights=weights))
        r["agr_rho_logit"] = float(fusion_score(r, mode="full_fusion", cal=cal, weights=weights))
        r["agr_p_rho"] = float(sigmoid(np.array([r["agr_rho_logit"]]))[0])
        r["_fusion_score"] = r["agr_rho_logit"]

    meta = {"calibration": cal, "fusion_weights": weights}
    return rows, meta


def _rows_to_gating_df(rows: list[dict[str, Any]]) -> Any:
    import pandas as pd

    instances = [
        {
            "instance_audit_key": r["instance_audit_key"],
            "trajectory_id": r["trajectory_id"],
            "split": r.get("prism_split") or r.get("agr_split"),
            "y": r["y_pm"],
            "y_pm": r["y_pm"],
            "gold_verdict": r.get("gold_verdict", ""),
            "response_quote": r.get("response_quote", ""),
        }
        for r in rows
    ]
    probe_map = {str(r["instance_audit_key"]): float(r["frozen_p_pm"]) for r in rows}
    df = instances_to_dataframe(instances, probe_map)
    for key in (
        "agr_split",
        "bind_surprise",
        "frozen_p_pm",
        "agr_probe_logit",
        "agr_rho_logit",
        "agr_p_rho",
        "z2_available",
        "slot_type_aligned",
        "prism_audit_state",
    ):
        df[key] = [r.get(key) for r in rows]
    return df


def eval_detector_gating(
    rows: list[dict[str, Any]],
    *,
    detector_id: str,
    fit_split: str = "D_f",
    eval_split: str = "test",
) -> dict[str, Any]:
    spec = DETECTOR_SPECS[detector_id]
    score_key = str(spec["score_key"])
    fit_rows = [r for r in rows if r.get("agr_split") == fit_split]
    test_rows = [r for r in rows if r.get("agr_split") == eval_split]
    if not fit_rows or not test_rows:
        return {"detector_id": detector_id, "error": "empty split"}

    tau = fit_threshold_f1(fit_rows, score_key)
    drop_fn: Callable[[dict], bool] = _score_drop_factory(score_key, float(tau))

    df_test = _rows_to_gating_df(test_rows)
    audit = audit_gating(df_test, drop_fn)
    det = detection_metrics(test_rows, score_key=score_key, threshold=float(tau))

    return {
        "detector_id": detector_id,
        "label": spec["label"],
        "score_key": score_key,
        "fit_split": fit_split,
        "eval_split": eval_split,
        "n_fit": len(fit_rows),
        "n_test": len(test_rows),
        "tau_D_f": float(tau),
        "detection_test": {
            "auroc": try_auroc(test_rows, score_key),
            **det,
        },
        "gating_test_delete_policy": audit.to_dict(),
    }


def run_detector_gating_comparison(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    results = [eval_detector_gating(rows, detector_id=did) for did in DETECTOR_SPECS]
    test_rows = [r for r in rows if r.get("agr_split") == "test"]
    for r in test_rows:
        r["_score_agr_rho"] = r["agr_rho_logit"]
        r["_score_bind_surprise"] = r["bind_surprise"]
    sig = compare_scores_on_rows(
        test_rows,
        "_score_agr_rho",
        "_score_bind_surprise",
        label_a="agr_rho",
        label_b="bind_surprise",
        n_boot=2000,
        seed=42,
    )
    return {
        "detectors": results,
        "significance_bind_surprise_vs_agr_rho": sig,
        "dual_signal_test_n": sum(1 for r in test_rows if r.get("z2_available")),
    }


def run_e2e_rewrite_comparison(
    rows: list[dict[str, Any]],
    *,
    fit_split: str = "D_f",
    eval_split: str = "test",
    use_llm: bool = False,
    tau_probe: float = 0.05,
) -> dict[str, Any]:
    """PRISM-L+ rewrite stack (arm A delete on probe_only) under two detectors."""
    fit_rows = [r for r in rows if r.get("agr_split") == fit_split]
    test_rows = [r for r in rows if r.get("agr_split") == eval_split]
    tau_bs = fit_threshold_f1(fit_rows, "bind_surprise")
    tau_agr = fit_threshold_f1(fit_rows, "agr_rho_logit")
    tau_agr_probe = fit_threshold_f1(fit_rows, "agr_probe_logit")
    traj_index = load_trajectory_index()

    def _bs_row(r: dict[str, Any]) -> dict[str, Any]:
        return {
            **r,
            "tau": tau_probe,
            "p_pm": r["frozen_p_pm"],
            "prism_audit_state": None,
            "slot_type_aligned": r.get("slot_type_aligned"),
        }

    def _agr_rho_row(r: dict[str, Any]) -> dict[str, Any]:
        return {
            **r,
            "tau": tau_probe,
            "p_pm": r["agr_p_rho"],
            "prism_audit_state": (
                "confirmed_risk" if float(r["agr_rho_logit"]) >= tau_agr else "clean"
            ),
        }

    def _agr_probe_row(r: dict[str, Any]) -> dict[str, Any]:
        z = float(r["agr_probe_logit"])
        return {
            **r,
            "tau": tau_probe,
            "p_pm": float(sigmoid(np.array([z]))[0]),
            "prism_audit_state": "confirmed_risk" if z >= tau_agr_probe else "clean",
        }

    arm_configs = (
        (
            "bindsurprise_prism_l_plus_a",
            "BindSurprise + fixed-anchor correction (empirical best)",
            tau_bs,
            _bs_row,
        ),
        (
            "agr_rho_prism_l_plus_a",
            "AGR ρ(c) + fixed-anchor correction (theory-led)",
            tau_agr,
            _agr_rho_row,
        ),
        (
            "agr_probe_prism_l_plus_a",
            "AGR z_rep only + fixed-anchor correction (ablation)",
            tau_agr_probe,
            _agr_probe_row,
        ),
    )
    arms: dict[str, Any] = {}
    for arm_id, label, tau_score, builder in arm_configs:
        logs: list[dict[str, Any]] = []
        actions: list[str] = []
        for r in test_rows:
            rec = builder(r)
            tid = str(r["trajectory_id"])
            traj = traj_index.get(tid) or {}
            anchor_pid = resolve_heuristic_anchor_pid(traj) or ""
            if not anchor_pid:
                outcome = {
                    "action": "delete",
                    "branch": "e2e_no_anchor",
                    "quote_after": None,
                }
            else:
                outcome = apply_prism_l_plus(
                    rec,
                    traj,
                    anchor_pid=anchor_pid,
                    arm=arm_id,
                    probe_only_mode="delete_always",
                    use_llm=use_llm,
                    gate_enabled=True,
                )
            q = assess_rewrite_quality({**outcome, **rec}, traj, anchor_pid=anchor_pid)
            logs.append({**rec, **outcome, "quality": q.to_dict()})
            actions.append(str(outcome.get("action") or "keep"))

        import pandas as pd

        df = _rows_to_gating_df(test_rows)
        df["_line_l_action"] = actions
        opt = audit_line_l(df)
        verified_fn = lambda row, t, ap=anchor_pid: False  # noqa: E731
        con = audit_line_l_conservative(df, verified_fn)
        arms[arm_id] = {
            "label": label,
            "tau_score": tau_score,
            "audit_optimistic": opt.to_dict(),
            "audit_conservative": con.to_dict(),
            "n_test": len(test_rows),
            "use_llm": use_llm,
        }

    return {
        "fit_split": fit_split,
        "eval_split": eval_split,
        "tau_bind_surprise_D_f": tau_bs,
        "tau_agr_rho_D_f": tau_agr,
        "tau_agr_probe_D_f": tau_agr_probe,
        "arms": arms,
    }
