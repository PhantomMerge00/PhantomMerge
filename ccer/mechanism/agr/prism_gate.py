"""G1–G6 gate evaluation for prism_audit integration (Round 3)."""
from __future__ import annotations

from typing import Any


def evaluate_prism_gate(summary: dict[str, Any]) -> dict[str, Any]:
    track_j = summary.get("track_j", {}).get("symbol_audit", {})
    track_f = summary.get("track_f", {})
    upgrade = track_f.get("upgrade_recommendation", {})
    boot = track_f.get("bootstrap_variances", {})
    nonlinear = track_f.get("nonlinear_ceiling", {})
    oof_r = track_f.get("residual_correlation_oof_D_f", {}).get("residual_corr_z1_z2")

    w2_uncal = float(track_j.get("fusion_w2_uncalibrated") or 0.0)
    w_fit = track_f.get("weight_ratio_fit")
    w_theory = track_f.get("weight_ratio_theory")
    w_err = track_f.get("weight_ratio_relative_error")

    g1 = track_j.get("z2_source") == "anchor_value" and "anchor evidence value" in str(
        track_j.get("slot_semantics") or ""
    )
    g2 = w2_uncal < 0.0
    g3 = bool(track_j.get("anchor_resolve_frac")) and track_j.get("z2_unavailable_frac") is not None
    g4 = bool(upgrade.get("pareto_dominates_bind_surprise"))
    weight_status = track_f.get("weight_ratio_status", "")
    dual_n = (track_f.get("dual_signal_ablation") or {}).get("counts", {}).get("D_f_z2_available")
    g5_evaluable = isinstance(dual_n, int) and dual_n >= 100
    g5 = g5_evaluable and w_err is not None and float(w_err) < 0.15
    gbm = float((nonlinear.get("gbm") or {}).get("auroc") or 0.0)
    mlp = float((nonlinear.get("mlp") or {}).get("auroc") or 0.0)
    g6 = gbm < 0.99 and mlp < 0.99

    gates = {
        "G1_semantics": {"pass": g1, "detail": track_j.get("slot_semantics")},
        "G2_sign_w2_negative": {"pass": g2, "w2_uncalibrated": w2_uncal},
        "G3_coverage_disclosed": {
            "pass": g3,
            "anchor_resolve_frac": track_j.get("anchor_resolve_frac"),
            "z2_unavailable_frac": track_j.get("z2_unavailable_frac"),
            "z2_available_frac": track_j.get("z2_available_frac"),
        },
        "G4_fusion_pareto_vs_bind_surprise": {
            "pass": g4,
            "full_fusion_auroc": upgrade.get("full_fusion_auroc"),
            "full_fusion_f1": upgrade.get("full_fusion_f1"),
            "bind_surprise_auroc": upgrade.get("bind_surprise_equal_auroc"),
            "bind_surprise_f1": upgrade.get("bind_surprise_equal_f1"),
        },
        "G5_oof_weight_ratio_error_lt_15pct": {
            "pass": g5 if g5_evaluable else None,
            "evaluable": g5_evaluable,
            "status": weight_status,
            "D_f_dual_signal_n": dual_n,
            "weight_ratio_fit": w_fit if g5_evaluable else None,
            "weight_ratio_theory": w_theory if g5_evaluable else None,
            "relative_error": w_err if g5_evaluable else None,
            "rho_proxy": boot.get("rho_proxy"),
            "note": (
                "Insufficient dual-signal n in D_f; Eq.8 weight-ratio gate not applicable"
                if not g5_evaluable
                else None
            ),
        },
        "G6_no_nonlinear_red_flag": {"pass": g6, "gbm_auroc": gbm, "mlp_auroc": mlp},
    }
    all_pass = all(g["pass"] for g in gates.values() if g.get("pass") is not None)
    return {
        "gates": gates,
        "all_pass": all_pass,
        "integrate_prism_audit": all_pass,
        "residual_corr_oof_D_f": oof_r,
        "recommendation": (
            "Proceed with prism_audit AGR-Fusion integration"
            if all_pass
            else "Do NOT integrate AGR-Fusion into prism_audit; keep legacy BindSurprise"
        ),
    }
