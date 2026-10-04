"""AGR Track F: fusion Eq.5/8/9."""
from __future__ import annotations

from typing import Any

from ccer.mechanism.agr.fusion import (
    bootstrap_signal_variances,
    bootstrap_signal_variances_label_proxy,
    complementarity_2x2,
    enrich_calibrated,
    eval_mode,
    fit_fusion_calibrations,
    fit_mle_fusion,
    fusion_score,
    nonlinear_ceiling,
    residual_correlation,
    residual_correlation_oof,
)
from ccer.mechanism.agr.calibration import fit_threshold_f1
from ccer.mechanism.agr.significance import compare_scores_on_rows


def _attach_mode_scores(
    rows: list[dict[str, Any]],
    *,
    modes: list[str],
    cal: dict[str, Any],
    weights: dict[str, Any],
) -> None:
    for r in rows:
        for mode in modes:
            r[f"_score_{mode}"] = float(
                fusion_score(r, mode=mode, cal=cal, weights=weights)
            )


def _eval_subset_ablation(
    subset: list[dict[str, Any]],
    *,
    cal: dict[str, Any],
    weights: dict[str, Any],
    all_rows: list[dict[str, Any]],
    modes: list[str],
) -> list[dict[str, Any]]:
    if not subset:
        return []
    _attach_mode_scores(subset, modes=modes, cal=cal, weights=weights)
    return [
        eval_mode(subset, mode=m, cal=cal, weights=weights, all_rows=all_rows)
        for m in modes
    ]


def run_track_f(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cal = fit_fusion_calibrations(rows, cal_split="D_c")
    enrich_calibrated(rows, cal)
    weights = fit_mle_fusion(rows, fit_split="D_f", use_calibrated=True, interaction=False)
    weights_int = fit_mle_fusion(rows, fit_split="D_f", use_calibrated=True, interaction=True)

    df = [r for r in rows if r.get("agr_split") == "D_f"]
    test = [r for r in rows if r.get("agr_split") == "test"]
    tau1 = fit_threshold_f1(df, "z1_prime")
    tau2 = fit_threshold_f1(df, "z2_prime")

    ablation_modes = [
        "probe_only",
        "causal_only",
        "naive_sum",
        "calibrated_equal",
        "full_fusion",
        "bind_surprise_equal",
    ]
    ablation = [
        eval_mode(test, mode=m, cal=cal, weights=weights, all_rows=rows) for m in ablation_modes
    ]

    dual_signal_all = [r for r in rows if r.get("z2_available")]
    dual_signal_test = [r for r in test if r.get("z2_available")]
    dual_signal_df = [r for r in df if r.get("z2_available")]
    dual_modes = ["probe_only", "full_fusion", "bind_surprise_equal"]
    dual_signal_ablation = {
        "test": _eval_subset_ablation(
            dual_signal_test,
            cal=cal,
            weights=weights,
            all_rows=rows,
            modes=dual_modes,
        ),
        "all_cohort": _eval_subset_ablation(
            dual_signal_all,
            cal=cal,
            weights=weights,
            all_rows=rows,
            modes=dual_modes,
        ),
        "counts": {
            "test_total": len(test),
            "test_z2_available": len(dual_signal_test),
            "all_cohort_z2_available": len(dual_signal_all),
            "D_f_z2_available": len(dual_signal_df),
        },
    }

    _attach_mode_scores(test, modes=["full_fusion", "bind_surprise_equal"], cal=cal, weights=weights)
    significance_vs_bind = compare_scores_on_rows(
        test,
        "_score_full_fusion",
        "_score_bind_surprise_equal",
        label_a="full_fusion",
        label_b="bind_surprise_equal",
        n_boot=2000,
        seed=42,
    )
    if dual_signal_test:
        _attach_mode_scores(
            dual_signal_test,
            modes=["probe_only", "full_fusion"],
            cal=cal,
            weights=weights,
        )
        significance_dual_probe_vs_fusion = compare_scores_on_rows(
            dual_signal_test,
            "_score_full_fusion",
            "_score_probe_only",
            label_a="full_fusion",
            label_b="probe_only",
            n_boot=2000,
            seed=42,
        )
    else:
        significance_dual_probe_vs_fusion = None

    boot = bootstrap_signal_variances(rows, fit_split="D_f", use_prime=True, n_boot=500)
    boot_label = bootstrap_signal_variances_label_proxy(rows, fit_split="D_f", use_prime=True, n_boot=500)
    oof_corr = residual_correlation_oof(rows, fit_split="D_f", use_prime=True)
    w_fit = abs(weights["w1"] / weights["w2"]) if weights["w2"] != 0 else None
    w_theory = boot.get("w1_over_w2_theory")
    dual_df_n = len(dual_signal_df)
    weight_ratio_evaluable = dual_df_n >= 100

    return {
        "calibration": cal,
        "fusion_weights": weights,
        "interaction_model": weights_int,
        "residual_correlation_oof_D_f": oof_corr,
        "residual_correlation_D_f_label_proxy": residual_correlation(df, use_prime=True),
        "residual_correlation_test_label_proxy": residual_correlation(test, use_prime=True),
        "disentangle_2x2_test": complementarity_2x2(test, tau1=tau1, tau2=tau2, use_prime=True),
        "bootstrap_variances": boot,
        "bootstrap_variances_label_proxy": boot_label,
        "weight_ratio_fit": w_fit,
        "weight_ratio_theory": w_theory,
        "weight_ratio_relative_error": (
            abs(w_fit - w_theory) / w_theory
            if weight_ratio_evaluable and w_fit and w_theory and w_theory > 0
            else None
        ),
        "weight_ratio_status": (
            "evaluable"
            if weight_ratio_evaluable
            else f"insufficient_n_D_f_dual_signal={dual_df_n}; not a theory failure"
        ),
        "ablation_test": ablation,
        "dual_signal_ablation": dual_signal_ablation,
        "significance_full_fusion_vs_bind_surprise_test": significance_vs_bind,
        "significance_dual_signal_probe_vs_fusion_test": significance_dual_probe_vs_fusion,
        "nonlinear_ceiling": nonlinear_ceiling(rows, fit_split="D_f", eval_split="test"),
        "upgrade_recommendation": _upgrade_decision(ablation),
    }


def _upgrade_decision(ablation: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode = {a["mode"]: a for a in ablation}
    full = by_mode.get("full_fusion", {})
    bind = by_mode.get("bind_surprise_equal", {})
    pareto = (full.get("f1", 0) >= bind.get("f1", 0)) and (full.get("auroc", 0) >= bind.get("auroc", 0))
    strict = (full.get("f1", 0) > bind.get("f1", 0)) or (
        full.get("f1", 0) == bind.get("f1", 0) and full.get("auroc", 0) > bind.get("auroc", 0)
    )
    return {
        "full_fusion_f1": full.get("f1"),
        "bind_surprise_equal_f1": bind.get("f1"),
        "full_fusion_auroc": full.get("auroc"),
        "bind_surprise_equal_auroc": bind.get("auroc"),
        "pareto_dominates_bind_surprise": pareto,
        "recommend_upgrade": strict,
    }
