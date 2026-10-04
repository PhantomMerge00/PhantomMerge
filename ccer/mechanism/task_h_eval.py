"""Task H evaluation: attribution specificity, McNemar, patch-tier stratification."""
from __future__ import annotations

from typing import Any

from ccer.mechanism.claim_attribution import aggregate_claim_attribution_metrics
from ccer.mechanism.stats import mcnemar_exact_p, wilson_ci


def _attribution_flip(detail: dict[str, Any]) -> bool:
    return bool(
        detail.get("patched")
        and detail.get("claim_attribution_scorable")
        and detail.get("claim_attribution_diff")
    )


def attribution_specificity_block(
    ti_details: list[dict[str, Any]],
    wo_details: list[dict[str, Any]],
) -> dict[str, Any]:
    """McNemar-style table for claim_attribution_diff (correct vs wrong_owner)."""
    by_pm_wrong = {str(d["pm_trajectory_id"]): d for d in wo_details}
    correct_only = wrong_only = both = neither = 0
    pair_cmp: list[dict[str, Any]] = []
    for cd in ti_details:
        pm = str(cd["pm_trajectory_id"])
        wd = by_pm_wrong.get(pm) or {}
        c_flip = _attribution_flip(cd)
        w_flip = _attribution_flip(wd)
        if c_flip and w_flip:
            both += 1
        elif c_flip and not w_flip:
            correct_only += 1
        elif not c_flip and w_flip:
            wrong_only += 1
        else:
            neither += 1
        pair_cmp.append(
            {
                "pm_trajectory_id": pm,
                "target_interchange_attribution_diff": c_flip,
                "wrong_owner_attribution_diff": w_flip,
                "target_patched": bool(cd.get("patched")),
                "wrong_patched": bool(wd.get("patched")),
                "target_patch_tier": cd.get("patch_tier"),
                "wrong_patch_tier": wd.get("patch_tier"),
            }
        )
    return {
        "primary_metric": "claim_attribution_diff",
        "denominator": "patched_and_scorable",
        "correct_donor_only": correct_only,
        "wrong_owner_only": wrong_only,
        "both_flip": both,
        "neither_flip": neither,
        "mcnemar_p": mcnemar_exact_p(wrong_only, correct_only),
        "pair_comparisons": pair_cmp,
    }


def patch_tier_stratum(pair_details: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize attribution metrics by dynamic_answer_anchor patch tier."""
    tier_names = sorted(
        {str(d.get("patch_tier") or "none") for d in pair_details}
        | {"live", "stored", "decode_head", "none"}
    )
    out: dict[str, Any] = {}
    for tier in tier_names:
        subset = [d for d in pair_details if str(d.get("patch_tier") or "none") == tier]
        if not subset:
            continue
        patched_k = sum(1 for d in subset if d.get("patched"))
        metrics = aggregate_claim_attribution_metrics(subset)
        out[tier] = {
            "n_rows": len(subset),
            "n_patched": patched_k,
            **metrics,
        }
    return out


def invalid_run_rows(pair_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pm_trajectory_id": d.get("pm_trajectory_id"),
            "clean_trajectory_id": d.get("clean_trajectory_id"),
            "control_id": d.get("control_id"),
            "patched": d.get("patched"),
            "patch_tier": d.get("patch_tier"),
        }
        for d in pair_details
        if not d.get("patched")
    ]


def summarize_task_h_arm(row: dict[str, Any]) -> dict[str, Any]:
    details = row.get("pair_details") or []
    patched_k = sum(1 for d in details if d.get("patched"))
    n = len(details)
    attr = aggregate_claim_attribution_metrics(details)
    pid_k = sum(1 for d in details if d.get("product_id_diff"))
    return {
        "n_pairs": n,
        "n_patched": patched_k,
        "patched_rate": patched_k / n if n else 0.0,
        "patched_ci95": wilson_ci(patched_k, n),
        "product_id_k": pid_k,
        "product_id_rate": pid_k / n if n else 0.0,
        "product_id_ci95": wilson_ci(pid_k, n),
        **attr,
        "patch_tier_stratum": patch_tier_stratum(details),
        "invalid_runs": invalid_run_rows(details),
    }


def _strip_text_fields(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    details = []
    for d in out.get("pair_details") or []:
        slim = {k: v for k, v in d.items() if k not in ("ni_text", "ti_text")}
        details.append(slim)
    if details:
        out["pair_details"] = details
    return out


def build_task_h_summary(
    *,
    ti_row: dict[str, Any],
    wo_row: dict[str, Any],
    ceiling_row: dict[str, Any] | None = None,
    position: str = "claim_onset",
    layer: int = 49,
) -> dict[str, Any]:
    ti_details = ti_row.get("pair_details") or []
    wo_details = wo_row.get("pair_details") or []
    spec = attribution_specificity_block(ti_details, wo_details)
    ti_sum = summarize_task_h_arm(ti_row)
    wo_sum = summarize_task_h_arm(wo_row)
    ceiling_sum = summarize_task_h_arm(ceiling_row) if ceiling_row else {}

    ti_rate = float(ti_sum.get("claim_attribution_diff_rate") or 0.0)
    wo_rate = float(wo_sum.get("claim_attribution_diff_rate") or 0.0)
    ti_ci = ti_sum.get("claim_attribution_diff_ci95") or (0.0, 0.0)
    wo_ci = wo_sum.get("claim_attribution_diff_ci95") or (0.0, 0.0)
    ci_non_overlap = ti_ci[0] > wo_ci[1] or wo_ci[0] > ti_ci[1]
    corrected_k = int(ti_sum.get("n_claim_attribution_corrected") or 0)

    if ti_rate > wo_rate and (spec["mcnemar_p"] < 0.1 or ci_non_overlap) and corrected_k > 0:
        verdict = "exploratory_positive_attribution_lever"
    elif ti_rate > wo_rate and (spec["mcnemar_p"] < 0.1 or ci_non_overlap):
        verdict = "weak_attribution_lever_no_correction"
    else:
        verdict = "null_claim_attribution_lever"

    return {
        "schema": "ccer_task_h_claim_attribution_v1",
        "position": position,
        "layer": layer,
        "target_interchange": _strip_text_fields({**ti_sum, "pair_details": ti_details}),
        "wrong_owner_donor": _strip_text_fields({**wo_sum, "pair_details": wo_details}),
        "full_vector_ceiling": _strip_text_fields({**ceiling_sum, "pair_details": (ceiling_row or {}).get("pair_details") or []}),
        "attribution_specificity": spec,
        "success_judgment": {
            "verdict": verdict,
            "claim_attribution_diff_correct_rate": ti_rate,
            "claim_attribution_diff_wrong_rate": wo_rate,
            "mcnemar_p": spec["mcnemar_p"],
            "ci_non_overlap": ci_non_overlap,
            "n_claim_attribution_corrected": corrected_k,
        },
    }
