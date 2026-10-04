"""AGR Track J: causal signal Eq.2 validation."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from ccer.mechanism.agr.fusion import fit_mle_fusion


def symbol_audit(rows: list[dict[str, Any]], *, z2_source: str = "anchor_value") -> dict[str, Any]:
    n = max(len(rows), 1)
    prov_counts = dict(Counter(str(r.get("anchor_provenance") or "missing") for r in rows))
    anchor_resolve = {k: v / n for k, v in prov_counts.items()}

    z2_avail = [r for r in rows if r.get("z2_available")]
    if z2_source == "anchor_value":
        unavailable_frac = 1.0 - (len(z2_avail) / n)
        masses = np.array(
            [
                float(math.exp(float(r.get("z2_anchor_log_mass") or float("-inf"))))
                if r.get("z2_anchor_log_mass") is not None
                and math.isfinite(float(r.get("z2_anchor_log_mass")))
                else 0.0
                for r in z2_avail
            ],
            dtype=float,
        )
        slot_semantics = (
            "T_s = anchor evidence value s_aτ token set "
            "(top-k tokens matching v_anchor via value_tokens_match); Eq.2 aligned."
        )
        code_def = "log_anchor_value_mass = logsumexp(value_token_logits) - logsumexp(all_topk_logits)"
        missing_note = (
            "unrecoverable Obs_τ(a_τ)[slot(c)] → z2 unavailable; fusion uses z_rep only (no floor)"
        )
    else:
        unavailable_frac = 0.0
        masses = np.array(
            [float(math.exp(float(r.get("z2_log_mass") or -12.0))) for r in rows],
            dtype=float,
        )
        slot_semantics = (
            "T_s = slot-TYPE lexicon (legacy BindSurprise); NOT Eq.2 anchor value."
        )
        code_def = "log_slot_mass = logsumexp(slot_logits) - logsumexp(all_topk_logits)"
        missing_note = "legacy lexicon path"

    out_of_range = sum(1 for m in masses if m < 0 or m > 1) if len(masses) else 0
    df = [r for r in rows if r.get("agr_split") == "D_f" and r.get("z2_available", True)]
    w = fit_mle_fusion(df, fit_split="D_f", use_calibrated=False, interaction=False)

    return {
        "z2_source": z2_source,
        "slot_semantics": slot_semantics,
        "code_z2_definition": code_def,
        "paper_eq2_alt": "log((1-pi)/pi) stored as z2_odds",
        "missing_handling": missing_note,
        "anchor_resolve_frac": anchor_resolve,
        "anchor_resolve_counts": prov_counts,
        "z2_available_frac": len(z2_avail) / n if z2_source == "anchor_value" else 1.0,
        "z2_unavailable_frac": unavailable_frac if z2_source == "anchor_value" else 0.0,
        "slot_mass_min": float(np.min(masses)) if len(masses) else 0.0,
        "slot_mass_max": float(np.max(masses)) if len(masses) else 0.0,
        "out_of_range_mass_frac": out_of_range / max(len(masses), 1),
        "fusion_w2_uncalibrated": w["w2"],
        "fusion_w1_uncalibrated": w["w1"],
        "sign_note": (
            "odds semantics: w2<0 expected (high anchor support → low PM odds); "
            "log_mass legacy: w2>0"
        ),
    }


def value_span_variance(rows: list[dict[str, Any]], *, split: str = "test") -> dict[str, Any]:
    sub = [r for r in rows if r.get("agr_split") == split and r.get("z2_available")]
    out_groups: dict[str, Any] = {}
    for gname in ("single_token", "multi_token"):
        grp = [r for r in sub if r.get("value_span_group") == gname]
        if not grp:
            out_groups[gname] = {"n": 0}
            continue
        yg = np.array([int(r["y_pm"]) for r in grp], dtype=int)
        lo = np.log((yg + 1e-9) / (1 - yg + 1e-9))
        resid = np.array([float(r["z2"]) for r in grp]) - lo
        out_groups[gname] = {
            "n": len(grp),
            "residual_var_z2": float(np.var(resid)),
            "mean_abs_resid": float(np.mean(np.abs(resid))),
        }
    return {
        "split": split,
        "groups": out_groups,
        "ratio_multi_over_single_var": (
            out_groups["multi_token"]["residual_var_z2"] / out_groups["single_token"]["residual_var_z2"]
            if out_groups.get("single_token", {}).get("residual_var_z2")
            else None
        ),
    }


def run_track_j(rows: list[dict[str, Any]], *, z2_source: str = "anchor_value") -> dict[str, Any]:
    return {
        "symbol_audit": symbol_audit(rows, z2_source=z2_source),
        "value_span_variance": value_span_variance(rows, split="test"),
    }
