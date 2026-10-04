"""IIA evaluation for interchange interventions (§10)."""
from __future__ import annotations

import re
from typing import Any

from ccer.eval.source_following import _contains_value, evaluate_source_following
from ccer.replay.answer_utils import extract_answer, validate_replay_output


def expected_change_for_direction(direction: str, *, claim_value: str, cf_value: str) -> dict[str, Any]:
    if direction == "PM_to_clean":
        return {
            "direction": direction,
            "expected": "anchor_supported_or_not_query_following",
            "query_value": cf_value,
            "claim_value": claim_value,
        }
    return {
        "direction": direction,
        "expected": "query_constraint_value_in_answer",
        "query_value": cf_value,
        "claim_value": claim_value,
    }


def evaluate_iia_row(
    *,
    direction: str,
    control_id: str,
    generated_text: str,
    original_answer: str,
    claim_value: str,
    cf_value: str,
    anchor_value: str | None = None,
    track: str = "CAP",
    condition_id: str | None = None,
) -> dict[str, Any]:
    excerpt, harness_err = validate_replay_output(generated_text)
    expected = expected_change_for_direction(direction, claim_value=claim_value, cf_value=cf_value)
    if harness_err:
        return {
            "iia_success": None,
            "scorable": False,
            "harness_valid": False,
            "reason": harness_err,
            "expected": expected,
            "control_id": control_id,
        }

    new_excerpt = extract_answer(generated_text)
    orig_excerpt = extract_answer(original_answer) if original_answer else ""

    cond = condition_id or ("query_value_swap" if track.upper() == "CAP" else "rival_value_swap")
    expected_key = "query_constraint_value" if cond == "query_value_swap" else "rival_value"
    follow = evaluate_source_following(
        condition_id=cond,
        original_answer=original_answer,
        new_answer=generated_text,
        expected={expected_key: cf_value, "rival_value_old": claim_value},
        edit_manifest=[],
    )

    if direction == "clean_to_PM":
        if track.upper() == "CEM":
            ok = bool(follow.get("source_following"))
            reason = "cem_rival_following" if ok else "no_cem_following"
        else:
            ok = _contains_value(new_excerpt, cf_value) or _contains_value(new_excerpt, claim_value)
            reason = "cap_like_query_following" if ok else "no_cap_following"
    elif track.upper() == "CEM":
        ok = not bool(follow.get("source_following"))
        reason = "cem_not_following_rival" if ok else "still_following_rival"
    else:
        still_query = _contains_value(new_excerpt, cf_value)
        anchor_ok = anchor_value and _contains_value(new_excerpt, anchor_value)
        ok = (not still_query) or bool(anchor_ok)
        reason = "pm_to_clean_correction" if ok else "still_query_following"
    return {
        "iia_success": bool(ok),
        "scorable": True,
        "harness_valid": True,
        "reason": reason,
        "expected": expected,
        "control_id": control_id,
        "source_following_check": follow,
        "answer_excerpt": new_excerpt[:300],
        "orig_excerpt": orig_excerpt[:300],
    }


def side_effect_metrics(
    *,
    baseline_text: str,
    intervened_text: str,
    h_norm: float | None = None,
    h_norm_baseline: float | None = None,
    mahalanobis: float | None = None,
) -> dict[str, Any]:
    base_excerpt = extract_answer(baseline_text)
    new_excerpt = extract_answer(intervened_text)
    kl_proxy = _token_overlap_kl_proxy(base_excerpt, new_excerpt)
    return {
        "activation_l2": h_norm,
        "activation_l2_baseline": h_norm_baseline,
        "mahalanobis": mahalanobis,
        "output_kl_proxy": kl_proxy,
        "anchor_changed": _anchor_pid_changed(base_excerpt, new_excerpt),
        "unrelated_claim_changed": base_excerpt != new_excerpt,
    }


def _token_overlap_kl_proxy(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta and not tb:
        return 0.0
    union = ta | tb
    pa = {t: (1.0 / len(ta) if t in ta else 1e-9) for t in union}
    pb = {t: (1.0 / len(tb) if t in tb else 1e-9) for t in union}
    s = sum(pa[t] for t in union)
    t = sum(pb[t] for t in union)
    pa = {k: v / s for k, v in pa.items()}
    pb = {k: v / t for k, v in pb.items()}
    kl = 0.0
    for w in union:
        kl += pa[w] * (np_log(pa[w] / pb[w]))
    return float(kl)


def np_log(x: float) -> float:
    import math

    return math.log(max(x, 1e-12))


def _anchor_pid_changed(a: str, b: str) -> bool:
    pa = re.search(r"Selected product ID:\s*(\d+)", a, re.I)
    pb = re.search(r"Selected product ID:\s*(\d+)", b, re.I)
    if pa and pb:
        return pa.group(1) != pb.group(1)
    return False


def aggregate_iia(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str, float], list[bool]] = {}
    for row in rows:
        if row.get("iia_success") is None:
            continue
        key = (
            str(row.get("direction")),
            str(row.get("control_id")),
            float(row.get("alpha") or 1.0),
        )
        by_key.setdefault(key, []).append(bool(row["iia_success"]))

    summary: dict[str, Any] = {"by_group": {}, "overall": {}}
    for key, vals in by_key.items():
        direction, control_id, alpha = key
        rate = sum(vals) / len(vals)
        summary["by_group"][f"{direction}|{control_id}|alpha={alpha}"] = {
            "iia_rate": rate,
            "n": len(vals),
        }

    for direction in ("PM_to_clean", "clean_to_PM"):
        dir_rows = [r for r in rows if r.get("direction") == direction and r.get("iia_success") is not None]
        if not dir_rows:
            continue
        summary["overall"][direction] = {
            "iia_rate": sum(bool(r["iia_success"]) for r in dir_rows) / len(dir_rows),
            "n": len(dir_rows),
        }
    return summary
