"""Line L+ attribution specificity evaluation (E3)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ccer.mechanism.claim_attribution import parse_claim_attribution_from_text
from ccer.mechanism.line_l_rewrite_quality import build_evidence_corpus, extract_value_from_quote, verify_grounded


def _committed_anchor(row: dict[str, Any], traj: dict[str, Any]) -> str:
    return str(
        row.get("committed_anchor_pid")
        or (traj.get("commitment") or {}).get("action_anchor")
        or (traj.get("text_anchor") or {}).get("selected_pid")
        or row.get("textual_selected_pid")
        or ""
    ).strip()


def attribution_correct_for_rewrite(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    quote_after: str | None = None,
) -> dict[str, Any]:
    """Check if rewritten claim value attributes to committed anchor entity."""
    quote = str(quote_after or row.get("quote_after") or "")
    anchor = _committed_anchor(row, traj)
    slot_norm = str(row.get("slot_norm") or "")
    claim_value = extract_value_from_quote(quote) or str(row.get("claim_value") or "")
    if not quote or not anchor:
        return {
            "scorable": False,
            "attribution_correct": None,
            "parse_error": "missing_quote_or_anchor",
        }
    # Build synthetic final answer matching parser section headers
    synthetic = (
        f"<response>\n"
        f"### About the selected product\n"
        f"Selected Product ID: {anchor}\n"
        f"{quote}\n"
        f"### Compared (not selected): 9999999999\n"
        f"Placeholder: x\n"
        f"</response>"
    )
    parsed = parse_claim_attribution_from_text(
        synthetic,
        slot_norm=slot_norm or None,
        claim_value=claim_value or None,
        response_quote=quote,
    )
    if not parsed.scorable or not parsed.claim_target_entity:
        return {
            "scorable": False,
            "attribution_correct": None,
            "parse_error": parsed.parse_error,
            "claim_target_entity": parsed.claim_target_entity,
        }
    correct = str(parsed.claim_target_entity) == str(anchor)
    return {
        "scorable": True,
        "attribution_correct": correct,
        "claim_target_entity": parsed.claim_target_entity,
        "expected_anchor": anchor,
        "parse_error": None,
    }


def anchor_evidence_attribution_rate(
    rows: list[dict[str, Any]],
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    anchor_pid_fn: Callable[[dict[str, Any], dict[str, Any]], str],
) -> dict[str, Any]:
    """Fraction of rewrites whose value is grounded in the specified anchor PID's evidence."""
    rewrites = [
        r
        for r in rows
        if float(r.get("p_pm", 0)) > float(tau) and str(r.get("action") or "") == "rewrite"
    ]
    scorable = grounded = 0
    for r in rewrites:
        tid = str(r.get("trajectory_id") or "")
        traj = traj_index.get(tid, {})
        pid = anchor_pid_fn(r, traj)
        if not pid:
            continue
        v = r.get("v_anchor") or extract_value_from_quote(str(r.get("quote_after") or ""))
        if not v:
            continue
        scorable += 1
        if verify_grounded(str(v), build_evidence_corpus(traj, pid)):
            grounded += 1
    return {
        "n_rewrite_flagged": len(rewrites),
        "n_anchor_grounded_scorable": scorable,
        "n_anchor_grounded": grounded,
        "anchor_evidence_attribution_rate": (grounded / scorable) if scorable else None,
    }


def evaluate_attribution_batch(
    rows: list[dict[str, Any]],
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
) -> dict[str, Any]:
    flagged_rewrites = [
        r
        for r in rows
        if float(r.get("p_pm", 0)) > float(tau) and str(r.get("action") or "") == "rewrite"
    ]
    scorable = 0
    correct = 0
    for r in flagged_rewrites:
        tid = str(r.get("trajectory_id") or "")
        traj = traj_index.get(tid, {})
        ev = attribution_correct_for_rewrite(r, traj)
        if ev.get("scorable"):
            scorable += 1
            if ev.get("attribution_correct"):
                correct += 1
    return {
        "n_rewrite_flagged": len(flagged_rewrites),
        "n_attribution_scorable": scorable,
        "n_attribution_correct": correct,
        "attribution_corrected_rate": (correct / scorable) if scorable else None,
    }


def permutation_attribution_diff(
    target_rows: list[dict[str, Any]],
    wrong_rows: list[dict[str, Any]],
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Permutation test on attribution_corrected_rate (target - wrong)."""
    t_map = {str(r.get("instance_audit_key") or r.get("claim_id")): r for r in target_rows}
    w_map = {str(r.get("instance_audit_key") or r.get("claim_id")): r for r in wrong_rows}
    keys = sorted(set(t_map) & set(w_map))

    def _rate(rows_map: dict[str, dict]) -> tuple[float, int]:
        sc = co = 0
        for k in keys:
            r = rows_map[k]
            if float(r.get("p_pm", 0)) <= float(tau):
                continue
            if str(r.get("action") or "") != "rewrite":
                continue
            tid = str(r.get("trajectory_id") or "")
            ev = attribution_correct_for_rewrite(r, traj_index.get(tid, {}))
            if ev.get("scorable"):
                sc += 1
                if ev.get("attribution_correct"):
                    co += 1
        return (co / sc if sc else 0.0, sc)

    t_rate, t_n = _rate(t_map)
    w_rate, w_n = _rate(w_map)
    observed = t_rate - w_rate

    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(n_perm):
        t_co = w_co = t_sc = w_sc = 0
        for k in keys:
            rt = t_map[k]
            rw = w_map[k]
            if float(rt.get("p_pm", 0)) <= float(tau):
                continue
            if str(rt.get("action") or "") != "rewrite" and str(rw.get("action") or "") != "rewrite":
                continue
            swap = rng.random() < 0.5
            r_a, r_b = (rw, rt) if swap else (rt, rw)
            for r in (r_a, r_b):
                if str(r.get("action") or "") != "rewrite":
                    continue
                tid = str(r.get("trajectory_id") or "")
                ev = attribution_correct_for_rewrite(r, traj_index.get(tid, {}))
                if not ev.get("scorable"):
                    continue
                if swap:
                    w_sc += 1
                    w_co += int(bool(ev.get("attribution_correct")))
                else:
                    t_sc += 1
                    t_co += int(bool(ev.get("attribution_correct")))
        # Simpler: swap per-key correct flags
    # Redo with per-key flags
    t_flags: list[int] = []
    w_flags: list[int] = []
    for k in keys:
        rt = t_map[k]
        rw = w_map[k]
        if float(rt.get("p_pm", 0)) <= float(tau):
            continue
        for r, bucket in ((rt, t_flags), (rw, w_flags)):
            if str(r.get("action") or "") != "rewrite":
                bucket.append(-1)
                continue
            tid = str(r.get("trajectory_id") or "")
            ev = attribution_correct_for_rewrite(r, traj_index.get(tid, {}))
            if not ev.get("scorable"):
                bucket.append(-1)
            else:
                bucket.append(int(bool(ev.get("attribution_correct"))))
    # align by key index
    paired: list[tuple[int, int]] = []
    for k in keys:
        rt = t_map[k]
        rw = w_map[k]
        if float(rt.get("p_pm", 0)) <= float(tau):
            continue
        tid_t = str(rt.get("trajectory_id") or "")
        tid_w = str(rw.get("trajectory_id") or "")
        et = attribution_correct_for_rewrite(rt, traj_index.get(tid_t, {}))
        ew = attribution_correct_for_rewrite(rw, traj_index.get(tid_w, {}))
        if et.get("scorable") and ew.get("scorable"):
            paired.append((int(bool(et.get("attribution_correct"))), int(bool(ew.get("attribution_correct")))))
    if not paired:
        return {
            "observed_diff": observed,
            "target_rate": t_rate,
            "wrong_rate": w_rate,
            "p_value_two_sided": 1.0,
            "n_paired": 0,
        }
    t_arr = np.array([p[0] for p in paired], dtype=float)
    w_arr = np.array([p[1] for p in paired], dtype=float)
    observed = float(t_arr.mean() - w_arr.mean())
    diffs = []
    for _ in range(n_perm):
        swap = rng.random(len(paired)) < 0.5
        t_p = np.where(swap, w_arr, t_arr)
        w_p = np.where(swap, t_arr, w_arr)
        diffs.append(float(t_p.mean() - w_p.mean()))
    p = float((np.abs(np.array(diffs)) >= abs(observed)).mean())
    return {
        "observed_diff": observed,
        "target_rate": float(t_arr.mean()),
        "wrong_rate": float(w_arr.mean()),
        "p_value_two_sided": p,
        "n_paired": len(paired),
        "n_perm": n_perm,
    }
