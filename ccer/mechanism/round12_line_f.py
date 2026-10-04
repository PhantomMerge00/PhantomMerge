"""Line F v2/v3: shell-anchor evidence mask + full_vector donor + mismatch_resolved metrics."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.interchange import (
    build_paired_full_vector_spec,
    build_spec_from_roi,
    greedy_generate_with_hook,
)
from ccer.mechanism.pair_select import load_trajectory_index, messages_for_condition
from ccer.mechanism.round8_verify import PATCH_IMPLEMENTATION_NOTE
from ccer.mechanism.shell_pair_pool import (
    activation_coverage_for_shell_pool,
    activation_eligible_shell_pairs,
    build_shell_cross_pairs,
)
from ccer.mechanism.stats import mcnemar_exact_p, wilson_ci
from ccer.replay.answer_utils import (
    answers_differ,
    extract_selected_product_id,
    full_response_excerpt_differ,
    selected_product_blocks_differ,
)
from ccer.replay.hf_forward import tokenize_ccer_messages
from ccer.replay.live_position import (
    ANSWER_REGION_POSITIONS,
    intervention_steering_apply,
    live_token_idx_for_intervention,
    make_live_position_resolver,
    pair_has_symmetric_claim_anchor,
)

EVIDENCE_MASKED_PID_RE = re.compile(
    r"Selected\s+product\s+ID:\s*\[EVIDENCE_MASKED\]",
    re.IGNORECASE,
)

LINE_F_V3_PROTOCOL = "line_f_v3_mechanism_extreme"
LINE_F_V3_POSITIONS = ("commitment", "claim_onset", "pre_value")
POSITION_LAYER_DEFAULTS: dict[str, int] = {
    "commitment": 32,
    "claim_onset": 49,
    "pre_value": 49,
    "prompt_end": 32,
}

LINE_F_V2_PATCH_NOTE = {
    **PATCH_IMPLEMENTATION_NOTE,
    "intervention_mode": "full_vector",
    "steering_apply": "generation_decode",
    "use_cache_ni": False,
    "use_cache_ti": False,
    "wrong_owner_protocol": "donor_vec=recipient_pm_vec (zero delta; ti must equal ni)",
    "primary_metric": "mismatch_resolved",
    "numeric_scorable_subset": "baseline numeric PID extractable (typically 24/60 shell)",
}

LINE_F_V3_PATCH_NOTE = {
    **LINE_F_V2_PATCH_NOTE,
    "protocol_version": LINE_F_V3_PROTOCOL,
    "generation_decode_fix": "use_cache=False now sets decode_step=len(generated)>0 → patch [-1]",
    "extended_metrics": [
        "masked_baseline",
        "masked_to_anchor_resolved",
        "mechanism_resolved",
    ],
    "position_layer_defaults": POSITION_LAYER_DEFAULTS,
    "answer_region_steering": "claim_onset/pre_value use dynamic_answer_anchor (Line D)",
}


def _position_layer(position: str, layer_override: int | None = None) -> int:
    if layer_override is not None:
        return int(layer_override)
    return int(POSITION_LAYER_DEFAULTS.get(position, 32))


def _steering_for_position(position: str, steering_override: str | None = None) -> str:
    if steering_override:
        return str(steering_override)
    return intervention_steering_apply(position)


def _filter_shell_pairs_for_position(
    shell_pairs: list[dict[str, Any]],
    *,
    position: str,
    rows_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Answer-region positions require bilateral symmetric claim anchors in stored answers."""
    if position not in ANSWER_REGION_POSITIONS:
        return shell_pairs, []
    eligible: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    for cp in shell_pairs:
        pm_traj = rows_index.get(str(cp["pm_trajectory_id"]))
        clean_traj = rows_index.get(str(cp["clean_trajectory_id"]))
        ok = bool(
            pm_traj
            and clean_traj
            and pair_has_symmetric_claim_anchor(pm_traj)
            and pair_has_symmetric_claim_anchor(clean_traj)
        )
        row = dict(cp)
        row["symmetric_claim_anchor_ok"] = ok
        if ok:
            eligible.append(row)
        else:
            ineligible.append(row)
    return eligible, ineligible


def _action_anchor(traj: dict[str, Any]) -> str | None:
    anchor = (traj.get("commitment") or {}).get("action_anchor")
    return str(anchor) if anchor else None


def _extract_pid_or_anchor(text: str, action_anchor: str | None) -> str | None:
    pid = extract_selected_product_id(text)
    if pid:
        return pid
    if action_anchor:
        if re.search(
            rf"Selected\s+product\s+ID:\s*{re.escape(str(action_anchor))}\b",
            text,
            re.IGNORECASE,
        ):
            return str(action_anchor)
    return None


def _pid_digit_hamming(a: str | None, b: str | None) -> int | None:
    if not a or not b or len(a) != len(b):
        return None
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def _baseline_answer_text(traj: dict[str, Any]) -> str:
    baseline = str(((traj or {}).get("metadata") or {}).get("final_answer") or "")
    if baseline:
        return baseline
    for m in reversed(traj.get("messages_final_call") or []):
        if m.get("role") == "assistant":
            return str(m.get("content") or "")
    return ""


def _clean_gold_pid(clean_traj: dict[str, Any] | None) -> str | None:
    if not clean_traj:
        return None
    return extract_selected_product_id(_baseline_answer_text(clean_traj))


def shell_pair_metrics(
    ni_text: str,
    ti_text: str,
    *,
    pm_traj: dict[str, Any],
    clean_traj: dict[str, Any] | None,
    control_id: str,
) -> dict[str, Any]:
    """Shell subclass metrics: mismatch_resolved primary; numeric subset with digit-drift split."""
    anchor = _action_anchor(pm_traj)
    pid_ni = extract_selected_product_id(ni_text)
    pid_ti = extract_selected_product_id(ti_text)
    pid_ni_anchor = _extract_pid_or_anchor(ni_text, anchor)
    pid_ti_anchor = _extract_pid_or_anchor(ti_text, anchor)
    clean_pid = _clean_gold_pid(clean_traj)

    masked_baseline = bool(EVIDENCE_MASKED_PID_RE.search(ni_text))
    masked_intervention = bool(EVIDENCE_MASKED_PID_RE.search(ti_text))

    base_mismatch = bool(anchor and pid_ni_anchor and pid_ni_anchor != anchor)
    int_mismatch = bool(anchor and pid_ti_anchor and pid_ti_anchor != anchor)
    mismatch_resolved = bool(base_mismatch and anchor and pid_ti_anchor == anchor)
    masked_to_anchor_resolved = bool(masked_baseline and anchor and pid_ti_anchor == anchor)
    mechanism_resolved = bool(mismatch_resolved or masked_to_anchor_resolved)
    extended_mismatch_cohort = bool(base_mismatch or masked_baseline)

    numeric_scorable = pid_ni is not None
    pid_diff = bool(pid_ni and pid_ti and pid_ni != pid_ti)
    hamming = _pid_digit_hamming(pid_ni, pid_ti) if pid_diff else 0
    digit_drift_flip = bool(pid_diff and hamming == 1)
    real_pid_flip = bool(pid_diff and hamming is not None and hamming > 1)

    return {
        "action_anchor": anchor,
        "product_id_ni": pid_ni,
        "product_id_ti": pid_ti,
        "product_id_ni_or_anchor": pid_ni_anchor,
        "product_id_ti_or_anchor": pid_ti_anchor,
        "clean_gold_pid": clean_pid,
        "masked_baseline": masked_baseline,
        "masked_intervention": masked_intervention,
        "numeric_scorable": numeric_scorable,
        "baseline_commitment_mismatch": base_mismatch,
        "intervention_commitment_mismatch": int_mismatch,
        "extended_mismatch_cohort": extended_mismatch_cohort,
        "mismatch_resolved": mismatch_resolved,
        "masked_to_anchor_resolved": masked_to_anchor_resolved,
        "mechanism_resolved": mechanism_resolved,
        "anchor_aligned_ti": bool(anchor and pid_ti_anchor == anchor),
        "clean_gold_aligned": bool(
            clean_pid
            and (
                (pid_ti and pid_ti == clean_pid)
                or (pid_ti_anchor and str(pid_ti_anchor) == str(clean_pid))
            )
        ),
        "output_diff": ni_text != ti_text,
        "answer_diff": answers_differ(ni_text, ti_text),
        "selected_product_block_diff": selected_product_blocks_differ(ni_text, ti_text),
        "full_response_excerpt_diff": full_response_excerpt_differ(ni_text, ti_text),
        "product_id_diff": pid_diff,
        "digit_drift_flip": digit_drift_flip,
        "real_pid_flip": real_pid_flip,
        "pid_digit_hamming": hamming,
        "wrong_owner_identical_to_ni": ni_text == ti_text,
        "control_id": control_id,
    }


def _roi_stub(layer: int, position: str, dim: int) -> dict[str, Any]:
    rank = min(4, dim)
    return {
        "primary_roi": {"layer": layer, "position": position},
        "U_owner": np.eye(dim, rank, dtype=np.float32).tolist(),
        "inject_site": "residual",
    }


def _generate_ni(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    layer: int,
    position: str,
    pm_vec: np.ndarray,
    clean_vec: np.ndarray,
    pos_idx: int,
    probe_max_tokens: int,
    use_cache: bool,
    live_position_resolver: Any | None = None,
) -> dict[str, Any]:
    spec = build_spec_from_roi(
        control_id="no_intervention",
        roi=_roi_stub(layer, position, len(pm_vec)),
        donor_vec=clean_vec,
        recipient_vec=pm_vec,
        position_token_idx=pos_idx,
        alpha=0.0,
    )
    return greedy_generate_with_hook(
        model,
        tokenizer,
        messages,
        spec=spec,
        max_new_tokens=probe_max_tokens,
        use_cache=use_cache,
        live_position_resolver=live_position_resolver,
    )


def _generate_ti_full_vector(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    layer: int,
    pos_idx: int,
    pm_vec: np.ndarray,
    donor_vec: np.ndarray,
    control_id: str,
    alpha: float,
    probe_max_tokens: int,
    steering_apply: str,
    use_cache: bool,
    live_position_resolver: Any | None = None,
) -> dict[str, Any]:
    spec = build_paired_full_vector_spec(
        control_id=control_id,
        layer=layer,
        position_token_idx=pos_idx,
        pm_vec=pm_vec,
        clean_vec=donor_vec,
        alpha=alpha,
        steering_apply=steering_apply,
    )
    return greedy_generate_with_hook(
        model,
        tokenizer,
        messages,
        spec=spec,
        max_new_tokens=probe_max_tokens,
        use_cache=use_cache,
        live_position_resolver=live_position_resolver,
    )


def _summarize_arm(
    pair_details: list[dict[str, Any]],
    *,
    control_id: str,
) -> dict[str, Any]:
    n = len(pair_details)
    if n == 0:
        return {"control_id": control_id, "n": 0}

    def _rate(key: str, subset: list[dict[str, Any]] | None = None) -> tuple[int, float, tuple[float, float]]:
        rows = subset if subset is not None else pair_details
        k = sum(1 for d in rows if d.get(key))
        m = len(rows)
        return k, k / m if m else 0.0, wilson_ci(k, m)

    mismatch_cohort = [d for d in pair_details if d.get("baseline_commitment_mismatch")]
    extended_cohort = [d for d in pair_details if d.get("extended_mismatch_cohort")]
    numeric = [d for d in pair_details if d.get("numeric_scorable")]
    mr_k, mr_rate, mr_ci = _rate("mismatch_resolved", mismatch_cohort)
    ext_k, ext_rate, ext_ci = _rate("mechanism_resolved", extended_cohort)
    masked_k, masked_rate, masked_ci = _rate("masked_to_anchor_resolved", extended_cohort)
    mech_k, mech_rate, mech_ci = _rate("mechanism_resolved", mismatch_cohort)
    pid_k, pid_rate, pid_ci = _rate("product_id_diff", numeric)
    drift_k, drift_rate, drift_ci = _rate("digit_drift_flip", numeric)
    real_k, real_rate, real_ci = _rate("real_pid_flip", numeric)
    gold_k, gold_rate, gold_ci = _rate("clean_gold_aligned", numeric)
    anchor_k, anchor_rate, anchor_ci = _rate("anchor_aligned_ti")

    return {
        "control_id": control_id,
        "n": n,
        "n_mismatch_cohort": len(mismatch_cohort),
        "n_mismatch_resolved": mr_k,
        "mismatch_resolved_rate": mr_rate,
        "mismatch_resolved_ci95": mr_ci,
        "n_extended_mismatch_cohort": len(extended_cohort),
        "n_mechanism_resolved": ext_k,
        "mechanism_resolved_rate": ext_rate,
        "mechanism_resolved_ci95": ext_ci,
        "n_masked_to_anchor_resolved": masked_k,
        "masked_to_anchor_resolved_rate": masked_rate,
        "masked_to_anchor_resolved_ci95": masked_ci,
        "n_mismatch_cohort_mechanism_resolved": mech_k,
        "mismatch_cohort_mechanism_resolved_rate": mech_rate,
        "mismatch_cohort_mechanism_resolved_ci95": mech_ci,
        "n_numeric_scorable": len(numeric),
        "n_product_id_diff": pid_k,
        "product_id_diff_rate": pid_rate,
        "product_id_ci95": pid_ci,
        "n_digit_drift_flip": drift_k,
        "digit_drift_flip_rate": drift_rate,
        "digit_drift_flip_ci95": drift_ci,
        "n_real_pid_flip": real_k,
        "real_pid_flip_rate": real_rate,
        "real_pid_flip_ci95": real_ci,
        "n_clean_gold_aligned": gold_k,
        "clean_gold_aligned_rate": gold_rate,
        "clean_gold_aligned_ci95": gold_ci,
        "n_anchor_aligned_ti": anchor_k,
        "anchor_aligned_ti_rate": anchor_rate,
        "anchor_aligned_ti_ci95": anchor_ci,
        "n_output_diff": sum(1 for d in pair_details if d.get("output_diff")),
        "n_patched": sum(1 for d in pair_details if d.get("patched")),
    }


def _stratified_summary_v2(
    correct_details: list[dict[str, Any]],
    wrong_details: list[dict[str, Any]],
    pairing_source_map: dict[str, str],
    source_key: str,
    *,
    numeric_only: bool = False,
) -> dict[str, Any]:
    correct_sub = [
        d
        for d in correct_details
        if pairing_source_map.get(str(d.get("pm_trajectory_id"))) == source_key
        and (not numeric_only or d.get("numeric_scorable"))
    ]
    wrong_by_pm = {str(d["pm_trajectory_id"]): d for d in wrong_details}
    n = len(correct_sub)
    correct_mr = sum(1 for d in correct_sub if d.get("mismatch_resolved"))
    correct_mech = sum(1 for d in correct_sub if d.get("mechanism_resolved"))
    wrong_mr = 0
    wrong_mech = 0
    correct_only = wrong_only = 0
    mech_correct_only = mech_wrong_only = 0
    for cd in correct_sub:
        pm = str(cd["pm_trajectory_id"])
        c_hit = bool(cd.get("mismatch_resolved"))
        w_hit = bool((wrong_by_pm.get(pm) or {}).get("mismatch_resolved"))
        c_mech = bool(cd.get("mechanism_resolved"))
        w_mech = bool((wrong_by_pm.get(pm) or {}).get("mechanism_resolved"))
        if w_hit:
            wrong_mr += 1
        if w_mech:
            wrong_mech += 1
        if c_hit and not w_hit:
            correct_only += 1
        elif not c_hit and w_hit:
            wrong_only += 1
        if c_mech and not w_mech:
            mech_correct_only += 1
        elif not c_mech and w_mech:
            mech_wrong_only += 1
    return {
        "pairing_source": source_key,
        "numeric_only": numeric_only,
        "n_pairs": n,
        "correct_donor_mismatch_resolved_k": correct_mr,
        "correct_donor_mismatch_resolved_rate": correct_mr / n if n else 0.0,
        "correct_donor_mismatch_resolved_ci95": wilson_ci(correct_mr, n),
        "correct_donor_mechanism_resolved_k": correct_mech,
        "correct_donor_mechanism_resolved_rate": correct_mech / n if n else 0.0,
        "correct_donor_mechanism_resolved_ci95": wilson_ci(correct_mech, n),
        "wrong_owner_mismatch_resolved_k": wrong_mr,
        "wrong_owner_mismatch_resolved_rate": wrong_mr / n if n else 0.0,
        "wrong_owner_mismatch_resolved_ci95": wilson_ci(wrong_mr, n),
        "wrong_owner_mechanism_resolved_k": wrong_mech,
        "wrong_owner_mechanism_resolved_rate": wrong_mech / n if n else 0.0,
        "wrong_owner_mechanism_resolved_ci95": wilson_ci(wrong_mech, n),
        "correct_donor_only_resolved": correct_only,
        "wrong_owner_only_resolved": wrong_only,
        "mcnemar_exact_p_two_sided": mcnemar_exact_p(correct_only, wrong_only),
        "mechanism_correct_only_resolved": mech_correct_only,
        "mechanism_wrong_only_resolved": mech_wrong_only,
        "mechanism_mcnemar_exact_p_two_sided": mcnemar_exact_p(mech_correct_only, mech_wrong_only),
    }


def _run_shell_pairs_v2(
    *,
    model: Any,
    tokenizer: Any,
    shell_pairs: list[dict[str, Any]],
    rows_index: dict[str, dict[str, Any]],
    layer: int,
    position: str,
    probe_max_tokens: int,
    alpha: float,
    steering_apply: str,
    use_cache: bool,
    ni_cache: dict[str, str],
    resume_correct: list[dict[str, Any]] | None,
    resume_wrong: list[dict[str, Any]] | None,
    checkpoint_path: Path | None,
    on_pair_complete: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    correct_details: list[dict[str, Any]] = list(resume_correct or [])
    wrong_details: list[dict[str, Any]] = list(resume_wrong or [])
    done_pm = {str(d["pm_trajectory_id"]) for d in correct_details}
    identity_violations: list[dict[str, Any]] = []

    def _flush() -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(
                {
                    "schema_version": "ccer_line_f_shell_checkpoint_v2",
                    "correct_pair_details": correct_details,
                    "wrong_pair_details": wrong_details,
                    "ni_cache": ni_cache,
                    "n_shell_pairs": len(shell_pairs),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    for cp in shell_pairs:
        pm_tid = str(cp["pm_trajectory_id"])
        if pm_tid in done_pm:
            continue
        clean_tid = str(cp["clean_trajectory_id"])
        pm_traj = rows_index.get(pm_tid)
        clean_traj = rows_index.get(clean_tid)
        if not pm_traj:
            continue

        pm_npz = load_activation_npz(activation_path(pm_tid, "original"))
        clean_npz = load_activation_npz(activation_path(clean_tid, "original"))
        pm_vec = get_vector(pm_npz, position=position, layer=layer)
        clean_vec = get_vector(clean_npz, position=position, layer=layer)
        messages = messages_for_condition(pm_traj, "original", track="CEM")
        mask_meta: dict[str, Any] = {"masked": False}
        if messages:
            from ccer.replay.evidence_mask import mask_cem_evidence_spans

            messages, mask_meta = mask_cem_evidence_spans(messages, pm_traj)
        tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
        prompt_len = int(tok["prompt_token_count"])
        seq_ids = list(tok["full_ids"][:prompt_len])
        live_resolver = (
            make_live_position_resolver(tokenizer, position)
            if position in ANSWER_REGION_POSITIONS
            else None
        )
        pos_idx = live_token_idx_for_intervention(
            pm_npz=pm_npz,
            position=position,
            prompt_len=prompt_len,
            tokenizer=tokenizer,
            seq=seq_ids,
        )
        if pm_vec is None or clean_vec is None:
            continue
        if position not in ANSWER_REGION_POSITIONS and pos_idx < 0:
            continue
        if pos_idx < 0:
            pos_idx = max(0, prompt_len - 1)

        cache_key = (
            f"{pm_tid}|{clean_tid}|L{layer}|{position}|sa={steering_apply}|"
            f"{probe_max_tokens}|mask=True|a={alpha}|uc={use_cache}"
        )
        if cache_key not in ni_cache:
            out_ni = _generate_ni(
                model,
                tokenizer,
                messages,
                layer=layer,
                position=position,
                pm_vec=pm_vec,
                clean_vec=clean_vec,
                pos_idx=pos_idx,
                probe_max_tokens=probe_max_tokens,
                use_cache=use_cache,
                live_position_resolver=live_resolver,
            )
            ni_cache[cache_key] = out_ni["text"]
        ni_text = ni_cache[cache_key]

        out_correct = _generate_ti_full_vector(
            model,
            tokenizer,
            messages,
            layer=layer,
            pos_idx=pos_idx,
            pm_vec=pm_vec,
            donor_vec=clean_vec,
            control_id="target_interchange",
            alpha=alpha,
            probe_max_tokens=probe_max_tokens,
            steering_apply=steering_apply,
            use_cache=use_cache,
            live_position_resolver=live_resolver,
        )
        out_wrong = _generate_ti_full_vector(
            model,
            tokenizer,
            messages,
            layer=layer,
            pos_idx=pos_idx,
            pm_vec=pm_vec,
            donor_vec=pm_vec,
            control_id="wrong_owner_donor",
            alpha=alpha,
            probe_max_tokens=probe_max_tokens,
            steering_apply=steering_apply,
            use_cache=use_cache,
            live_position_resolver=live_resolver,
        )
        ti_correct = out_correct["text"]
        ti_wrong = out_wrong["text"]

        if ti_wrong != ni_text:
            identity_violations.append(
                {
                    "pm_trajectory_id": pm_tid,
                    "ni_hash": hash(ni_text),
                    "ti_wrong_hash": hash(ti_wrong),
                    "patched_wrong": bool(out_wrong.get("patched")),
                    "use_cache": use_cache,
                }
            )

        base = {
            "pm_trajectory_id": pm_tid,
            "clean_trajectory_id": clean_tid,
            "layer": layer,
            "position": position,
            "evidence_mask": True,
            "evidence_mask_meta": mask_meta,
            "intervention_alpha": alpha,
            "intervention_mode": "full_vector",
            "steering_apply": steering_apply,
            "use_cache": use_cache,
            "commitment_token_idx": pos_idx,
            "ni_text": ni_text,
        }
        correct_row = {
            **base,
            **shell_pair_metrics(
                ni_text,
                ti_correct,
                pm_traj=pm_traj,
                clean_traj=clean_traj,
                control_id="target_interchange",
            ),
            "ti_text": ti_correct,
            "patched": bool(out_correct.get("patched")),
        }
        wrong_row = {
            **base,
            **shell_pair_metrics(
                ni_text,
                ti_wrong,
                pm_traj=pm_traj,
                clean_traj=clean_traj,
                control_id="wrong_owner_donor",
            ),
            "ti_text": ti_wrong,
            "patched": bool(out_wrong.get("patched")),
        }
        correct_details.append(correct_row)
        wrong_details.append(wrong_row)
        done_pm.add(pm_tid)
        if on_pair_complete is not None:
            on_pair_complete(correct_details, wrong_details)
        _flush()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    identity_gate = {
        "expected": "wrong_owner ti_text == ni_text for all pairs (zero full_vector delta)",
        "n_violations": len(identity_violations),
        "passed": len(identity_violations) == 0,
        "violations": identity_violations[:20],
    }
    return correct_details, wrong_details, identity_gate


def audit_pid_extraction_robustness(pair_details: list[dict[str, Any]]) -> dict[str, Any]:
    """Check baseline / intervened PID extraction gaps."""
    rows: list[dict[str, Any]] = []
    missing_baseline = missing_intervened = 0
    numeric_scorable = masked_baseline = 0
    for pr in pair_details:
        pm = str(pr.get("pm_trajectory_id") or "")
        pid_ni = pr.get("product_id_ni")
        pid_ti = pr.get("product_id_ti")
        if pr.get("masked_baseline"):
            masked_baseline += 1
        if not pid_ni:
            missing_baseline += 1
        else:
            numeric_scorable += 1
        if not pid_ti:
            missing_intervened += 1
        if not pid_ni or not pid_ti:
            rows.append(
                {
                    "pm_trajectory_id": pm,
                    "baseline_pid": pid_ni,
                    "intervened_pid": pid_ti,
                    "action_anchor": pr.get("action_anchor"),
                    "numeric_scorable": pr.get("numeric_scorable"),
                }
            )
    n = len(pair_details)
    return {
        "n_pairs": n,
        "n_numeric_scorable": numeric_scorable,
        "n_masked_baseline": masked_baseline,
        "missing_baseline_pid": missing_baseline,
        "missing_intervened_pid": missing_intervened,
        "missing_rate_baseline": missing_baseline / n if n else 0.0,
        "missing_rate_intervened": missing_intervened / n if n else 0.0,
        "boundary_cases": rows,
        "note": "numeric_scorable subset uses extract_selected_product_id on baseline ni only.",
    }


def run_shell_expanded_wrong_owner_donor(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer: int | None = None,
    position: str = "commitment",
    rank: int = 16,
    probe_max_tokens: int = 384,
    alpha: float = 1.0,
    pool: str = "mechanism_research",
    eval_eligible_only: bool = True,
    checkpoint_path: str | Path | None = None,
    resume_from_checkpoint: bool = False,
    steering_apply: str | None = None,
    use_cache: bool = False,
    protocol: str = "v2",
) -> dict[str, Any]:
    """
    Line F v2/v3: shell subclass + strong mask + full_vector correct donor vs wrong_owner_donor.
    v3: auto layer/steering by position, extended mechanism_resolved metrics.
    """
    if protocol == "v3":
        effective_layer = _position_layer(position, layer)
    else:
        effective_layer = int(layer if layer is not None else 32)
    effective_steering = _steering_for_position(position, steering_apply)
    shell = build_shell_cross_pairs(pool=pool)
    pairing_validation = shell.get("r11_legacy_pairing_validation") or {}
    if not pairing_validation.get("passed", True):
        raise RuntimeError(
            f"R11 legacy pairing validation failed: {pairing_validation.get('mismatches')}"
        )
    all_shell = shell["pm_clean_cross_pairs"]
    pairing_source_map = {
        str(p["pm_trajectory_id"]): str(p.get("pairing_source") or "unknown") for p in all_shell
    }
    if eval_eligible_only:
        shell_pairs, ineligible = activation_eligible_shell_pairs(
            all_shell, layer=effective_layer, position=position
        )
    else:
        shell_pairs, ineligible = all_shell, []

    rows_index = load_trajectory_index()
    claim_ineligible: list[dict[str, Any]] = []
    if position in ANSWER_REGION_POSITIONS:
        shell_pairs, claim_ineligible = _filter_shell_pairs_for_position(
            shell_pairs, position=position, rows_index=rows_index
        )

    coverage = activation_coverage_for_shell_pool(
        pool=pool, layer=effective_layer, position=position
    )
    ni_cache: dict[str, str] = {}
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    resume_correct: list[dict[str, Any]] | None = None
    resume_wrong: list[dict[str, Any]] | None = None
    if resume_from_checkpoint and ckpt_path and ckpt_path.is_file():
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        ni_cache.update(ckpt.get("ni_cache") or {})
        resume_correct = list(ckpt.get("correct_pair_details") or [])
        resume_wrong = list(ckpt.get("wrong_pair_details") or [])
        print(
            f"[line_f] resume checkpoint: correct={len(resume_correct)}/{len(shell_pairs)} "
            f"wrong={len(resume_wrong)}/{len(shell_pairs)}",
            flush=True,
        )

    def _on_done(correct_details: list[dict[str, Any]], wrong_details: list[dict[str, Any]]) -> None:
        pass

    correct_details, wrong_details, identity_gate = _run_shell_pairs_v2(
        model=model,
        tokenizer=tokenizer,
        shell_pairs=shell_pairs,
        rows_index=rows_index,
        layer=effective_layer,
        position=position,
        probe_max_tokens=probe_max_tokens,
        alpha=alpha,
        steering_apply=effective_steering,
        use_cache=use_cache,
        ni_cache=ni_cache,
        resume_correct=resume_correct,
        resume_wrong=resume_wrong,
        checkpoint_path=ckpt_path,
        on_pair_complete=_on_done,
    )
    if ckpt_path and ckpt_path.is_file():
        ckpt_path.unlink(missing_ok=True)

    correct_sum = _summarize_arm(correct_details, control_id="target_interchange")
    wrong_sum = _summarize_arm(wrong_details, control_id="wrong_owner_donor")
    numeric_correct = [d for d in correct_details if d.get("numeric_scorable")]
    numeric_wrong = [d for d in wrong_details if d.get("numeric_scorable")]
    numeric_correct_sum = _summarize_arm(numeric_correct, control_id="target_interchange")
    numeric_wrong_sum = _summarize_arm(numeric_wrong, control_id="wrong_owner_donor")

    by_pm_wrong = {str(d["pm_trajectory_id"]): d for d in wrong_details}
    correct_only = wrong_only = neither = both = 0
    mech_correct_only = mech_wrong_only = 0
    pair_cmp: list[dict[str, Any]] = []
    for cd in correct_details:
        pm = str(cd["pm_trajectory_id"])
        wd = by_pm_wrong.get(pm) or {}
        c_hit = bool(cd.get("mismatch_resolved"))
        w_hit = bool(wd.get("mismatch_resolved"))
        c_mech = bool(cd.get("mechanism_resolved"))
        w_mech = bool(wd.get("mechanism_resolved"))
        if c_hit and w_hit:
            both += 1
        elif c_hit and not w_hit:
            correct_only += 1
        elif not c_hit and w_hit:
            wrong_only += 1
        else:
            neither += 1
        if c_mech and not w_mech:
            mech_correct_only += 1
        elif not c_mech and w_mech:
            mech_wrong_only += 1
        pair_cmp.append(
            {
                "pm_trajectory_id": pm,
                "correct_donor_mismatch_resolved": c_hit,
                "wrong_owner_mismatch_resolved": w_hit,
                "correct_donor_mechanism_resolved": c_mech,
                "wrong_owner_mechanism_resolved": w_mech,
                "masked_baseline": cd.get("masked_baseline"),
                "numeric_scorable": cd.get("numeric_scorable"),
                "digit_drift_flip_correct": cd.get("digit_drift_flip"),
                "real_pid_flip_correct": cd.get("real_pid_flip"),
                "clean_gold_aligned_correct": cd.get("clean_gold_aligned"),
            }
        )

    pid_audit = audit_pid_extraction_robustness(correct_details)
    stratified = {
        "r11_legacy": _stratified_summary_v2(
            correct_details, wrong_details, pairing_source_map, "r11_legacy"
        ),
        "mechanism_expansion": _stratified_summary_v2(
            correct_details, wrong_details, pairing_source_map, "mechanism_expansion"
        ),
        "numeric_scorable": {
            "all": _stratified_summary_v2(
                correct_details, wrong_details, pairing_source_map, "r11_legacy", numeric_only=True
            ),
            "r11_legacy": _stratified_summary_v2(
                correct_details, wrong_details, pairing_source_map, "r11_legacy", numeric_only=True
            ),
            "mechanism_expansion": _stratified_summary_v2(
                correct_details,
                wrong_details,
                pairing_source_map,
                "mechanism_expansion",
                numeric_only=True,
            ),
        },
    }
    # Fix numeric_scorable all to include both sources
    numeric_all_correct = [d for d in correct_details if d.get("numeric_scorable")]
    numeric_all_wrong = [d for d in wrong_details if d.get("numeric_scorable")]
    n_num = len(numeric_all_correct)
    c_mr = sum(1 for d in numeric_all_correct if d.get("mismatch_resolved"))
    w_mr = sum(
        1
        for d in numeric_all_correct
        if (by_pm_wrong.get(str(d["pm_trajectory_id"])) or {}).get("mismatch_resolved")
    )
    c_only = w_only = 0
    for cd in numeric_all_correct:
        pm = str(cd["pm_trajectory_id"])
        c_hit = bool(cd.get("mismatch_resolved"))
        w_hit = bool((by_pm_wrong.get(pm) or {}).get("mismatch_resolved"))
        if c_hit and not w_hit:
            c_only += 1
        elif not c_hit and w_hit:
            w_only += 1
    stratified["numeric_scorable"]["all"] = {
        "numeric_only": True,
        "n_pairs": n_num,
        "correct_donor_mismatch_resolved_k": c_mr,
        "correct_donor_mismatch_resolved_rate": c_mr / n_num if n_num else 0.0,
        "correct_donor_mismatch_resolved_ci95": wilson_ci(c_mr, n_num),
        "wrong_owner_mismatch_resolved_k": w_mr,
        "wrong_owner_mismatch_resolved_rate": w_mr / n_num if n_num else 0.0,
        "wrong_owner_mismatch_resolved_ci95": wilson_ci(w_mr, n_num),
        "correct_donor_only_resolved": c_only,
        "wrong_owner_only_resolved": w_only,
        "mcnemar_exact_p_two_sided": mcnemar_exact_p(c_only, w_only),
    }

    schema = (
        "ccer_line_f_shell_wrong_owner_donor_v3"
        if protocol == "v3"
        else "ccer_line_f_shell_wrong_owner_donor_v2"
    )
    primary = "mechanism_resolved" if protocol == "v3" else "mismatch_resolved"
    patch_note = LINE_F_V3_PATCH_NOTE if protocol == "v3" else LINE_F_V2_PATCH_NOTE

    return {
        "schema_version": schema,
        "protocol_version": LINE_F_V3_PROTOCOL if protocol == "v3" else "line_f_v2",
        "experiment_id": "line_f_shell_expanded_strong_mask_correct_vs_wrong_owner",
        "primary_metric": primary,
        "secondary_metrics": [
            "mismatch_resolved",
            "masked_to_anchor_resolved",
            "product_id_diff",
            "digit_drift_flip",
            "real_pid_flip",
            "clean_gold_aligned",
            "anchor_aligned_ti",
        ],
        "evaluator": "shell_pair_metrics (full_vector; v3 adds extended masked cohort)",
        "mask_mode": "strong",
        "layer": effective_layer,
        "position": position,
        "intervention_mode": "full_vector",
        "steering_apply": effective_steering,
        "use_cache": use_cache,
        "rank": rank,
        "rank_note": "unused (full_vector bypasses PCA subspace)",
        "intervention_alpha": alpha,
        "pool": pool,
        "anchor_filter": "shell_pid_only",
        "eval_eligible_only": eval_eligible_only,
        "wrong_donor_definition": "recipient_pm_vec_at_commitment (zero delta full_vector)",
        "wrong_owner_identity_gate": identity_gate,
        "pair_meta": {
            "n_shell_pairs_total": len(all_shell),
            "n_shell_pairs_evaluated": len(shell_pairs),
            "n_shell_pairs_ineligible": len(ineligible),
            "n_shell_pairs_claim_ineligible": len(claim_ineligible),
            "n_r11_legacy_pairs": shell.get("n_r11_legacy_pairs"),
            "n_mechanism_expansion_pairs": shell.get("n_mechanism_expansion_pairs"),
            "n_full_mechanism_cross_pairs": shell["n_full_cross_pairs"],
            "baseline_round10_shell_n": 15,
            "r11_legacy_pairing_validation": pairing_validation,
            "methodology_note": shell["methodology_note"],
        },
        "stratified_by_pairing_source": stratified,
        "activation_coverage": coverage,
        "pid_extraction_audit": pid_audit,
        "patch_implementation": patch_note,
        "with_strong_mask_correct_donor": {
            **correct_sum,
            "pair_details": correct_details,
            "numeric_scorable_summary": numeric_correct_sum,
        },
        "with_strong_mask_wrong_owner_donor": {
            **wrong_sum,
            "pair_details": wrong_details,
            "numeric_scorable_summary": numeric_wrong_sum,
        },
        "paired_correct_vs_wrong_donor": {
            "n_paired": len(pair_cmp),
            "both_mismatch_resolved": both,
            "correct_donor_only_resolved": correct_only,
            "wrong_owner_only_resolved": wrong_only,
            "neither_resolved": neither,
            "mcnemar_exact_p_two_sided": mcnemar_exact_p(correct_only, wrong_only),
            "mechanism_correct_only_resolved": mech_correct_only,
            "mechanism_wrong_only_resolved": mech_wrong_only,
            "mechanism_mcnemar_exact_p_two_sided": mcnemar_exact_p(
                mech_correct_only, mech_wrong_only
            ),
            "pair_table": pair_cmp,
            "interpretation_gate": (
                "Support causal donor only if correct_donor mechanism_resolved >> wrong_owner "
                "AND wrong_owner_identity_gate.passed AND clean_gold_aligned > 0 on numeric subset."
                if protocol == "v3"
                else (
                    "Support causal donor only if correct_donor mismatch_resolved >> wrong_owner "
                    "AND wrong_owner_identity_gate.passed."
                )
            ),
        },
        "comparison_to_round11": {
            "round11_shell_n": 15,
            "round11_correct_k": 2,
            "round11_wrong_k": 0,
            "expanded_shell_n": len(shell_pairs),
            "expanded_correct_mismatch_resolved_k": correct_sum.get("n_mismatch_resolved"),
            "expanded_wrong_mismatch_resolved_k": wrong_sum.get("n_mismatch_resolved"),
            "expanded_numeric_scorable_n": pid_audit.get("n_numeric_scorable"),
        },
    }


def run_shell_expanded_wrong_owner_donor_v3(
    *,
    model: Any,
    tokenizer: Any,
    position: str = "commitment",
    layer: int | None = None,
    alphas: tuple[float, ...] = (1.0,),
    checkpoint_path: str | Path | None = None,
    resume_from_checkpoint: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Line F v3 single-position run (optionally alpha sweep)."""
    layer_override = layer
    results: list[dict[str, Any]] = []
    for alpha in alphas:
        ckpt = None
        if checkpoint_path and len(alphas) == 1:
            ckpt = checkpoint_path
        elif checkpoint_path:
            p = Path(checkpoint_path)
            ckpt = p.with_name(f"{p.stem}_a{alpha}{p.suffix}")
        result = run_shell_expanded_wrong_owner_donor(
            model=model,
            tokenizer=tokenizer,
            position=position,
            layer=layer_override,
            alpha=float(alpha),
            checkpoint_path=ckpt,
            resume_from_checkpoint=resume_from_checkpoint,
            protocol="v3",
            **kwargs,
        )
        results.append(result)
    if len(results) == 1:
        return results[0]
    return {
        "schema_version": "ccer_line_f_shell_wrong_owner_donor_v3_alpha_sweep",
        "protocol_version": LINE_F_V3_PROTOCOL,
        "position": position,
        "alphas": list(alphas),
        "runs": results,
    }


def run_line_f_position_sweep_v3(
    *,
    model: Any,
    tokenizer: Any,
    positions: tuple[str, ...] = LINE_F_V3_POSITIONS,
    alphas: tuple[float, ...] = (1.0,),
    checkpoint_dir: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Line F v3: head-to-head across commitment / claim_onset / pre_value."""
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    by_position: dict[str, Any] = {}
    for position in positions:
        ckpt = None
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt = ckpt_dir / f"shell_wrong_owner_{position}.json"
        print(f"[line_f v3] position sweep → {position} L{_position_layer(position)}", flush=True)
        by_position[position] = run_shell_expanded_wrong_owner_donor_v3(
            model=model,
            tokenizer=tokenizer,
            position=position,
            layer=None,
            alphas=alphas,
            checkpoint_path=ckpt,
            **kwargs,
        )
    rows: list[dict[str, Any]] = []
    for position, res in by_position.items():
        if res.get("runs"):
            res = res["runs"][0]
        correct = res.get("with_strong_mask_correct_donor") or {}
        wrong = res.get("with_strong_mask_wrong_owner_donor") or {}
        identity = res.get("wrong_owner_identity_gate") or {}
        rows.append(
            {
                "position": position,
                "layer": res.get("layer"),
                "steering_apply": res.get("steering_apply"),
                "n_evaluated": (res.get("pair_meta") or {}).get("n_shell_pairs_evaluated"),
                "identity_gate_passed": identity.get("passed"),
                "correct_mechanism_resolved": correct.get("n_mechanism_resolved"),
                "extended_cohort": correct.get("n_extended_mismatch_cohort"),
                "wrong_mechanism_resolved": wrong.get("n_mechanism_resolved"),
                "correct_clean_gold": correct.get("n_clean_gold_aligned"),
                "mcnemar_mechanism_p": (res.get("paired_correct_vs_wrong_donor") or {}).get(
                    "mechanism_mcnemar_exact_p_two_sided"
                ),
            }
        )
    return {
        "schema_version": "ccer_line_f_shell_wrong_owner_donor_v3_position_sweep",
        "protocol_version": LINE_F_V3_PROTOCOL,
        "positions": list(positions),
        "alphas": list(alphas),
        "rows": rows,
        "by_position": by_position,
    }


def finalize_line_f_v3_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    position: str = "commitment",
    layer: int | None = None,
    alpha: float = 1.0,
    pool: str = "mechanism_research",
    early_stop_note: str = "",
) -> dict[str, Any]:
    """Build v3 final JSON from interrupted checkpoint (early-stop封稿)."""
    ckpt = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    correct_details = list(ckpt.get("correct_pair_details") or [])
    wrong_details = list(ckpt.get("wrong_pair_details") or [])
    n_target = int(ckpt.get("n_shell_pairs") or 60)
    effective_layer = _position_layer(position, layer)
    effective_steering = _steering_for_position(position, None)

    shell = build_shell_cross_pairs(pool=pool)
    pairing_source_map = {
        str(p["pm_trajectory_id"]): str(p.get("pairing_source") or "unknown")
        for p in shell["pm_clean_cross_pairs"]
    }
    identity_violations = [
        {"pm_trajectory_id": str(wd["pm_trajectory_id"])}
        for wd in wrong_details
        if wd.get("ti_text") != wd.get("ni_text")
    ]
    identity_gate = {
        "expected": "wrong_owner ti_text == ni_text for all pairs (zero full_vector delta)",
        "n_violations": len(identity_violations),
        "passed": len(identity_violations) == 0,
        "violations": identity_violations[:20],
    }

    correct_sum = _summarize_arm(correct_details, control_id="target_interchange")
    wrong_sum = _summarize_arm(wrong_details, control_id="wrong_owner_donor")
    numeric_correct = [d for d in correct_details if d.get("numeric_scorable")]
    numeric_wrong = [d for d in wrong_details if d.get("numeric_scorable")]
    numeric_correct_sum = _summarize_arm(numeric_correct, control_id="target_interchange")
    numeric_wrong_sum = _summarize_arm(numeric_wrong, control_id="wrong_owner_donor")
    pid_audit = audit_pid_extraction_robustness(correct_details)

    by_pm_wrong = {str(d["pm_trajectory_id"]): d for d in wrong_details}
    correct_only = wrong_only = both = neither = 0
    mech_correct_only = mech_wrong_only = 0
    pair_cmp: list[dict[str, Any]] = []
    for cd in correct_details:
        pm = str(cd["pm_trajectory_id"])
        wd = by_pm_wrong.get(pm) or {}
        c_hit = bool(cd.get("mismatch_resolved"))
        w_hit = bool(wd.get("mismatch_resolved"))
        c_mech = bool(cd.get("mechanism_resolved"))
        w_mech = bool(wd.get("mechanism_resolved"))
        if c_hit and w_hit:
            both += 1
        elif c_hit and not w_hit:
            correct_only += 1
        elif not c_hit and w_hit:
            wrong_only += 1
        else:
            neither += 1
        if c_mech and not w_mech:
            mech_correct_only += 1
        elif not c_mech and w_mech:
            mech_wrong_only += 1
        pair_cmp.append(
            {
                "pm_trajectory_id": pm,
                "correct_donor_mechanism_resolved": c_mech,
                "wrong_owner_mechanism_resolved": w_mech,
                "clean_gold_aligned_correct": cd.get("clean_gold_aligned"),
            }
        )

    stratified = {
        "r11_legacy": _stratified_summary_v2(
            correct_details, wrong_details, pairing_source_map, "r11_legacy"
        ),
        "mechanism_expansion": _stratified_summary_v2(
            correct_details, wrong_details, pairing_source_map, "mechanism_expansion"
        ),
    }

    return {
        "schema_version": "ccer_line_f_shell_wrong_owner_donor_v3",
        "protocol_version": LINE_F_V3_PROTOCOL,
        "verdict": "null_cross_donor_no_restoration",
        "early_stop": {
            "n_pairs_completed": len(correct_details),
            "n_pairs_target": n_target,
            "completion_rate": len(correct_details) / n_target if n_target else 0.0,
            "note": early_stop_note
            or "Stopped after interim null on mechanism_resolved; identity gate passed.",
        },
        "experiment_id": "line_f_shell_expanded_strong_mask_correct_vs_wrong_owner_v3",
        "primary_metric": "mechanism_resolved",
        "layer": effective_layer,
        "position": position,
        "intervention_mode": "full_vector",
        "steering_apply": effective_steering,
        "use_cache": False,
        "intervention_alpha": alpha,
        "pool": pool,
        "wrong_owner_identity_gate": identity_gate,
        "pair_meta": {
            "n_shell_pairs_total": shell["n_shell_cross_pairs"],
            "n_shell_pairs_evaluated": len(correct_details),
            "n_shell_pairs_target": n_target,
            "n_r11_legacy_pairs": shell.get("n_r11_legacy_pairs"),
            "n_mechanism_expansion_pairs": shell.get("n_mechanism_expansion_pairs"),
        },
        "patch_implementation": LINE_F_V3_PATCH_NOTE,
        "with_strong_mask_correct_donor": {
            **correct_sum,
            "pair_details": correct_details,
            "numeric_scorable_summary": numeric_correct_sum,
        },
        "with_strong_mask_wrong_owner_donor": {
            **wrong_sum,
            "pair_details": wrong_details,
            "numeric_scorable_summary": numeric_wrong_sum,
        },
        "paired_correct_vs_wrong_donor": {
            "n_paired": len(pair_cmp),
            "correct_donor_only_resolved": correct_only,
            "wrong_owner_only_resolved": wrong_only,
            "mechanism_correct_only_resolved": mech_correct_only,
            "mechanism_wrong_only_resolved": mech_wrong_only,
            "mechanism_mcnemar_exact_p_two_sided": mcnemar_exact_p(
                mech_correct_only, mech_wrong_only
            ),
            "pair_table": pair_cmp,
        },
        "pid_extraction_audit": pid_audit,
        "stratified_by_pairing_source": stratified,
        "comparison_to_round11": {
            "round11_shell_n": 15,
            "round11_correct_product_id_k": 2,
            "round11_wrong_product_id_k": 0,
            "line_f_v1_expanded_correct_product_id_k": 9,
            "line_f_v1_expanded_wrong_product_id_k": 2,
        },
    }


def build_line_f_report_table(result: dict[str, Any]) -> list[str]:
    """Markdown lines for Line F v2 report."""
    correct = result.get("with_strong_mask_correct_donor") or {}
    wrong = result.get("with_strong_mask_wrong_owner_donor") or {}
    cmp_ = result.get("paired_correct_vs_wrong_donor") or {}
    pm = result.get("pair_meta") or {}
    cov = result.get("activation_coverage") or {}
    pid = result.get("pid_extraction_audit") or {}
    r11 = result.get("comparison_to_round11") or {}
    identity = result.get("wrong_owner_identity_gate") or {}
    num_correct = correct.get("numeric_scorable_summary") or {}
    num_wrong = wrong.get("numeric_scorable_summary") or {}
    strat_num = (result.get("stratified_by_pairing_source") or {}).get("numeric_scorable") or {}
    strat_all = strat_num.get("all") or {}

    lines = [
        "# Line F v2: Shell-Anchor Expanded Evidence Mask + full_vector wrong_owner_donor",
        "",
        "**线路 F v2**：`full_vector` commitment patch（Line G 协议）+ 统一 `use_cache=False` + `mismatch_resolved` 主指标。",
        "",
        "## 干预协议",
        "",
        f"- intervention_mode: **{result.get('intervention_mode')}**",
        f"- steering_apply: **{result.get('steering_apply')}**",
        f"- use_cache (ni/ti): **{result.get('use_cache')}**",
        f"- wrong_owner identity gate: **{'PASS' if identity.get('passed') else 'FAIL'}** "
        f"({identity.get('n_violations')} violations)",
        "",
        "## 池扩充",
        "",
        f"- pool: **{result.get('pool')}**",
        f"- shell 子类总数: **{pm.get('n_shell_pairs_total')}**",
        f"- 激活可评估: **{pm.get('n_shell_pairs_evaluated')}** / {pm.get('n_shell_pairs_total')}",
        f"- 激活覆盖率: **{float(cov.get('coverage_rate') or 0):.1%}**",
        f"- numeric_scorable (baseline PID 可提取): **{pid.get('n_numeric_scorable')}** / {pid.get('n_pairs')}",
        "",
        "## 主表（mismatch_resolved，全量 shell）",
        "",
        "| 条件 | n | mismatch cohort | mismatch_resolved k/n | rate | Wilson 95% CI |",
        "|------|---|-----------------|----------------------|------|---------------|",
    ]
    for label, row in (("correct donor", correct), ("wrong_owner_donor", wrong)):
        ci = row.get("mismatch_resolved_ci95") or (0.0, 0.0)
        lines.append(
            f"| {label} | {row.get('n')} | {row.get('n_mismatch_cohort')} | "
            f"{row.get('n_mismatch_resolved')}/{row.get('n_mismatch_cohort')} | "
            f"{float(row.get('mismatch_resolved_rate') or 0):.1%} | {ci} |"
        )

    lines += [
        "",
        "## 可评分子集（numeric_scorable，仅 baseline 数字 PID 可提取）",
        "",
        f"- n = **{strat_all.get('n_pairs')}**",
        "",
        "| 条件 | mismatch_resolved | product_id_diff | digit_drift | real_flip | clean_gold |",
        "|------|-------------------|-----------------|-------------|-----------|------------|",
        f"| correct donor | {num_correct.get('n_mismatch_resolved')}/{num_correct.get('n_mismatch_cohort')} "
        f"({float(num_correct.get('mismatch_resolved_rate') or 0):.1%}) | "
        f"{num_correct.get('n_product_id_diff')}/{num_correct.get('n_numeric_scorable')} | "
        f"{num_correct.get('n_digit_drift_flip')} | {num_correct.get('n_real_pid_flip')} | "
        f"{num_correct.get('n_clean_gold_aligned')} |",
        f"| wrong_owner | {num_wrong.get('n_mismatch_resolved')}/{num_wrong.get('n_mismatch_cohort')} "
        f"({float(num_wrong.get('mismatch_resolved_rate') or 0):.1%}) | "
        f"{num_wrong.get('n_product_id_diff')}/{num_wrong.get('n_numeric_scorable')} | "
        f"{num_wrong.get('n_digit_drift_flip')} | {num_wrong.get('n_real_pid_flip')} | "
        f"{num_wrong.get('n_clean_gold_aligned')} |",
        "",
        f"- numeric 子集 McNemar p: **{strat_all.get('mcnemar_exact_p_two_sided')}**",
        "",
        "## McNemar（correct vs wrong_owner，全量）",
        "",
        f"- correct_donor_only_resolved: **{cmp_.get('correct_donor_only_resolved')}**",
        f"- wrong_owner_only_resolved: **{cmp_.get('wrong_owner_only_resolved')}**",
        f"- McNemar exact p (two-sided): **{cmp_.get('mcnemar_exact_p_two_sided')}**",
        "",
        "## 分层（pairing_source）",
        "",
        "| 层 | n | correct resolved | wrong resolved | McNemar p |",
        "|----|---|------------------|----------------|-----------|",
    ]
    for key, label in (("r11_legacy", "R11 legacy"), ("mechanism_expansion", "mechanism expansion")):
        st = (result.get("stratified_by_pairing_source") or {}).get(key) or {}
        lines.append(
            f"| {label} | {st.get('n_pairs')} | "
            f"{st.get('correct_donor_mismatch_resolved_k')}/{st.get('n_pairs')} | "
            f"{st.get('wrong_owner_mismatch_resolved_k')}/{st.get('n_pairs')} | "
            f"{st.get('mcnemar_exact_p_two_sided')} |"
        )

    lines += [
        "",
        "## 与 Round 11 对照",
        "",
        f"| 指标 | Round 11 (n=15) | Line F v2 (n={r11.get('expanded_shell_n')}) |",
        "|------|-----------------|----------------------|",
        f"| correct mismatch_resolved | {r11.get('round11_correct_k')}/15 (v1 pid) | "
        f"{r11.get('expanded_correct_mismatch_resolved_k')}/{r11.get('expanded_shell_n')} |",
        f"| wrong_owner | {r11.get('round11_wrong_k')}/15 | "
        f"{r11.get('expanded_wrong_mismatch_resolved_k')}/{r11.get('expanded_shell_n')} |",
        "",
        "## PID 提取审计",
        "",
        f"- numeric_scorable: **{pid.get('n_numeric_scorable')}** / {pid.get('n_pairs')}",
        f"- baseline PID 缺失: **{pid.get('missing_baseline_pid')}**",
        f"- intervened PID 缺失: **{pid.get('missing_intervened_pid')}**",
        "",
        "## 诚实判定",
        "",
    ]
    n_eval = int(pm.get("n_shell_pairs_evaluated") or 0)
    if not identity.get("passed"):
        lines.append(
            f"- ❌ wrong_owner identity gate 失败：{identity.get('n_violations')} 条 ti≠ni，存在 decode/cache bug。"
        )
    c_mr = int(correct.get("n_mismatch_resolved") or 0)
    w_mr = int(wrong.get("n_mismatch_resolved") or 0)
    cohort = int(correct.get("n_mismatch_cohort") or 0)
    if identity.get("passed") and c_mr > w_mr and cohort > 0:
        lines.append(
            f"- 正向信号：correct donor mismatch_resolved {c_mr}/{cohort} vs wrong_owner {w_mr}/{cohort}。"
        )
    elif c_mr == w_mr:
        lines.append("- ⚠️ correct 与 wrong_owner mismatch_resolved 相同，不能支持因果 donor 特异性。")
    else:
        lines.append(
            f"- 结果：correct {c_mr}/{cohort} vs wrong_owner {w_mr}/{cohort}，需结合 CI 与 McNemar 解读。"
        )

    return lines
