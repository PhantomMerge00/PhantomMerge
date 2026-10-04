"""Task I: simultaneous four-position joint interchange patch."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.controls import build_control_specs
from ccer.mechanism.interchange import (
    JointInterventionSlot,
    greedy_generate_with_hook,
    greedy_generate_with_joint_hooks,
)
from ccer.mechanism.iia_roi_probe import _outputs_differ, _probe_log
from ccer.mechanism.line_b_protocol import resolve_wrong_owner_donor_vec
from ccer.mechanism.owner_pca import fit_owner_pca
from ccer.mechanism.rank_sweep import _cross_pairs, _line_d_eligible_cross_pairs, _shard_pairs
from ccer.mechanism.round8_verify import LINE_D_POSITIONS, _specificity_block
from ccer.mechanism.stats import wilson_ci
from ccer.mechanism.pair_select import load_trajectory_index, messages_for_condition
from ccer.replay.answer_utils import (
    extract_selected_product_id,
    full_response_excerpt_differ,
    selected_product_blocks_differ,
)
from ccer.replay.hf_forward import tokenize_ccer_messages
from ccer.replay.live_position import (
    LINE_D_PROTOCOL_VERSION,
    cross_pair_passes_line_d_v3_gate,
    intervention_steering_apply,
    line_d_layer_for_position,
    live_token_idx_for_intervention,
    make_live_position_resolver,
)

LINE_I_PROTOCOL_VERSION = "line_i_joint_patch_v1"
LINE_I_INTERVENTION_PROTOCOL = "joint_simultaneous_four_position"

JOINT_ARMS: tuple[str, ...] = (
    "joint_target_interchange",
    "joint_wrong_owner",
    "joint_minus_prompt_end",
    "joint_minus_commitment",
    "joint_minus_claim_onset",
    "joint_minus_pre_value",
)

MINUS_ONE_POSITION_BY_ARM: dict[str, str] = {
    "joint_minus_prompt_end": "prompt_end",
    "joint_minus_commitment": "commitment",
    "joint_minus_claim_onset": "claim_onset",
    "joint_minus_pre_value": "pre_value",
}

HISTORICAL_SINGLE_POINT_BASELINE = [
    {
        "source": "Line D v3",
        "position": "prompt_end",
        "product_id_k": 0,
        "product_id_n": 22,
        "patched_k": 22,
        "patched_n": 22,
    },
    {
        "source": "Round 8 / Line D history",
        "position": "commitment",
        "product_id_k": 1,
        "product_id_n": 22,
        "patched_k": 22,
        "patched_n": 22,
        "note": "approx 4.5%",
    },
    {
        "source": "Line C Arm1",
        "position": "commitment (attention path)",
        "product_id_k": 4,
        "product_id_n": 91,
        "patched_k": 91,
        "patched_n": 91,
        "note": "approx 4.4%",
    },
]


def positions_for_arm(arm_id: str) -> tuple[str, ...]:
    if arm_id == "joint_target_interchange" or arm_id == "joint_wrong_owner":
        return LINE_D_POSITIONS
    drop = MINUS_ONE_POSITION_BY_ARM.get(arm_id)
    if drop is None:
        raise ValueError(f"unknown joint arm: {arm_id}")
    return tuple(p for p in LINE_D_POSITIONS if p != drop)


def _eligible_joint_pairs(
    cross_pairs: list[dict[str, Any]],
    *,
    track_u: str,
    rows_index: dict[str, Any],
) -> list[dict[str, Any]]:
    return _line_d_eligible_cross_pairs(
        cross_pairs,
        position="claim_onset",
        track_u=track_u,
        rows_index=rows_index,
    )


def _fit_position_pca(
    *,
    position: str,
    layer: int,
    rank: int,
    track_u: str,
    pm_allowlist: set[str],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    line_d_v3 = position in {"claim_onset", "pre_value"}
    fit = fit_owner_pca(
        layer=layer,
        position=position,
        rank=rank,
        track=track_u,
        pool="dev",
        pm_trajectory_allowlist=pm_allowlist if line_d_v3 else None,
        require_line_d_v3=line_d_v3,
        cross_only=line_d_v3,
    )
    u = fit.get("U_owner")
    if u is None:
        return None, fit
    return np.asarray(u, dtype=np.float32), fit


def _build_joint_slots(
    *,
    arm_id: str,
    positions: tuple[str, ...],
    roi_by_position: dict[str, dict[str, Any]],
    pm_vecs: dict[str, np.ndarray],
    donor_vecs: dict[str, np.ndarray],
    pos_idx_by_position: dict[str, int],
    tokenizer: Any,
    claim_layer: int,
) -> list[JointInterventionSlot]:
    slots: list[JointInterventionSlot] = []
    for position in positions:
        layer = line_d_layer_for_position(position, claim_layer=claim_layer)
        roi = dict(roi_by_position[position])
        roi["primary_roi"] = {"layer": layer, "position": position}
        roi["steering_apply"] = intervention_steering_apply(position)
        spec = build_control_specs(
            control_id="target_interchange",
            roi=roi,
            donor_vec=donor_vecs[position],
            recipient_vec=pm_vecs[position],
            position_token_idx=pos_idx_by_position[position],
            alpha=1.0,
        )
        spec.steering_apply = intervention_steering_apply(position)
        live_resolver = (
            make_live_position_resolver(tokenizer, position)
            if position in {"claim_onset", "pre_value"}
            else None
        )
        slots.append(
            JointInterventionSlot(
                position=position,
                spec=spec,
                anchor_pos=int(pos_idx_by_position[position]),
                live_resolver=live_resolver,
                patched=False,
            )
        )
    return slots


def _resolve_donor_vecs_for_arm(
    *,
    arm_id: str,
    pm_tid: str,
    clean_tid: str,
    cross_pairs: list[dict[str, Any]],
    positions: tuple[str, ...],
    pm_vecs: dict[str, np.ndarray],
    clean_vecs: dict[str, np.ndarray],
    claim_layer: int,
) -> tuple[dict[str, np.ndarray], dict[str, str | None]]:
    donor_vecs: dict[str, np.ndarray] = {}
    wrong_impl: dict[str, str | None] = {}
    for position in positions:
        layer = line_d_layer_for_position(position, claim_layer=claim_layer)
        if arm_id == "joint_wrong_owner":
            wrong_vec, impl = resolve_wrong_owner_donor_vec(
                pm_tid=pm_tid,
                paired_clean_tid=clean_tid,
                cross_pairs=cross_pairs,
                layer=layer,
                position=position,
            )
            if wrong_vec is not None:
                donor_vecs[position] = wrong_vec
                wrong_impl[position] = impl
            else:
                donor_vecs[position] = clean_vecs[position] * -1.0
                wrong_impl[position] = "negated_clean_fallback"
        else:
            donor_vecs[position] = clean_vecs[position]
            wrong_impl[position] = None
    return donor_vecs, wrong_impl


def _aggregate_arm_row(pair_details: list[dict[str, Any]]) -> dict[str, Any]:
    valid_details = [d for d in pair_details if d.get("all_patched")]
    invalid_details = [d for d in pair_details if not d.get("all_patched")]
    n_eligible = len(pair_details)
    n_valid = len(valid_details)
    n_invalid = len(invalid_details)
    pid = sum(1 for d in valid_details if d.get("product_id_diff"))
    out = sum(1 for d in valid_details if d.get("output_diff"))
    block = sum(1 for d in valid_details if d.get("selected_product_block_diff"))
    patched_all = sum(1 for d in pair_details if d.get("all_patched"))
    fallback_counts: dict[str, int] = {}
    for d in pair_details:
        fb = d.get("patch_fallback_used") or {}
        for pos, path in fb.items():
            if path:
                key = f"{pos}:{path}"
                fallback_counts[key] = fallback_counts.get(key, 0) + 1
    return {
        "n_eligible": n_eligible,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "n_all_patched": patched_all,
        "n_product_id_diff": pid,
        "product_id_diff_rate": pid / n_valid if n_valid else 0.0,
        "product_id_ci95": wilson_ci(pid, n_valid),
        "n_output_diff": out,
        "output_diff_rate": out / n_valid if n_valid else 0.0,
        "output_diff_ci95": wilson_ci(out, n_valid),
        "n_selected_product_block_diff": block,
        "selected_product_block_diff_rate": block / n_valid if n_valid else 0.0,
        "selected_product_block_ci95": wilson_ci(block, n_valid),
        "patch_fallback_counts": fallback_counts,
        "pair_details": pair_details,
    }


def _infer_verdict(
    *,
    target_row: dict[str, Any],
    wrong_row: dict[str, Any],
    specificity: dict[str, Any],
) -> str:
    n_valid = int(target_row.get("n_valid") or 0)
    n_invalid = int(target_row.get("n_invalid") or 0)
    if n_valid == 0 and n_invalid > 0:
        return "invalid_patch_engineering"
    if n_valid < 5:
        return "inconclusive_low_n_valid"
    target_pid = float(target_row.get("product_id_diff_rate") or 0.0)
    best_single = max(
        (row["product_id_k"] / row["product_id_n"] if row["product_id_n"] else 0.0)
        for row in HISTORICAL_SINGLE_POINT_BASELINE
    )
    pid_correct_only = int(specificity.get("product_id_correct_only") or 0)
    if target_pid > best_single and pid_correct_only > 0:
        return "positive_joint_lever"
    if n_valid > 0:
        return "null_joint_with_valid_patch"
    return "inconclusive_low_n_valid"


def measure_joint_diff_rate(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    rank: int = 16,
    claim_layer: int = 49,
    pair_limit: int | None = None,
    probe_max_tokens: int = 384,
    arms: tuple[str, ...] = JOINT_ARMS,
    shard_index: int = 0,
    num_shards: int = 1,
    save_texts: bool = False,
    use_cache: bool | None = False,
    cross_pairs_override: list[dict[str, Any]] | None = None,
    resume_by_arm: dict[str, list[dict[str, Any]]] | None = None,
    on_pair_complete: Callable[[str, list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    track_u = track.upper()
    cross_pairs_all = list(cross_pairs_override) if cross_pairs_override is not None else _cross_pairs(track_u)
    rows_index = load_trajectory_index()
    cross_pairs_all = _eligible_joint_pairs(cross_pairs_all, track_u=track_u, rows_index=rows_index)
    pm_key = "pm_trajectory_id" if track_u == "CEM" else "cap_trajectory_id"
    pm_allowlist = {
        str(cp.get(pm_key) or cp.get("cap_trajectory_id"))
        for cp in cross_pairs_all
    }
    cross_pairs = list(cross_pairs_all)
    if pair_limit is not None and cross_pairs_override is None:
        cross_pairs = cross_pairs[:pair_limit]
    cross_pairs = _shard_pairs(cross_pairs, shard_index, num_shards)

    roi_by_position: dict[str, dict[str, Any]] = {}
    layer_by_position = {
        pos: line_d_layer_for_position(pos, claim_layer=claim_layer) for pos in LINE_D_POSITIONS
    }
    for position in LINE_D_POSITIONS:
        layer = layer_by_position[position]
        u, fit = _fit_position_pca(
            position=position,
            layer=layer,
            rank=rank,
            track_u=track_u,
            pm_allowlist=pm_allowlist,
        )
        if u is None:
            return {
                "schema_version": LINE_I_PROTOCOL_VERSION,
                "error": fit.get("error", f"pca_fit_failed_{position}"),
                "position": position,
                "n_eligible": 0,
            }
        roi_by_position[position] = {
            "primary_roi": {"layer": layer, "position": position},
            "U_owner": u.tolist(),
            "inject_site": "residual",
            "steering_apply": intervention_steering_apply(position),
            "pc_variance_explained": fit.get("pc_variance_explained"),
        }

    ni_cache: dict[str, str] = {}
    arm_rows: dict[str, dict[str, Any]] = {}
    resume_by_arm = resume_by_arm or {}
    t0 = time.time()

    for arm_id in arms:
        positions = positions_for_arm(arm_id)
        pair_details: list[dict[str, Any]] = list(resume_by_arm.get(arm_id) or [])
        done_pm = {str(d["pm_trajectory_id"]) for d in pair_details}

        for cp in cross_pairs:
            pm_tid = str(cp.get(pm_key) or cp.get("cap_trajectory_id"))
            if pm_tid in done_pm:
                continue
            clean_tid = str(cp["clean_trajectory_id"])
            pm_traj = rows_index.get(pm_tid)
            if not pm_traj:
                continue
            pm_npz = load_activation_npz(activation_path(pm_tid, "original"))
            clean_npz = load_activation_npz(activation_path(clean_tid, "original"))
            skip_pair = False
            for position in positions:
                if position in {"claim_onset", "pre_value"} and not cross_pair_passes_line_d_v3_gate(
                    pm_npz, clean_npz, position
                ):
                    _probe_log(f"joint_patch skip {pm_tid}: missing v3 gate for {position}")
                    skip_pair = True
                    break
            if skip_pair:
                continue

            pm_vecs: dict[str, np.ndarray] = {}
            clean_vecs: dict[str, np.ndarray] = {}
            pos_idx_by_position: dict[str, int] = {}
            skip_pair = False
            for position in positions:
                layer = layer_by_position[position]
                pm_vec = get_vector(pm_npz, position=position, layer=layer)
                clean_vec = get_vector(clean_npz, position=position, layer=layer)
                if pm_vec is None or clean_vec is None:
                    skip_pair = True
                    break
                pm_vecs[position] = pm_vec
                clean_vecs[position] = clean_vec
            if skip_pair:
                continue

            messages = messages_for_condition(pm_traj, "original", track=track_u)
            tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
            prompt_len = int(tok["prompt_token_count"])
            for position in positions:
                pos_idx_by_position[position] = live_token_idx_for_intervention(
                    pm_npz=pm_npz,
                    position=position,
                    prompt_len=prompt_len,
                )
                if position not in {"claim_onset", "pre_value"} and pos_idx_by_position[position] < 0:
                    skip_pair = True
                    break
            if skip_pair:
                continue

            cache_key = f"{pm_tid}|{clean_tid}|joint|{probe_max_tokens}"
            if cache_key not in ni_cache:
                out_ni = greedy_generate_with_hook(
                    model,
                    tokenizer,
                    messages,
                    spec=build_control_specs(
                        control_id="no_intervention",
                        roi=roi_by_position[positions[0]],
                        donor_vec=clean_vecs[positions[0]],
                        recipient_vec=pm_vecs[positions[0]],
                        position_token_idx=pos_idx_by_position[positions[0]],
                        alpha=0.0,
                    ),
                    max_new_tokens=probe_max_tokens,
                    use_cache=use_cache,
                )
                ni_cache[cache_key] = out_ni["text"]

            donor_vecs, wrong_impl = _resolve_donor_vecs_for_arm(
                arm_id=arm_id,
                pm_tid=pm_tid,
                clean_tid=clean_tid,
                cross_pairs=cross_pairs,
                positions=positions,
                pm_vecs=pm_vecs,
                clean_vecs=clean_vecs,
                claim_layer=claim_layer,
            )
            slots = _build_joint_slots(
                arm_id=arm_id,
                positions=positions,
                roi_by_position=roi_by_position,
                pm_vecs=pm_vecs,
                donor_vecs=donor_vecs,
                pos_idx_by_position=pos_idx_by_position,
                tokenizer=tokenizer,
                claim_layer=claim_layer,
            )
            out_ti = greedy_generate_with_joint_hooks(
                model,
                tokenizer,
                messages,
                slots=slots,
                max_new_tokens=probe_max_tokens,
                use_cache=use_cache,
            )
            ni_text = ni_cache[cache_key]
            ti_text = out_ti["text"]
            pid_ni = extract_selected_product_id(ni_text)
            pid_ti = extract_selected_product_id(ti_text)
            pid_differs = bool(pid_ni and pid_ti and pid_ni != pid_ti)
            detail = {
                "pm_trajectory_id": pm_tid,
                "clean_trajectory_id": clean_tid,
                "arm_id": arm_id,
                "positions_active": list(positions),
                "patched_by_position": out_ti.get("patched_by_position") or {},
                "all_patched": bool(out_ti.get("all_patched")),
                "patch_fallback_used": out_ti.get("patch_fallback_used") or {},
                "wrong_owner_impl": wrong_impl if arm_id == "joint_wrong_owner" else None,
                "output_diff": _outputs_differ(ni_text, ti_text),
                "selected_product_block_diff": selected_product_blocks_differ(ni_text, ti_text),
                "full_response_excerpt_diff": full_response_excerpt_differ(ni_text, ti_text),
                "product_id_ni": pid_ni,
                "product_id_ti": pid_ti,
                "product_id_diff": pid_differs,
                "claim_attribution_diff": None,
                "claim_attribution_corrected": None,
                "generated_tokens": len(out_ti.get("generated_token_ids") or []),
            }
            if save_texts and (detail["output_diff"] or detail["product_id_diff"]):
                detail["ni_text"] = ni_text
                detail["ti_text"] = ti_text
            pair_details.append(detail)
            done_pm.add(pm_tid)
            if on_pair_complete is not None:
                on_pair_complete(arm_id, pair_details)
            _probe_log(
                f"joint_patch arm={arm_id} pair={len(pair_details)}/{len(cross_pairs)} "
                f"all_patched={detail['all_patched']} pid_diff={pid_differs}"
            )

        arm_rows[arm_id] = _aggregate_arm_row(pair_details)

    target_details = (arm_rows.get("joint_target_interchange") or {}).get("pair_details") or []
    wrong_details = (arm_rows.get("joint_wrong_owner") or {}).get("pair_details") or []
    valid_target = [d for d in target_details if d.get("all_patched")]
    valid_wrong = [d for d in wrong_details if d.get("all_patched")]
    specificity = _specificity_block(valid_target, valid_wrong)
    target_row = arm_rows.get("joint_target_interchange") or {}
    wrong_row = arm_rows.get("joint_wrong_owner") or {}
    verdict = _infer_verdict(
        target_row=target_row,
        wrong_row=wrong_row,
        specificity=specificity,
    )

    return {
        "schema_version": LINE_I_PROTOCOL_VERSION,
        "protocol_version": LINE_D_PROTOCOL_VERSION,
        "intervention_protocol": LINE_I_INTERVENTION_PROTOCOL,
        "primary_metric": "product_id_diff",
        "secondary_metric": "selected_product_block_diff",
        "claim_metric_reserved": "claim_attribution_diff",
        "track": track_u,
        "rank": rank,
        "claim_layer": claim_layer,
        "layer_by_position": layer_by_position,
        "probe_max_tokens": probe_max_tokens,
        "n_eligible_pairs": len(cross_pairs_all),
        "n_eval_pairs": len(cross_pairs),
        "arms": list(arms),
        "arm_results": arm_rows,
        "specificity": specificity,
        "historical_single_point_baseline": HISTORICAL_SINGLE_POINT_BASELINE,
        "verdict": verdict,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "elapsed_s": round(time.time() - t0, 1),
    }


def merge_joint_patch_shards(shard_results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(shard_results) == 1:
        return shard_results[0]
    base = dict(shard_results[0])
    merged_arms: dict[str, dict[str, Any]] = {}
    for arm_id in JOINT_ARMS:
        parts = []
        for shard in shard_results:
            arm = (shard.get("arm_results") or {}).get(arm_id) or {}
            parts.extend(arm.get("pair_details") or [])
        if parts:
            merged_arms[arm_id] = _aggregate_arm_row(parts)
    base["arm_results"] = merged_arms
    base["num_shards_merged"] = len(shard_results)
    target_details = (merged_arms.get("joint_target_interchange") or {}).get("pair_details") or []
    wrong_details = (merged_arms.get("joint_wrong_owner") or {}).get("pair_details") or []
    valid_target = [d for d in target_details if d.get("all_patched")]
    valid_wrong = [d for d in wrong_details if d.get("all_patched")]
    base["specificity"] = _specificity_block(valid_target, valid_wrong)
    base["verdict"] = _infer_verdict(
        target_row=merged_arms.get("joint_target_interchange") or {},
        wrong_row=merged_arms.get("joint_wrong_owner") or {},
        specificity=base["specificity"],
    )
    base["n_eligible_pairs"] = sum(int(s.get("n_eligible_pairs") or 0) for s in shard_results)
    base["elapsed_s"] = round(sum(float(s.get("elapsed_s") or 0) for s in shard_results), 1)
    return base
