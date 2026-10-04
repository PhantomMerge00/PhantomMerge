"""Line E evaluation: CAA steering curves on held-out PM trajectories."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ccer.io_utils import append_jsonl, load_jsonl, write_json
from ccer.mechanism.iia_roi_probe import _live_token_idx
from ccer.mechanism.interchange import (
    build_donor_restore_spec,
    build_paired_full_vector_spec,
    build_steering_spec,
    greedy_generate_with_hook,
)
from ccer.mechanism.line_e_donor import trajectory_commitment_stratum
from ccer.mechanism.line_e_mechanism_obs import measure_claim_onset_mechanism
from ccer.mechanism.line_e_protocol import (
    LINE_E_MECHANISM_OBS_ALPHAS,
    LINE_E_V3_ALPHA_SWEEP,
    LINE_E_V3_PRIMARY_POSITION,
    LINE_E_V3_PROTOCOL_VERSION,
    LineESteeringConfig,
    audit_v3_pool_coverage,
    build_trajectory_steering_bundle,
    filter_v3_eval_trajectories,
    line_e_v3_config,
    mechanism_research_pm_v3_pool,
    trajectory_has_position_activation,
)
from ccer.replay.live_position import make_live_position_resolver
from ccer.mechanism.pair_select import TrackName, load_trajectory_index, messages_for_condition
from ccer.mechanism.stats import mcnemar_exact_p, wilson_ci
from ccer.mechanism.mechanism_pool import PoolName
from ccer.mechanism.steering_vector import (
    SteeringControlId,
    assign_build_eval_holdout,
    assign_split_manifest_holdout,
    build_caa_vector,
    build_paired_steering_controls,
    build_steering_controls,
    clean_pool_with_activations,
    mechanism_research_clean_for_build,
    mechanism_research_pm_with_activations,
    paired_activation_vectors,
    pm_pool_with_activations,
    save_caa_bundle,
)
from ccer.mechanism.activation_store import activation_path, load_activation_npz
from ccer.paths import P3_DIR
from ccer.replay.answer_utils import extract_selected_product_id
from ccer.replay.hf_forward import tokenize_ccer_messages

LINE_E_ALPHA_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
STEERING_CONTROLS: tuple[SteeringControlId, ...] = ("caa_true", "caa_random", "caa_orthogonal")
STEERING_CONTROLS_V2: tuple[str, ...] = ("paired_caa", "caa_random", "caa_orthogonal", "paired_full")
STEERING_APPLY_V2 = "generation_decode"
STEERING_DIRECTION_V2 = "subtract"
P3_LINE_E_DIR = P3_DIR / "line_e"
_DEBUG_LOG = Path("${PHANTOM_MERGE_ROOT}/logs/debug")


def _protocol_stem(*, steering_v2: bool = False, steering_v3: bool = False) -> str:
    if steering_v3:
        return "steering_eval_rows_v4_ultimate"
    if steering_v2:
        return "steering_eval_rows_v2"
    return "steering_eval_rows"


def eval_rows_path(
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    steering_v2: bool = False,
    steering_v3: bool = False,
    use_cache: bool = False,
) -> Path:
    stem = _protocol_stem(steering_v2=steering_v2, steering_v3=steering_v3)
    if use_cache:
        stem = f"{stem}_uc"
    if num_shards <= 1:
        return P3_LINE_E_DIR / f"{stem}.jsonl"
    return P3_LINE_E_DIR / f"{stem}_s{shard_index}.jsonl"


def eval_checkpoint_path(
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    steering_v2: bool = False,
    steering_v3: bool = False,
    use_cache: bool = False,
) -> Path:
    if steering_v3:
        stem = "steering_eval_checkpoint_v4_ultimate"
    elif steering_v2:
        stem = "steering_eval_checkpoint_v2"
    else:
        stem = "steering_eval_checkpoint"
    if use_cache:
        stem = f"{stem}_uc"
    if num_shards <= 1:
        return P3_LINE_E_DIR / f"{stem}.json"
    return P3_LINE_E_DIR / f"{stem}_s{shard_index}.json"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    # region agent log
    try:
        payload = {
            "sessionId": "22e232",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass
    # endregion


def _row_key(row: dict[str, Any]) -> tuple[str, str, float]:
    return (str(row["trajectory_id"]), str(row["control_id"]), float(row["alpha"]))


def _steering_vec(controls: dict[str, Any], key: str, *, fallback: dict[str, Any] | None = None, fallback_key: str = "caa_true") -> np.ndarray:
    """Fetch steering vector without numpy truthiness (`or` on ndarray is ambiguous)."""
    vec = controls.get(key)
    if vec is None and fallback is not None:
        vec = fallback.get(fallback_key)
    if vec is None:
        raise KeyError(f"missing steering vector: {key}")
    return vec


def _rebuild_baseline_cache(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        tid = str(row.get("trajectory_id") or "")
        if not tid:
            continue
        cache[tid] = {
            "wrong_owner": bool(row.get("baseline_wrong_owner")),
            "product_id_correct": bool(row.get("baseline_product_id_correct")),
            "selected_product_id": row.get("baseline_selected_product_id"),
            "action_anchor": row.get("action_anchor"),
            "product_id_missing": row.get("baseline_selected_product_id") is None,
        }
    return cache


def _trajectory_track(traj: dict[str, Any]) -> TrackName:
    for claim in traj.get("claims") or []:
        label = claim.get("legacy_label")
        if label == "cross_object_merge":
            return "CEM"
        if label == "constraint_projection":
            return "CAP"
    return "CEM"


def _action_anchor(traj: dict[str, Any]) -> str | None:
    anchor = (traj.get("commitment") or {}).get("action_anchor")
    return str(anchor) if anchor else None


def behavioral_outcome(text: str, traj: dict[str, Any]) -> dict[str, Any]:
    """Behavioral PM proxy on regenerated output (product_id vs action_anchor)."""
    pid = extract_selected_product_id(text)
    anchor = _action_anchor(traj)
    wrong_owner = bool(pid and anchor and pid != anchor)
    pid_correct = bool(pid and anchor and pid == anchor)
    pid_missing = pid is None
    return {
        "selected_product_id": pid,
        "action_anchor": anchor,
        "wrong_owner": wrong_owner,
        "product_id_correct": pid_correct,
        "product_id_missing": pid_missing,
        "pm_persist": wrong_owner,
    }


def build_line_e_bundle(
    *,
    layer: int,
    position: str,
    holdout_frac: float = 0.35,
    seed: int = 42,
    track: TrackName | None = None,
    pool: PoolName = "mechanism_research",
    steering_v3: bool = False,
) -> dict[str, Any]:
    if steering_v3:
        v3_cfg = line_e_v3_config(position=position)
        if position == "commitment" and layer == 32:
            position = v3_cfg.position
            layer = v3_cfg.layer
        if pool == "mechanism_research":
            pm_pool = mechanism_research_pm_v3_pool(position=position)
        else:
            pm_pool = [
                tid
                for tid in pm_pool_with_activations(track=track)
                if trajectory_has_position_activation(tid, layer=layer, position=position)
            ]
        holdout = assign_split_manifest_holdout(pm_pool)
        build_pm = holdout["build_pm_trajectory_ids"]
        clean_pool = mechanism_research_clean_for_build(build_pm, layer=layer, position=position)
    elif pool == "mechanism_research":
        pm_pool = mechanism_research_pm_with_activations(layer=layer, position=position)
        holdout = assign_split_manifest_holdout(pm_pool)
        build_pm = holdout["build_pm_trajectory_ids"]
        clean_pool = mechanism_research_clean_for_build(build_pm, layer=layer, position=position)
    else:
        pm_pool = pm_pool_with_activations(track=track)
        clean_pool = clean_pool_with_activations()
        holdout = assign_build_eval_holdout(pm_pool, holdout_frac=holdout_frac, seed=seed)
        build_pm = holdout["build_pm_trajectory_ids"]
    caa = build_caa_vector(build_pm, clean_pool, layer=layer, position=position)
    if isinstance(caa, dict) and caa.get("error"):
        return {"error": caa["error"], "holdout": holdout, "layer": layer, "position": position}
    controls = build_steering_controls(caa.vector, seed=seed)
    out = {
        "caa": caa,
        "controls": controls,
        "holdout": holdout,
        "clean_trajectory_ids": clean_pool,
        "layer": layer,
        "position": position,
        "seed": seed,
    }
    if steering_v3:
        out["protocol_version"] = LINE_E_V3_PROTOCOL_VERSION
        out["steering_config"] = line_e_v3_config(position=position)
    return out


def _generate_with_steering(
    model: Any,
    tokenizer: Any,
    traj: dict[str, Any],
    *,
    layer: int,
    position: str,
    position_token_idx: int,
    steering_vec: np.ndarray,
    control_id: str,
    alpha: float,
    max_new_tokens: int,
    steering_direction: str = "subtract",
    steering_apply: str = "anchor_once",
    paired_full: dict[str, np.ndarray] | None = None,
    donor_restore: dict[str, np.ndarray] | None = None,
    slot_consistent_restore: dict[str, np.ndarray] | None = None,
    anchor_aligned_restore: dict[str, np.ndarray] | None = None,
    use_live_position_resolver: bool = False,
    capture_mechanism_obs: bool = False,
    use_cache: bool | None = None,
) -> dict[str, Any]:
    track = _trajectory_track(traj)
    messages = messages_for_condition(traj, "original", track=track)
    if not messages:
        return {"error": "messages_unavailable", "text": ""}
    live_resolver = make_live_position_resolver(tokenizer, position) if use_live_position_resolver else None
    donor_vec_for_obs: np.ndarray | None = None
    if control_id == "donor_restore" and donor_restore is not None and alpha > 0:
        donor_vec_for_obs = donor_restore["clean_vec"]
        spec = build_donor_restore_spec(
            control_id=control_id,
            layer=layer,
            position_token_idx=position_token_idx,
            donor_vec=donor_restore["clean_vec"],
            alpha=alpha,
            steering_apply=steering_apply,
        )
    elif control_id == "slot_consistent_restore" and slot_consistent_restore is not None and alpha > 0:
        donor_vec_for_obs = slot_consistent_restore["donor_vec"]
        spec = build_donor_restore_spec(
            control_id=control_id,
            layer=layer,
            position_token_idx=position_token_idx,
            donor_vec=slot_consistent_restore["donor_vec"],
            alpha=alpha,
            steering_apply=steering_apply,
        )
    elif control_id == "anchor_aligned_restore" and anchor_aligned_restore is not None and alpha > 0:
        donor_vec_for_obs = anchor_aligned_restore["donor_vec"]
        spec = build_donor_restore_spec(
            control_id=control_id,
            layer=layer,
            position_token_idx=position_token_idx,
            donor_vec=anchor_aligned_restore["donor_vec"],
            alpha=alpha,
            steering_apply=steering_apply,
        )
    elif control_id == "paired_full" and paired_full is not None and alpha > 0:
        spec = build_paired_full_vector_spec(
            control_id=control_id,
            layer=layer,
            position_token_idx=position_token_idx,
            pm_vec=paired_full["pm_vec"],
            clean_vec=paired_full["clean_vec"],
            alpha=alpha,
            steering_apply=steering_apply,
        )
    elif control_id == "no_intervention" or alpha == 0.0:
        spec = build_steering_spec(
            control_id="no_intervention",
            layer=layer,
            position_token_idx=position_token_idx,
            steering_vec=steering_vec,
            alpha=0.0,
            steering_direction=steering_direction,
            steering_apply="anchor_once",
        )
    else:
        spec = build_steering_spec(
            control_id=control_id,
            layer=layer,
            position_token_idx=position_token_idx,
            steering_vec=steering_vec,
            alpha=alpha,
            steering_direction=steering_direction,
            steering_apply=steering_apply,
        )
    mechanism_obs = None
    if capture_mechanism_obs and alpha > 0 and donor_vec_for_obs is not None:
        mechanism_obs = measure_claim_onset_mechanism(
            model,
            tokenizer,
            messages,
            spec=spec,
            claim_token_idx=position_token_idx,
            donor_vec=donor_vec_for_obs,
            action_anchor=_action_anchor(traj),
        )
    elif capture_mechanism_obs and alpha > 0 and control_id == "paired_caa":
        mechanism_obs = measure_claim_onset_mechanism(
            model,
            tokenizer,
            messages,
            spec=spec,
            claim_token_idx=position_token_idx,
            donor_vec=None,
            action_anchor=_action_anchor(traj),
        )

    out = greedy_generate_with_hook(
        model,
        tokenizer,
        messages,
        spec=spec,
        max_new_tokens=max_new_tokens,
        live_position_resolver=live_resolver,
        use_cache=use_cache,
    )
    if mechanism_obs:
        out["mechanism_obs"] = mechanism_obs
    return out


def evaluate_steering_on_holdout(
    *,
    model: Any,
    tokenizer: Any,
    eval_pm_trajectory_ids: list[str],
    steering_controls: dict[str, np.ndarray],
    layer: int,
    position: str,
    alphas: list[float] | None = None,
    max_new_tokens: int = 512,
    steering_direction: str = "subtract",
    steering_v2: bool = False,
    steering_v3: bool = False,
    limit: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    resume: bool = True,
    seed: int = 42,
    use_cache: bool = False,
) -> dict[str, Any]:
    v3_cfg = line_e_v3_config(position=position) if steering_v3 else None
    if steering_v3:
        alphas = list(alphas if alphas is not None else LINE_E_V3_ALPHA_SWEEP)
        layer = v3_cfg.layer
        position = v3_cfg.position
        use_direction = v3_cfg.steering_direction
        use_apply = v3_cfg.steering_apply
        control_ids = v3_cfg.control_ids
        use_live_resolver = v3_cfg.use_live_position_resolver
        baseline_control = v3_cfg.baseline_control
    else:
        alphas = list(alphas if alphas is not None else LINE_E_ALPHA_SWEEP)
        use_direction = STEERING_DIRECTION_V2 if steering_v2 else steering_direction
        use_apply = STEERING_APPLY_V2 if steering_v2 else "anchor_once"
        control_ids = STEERING_CONTROLS_V2 if steering_v2 else STEERING_CONTROLS
        use_live_resolver = False
        baseline_control = "paired_caa" if steering_v2 else "caa_true"
    rows_index = load_trajectory_index()
    eval_ids = sorted(eval_pm_trajectory_ids)
    if steering_v3:
        eval_ids = filter_v3_eval_trajectories(eval_ids, rows_index, layer=layer, position=position)
    if limit is not None:
        eval_ids = eval_ids[:limit]
    if num_shards > 1:
        eval_ids = [tid for i, tid in enumerate(eval_ids) if i % num_shards == shard_index]

    rows_path = eval_rows_path(
        shard_index=shard_index,
        num_shards=num_shards,
        steering_v2=steering_v2,
        steering_v3=steering_v3,
        use_cache=use_cache,
    )
    checkpoint_path = eval_checkpoint_path(
        shard_index=shard_index,
        num_shards=num_shards,
        steering_v2=steering_v2,
        steering_v3=steering_v3,
        use_cache=use_cache,
    )
    P3_LINE_E_DIR.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str, float]] = set()
    if resume and rows_path.is_file():
        detail_rows = load_jsonl(rows_path)
        done_keys = {_row_key(r) for r in detail_rows}
        print(f"[line_e] resume {len(done_keys)} rows from {rows_path.name}", flush=True)
    if use_cache:
        print("[line_e] use_cache=True (KV-cache decode; separate jsonl _uc)", flush=True)
        _debug_log("H3", "line_e_eval.py:resume", "loaded prior rows", {"n_rows": len(done_keys), "path": str(rows_path)})

    baseline_cache = _rebuild_baseline_cache(detail_rows)
    t0 = time.time()
    n_skipped = 0
    n_new = 0

    for ti, tid in enumerate(eval_ids):
        traj = rows_index.get(tid)
        if not traj:
            continue
        npz_path = activation_path(tid, "original")
        if not npz_path.is_file():
            continue
        pm_npz = load_activation_npz(npz_path)
        track = _trajectory_track(traj)
        messages = messages_for_condition(traj, "original", track=track)
        if not messages:
            continue
        tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
        pos_idx = _live_token_idx(pm_npz, position, int(tok["prompt_token_count"]))
        if pos_idx < 0:
            continue

        traj_t0 = time.time()
        print(f"[line_e] {ti+1}/{len(eval_ids)} {tid} (pos={pos_idx})", flush=True)

        paired = None
        traj_controls: dict[str, np.ndarray] = dict(steering_controls)
        paired_full = None
        donor_restore_vecs = None
        slot_restore_vecs = None
        anchor_restore_vecs = None
        commitment_stratum = trajectory_commitment_stratum(traj)
        if steering_v3 or steering_v2:
            bundle_fn = build_trajectory_steering_bundle if steering_v3 else paired_activation_vectors
            paired = (
                bundle_fn(tid, layer=layer, position=position, seed=seed)
                if steering_v3
                else paired_activation_vectors(tid, layer=layer, position=position)
            )
            if paired.get("error"):
                print(f"[line_e] skip {tid}: paired activation error {paired.get('error')}", flush=True)
                continue
            if steering_v3:
                traj_controls = {
                    "paired_caa": paired["mitigation_vector"],
                    **paired["controls"],
                }
                donor_restore_vecs = paired["donor_restore"]
                slot_restore_vecs = paired.get("slot_consistent_restore")
                anchor_restore_vecs = paired.get("anchor_aligned_restore")
                commitment_stratum = str(paired.get("commitment_stratum") or commitment_stratum)
            else:
                traj_controls = {
                    "paired_caa": paired["paired_caa_vector"],
                    **build_paired_steering_controls(paired["paired_caa_vector"], seed=seed + hash(tid) % 10000),
                }
                paired_full = {"pm_vec": paired["pm_vec"], "clean_vec": paired["clean_vec"]}
        if tid not in baseline_cache:
            base_key = (tid, baseline_control, 0.0)
            if base_key in done_keys:
                base_rows = [r for r in detail_rows if _row_key(r) == base_key]
                if base_rows:
                    br = base_rows[0]
                    baseline_cache[tid] = {
                        "wrong_owner": bool(br.get("baseline_wrong_owner")),
                        "product_id_correct": bool(br.get("baseline_product_id_correct")),
                        "selected_product_id": br.get("baseline_selected_product_id"),
                        "action_anchor": br.get("action_anchor"),
                        "product_id_missing": br.get("baseline_selected_product_id") is None,
                    }
            else:
                base_vec = _steering_vec(
                    traj_controls, baseline_control, fallback=steering_controls, fallback_key="caa_true"
                )
                out0 = _generate_with_steering(
                    model,
                    tokenizer,
                    traj,
                    layer=layer,
                    position=position,
                    position_token_idx=pos_idx,
                    steering_vec=base_vec,
                    control_id="no_intervention",
                    alpha=0.0,
                    max_new_tokens=max_new_tokens,
                    steering_direction=use_direction,
                    steering_apply=use_apply,
                    use_live_position_resolver=use_live_resolver,
                    use_cache=use_cache,
                )
                beh0 = behavioral_outcome(out0.get("text") or "", traj)
                baseline_cache[tid] = beh0
                baseline_cache[tid]["text"] = out0.get("text") or ""
                pid_base = beh0.get("selected_product_id")
                base_row = {
                    "trajectory_id": tid,
                    "track": track,
                    "control_id": baseline_control,
                    "alpha": 0.0,
                    "steering_direction": use_direction,
                    "steering_apply": use_apply,
                    "steering_v2": steering_v2,
                    "steering_v3": steering_v3,
                    "protocol_version": v3_cfg.protocol_version if v3_cfg else None,
                    "layer": layer,
                    "position": position,
                    "patched": False,
                    "use_cache": bool(out0.get("use_cache")),
                    "baseline_wrong_owner": beh0["wrong_owner"],
                    "baseline_product_id_correct": beh0["product_id_correct"],
                    "steered_wrong_owner": beh0["wrong_owner"],
                    "steered_product_id_correct": beh0["product_id_correct"],
                    "baseline_selected_product_id": pid_base,
                    "steered_selected_product_id": pid_base,
                    "action_anchor": beh0["action_anchor"],
                    "product_id_diff": False,
                    "wrong_owner_reduced": False,
                    "product_id_correct_gained": False,
                    "clean_trajectory_id": (paired or {}).get("clean_trajectory_id"),
                    "paired_norm": (paired or {}).get("paired_norm"),
                }
                detail_rows.append(base_row)
                append_jsonl(rows_path, base_row)
                done_keys.add(base_key)
                n_new += 1
                _debug_log("H1", "line_e_eval.py:baseline", "baseline row flushed", {"tid": tid, "use_cache": base_row["use_cache"]})

        base = baseline_cache[tid]
        traj_new = 0
        for control_id in control_ids:
            for alpha in alphas:
                if alpha == 0.0 and control_id != baseline_control:
                    continue
                key = (tid, control_id, float(alpha))
                if key in done_keys:
                    n_skipped += 1
                    continue
                capture_obs = False
                if control_id == "slot_consistent_restore" and not slot_restore_vecs:
                    continue
                if control_id == "anchor_aligned_restore" and not anchor_restore_vecs:
                    continue
                capture_obs = steering_v3 and float(alpha) in LINE_E_MECHANISM_OBS_ALPHAS and control_id in (
                    "donor_restore",
                    "slot_consistent_restore",
                    "anchor_aligned_restore",
                    "paired_caa",
                )
                if control_id == "donor_restore":
                    out = _generate_with_steering(
                        model,
                        tokenizer,
                        traj,
                        layer=layer,
                        position=position,
                        position_token_idx=pos_idx,
                        steering_vec=traj_controls.get("paired_caa", steering_controls.get("caa_true")),
                        control_id=control_id,
                        alpha=alpha,
                        max_new_tokens=max_new_tokens,
                        steering_direction=use_direction,
                        steering_apply=use_apply,
                        donor_restore=donor_restore_vecs,
                        use_live_position_resolver=use_live_resolver,
                        capture_mechanism_obs=capture_obs,
                        use_cache=use_cache,
                    )
                elif control_id == "slot_consistent_restore":
                    out = _generate_with_steering(
                        model,
                        tokenizer,
                        traj,
                        layer=layer,
                        position=position,
                        position_token_idx=pos_idx,
                        steering_vec=traj_controls.get("paired_caa", steering_controls.get("caa_true")),
                        control_id=control_id,
                        alpha=alpha,
                        max_new_tokens=max_new_tokens,
                        steering_direction=use_direction,
                        steering_apply=use_apply,
                        slot_consistent_restore=slot_restore_vecs,
                        use_live_position_resolver=use_live_resolver,
                        capture_mechanism_obs=capture_obs,
                        use_cache=use_cache,
                    )
                elif control_id == "anchor_aligned_restore":
                    out = _generate_with_steering(
                        model,
                        tokenizer,
                        traj,
                        layer=layer,
                        position=position,
                        position_token_idx=pos_idx,
                        steering_vec=traj_controls.get("paired_caa", steering_controls.get("caa_true")),
                        control_id=control_id,
                        alpha=alpha,
                        max_new_tokens=max_new_tokens,
                        steering_direction=use_direction,
                        steering_apply=use_apply,
                        anchor_aligned_restore=anchor_restore_vecs,
                        use_live_position_resolver=use_live_resolver,
                        capture_mechanism_obs=capture_obs,
                        use_cache=use_cache,
                    )
                elif control_id == "paired_full":
                    out = _generate_with_steering(
                        model,
                        tokenizer,
                        traj,
                        layer=layer,
                        position=position,
                        position_token_idx=pos_idx,
                        steering_vec=traj_controls.get("paired_caa", steering_controls.get("caa_true")),
                        control_id=control_id,
                        alpha=alpha,
                        max_new_tokens=max_new_tokens,
                        steering_direction=use_direction,
                        steering_apply=use_apply,
                        paired_full=paired_full,
                        use_live_position_resolver=use_live_resolver,
                        use_cache=use_cache,
                    )
                else:
                    vec = traj_controls.get(control_id)
                    if vec is None:
                        continue
                    out = _generate_with_steering(
                        model,
                        tokenizer,
                        traj,
                        layer=layer,
                        position=position,
                        position_token_idx=pos_idx,
                        steering_vec=vec,
                        control_id=control_id if control_id != "paired_caa" else "paired_caa",
                        alpha=alpha,
                        max_new_tokens=max_new_tokens,
                        steering_direction=use_direction,
                        steering_apply=use_apply,
                        use_live_position_resolver=use_live_resolver,
                        capture_mechanism_obs=capture_obs if control_id == "paired_caa" else False,
                        use_cache=use_cache,
                    )
                beh = behavioral_outcome(out.get("text") or "", traj)
                mech = out.get("mechanism_obs") or {}
                pid_base = base.get("selected_product_id")
                pid_steer = beh.get("selected_product_id")
                row = {
                    "trajectory_id": tid,
                    "track": track,
                    "control_id": control_id,
                    "alpha": float(alpha),
                    "steering_direction": use_direction,
                    "steering_apply": use_apply,
                    "steering_v2": steering_v2,
                    "steering_v3": steering_v3,
                    "protocol_version": v3_cfg.protocol_version if v3_cfg else None,
                    "layer": layer,
                    "position": position,
                    "patched": bool(out.get("patched")),
                    "use_cache": bool(out.get("use_cache")),
                    "baseline_wrong_owner": base["wrong_owner"],
                    "baseline_product_id_correct": base["product_id_correct"],
                    "steered_wrong_owner": beh["wrong_owner"],
                    "steered_product_id_correct": beh["product_id_correct"],
                    "baseline_selected_product_id": pid_base,
                    "steered_selected_product_id": pid_steer,
                    "action_anchor": base["action_anchor"],
                    "product_id_diff": bool(pid_base and pid_steer and pid_base != pid_steer),
                    "wrong_owner_reduced": bool(base["wrong_owner"] and not beh["wrong_owner"]),
                    "product_id_correct_gained": bool(not base["product_id_correct"] and beh["product_id_correct"]),
                    "clean_trajectory_id": (paired or {}).get("clean_trajectory_id"),
                    "paired_norm": (paired or {}).get("paired_norm"),
                    "commitment_stratum": commitment_stratum,
                    "vector_transfer": (paired or {}).get("vector_transfer"),
                    "slot_consistent_donor_id": (slot_restore_vecs or {}).get("donor_trajectory_id"),
                    "anchor_aligned_donor_id": (anchor_restore_vecs or {}).get("donor_trajectory_id"),
                    "hidden_cosine_to_donor_after": mech.get("hidden_cosine_to_donor_after"),
                    "hidden_cosine_to_donor_delta": mech.get("hidden_cosine_to_donor_delta"),
                    "anchor_logit_rank_mean": mech.get("anchor_logit_rank_mean"),
                }
                detail_rows.append(row)
                append_jsonl(rows_path, row)
                done_keys.add(key)
                n_new += 1
                traj_new += 1
                print(
                    f"[line_e] +row {control_id} a={alpha} patched={row.get('patched')} "
                    f"pid_diff={row.get('product_id_diff')} ({ti+1}/{len(eval_ids)} {tid[:20]}…)",
                    flush=True,
                )
                _debug_log(
                    "H2",
                    "line_e_eval.py:append",
                    "row flushed",
                    {"tid": tid, "control_id": control_id, "alpha": alpha, "use_cache": row["use_cache"]},
                )

        elapsed_traj = round(time.time() - traj_t0, 1)
        print(
            f"[line_e] done {ti+1}/{len(eval_ids)} {tid} new_rows={traj_new} elapsed={elapsed_traj}s",
            flush=True,
        )
        curves_partial = summarize_steering_curves(detail_rows, eval_n=len(baseline_cache))
        write_json(
            checkpoint_path,
            {
                "schema_version": (
                    "ccer_line_e_steering_checkpoint_v3"
                    if steering_v3
                    else "ccer_line_e_steering_checkpoint_v2"
                    if steering_v2
                    else "ccer_line_e_steering_checkpoint_v1"
                ),
                "steering_v2": steering_v2,
                "steering_v3": steering_v3,
                "protocol_version": v3_cfg.protocol_version if v3_cfg else None,
                "steering_apply": use_apply,
                "steering_direction": use_direction,
                "layer": layer,
                "position": position,
                "shard_index": shard_index,
                "num_shards": num_shards,
                "n_eval_trajectories": len(baseline_cache),
                "n_detail_rows": len(detail_rows),
                "n_skipped_resume": n_skipped,
                "n_new_this_run": n_new,
                "elapsed_s": round(time.time() - t0, 1),
                "curves": curves_partial,
                "baseline_summary": summarize_baseline(baseline_cache),
                "rows_path": str(rows_path),
            },
        )

    curves = summarize_steering_curves(detail_rows, eval_n=len(baseline_cache))
    strata_curves = summarize_steering_curves_by_stratum(detail_rows) if steering_v3 else {}
    return {
        "schema_version": (
            "ccer_line_e_steering_v4_ultimate"
            if steering_v3
            else "ccer_line_e_steering_v2"
            if steering_v2
            else "ccer_line_e_steering_v1"
        ),
        "steering_v2": steering_v2,
        "steering_v3": steering_v3,
        "protocol_version": v3_cfg.protocol_version if v3_cfg else None,
        "steering_apply": use_apply,
        "layer": layer,
        "position": position,
        "steering_direction": use_direction,
        "alphas": alphas,
        "n_eval_trajectories": len(baseline_cache),
        "n_detail_rows": len(detail_rows),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "elapsed_s": round(time.time() - t0, 1),
        "n_skipped_resume": n_skipped,
        "n_new_this_run": n_new,
        "rows_path": str(rows_path),
        "checkpoint_path": str(checkpoint_path),
        "use_cache": use_cache,
        "curves": curves,
        "curves_by_stratum": strata_curves,
        "detail_rows": detail_rows,
        "baseline_summary": summarize_baseline(baseline_cache),
    }


def summarize_baseline(baseline_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n = len(baseline_cache)
    if n == 0:
        return {"n": 0}
    wrong = sum(1 for v in baseline_cache.values() if v.get("wrong_owner"))
    correct = sum(1 for v in baseline_cache.values() if v.get("product_id_correct"))
    missing = sum(1 for v in baseline_cache.values() if v.get("product_id_missing"))
    return {
        "n": n,
        "wrong_owner_k": wrong,
        "wrong_owner_rate": wrong / n,
        "wrong_owner_ci95": wilson_ci(wrong, n),
        "product_id_correct_k": correct,
        "product_id_correct_rate": correct / n,
        "product_id_correct_ci95": wilson_ci(correct, n),
        "product_id_missing_k": missing,
        "product_id_missing_rate": missing / n,
    }


def summarize_steering_curves(detail_rows: list[dict[str, Any]], *, eval_n: int) -> dict[str, Any]:
    out: dict[str, Any] = {"by_control": {}}
    if not detail_rows:
        return out
    controls = sorted({str(r["control_id"]) for r in detail_rows})
    alphas = sorted({float(r["alpha"]) for r in detail_rows})
    for control_id in controls:
        points: list[dict[str, Any]] = []
        for alpha in alphas:
            subset = [r for r in detail_rows if r["control_id"] == control_id and float(r["alpha"]) == alpha]
            n = len(subset)
            if n == 0:
                continue
            wrong_k = sum(1 for r in subset if r.get("steered_wrong_owner"))
            correct_k = sum(1 for r in subset if r.get("steered_product_id_correct"))
            reduced_k = sum(1 for r in subset if r.get("wrong_owner_reduced"))
            gained_k = sum(1 for r in subset if r.get("product_id_correct_gained"))
            baseline_wrong = sum(1 for r in subset if r.get("baseline_wrong_owner"))
            only_reduced = sum(
                1 for r in subset if r.get("baseline_wrong_owner") and not r.get("steered_wrong_owner")
            )
            only_worsened = sum(
                1 for r in subset if not r.get("baseline_wrong_owner") and r.get("steered_wrong_owner")
            )
            pid_diff_k = sum(1 for r in subset if r.get("product_id_diff"))
            points.append(
                {
                    "alpha": alpha,
                    "n": n,
                    "wrong_owner_k": wrong_k,
                    "wrong_owner_rate": wrong_k / n,
                    "wrong_owner_ci95": wilson_ci(wrong_k, n),
                    "product_id_correct_k": correct_k,
                    "product_id_correct_rate": correct_k / n,
                    "product_id_correct_ci95": wilson_ci(correct_k, n),
                    "product_id_diff_k": pid_diff_k,
                    "product_id_diff_rate": pid_diff_k / n,
                    "product_id_diff_ci95": wilson_ci(pid_diff_k, n),
                    "wrong_owner_reduced_k": reduced_k,
                    "wrong_owner_reduced_rate": reduced_k / n,
                    "wrong_owner_reduced_ci95": wilson_ci(reduced_k, n),
                    "product_id_correct_gained_k": gained_k,
                    "product_id_correct_gained_rate": gained_k / n,
                    "product_id_correct_gained_ci95": wilson_ci(gained_k, n),
                    "paired_baseline_wrong_k": baseline_wrong,
                    "paired_only_reduced_k": only_reduced,
                    "paired_only_worsened_k": only_worsened,
                    "mcnemar_reduced_vs_worsened_p": mcnemar_exact_p(only_reduced, only_worsened),
                }
            )
        out["by_control"][control_id] = points
    out["eval_n"] = eval_n
    return out


def summarize_steering_curves_by_stratum(detail_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    strata = sorted({str(r.get("commitment_stratum") or "unknown") for r in detail_rows})
    for stratum in strata:
        subset = [r for r in detail_rows if str(r.get("commitment_stratum") or "unknown") == stratum]
        out[stratum] = summarize_steering_curves(subset, eval_n=len({r["trajectory_id"] for r in subset}))
    return out


def judge_line_e_ultimate_success(
    curves: dict[str, Any],
    curves_by_stratum: dict[str, Any] | None = None,
    *,
    alpha_min: float = 0.25,
    alpha_max: float = 1.25,
) -> dict[str, Any]:
    """protocol note Line E + mechanism-layer gates for v4 ultimate."""
    primary_controls = ("slot_consistent_restore", "donor_restore", "paired_caa")
    judgments = {
        cid: judge_line_e_success(curves, alpha_min=alpha_min, alpha_max=alpha_max, true_control=cid)
        for cid in primary_controls
    }
    best_control = min(
        primary_controls,
        key=lambda cid: float((judgments[cid].get("best_true_wrong_owner_rate") or 1.0)),
    )
    best = judgments[best_control]
    mismatch = (curves_by_stratum or {}).get("commitment_mismatch") or {}
    mismatch_pts = (mismatch.get("by_control") or {}).get("slot_consistent_restore") or []
    in_band = [p for p in mismatch_pts if alpha_min <= float(p["alpha"]) <= alpha_max and float(p["alpha"]) > 0]
    mismatch_best = min(in_band, key=lambda p: float(p.get("wrong_owner_rate") or 1.0)) if in_band else None
    return {
        **best,
        "best_primary_control": best_control,
        "judgments_by_control": judgments,
        "mismatch_stratum_best": mismatch_best,
        "verdict": best.get("verdict"),
        "protocol": LINE_E_V3_PROTOCOL_VERSION,
    }


def judge_line_e_success(
    curves: dict[str, Any],
    *,
    alpha_min: float = 0.25,
    alpha_max: float = 1.25,
    true_control: str = "caa_true",
) -> dict[str, Any]:
    """Honest success/fail against protocol note Line E criteria."""
    by_control = curves.get("by_control") or {}
    true_pts = by_control.get(true_control) or by_control.get("caa_true") or []
    rand_pts = by_control.get("caa_random") or []
    in_band = [p for p in true_pts if alpha_min <= float(p["alpha"]) <= alpha_max and float(p["alpha"]) > 0]
    if not in_band:
        return {"verdict": "insufficient_data", "reason": "no alpha points in evaluation band"}
    best_true = min(in_band, key=lambda p: float(p.get("wrong_owner_rate") or 1.0))
    alpha0_true = next((p for p in true_pts if float(p["alpha"]) == 0.0), None)
    alpha0_rand = next((p for p in rand_pts if float(p["alpha"]) == 0.0), None)
    baseline_wrong = float((alpha0_true or {}).get("wrong_owner_rate") or 1.0)
    best_wrong = float(best_true.get("wrong_owner_rate") or 1.0)
    reduction = baseline_wrong - best_wrong
    rand_in_band = [p for p in rand_pts if alpha_min <= float(p["alpha"]) <= alpha_max and float(p["alpha"]) > 0]
    orth_pts = by_control.get("caa_orthogonal") or []
    orth_in_band = [p for p in orth_pts if alpha_min <= float(p["alpha"]) <= alpha_max and float(p["alpha"]) > 0]
    best_rand_wrong = min((float(p.get("wrong_owner_rate") or 1.0) for p in rand_in_band), default=1.0)
    best_orth_wrong = min((float(p.get("wrong_owner_rate") or 1.0) for p in orth_in_band), default=1.0)
    specificity_random = best_wrong < best_rand_wrong - 0.03
    specificity_orthogonal = best_wrong < best_orth_wrong - 0.03
    specificity = specificity_random and specificity_orthogonal
    meaningful = reduction >= 0.05 and best_wrong < baseline_wrong
    if meaningful and specificity:
        verdict = "success"
    elif meaningful:
        verdict = "partial_no_specificity"
    else:
        verdict = "null"
    return {
        "verdict": verdict,
        "baseline_wrong_owner_rate": baseline_wrong,
        "best_true_wrong_owner_rate": best_wrong,
        "best_true_alpha": best_true.get("alpha"),
        "wrong_owner_reduction": reduction,
        "best_random_wrong_owner_rate_in_band": best_rand_wrong,
        "best_orthogonal_wrong_owner_rate_in_band": best_orth_wrong,
        "specificity_pass": specificity,
        "specificity_random_pass": specificity_random,
        "specificity_orthogonal_pass": specificity_orthogonal,
        "meaningful_reduction_pass": meaningful,
        "alpha_band": [alpha_min, alpha_max],
    }


def merge_line_e_shards(shard_results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(shard_results) == 1:
        return shard_results[0]
    details = [r for s in shard_results for r in (s.get("detail_rows") or [])]
    eval_n = sum(int(s.get("n_eval_trajectories") or 0) for s in shard_results)
    base = dict(shard_results[0])
    base["detail_rows"] = details
    base["n_eval_trajectories"] = eval_n
    base["curves"] = summarize_steering_curves(details, eval_n=eval_n)
    base["num_shards_merged"] = len(shard_results)
    base["elapsed_s"] = round(sum(float(s.get("elapsed_s") or 0) for s in shard_results), 1)
    return base
