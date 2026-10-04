"""Line E v5 extreme in-generation evaluation."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ccer.io_utils import append_jsonl, load_jsonl, write_json
from ccer.mechanism.activation_store import activation_path, load_activation_npz
from ccer.mechanism.iia_roi_probe import _live_token_idx
from ccer.mechanism.interchange import greedy_generate_with_hook
from ccer.mechanism.line_e_anchor_logit import build_anchor_logit_bias
from ccer.mechanism.line_e_donor import build_paired_cross_bundle, trajectory_commitment_stratum
from ccer.mechanism.line_e_eval import (
    _rebuild_baseline_cache,
    _row_key,
    _trajectory_track,
    behavioral_outcome,
    summarize_steering_curves,
    summarize_steering_curves_by_stratum,
)
from ccer.mechanism.line_e_probe_direction import fit_probe_steering_direction
from ccer.mechanism.line_e_protocol import filter_v3_eval_trajectories
from ccer.mechanism.line_e_v5_arms import (
    build_v5_interchange_spec,
    build_v5_pca_roi,
    build_v5_probe_steer_spec,
)
from ccer.mechanism.line_e_v5_protocol import LineEV5Config, line_e_v5_config
from ccer.mechanism.pair_select import load_trajectory_index, messages_for_condition
from ccer.paths import P3_DIR
from ccer.replay.hf_forward import tokenize_ccer_messages
from ccer.replay.live_position import make_live_position_resolver

P3_LINE_E_DIR = P3_DIR / "line_e"
V5_ROWS_STEM = "steering_eval_rows_v5_extreme"
V5_CHECKPOINT_STEM = "steering_eval_checkpoint_v5_extreme"


def v5_rows_path(
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    use_cache: bool = False,
) -> Path:
    stem = f"{V5_ROWS_STEM}_uc" if use_cache else V5_ROWS_STEM
    if num_shards <= 1:
        return P3_LINE_E_DIR / f"{stem}.jsonl"
    return P3_LINE_E_DIR / f"{stem}_s{shard_index}.jsonl"


def v5_checkpoint_path(
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    use_cache: bool = False,
) -> Path:
    stem = f"{V5_CHECKPOINT_STEM}_uc" if use_cache else V5_CHECKPOINT_STEM
    if num_shards <= 1:
        return P3_LINE_E_DIR / f"{stem}.json"
    return P3_LINE_E_DIR / f"{stem}_s{shard_index}.json"


def behavioral_outcome_v5(text: str, traj: dict[str, Any]) -> dict[str, Any]:
    beh = behavioral_outcome(text, traj)
    beh["anchor_aligned"] = bool(beh.get("product_id_correct"))
    return beh


def _build_v5_row(
    *,
    tid: str,
    track: str,
    control_id: str,
    alpha: float,
    cfg: LineEV5Config,
    base: dict[str, Any],
    beh: dict[str, Any],
    out: dict[str, Any],
    paired_meta: dict[str, Any],
    steering_v5: bool = True,
    use_cache: bool = False,
    wrong_owner_impl: str | None = None,
    arm_mode: str | None = None,
) -> dict[str, Any]:
    pid_base = base.get("selected_product_id")
    pid_steer = beh.get("selected_product_id")
    parseable = pid_steer is not None
    return {
        "trajectory_id": tid,
        "track": track,
        "control_id": control_id,
        "alpha": float(alpha),
        "steering_apply": cfg.steering_apply,
        "steering_v5": steering_v5,
        "protocol_version": cfg.protocol_version,
        "layer": cfg.layer,
        "position": cfg.position,
        "arm_mode": arm_mode,
        "patched": bool(out.get("patched")),
        "patch_tier": out.get("patch_tier"),
        "use_cache": bool(out.get("use_cache", use_cache)),
        "baseline_wrong_owner": base["wrong_owner"],
        "baseline_product_id_correct": base["product_id_correct"],
        "steered_wrong_owner": beh["wrong_owner"],
        "steered_product_id_correct": beh["product_id_correct"],
        "baseline_selected_product_id": pid_base,
        "steered_selected_product_id": pid_steer,
        "action_anchor": base.get("action_anchor"),
        "product_id_diff": bool(pid_base and pid_steer and pid_base != pid_steer),
        "wrong_owner_reduced": bool(base["wrong_owner"] and not beh["wrong_owner"]),
        "wrong_owner_reduced_parseable": bool(
            base["wrong_owner"] and not beh["wrong_owner"] and parseable
        ),
        "product_id_correct_gained": bool(not base["product_id_correct"] and beh["product_id_correct"]),
        "anchor_aligned": bool(beh.get("anchor_aligned")),
        "anchor_hit": bool(beh.get("anchor_aligned")),
        "steered_pid_parseable": parseable,
        "clean_trajectory_id": paired_meta.get("clean_trajectory_id"),
        "paired_norm": paired_meta.get("paired_norm"),
        "wrong_owner_impl": wrong_owner_impl,
        "commitment_stratum": paired_meta.get("commitment_stratum"),
        "logit_bias_n_tokens": out.get("logit_bias_n_tokens"),
    }


def evaluate_v5_on_holdout(
    model: Any,
    tokenizer: Any,
    eval_pm_trajectory_ids: list[str],
    *,
    cfg: LineEV5Config | None = None,
    max_new_tokens: int = 512,
    limit: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    resume: bool = True,
    seed: int = 42,
    use_cache: bool = False,
) -> dict[str, Any]:
    cfg = cfg or line_e_v5_config()
    rows_index = load_trajectory_index()
    eval_ids = filter_v3_eval_trajectories(
        sorted(eval_pm_trajectory_ids),
        rows_index,
        layer=cfg.layer,
        position=cfg.position,
    )
    if limit is not None:
        eval_ids = eval_ids[:limit]
    if num_shards > 1:
        eval_ids = [tid for i, tid in enumerate(eval_ids) if i % num_shards == shard_index]

    rows_path = v5_rows_path(shard_index=shard_index, num_shards=num_shards, use_cache=use_cache)
    checkpoint_path = v5_checkpoint_path(shard_index=shard_index, num_shards=num_shards, use_cache=use_cache)
    P3_LINE_E_DIR.mkdir(parents=True, exist_ok=True)

    roi = build_v5_pca_roi(cfg)
    if roi.get("error"):
        return {"error": f"pca_fit_failed:{roi.get('error')}", "roi": roi}

    probe_fit = fit_probe_steering_direction(layer=cfg.layer, position=cfg.position, seed=seed)
    if probe_fit.get("error"):
        return {"error": f"probe_fit_failed:{probe_fit.get('error')}", "probe_fit": probe_fit}
    probe_direction = np.asarray(probe_fit["direction"], dtype=np.float32)

    detail_rows: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str, float]] = set()
    if resume and rows_path.is_file():
        detail_rows = load_jsonl(rows_path)
        done_keys = {_row_key(r) for r in detail_rows}
        print(f"[line_e_v5] resume {len(done_keys)} rows from {rows_path.name}", flush=True)
    if use_cache:
        print("[line_e_v5] use_cache=True (KV-cache decode)", flush=True)

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
        pos_idx = _live_token_idx(pm_npz, cfg.position, int(tok["prompt_token_count"]))
        if pos_idx < 0:
            continue

        live_resolver = make_live_position_resolver(tokenizer, cfg.position)
        commitment_stratum = trajectory_commitment_stratum(traj)
        paired_bundle = build_paired_cross_bundle(tid, layer=cfg.layer, position=cfg.position)
        if paired_bundle.get("error"):
            print(f"[line_e_v5] skip {tid}: {paired_bundle.get('error')}", flush=True)
            continue
        paired_meta: dict[str, Any] = {
            "clean_trajectory_id": paired_bundle.get("clean_trajectory_id"),
            "paired_norm": paired_bundle.get("paired_norm"),
            "commitment_stratum": commitment_stratum,
        }

        traj_t0 = time.time()
        print(f"[line_e_v5] {ti+1}/{len(eval_ids)} {tid} (pos={pos_idx})", flush=True)

        if tid not in baseline_cache:
            base_key = (tid, "no_intervention", 0.0)
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
                out0 = greedy_generate_with_hook(
                    model,
                    tokenizer,
                    messages,
                    spec=None,
                    max_new_tokens=max_new_tokens,
                    use_cache=use_cache,
                    live_position_resolver=live_resolver,
                )
                beh0 = behavioral_outcome_v5(out0.get("text") or "", traj)
                baseline_cache[tid] = beh0
                base_row = _build_v5_row(
                    tid=tid,
                    track=track,
                    control_id="no_intervention",
                    alpha=0.0,
                    cfg=cfg,
                    base=beh0,
                    beh=beh0,
                    out=out0,
                    paired_meta=paired_meta,
                    use_cache=use_cache,
                    arm_mode="baseline",
                )
                detail_rows.append(base_row)
                append_jsonl(rows_path, base_row)
                done_keys.add(base_key)
                n_new += 1

        base = baseline_cache[tid]
        anchor = base.get("action_anchor")
        traj_new = 0

        for control_label, control_id, mode, alphas in cfg.interchange_arms:
            for alpha in alphas:
                key = (tid, control_label, float(alpha))
                if key in done_keys:
                    n_skipped += 1
                    continue
                built = build_v5_interchange_spec(
                    pm_trajectory_id=tid,
                    control_label=control_label,
                    control_id=control_id,
                    mode=mode,
                    alpha=float(alpha),
                    roi=roi,
                    position_token_idx=pos_idx,
                    cfg=cfg,
                )
                if built.get("error"):
                    print(f"[line_e_v5] skip {tid}/{control_label}: {built.get('error')}", flush=True)
                    continue
                spec = built["spec"]
                out = greedy_generate_with_hook(
                    model,
                    tokenizer,
                    messages,
                    spec=spec,
                    max_new_tokens=max_new_tokens,
                    use_cache=use_cache,
                    live_position_resolver=live_resolver,
                )
                beh = behavioral_outcome_v5(out.get("text") or "", traj)
                row = _build_v5_row(
                    tid=tid,
                    track=track,
                    control_id=control_label,
                    alpha=float(alpha),
                    cfg=cfg,
                    base=base,
                    beh=beh,
                    out=out,
                    paired_meta=paired_meta,
                    use_cache=use_cache,
                    wrong_owner_impl=built.get("wrong_owner_impl"),
                    arm_mode=mode,
                )
                detail_rows.append(row)
                append_jsonl(rows_path, row)
                done_keys.add(key)
                n_new += 1
                traj_new += 1
                print(
                    f"[line_e_v5] +row {control_label} a={alpha} patched={row.get('patched')} "
                    f"anchor_hit={row.get('anchor_hit')} gained={row.get('product_id_correct_gained')} "
                    f"({ti+1}/{len(eval_ids)})",
                    flush=True,
                )

        for control_label, alphas in cfg.steer_arms:
            for alpha in alphas:
                key = (tid, control_label, float(alpha))
                if key in done_keys:
                    n_skipped += 1
                    continue
                if control_label == "probe_steer":
                    spec = build_v5_probe_steer_spec(
                        probe_direction=probe_direction,
                        position_token_idx=pos_idx,
                        alpha=float(alpha),
                        cfg=cfg,
                    )
                    out = greedy_generate_with_hook(
                        model,
                        tokenizer,
                        messages,
                        spec=spec,
                        max_new_tokens=max_new_tokens,
                        use_cache=use_cache,
                        live_position_resolver=live_resolver,
                    )
                elif control_label == "anchor_logit_bias":
                    bias = build_anchor_logit_bias(
                        tokenizer,
                        anchor,
                        alpha=float(alpha),
                        bias_per_token=cfg.anchor_bias_per_token,
                    )
                    out = greedy_generate_with_hook(
                        model,
                        tokenizer,
                        messages,
                        spec=None,
                        max_new_tokens=max_new_tokens,
                        use_cache=use_cache,
                        live_position_resolver=live_resolver,
                        logit_bias=bias,
                    )
                    out["logit_bias_n_tokens"] = len(bias)
                    out["patched"] = bool(bias)
                else:
                    continue

                beh = behavioral_outcome_v5(out.get("text") or "", traj)
                row = _build_v5_row(
                    tid=tid,
                    track=track,
                    control_id=control_label,
                    alpha=float(alpha),
                    cfg=cfg,
                    base=base,
                    beh=beh,
                    out=out,
                    paired_meta=paired_meta,
                    use_cache=use_cache,
                    arm_mode=control_label,
                )
                detail_rows.append(row)
                append_jsonl(rows_path, row)
                done_keys.add(key)
                n_new += 1
                traj_new += 1
                print(
                    f"[line_e_v5] +row {control_label} a={alpha} patched={row.get('patched')} "
                    f"anchor_hit={row.get('anchor_hit')} gained={row.get('product_id_correct_gained')} "
                    f"({ti+1}/{len(eval_ids)})",
                    flush=True,
                )

        elapsed_traj = round(time.time() - traj_t0, 1)
        print(
            f"[line_e_v5] done traj {tid[:24]}… +{traj_new} rows in {elapsed_traj}s "
            f"(skipped={n_skipped} cumulative_new={n_new})",
            flush=True,
        )

    curves = summarize_steering_curves(detail_rows, eval_n=len(eval_ids))
    curves_by_stratum = summarize_steering_curves_by_stratum(detail_rows)
    checkpoint = {
        "protocol_version": cfg.protocol_version,
        "layer": cfg.layer,
        "position": cfg.position,
        "n_eval_trajectories": len(eval_ids),
        "n_rows": len(detail_rows),
        "n_new": n_new,
        "n_skipped": n_skipped,
        "elapsed_s": round(time.time() - t0, 1),
        "rows_path": str(rows_path),
        "pca_roi": {k: roi.get(k) for k in ("primary_roi", "pca_fit")},
        "probe_fit": {k: probe_fit.get(k) for k in ("n_pm", "n_clean", "source", "norm")},
        "use_cache": use_cache,
    }
    write_json(checkpoint_path, checkpoint)

    return {
        "protocol_version": cfg.protocol_version,
        "steering_v5": True,
        "layer": cfg.layer,
        "position": cfg.position,
        "n_eval_trajectories": len(eval_ids),
        "n_rows": len(detail_rows),
        "n_new": n_new,
        "n_skipped": n_skipped,
        "elapsed_s": checkpoint["elapsed_s"],
        "rows_path": str(rows_path),
        "checkpoint_path": str(checkpoint_path),
        "curves": curves,
        "curves_by_stratum": curves_by_stratum,
        "pca_roi": checkpoint["pca_roi"],
        "probe_fit": checkpoint["probe_fit"],
        "use_cache": use_cache,
    }
