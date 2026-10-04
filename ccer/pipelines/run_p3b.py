"""P3b: bidirectional interchange + controls + IIA report (multi-position/layer)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from ccer.eval.iia_eval import aggregate_iia, evaluate_iia_row, side_effect_metrics
from ccer.io_utils import append_jsonl, load_json, load_jsonl, write_json, write_jsonl
from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.controls import ALPHA_SWEEP, CONTROL_IDS, build_control_specs
from ccer.mechanism.interchange import greedy_generate_with_hook
from ccer.mechanism.owner_pca import POSITION_CONTROL_POSITIONS
from ccer.mechanism.pair_select import cap_valid_pair_ids, cem_valid_pair_ids, load_trajectory_index, messages_for_condition
from ccer.paths import P3_BINDING_ROI, P3_DIR, P3_IIA_SUMMARY, P3_INTERCHANGE_ROWS, P3_U_OWNER_NPZ, REPORTS
from ccer.replay.answer_utils import answers_differ, extract_answer
from ccer.replay.hf_forward import dense_layer_indices, load_hf_model


def _output_text(traj: dict[str, Any]) -> str:
    return str((traj.get("metadata") or {}).get("final_answer") or "")


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("track") or "CAP"),
        str(row.get("cap_trajectory_id") or row.get("pm_trajectory_id") or ""),
        str(row.get("clean_trajectory_id") or ""),
        str(row.get("direction") or ""),
        str(row.get("control_id") or ""),
        float(row.get("alpha") or 1.0),
        int(row.get("layer") or -1),
        str(row.get("position") or ""),
    )


def _shard_rows_path(track: str, shard_index: int, num_shards: int, *, inject_site: str = "residual") -> Path:
    track_u = track.upper()
    suffix = "" if inject_site == "residual" else f"_{inject_site}"
    if num_shards <= 1:
        if inject_site == "residual":
            return P3_INTERCHANGE_ROWS
        return P3_DIR / f"interchange_rows_{track_u.lower()}{suffix}.jsonl"
    return P3_DIR / f"interchange_rows_{track_u.lower()}{suffix}_s{shard_index}.jsonl"


def _prepare_interchange_store(
    track: str,
    *,
    resume: bool,
    rows_path: Path,
    num_shards: int,
) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    track_u = track.upper()
    if not resume:
        if num_shards <= 1 and rows_path == P3_INTERCHANGE_ROWS:
            existing = load_jsonl(P3_INTERCHANGE_ROWS) if P3_INTERCHANGE_ROWS.is_file() else []
            kept = [r for r in existing if str(r.get("track") or "CAP").upper() != track_u]
            write_jsonl(P3_INTERCHANGE_ROWS, kept)
        elif rows_path.is_file():
            rows_path.unlink()
        return [], set()
    if rows_path.is_file():
        track_rows = load_jsonl(rows_path)
        return track_rows, {_row_key(r) for r in track_rows}
    return [], set()


def merge_shard_rows(track: str, *, num_shards: int) -> list[dict[str, Any]]:
    track_u = track.upper()
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    if num_shards <= 1:
        rows = load_jsonl(P3_INTERCHANGE_ROWS) if P3_INTERCHANGE_ROWS.is_file() else []
        return [r for r in rows if str(r.get("track") or "CAP").upper() == track_u]
    for shard in range(num_shards):
        path = _shard_rows_path(track_u, shard, num_shards)
        if not path.is_file():
            continue
        for row in load_jsonl(path):
            key = _row_key(row)
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def run_p3b(
    *,
    dry_run: bool = False,
    limit: int | None = None,
    alpha_sweep: bool = False,
    device_map: str = "auto",
    resume: bool = True,
    track: str = "CAP",
    positions: list[str] | None = None,
    layer_sweep: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    control_ids: list[str] | None = None,
    max_new_tokens: int = 512,
    merge_shards: bool = False,
    layer_override: int | None = None,
    position_override: str | None = None,
    inject_site: str = "residual",
) -> dict[str, Any]:
    if not P3_BINDING_ROI.is_file():
        raise FileNotFoundError(f"missing binding ROI: {P3_BINDING_ROI}")
    roi = load_json(P3_BINDING_ROI)
    roi = {**roi, "inject_site": inject_site}
    primary = roi.get("primary_roi") or {}
    base_layer = int(layer_override if layer_override is not None else primary.get("layer") or 0)
    base_position = str(position_override or primary.get("position") or "claim_onset")
    positions = positions or [base_position]
    u = np.array([], dtype=np.float32)
    if P3_U_OWNER_NPZ.is_file():
        data = np.load(P3_U_OWNER_NPZ, allow_pickle=True)
        u = data["U_pca"] if "U_pca" in data.files else np.array(roi.get("U_owner") or [], dtype=np.float32)
    elif roi.get("U_owner"):
        u = np.array(roi.get("U_owner") or [], dtype=np.float32)
    if u.size and u.ndim == 1:
        u = u.reshape(-1, 1)

    track_u = track.upper()
    if merge_shards:
        roi = load_json(P3_BINDING_ROI) if P3_BINDING_ROI.is_file() else {}
        primary = roi.get("primary_roi") or {}
        track_rows = merge_shard_rows(track_u, num_shards=num_shards)
        summary = aggregate_iia(track_rows)
        summary["schema_version"] = "ccer_iia_summary_v2"
        summary["n_rows"] = len(track_rows)
        summary["primary_roi"] = primary
        summary["track"] = track_u
        summary["merged_shards"] = num_shards
        write_json(P3_IIA_SUMMARY, summary)
        return summary

    rows_path = _shard_rows_path(track_u, shard_index, num_shards, inject_site=inject_site)
    selection = cap_valid_pair_ids(include_matched_clean=True) if track_u == "CAP" else cem_valid_pair_ids(include_matched_clean=True)
    cross_pairs = selection["pm_clean_cross_pairs"]
    if num_shards > 1:
        cross_pairs = [cp for i, cp in enumerate(cross_pairs) if i % num_shards == shard_index]
    if limit:
        cross_pairs = cross_pairs[:limit]

    pm_key = "cap_trajectory_id" if track_u == "CAP" else "pm_trajectory_id"
    swap_cond = "query_value_swap" if track_u == "CAP" else "rival_value_swap"
    interchange_rows, done_keys = _prepare_interchange_store(
        track_u,
        resume=resume,
        rows_path=rows_path,
        num_shards=num_shards,
    )

    model = tokenizer = None
    if not dry_run:
        model, tokenizer, n_layers = load_hf_model(device_map=device_map)
    else:
        n_layers = 64

    layers = [base_layer]
    if layer_sweep:
        layers = dense_layer_indices(n_layers, base_layer, radius=2)

    alphas = ALPHA_SWEEP if alpha_sweep else [1.0]
    controls = control_ids or CONTROL_IDS
    rows_index = load_trajectory_index()

    for ci, cp in enumerate(cross_pairs, 1):
        pm_tid = cp.get(pm_key) or cp.get("cap_trajectory_id")
        clean_tid = cp["clean_trajectory_id"]
        cap_traj = rows_index.get(str(pm_tid))
        clean_traj = rows_index.get(clean_tid)
        if not cap_traj or not clean_traj:
            continue

        if track_u == "CAP":
            cap_claim = next(
                (c for c in (cap_traj.get("claims") or []) if c.get("legacy_label") == "constraint_projection"),
                None,
            )
        else:
            cap_claim = next(
                (c for c in (cap_traj.get("claims") or []) if c.get("legacy_label") == "cross_object_merge"),
                None,
            )
        claim_value = str((cap_claim or {}).get("value") or "")
        cf_value = f"CF_{claim_value}"

        for position in positions:
            for layer in layers:
                cap_npz = load_activation_npz(activation_path(str(pm_tid), "original"))
                clean_npz = load_activation_npz(activation_path(clean_tid, "original"))
                cap_vec = get_vector(cap_npz, position=position, layer=layer)
                clean_vec = get_vector(clean_npz, position=position, layer=layer)
                if cap_vec is None or clean_vec is None:
                    continue

                cap_pos = int((cap_npz.get("positions") or {}).get(position) or -1)
                clean_pos = int((clean_npz.get("positions") or {}).get(position) or -1)

                directions = [
                    ("PM_to_clean", cap_traj, clean_traj, cap_vec, clean_vec, cap_pos, clean_vec),
                    ("clean_to_PM", clean_traj, cap_traj, clean_vec, cap_vec, clean_pos, cap_vec),
                ]

                for direction, recipient_traj, donor_traj, rec_vec, don_vec, pos_idx, wrong_donor in directions:
                    messages = messages_for_condition(recipient_traj, "original", track=track_u)
                    if not messages or pos_idx < 0:
                        continue
                    original_answer = _output_text(recipient_traj)
                    baseline = None

                    for control_id in controls:
                        for alpha in alphas:
                            if control_id in ("no_intervention", "self_donor") and alpha != 1.0:
                                continue
                            if control_id == "target_interchange" and alpha == 0.0:
                                continue
                            preview = {
                                pm_key: pm_tid,
                                "clean_trajectory_id": clean_tid,
                                "direction": direction,
                                "control_id": control_id,
                                "alpha": alpha,
                                "layer": layer,
                                "position": position,
                            }
                            if resume and _row_key(preview) in done_keys:
                                continue
                            print(
                                f"[P3b] {ci}/{len(cross_pairs)} {direction} L{layer} {position} {control_id} a={alpha}",
                                flush=True,
                            )
                            if dry_run:
                                continue

                            spec = build_control_specs(
                                control_id=control_id if control_id != "target_interchange" else "target_interchange",
                                roi={**roi, "U_owner": u.tolist() if u.size else roi.get("U_owner")},
                                donor_vec=don_vec,
                                recipient_vec=rec_vec,
                                position_token_idx=pos_idx,
                                alpha=alpha,
                                wrong_donor_vec=wrong_donor,
                                seed=hash((pm_tid, clean_tid, direction, control_id, layer, position)) % 2**31,
                            )
                            out = greedy_generate_with_hook(
                                model,
                                tokenizer,
                                messages,
                                spec=spec,
                                output_text=None,
                                max_new_tokens=max_new_tokens,
                            )
                            if baseline is None and control_id == "no_intervention":
                                baseline = out["text"]

                            iia = evaluate_iia_row(
                                direction=direction,
                                control_id=control_id,
                                generated_text=out["text"],
                                original_answer=original_answer,
                                claim_value=claim_value,
                                cf_value=cf_value,
                                anchor_value=claim_value,
                                track=track_u,
                                condition_id=swap_cond,
                            )
                            side = side_effect_metrics(
                                baseline_text=baseline or original_answer,
                                intervened_text=out["text"],
                                h_norm=float(np.linalg.norm(rec_vec)),
                                h_norm_baseline=float(np.linalg.norm(rec_vec)),
                            )
                            row = {
                                pm_key: pm_tid,
                                "cap_trajectory_id": pm_tid if track_u == "CAP" else None,
                                "pm_trajectory_id": pm_tid,
                                "clean_trajectory_id": clean_tid,
                                "track": track_u,
                                "direction": direction,
                                "control_id": control_id,
                                "alpha": alpha,
                                "layer": layer,
                                "position": position,
                                "inject_site": inject_site,
                                "patched": out.get("patched"),
                                "output_diff_vs_no_intervention": (
                                    answers_differ(out["text"], baseline or original_answer)
                                    if baseline and control_id != "no_intervention"
                                    else None
                                ),
                                **iia,
                                **side,
                            }
                            interchange_rows.append(row)
                            append_jsonl(rows_path, row)
                            done_keys.add(_row_key(row))

    track_rows = interchange_rows
    if num_shards > 1:
        track_rows = merge_shard_rows(track_u, num_shards=num_shards)
    summary = aggregate_iia(track_rows)
    summary["schema_version"] = "ccer_iia_summary_v2"
    summary["n_rows"] = len(track_rows)
    summary["primary_roi"] = primary
    summary["primary_roi_note"] = (
        "candidate_only: layer=48/prompt_end v1 run; see factorial_disentanglement_table in binding_roi.json"
        if primary.get("roi_status") != "frozen"
        else "frozen_after_position_control"
    )
    summary["alpha_sweep"] = alpha_sweep
    summary["positions_evaluated"] = positions
    summary["layers_evaluated"] = layers
    summary["track"] = track_u
    summary["shard_index"] = shard_index
    summary["num_shards"] = num_shards
    summary["controls_evaluated"] = controls
    summary["rows_path"] = str(rows_path)
    summary["inject_site"] = inject_site
    ti_rows = [r for r in track_rows if r.get("control_id") == "target_interchange"]
    summary["target_interchange_output_diff_rate"] = (
        sum(1 for r in ti_rows if r.get("output_diff_vs_no_intervention")) / len(ti_rows) if ti_rows else None
    )
    summary["by_position"] = _aggregate_by_field(track_rows, "position")
    summary["by_layer"] = _aggregate_by_field(track_rows, "layer")
    write_json(P3_IIA_SUMMARY, summary)

    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P3b IIA Report",
        "",
        f"- track: {track_u}",
        f"- n_rows: {len(track_rows)}",
        f"- primary ROI (artifact): layer={base_layer} position={base_position}",
        f"- roi_status: {primary.get('roi_status', 'candidate')}",
        f"- positions evaluated: {positions}",
        f"- layers evaluated: {layers}",
        "",
        "> CAP 80.4% Tier-1 follow proves harness works, not fine-grained binding mechanism.",
        "",
        "## Overall",
        json.dumps(summary.get("overall") or {}, indent=2),
        "",
        "## By position",
        json.dumps(summary.get("by_position") or {}, indent=2),
        "",
        "## By layer (sample)",
        json.dumps(dict(list((summary.get("by_layer") or {}).items())[:12]), indent=2),
    ]
    (REPORTS / "p3_iia_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _aggregate_by_field(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    buckets: dict[str, list[bool]] = {}
    for row in rows:
        if row.get("iia_success") is None:
            continue
        key = str(row.get(field) or "unknown")
        buckets.setdefault(key, []).append(bool(row["iia_success"]))
    return {k: {"iia_rate": sum(v) / len(v), "n": len(v)} for k, v in buckets.items()}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--alpha-sweep", action="store_true")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--track", default="CAP", choices=["CAP", "CEM"])
    ap.add_argument("--positions", default="", help="Comma-separated; default primary ROI position")
    ap.add_argument("--all-positions", action="store_true")
    ap.add_argument("--layer-sweep", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--controls", default="", help="Comma-separated control ids; default all")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--merge-shards", action="store_true", help="Merge shard jsonl files and write iia_summary only")
    ap.add_argument("--layer", type=int, default=None, help="Override ROI layer")
    ap.add_argument("--position", default=None, help="Override ROI position")
    ap.add_argument("--inject-site", default="residual", choices=["residual", "attention", "mlp"])
    args = ap.parse_args()
    positions = list(POSITION_CONTROL_POSITIONS) if args.all_positions else (
        [p.strip() for p in args.positions.split(",") if p.strip()] or None
    )
    control_ids = [c.strip() for c in args.controls.split(",") if c.strip()] or None
    print(
        json.dumps(
            run_p3b(
                dry_run=args.dry_run,
                limit=args.limit,
                alpha_sweep=args.alpha_sweep,
                device_map=args.device_map,
                resume=not args.no_resume,
                track=args.track,
                positions=positions,
                layer_sweep=args.layer_sweep,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                control_ids=control_ids,
                max_new_tokens=args.max_new_tokens,
                merge_shards=args.merge_shards,
                layer_override=args.layer,
                position_override=args.position,
                inject_site=args.inject_site,
            ),
            indent=2,
        )
    )
