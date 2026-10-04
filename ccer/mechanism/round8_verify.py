"""ROUND8 verification: dual-metric expansion, activation refresh, cross-position, DAS, path patch."""
from __future__ import annotations

import json
import time
from typing import Any, Literal

import numpy as np

from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz, save_activation_npz
from ccer.mechanism.iia_roi_probe import _live_token_idx
from ccer.mechanism.pair_select import cem_valid_pair_ids, load_trajectory_index, messages_for_condition
from ccer.mechanism.rank_sweep import POSITION_SWEEP, measure_diff_rate
from ccer.mechanism.round6_verify import verify_dual_metrics_from_texts
from ccer.replay.hf_forward import extract_activations, line_d_activation_layer_indices, sparse_layer_indices
from ccer.replay.live_position import (
    ANSWER_REGION_POSITIONS,
    LINE_D_PROTOCOL_VERSION,
    LINE_D_REFRESH_TAG,
    activation_passes_line_d_v3_gate,
    cross_pair_passes_line_d_v3_gate,
    line_d_layer_for_position,
)
from ccer.replay.position_registry import POSITION_NAMES

PATCH_IMPLEMENTATION_NOTE = {
    "use_cache_on_intervention": False,
    "hook_reapplied_each_decode_step": True,
    "patch_position": "fixed commitment token index (not current decode token)",
    "interpretation": (
        "Intervention disables KV cache and re-runs full-sequence forward each decode step; "
        "hook patches the commitment position on every step. "
        "0% at answer layer is therefore a stronger negative result (patch is persistent), "
        "not a failure of one-shot prefill injection."
    ),
}


def cross_pair_trajectory_ids(track: str = "CEM") -> set[str]:
    if track.upper() == "CEM":
        cross = cem_valid_pair_ids(include_matched_clean=True)["pm_clean_cross_pairs"]
        pm_key = "pm_trajectory_id"
    else:
        from ccer.mechanism.pair_select import cap_valid_pair_ids

        cross = cap_valid_pair_ids(include_matched_clean=True)["pm_clean_cross_pairs"]
        pm_key = "cap_trajectory_id"
    tids: set[str] = set()
    for cp in cross:
        tids.add(str(cp.get(pm_key) or cp.get("cap_trajectory_id")))
        tids.add(str(cp["clean_trajectory_id"]))
    return tids


def activation_coverage_audit(
    *,
    track: str = "CEM",
    layer: int = 32,
    layer_by_position: dict[str, int] | None = None,
    positions: tuple[str, ...] = ("commitment", "prompt_end", "claim_onset", "pre_value"),
) -> dict[str, Any]:
    """CPU audit: which positions have vectors for cross-pair trajectories."""
    if track.upper() == "CEM":
        cross = cem_valid_pair_ids(include_matched_clean=True)["pm_clean_cross_pairs"]
        pm_key = "pm_trajectory_id"
    else:
        from ccer.mechanism.pair_select import cap_valid_pair_ids

        cross = cap_valid_pair_ids(include_matched_clean=True)["pm_clean_cross_pairs"]
        pm_key = "cap_trajectory_id"

    rows: list[dict[str, Any]] = []
    pos_counts = {p: 0 for p in positions}
    n_pairs = 0
    for cp in cross:
        pm_tid = str(cp.get(pm_key) or cp.get("cap_trajectory_id"))
        clean_tid = str(cp["clean_trajectory_id"])
        row: dict[str, Any] = {"pm_trajectory_id": pm_tid, "clean_trajectory_id": clean_tid}
        ok = True
        for pos in positions:
            pos_layer = int((layer_by_position or {}).get(pos, layer))
            pm_npz = load_activation_npz(activation_path(pm_tid, "original"))
            clean_npz = load_activation_npz(activation_path(clean_tid, "original"))
            pm_vec = get_vector(pm_npz, position=pos, layer=pos_layer)
            clean_vec = get_vector(clean_npz, position=pos, layer=pos_layer)
            v3_ok = True
            if pos in ANSWER_REGION_POSITIONS:
                v3_ok = cross_pair_passes_line_d_v3_gate(pm_npz, clean_npz, pos)
            eligible = pm_vec is not None and clean_vec is not None and v3_ok
            row[f"{pos}_ok"] = eligible
            row[f"{pos}_v3_gate"] = v3_ok if pos in ANSWER_REGION_POSITIONS else None
            if not eligible:
                ok = False
            else:
                pos_counts[pos] += 1
        row["all_positions_ok"] = ok
        rows.append(row)
        n_pairs += 1

    return {
        "schema_version": "ccer_round8_activation_audit_v1",
        "track": track.upper(),
        "layer": layer,
        "layer_by_position": layer_by_position or {p: layer for p in positions},
        "n_cross_pairs": n_pairs,
        "eligible_by_position": {p: f"{pos_counts[p]}/{n_pairs}" for p in positions},
        "pair_rows": rows,
    }


def refresh_cross_pair_activations(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    n_layers: int,
    resume: bool = True,
    force: bool = False,
    refresh_tag: str = LINE_D_REFRESH_TAG,
    claim_anchor_mode: str = "symmetric",
) -> dict[str, Any]:
    """Re-extract activations for all cross-pair trajectories (force overwrite when force=True)."""
    tids = cross_pair_trajectory_ids(track)
    rows_index = load_trajectory_index()
    layer_indices = line_d_activation_layer_indices(n_layers)
    counts = {
        "trajectories_refreshed": 0,
        "trajectories_skipped": 0,
        "claim_onset_ok": 0,
        "pre_value_ok": 0,
        "commitment_ok": 0,
        "position_fail": 0,
    }
    position_totals = {p: 0 for p in POSITION_NAMES}

    for tid in sorted(tids):
        out_path = activation_path(tid, "original")
        if resume and not force and out_path.is_file():
            existing = load_activation_npz(out_path)
            meta = existing.get("metadata") or {}
            if meta.get(refresh_tag) and str(meta.get("claim_anchor_mode") or "") == claim_anchor_mode:
                counts["trajectories_skipped"] += 1
                positions = existing.get("positions") or {}
                for pos in POSITION_NAMES:
                    if positions.get(pos) is not None:
                        position_totals[pos] += 1
                if positions.get("claim_onset") is not None:
                    counts["claim_onset_ok"] += 1
                if positions.get("pre_value") is not None:
                    counts["pre_value_ok"] += 1
                if positions.get("commitment") is not None:
                    counts["commitment_ok"] += 1
                continue
        traj = rows_index.get(tid)
        if not traj:
            continue
        messages = messages_for_condition(traj, "original", track=track)
        if not messages:
            continue
        output_text = str((traj.get("metadata") or {}).get("final_answer") or "")
        act = extract_activations(
            model,
            tokenizer,
            messages=messages,
            trajectory=traj,
            output_text=output_text,
            layer_indices=layer_indices,
            claim_anchor_mode=claim_anchor_mode,
        )
        positions = act.get("positions") or {}
        for pos in POSITION_NAMES:
            if positions.get(pos) is not None:
                position_totals[pos] += 1
        if positions.get("claim_onset") is not None:
            counts["claim_onset_ok"] += 1
        if positions.get("pre_value") is not None:
            counts["pre_value_ok"] += 1
        if positions.get("commitment") is not None:
            counts["commitment_ok"] += 1
        if not act.get("position_ok"):
            counts["position_fail"] += 1

        save_activation_npz(
            out_path,
            trajectory_id=tid,
            condition_id="original",
            cohort=track,
            layer_indices=act["layer_indices"],
            positions=positions,
            vectors=act["vectors"],
            metadata={
                "position_errors": act.get("position_errors"),
                "claim_anchor_mode": claim_anchor_mode,
                "claim_anchor_source": act.get("claim_anchor_source"),
                "refreshed_round8": True,
                refresh_tag: True,
            },
        )
        counts["trajectories_refreshed"] += 1

    n = counts["trajectories_refreshed"] + counts["trajectories_skipped"]
    return {
        "schema_version": "ccer_round8_activation_refresh_v1",
        "claim_anchor_mode": claim_anchor_mode,
        "refresh_tag": refresh_tag,
        "track": track.upper(),
        "n_trajectories": n,
        "counts": counts,
        "position_ok_rates": {p: position_totals[p] / n if n else 0.0 for p in POSITION_NAMES},
        "layer_indices": layer_indices,
        "resume": resume,
        "force": force,
    }


def run_dual_metric_expansion(
    *,
    model: Any,
    tokenizer: Any,
    pair_limit: int = 18,
    probe_max_tokens: int = 96,
    layer: int = 32,
    position: str = "commitment",
    rank: int = 16,
    track: str = "CEM",
) -> dict[str, Any]:
    row = measure_diff_rate(
        model=model,
        tokenizer=tokenizer,
        track=track,
        layer=layer,
        position=position,
        rank=rank,
        mode="interchange",
        pair_limit=pair_limit,
        probe_max_tokens=probe_max_tokens,
        save_texts=True,
    )
    dual = verify_dual_metrics_from_texts(row.get("pair_details") or [])
    verified_pairs = {d["pm_trajectory_id"]: d for d in dual.get("pair_details") or []}
    synced_details: list[dict[str, Any]] = []
    for pr in row.get("pair_details") or []:
        key = pr.get("pm_trajectory_id")
        merged = dict(pr)
        if key in verified_pairs:
            for field in (
                "answer_diff",
                "extract_answer_diff",
                "selected_product_block_diff",
                "full_response_excerpt_diff",
                "product_id_diff",
                "compared_section_only_diff",
            ):
                if field in verified_pairs[key]:
                    merged[field] = verified_pairs[key][field]
        synced_details.append(merged)
    row["pair_details"] = synced_details
    for metric_key, row_fields in (
        ("answer_diff", ("n_extract_answer_diff", "extract_answer_diff_rate", "extract_answer_ci95")),
        ("selected_product_block_diff", ("n_selected_product_block_diff", "selected_product_block_diff_rate", "selected_product_block_ci95")),
        ("full_response_excerpt_diff", ("n_full_response_excerpt_diff", "full_response_excerpt_diff_rate", "full_response_excerpt_ci95")),
        ("product_id_diff", ("n_product_id_diff", "product_id_diff_rate", "product_id_ci95")),
    ):
        block = dual.get(metric_key) or {}
        if isinstance(block, dict) and block.get("k") is not None:
            row[row_fields[0]] = block["k"]
            row[row_fields[1]] = block["rate"]
            row[row_fields[2]] = block["ci95"]
    row["answer_diff_rate"] = (dual.get("answer_diff") or {}).get("rate")
    row["answer_diff_ci95"] = (dual.get("answer_diff") or {}).get("ci95")
    od = dual.get("outputs_differ") or {}
    ea = dual.get("answer_diff") or dual.get("extract_answer_diff") or {}
    ci_overlap = (
        od.get("ci95") and ea.get("ci95")
        and od["ci95"][0] <= ea["ci95"][1]
        and ea["ci95"][0] <= od["ci95"][1]
    )
    return {
        "schema_version": "ccer_round8_dual_metric_v1",
        "layer": layer,
        "position": position,
        "rank": rank,
        "pair_limit": pair_limit,
        "probe_max_tokens": probe_max_tokens,
        "measure_row": row,
        "dual_metrics": dual,
        "ci_overlap": ci_overlap,
        "metrics_agree_statistically_supported": not ci_overlap and dual.get("n_pairs", 0) >= 15,
    }


def _row_behavioral_summary(ic_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_pairs": ic_row.get("n_pairs"),
        "product_id_k": ic_row.get("n_product_id_diff"),
        "product_id_rate": ic_row.get("product_id_diff_rate"),
        "product_id_ci95": ic_row.get("product_id_ci95"),
        "selected_block_k": ic_row.get("n_selected_product_block_diff"),
        "selected_block_rate": ic_row.get("selected_product_block_diff_rate"),
        "selected_block_ci95": ic_row.get("selected_product_block_ci95"),
        "full_excerpt_k": ic_row.get("n_full_response_excerpt_diff"),
        "full_excerpt_rate": ic_row.get("full_response_excerpt_diff_rate"),
        "outputs_differ_rate": ic_row.get("output_diff_rate"),
        "error": ic_row.get("error"),
    }


LINE_D_POSITIONS = ("prompt_end", "commitment", "claim_onset", "pre_value")


def _behavioral_flip(detail: dict[str, Any]) -> tuple[bool, str]:
    """Primary product_id; fallback selected_block for sparse pid (Line D v3)."""
    if detail.get("product_id_diff"):
        return True, "product_id_diff"
    if detail.get("selected_product_block_diff"):
        return True, "selected_block_diff"
    return False, "none"


def _specificity_block(
    ti_details: list[dict[str, Any]],
    wo_details: list[dict[str, Any]],
) -> dict[str, Any]:
    by_pm_wrong = {str(d["pm_trajectory_id"]): d for d in wo_details}
    correct_only = wrong_only = both = neither = 0
    pid_correct_only = pid_wrong_only = pid_both = pid_neither = 0
    pair_cmp: list[dict[str, Any]] = []
    for cd in ti_details:
        pm = str(cd["pm_trajectory_id"])
        wd = by_pm_wrong.get(pm) or {}
        c_flip, c_metric = _behavioral_flip(cd)
        w_flip, w_metric = _behavioral_flip(wd)
        c_pid = bool(cd.get("product_id_diff"))
        w_pid = bool(wd.get("product_id_diff"))
        if c_flip and w_flip:
            both += 1
        elif c_flip and not w_flip:
            correct_only += 1
        elif not c_flip and w_flip:
            wrong_only += 1
        else:
            neither += 1
        if c_pid and w_pid:
            pid_both += 1
        elif c_pid and not w_pid:
            pid_correct_only += 1
        elif not c_pid and w_pid:
            pid_wrong_only += 1
        else:
            pid_neither += 1
        pair_cmp.append(
            {
                "pm_trajectory_id": pm,
                "target_interchange_diff": c_flip,
                "wrong_owner_donor_diff": w_flip,
                "target_metric": c_metric,
                "wrong_owner_metric": w_metric,
                "target_product_id_diff": c_pid,
                "wrong_owner_product_id_diff": w_pid,
            }
        )
    return {
        "primary_metric": "behavioral_diff",
        "behavioral_fallback": "selected_block_diff_when_pid_sparse",
        "correct_donor_only": correct_only,
        "wrong_owner_only": wrong_only,
        "both_flip": both,
        "neither_flip": neither,
        "product_id_correct_only": pid_correct_only,
        "product_id_wrong_only": pid_wrong_only,
        "product_id_both_flip": pid_both,
        "product_id_neither_flip": pid_neither,
        "pair_comparisons": pair_cmp,
    }


def _agent_log_line_d(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    # #region agent log
    import time

    payload = {
        "sessionId": "f40181",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open("${PHANTOM_MERGE_ROOT}/logs/debug", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass
    # #endregion


def run_position_sweep_with_controls(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer: int = 32,
    claim_layer: int = 49,
    pair_limit: int | None = None,
    probe_max_tokens: int = 96,
    rank: int = 16,
    positions: tuple[str, ...] = LINE_D_POSITIONS,
    control_ids: tuple[str, ...] = ("target_interchange", "wrong_owner_donor"),
    include_full_vector_ceiling: bool = True,
    save_texts: bool = False,
    enrich_claim_attribution: bool = False,
    text_sidecar_path: str | None = None,
) -> dict[str, Any]:
    """Line D: four-position interchange vs wrong_owner_donor head-to-head (product_id primary)."""
    from ccer.mechanism.round9_verify import _aggregate_from_pair_details

    rows_out: list[dict[str, Any]] = []
    ni_cache: dict[str, str] = {}
    layer_by_position = {
        str(pos): line_d_layer_for_position(str(pos), claim_layer=claim_layer) for pos in positions
    }
    for pos in positions:
        pos_layer = int(layer_by_position[str(pos)])
        if "target_interchange" in control_ids:
            ti_row = measure_diff_rate(
                model=model,
                tokenizer=tokenizer,
                track=track,
                layer=pos_layer,
                position=str(pos),
                rank=rank,
                mode="interchange",
                pair_limit=pair_limit,
                probe_max_tokens=probe_max_tokens,
                ni_cache=ni_cache,
                control_id="target_interchange",
                save_texts=save_texts,
                enrich_claim_attribution=enrich_claim_attribution,
                text_sidecar_path=text_sidecar_path,
            )
            ti_sum = _row_behavioral_summary(ti_row)
            ti_details = ti_row.get("pair_details") or []
            if ti_details:
                ti_sum.update(_aggregate_from_pair_details(ti_details))
        else:
            ti_sum = {}
            ti_details = []

        if "wrong_owner_donor" in control_ids:
            wo_row = measure_diff_rate(
                model=model,
                tokenizer=tokenizer,
                track=track,
                layer=pos_layer,
                position=str(pos),
                rank=rank,
                mode="interchange",
                pair_limit=pair_limit,
                probe_max_tokens=probe_max_tokens,
                ni_cache=ni_cache,
                control_id="wrong_owner_donor",
                save_texts=save_texts,
                enrich_claim_attribution=enrich_claim_attribution,
                text_sidecar_path=(
                    str(text_sidecar_path).replace("target_interchange", "wrong_owner_donor")
                    if text_sidecar_path
                    else None
                ),
            )
            wo_sum = _row_behavioral_summary(wo_row)
            wo_details = wo_row.get("pair_details") or []
            if wo_details:
                wo_sum.update(_aggregate_from_pair_details(wo_details))
        else:
            wo_sum = {}
            wo_details = []

        spec = _specificity_block(ti_details, wo_details)
        ceiling_sum: dict[str, Any] = {}
        if include_full_vector_ceiling:
            ceil_row = measure_diff_rate(
                model=model,
                tokenizer=tokenizer,
                track=track,
                layer=pos_layer,
                position=str(pos),
                rank=rank,
                mode="full_vector",
                pair_limit=pair_limit,
                probe_max_tokens=probe_max_tokens,
                ni_cache=ni_cache,
                control_id="target_interchange",
                save_texts=save_texts,
                enrich_claim_attribution=enrich_claim_attribution,
                text_sidecar_path=(
                    str(text_sidecar_path).replace("target_interchange", "full_vector_ceiling")
                    if text_sidecar_path
                    else None
                ),
            )
            ceiling_sum = _row_behavioral_summary(ceil_row)
        row_out = {
            "position": pos,
            "layer": pos_layer,
            "target_interchange": ti_sum,
            "wrong_owner_donor": wo_sum,
            "full_vector_ceiling": ceiling_sum,
            "specificity": spec,
        }
        rows_out.append(row_out)
        _agent_log_line_d(
            "H_complete",
            "round8_verify.run_position_sweep_with_controls",
            "position_sweep_row_done",
            {
                "position": pos,
                "n_pairs": ti_sum.get("n_pairs"),
                "target_pid_k": ti_sum.get("product_id_k", ti_sum.get("n_product_id_diff")),
                "wrong_pid_k": wo_sum.get("product_id_k", wo_sum.get("n_product_id_diff")),
                "pair_comparisons": len(spec.get("pair_comparisons") or []),
                "reused_from": ti_sum.get("reused_from"),
            },
        )

    def _pid_rate(row: dict[str, Any]) -> float:
        ti = row.get("target_interchange") or {}
        return float(ti.get("product_id_diff_rate") or ti.get("product_id_rate") or 0.0)

    best = max(rows_out, key=_pid_rate)
    best_ti = best.get("target_interchange") or {}
    def _behavioral_rate(row: dict[str, Any]) -> float:
        ti = row.get("target_interchange") or {}
        pid = float(ti.get("product_id_rate") or 0.0)
        if pid > 0:
            return pid
        return float(ti.get("selected_block_rate") or 0.0)

    best_behavioral = max(rows_out, key=_behavioral_rate)
    best_ti = best_behavioral.get("target_interchange") or {}
    return {
        "schema_version": "ccer_line_d_position_causal_v2",
        "protocol_version": LINE_D_PROTOCOL_VERSION,
        "primary_metric": "product_id_diff",
        "secondary_metric": "selected_product_block_diff",
        "specificity_metric": "behavioral_diff",
        "claim_anchor_mode": "symmetric",
        "intervention_mode": "dynamic_answer_anchor_for_claim_positions",
        "track": track.upper(),
        "layer": layer,
        "claim_layer": claim_layer,
        "layer_by_position": layer_by_position,
        "rank": rank,
        "probe_max_tokens": probe_max_tokens,
        "pair_limit": pair_limit,
        "include_full_vector_ceiling": include_full_vector_ceiling,
        "positions": list(positions),
        "rows": rows_out,
        "best_position_product_id": best.get("position"),
        "best_product_id_rate": _pid_rate(best),
        "best_position_behavioral": best_behavioral.get("position"),
        "best_behavioral_rate": _behavioral_rate(best_behavioral),
        "best_behavioral_metric": (
            "product_id_diff"
            if float(best_ti.get("product_id_rate") or 0.0) > 0
            else "selected_block_diff"
        ),
    }


def run_position_sweep_extract_answer(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer: int = 32,
    pair_limit: int | None = None,
    probe_max_tokens: int = 96,
    rank: int = 16,
) -> dict[str, Any]:
    """Cross-position interchange; primary metric = product_id_diff."""
    rows_out: list[dict[str, Any]] = []
    ni_cache: dict[str, str] = {}
    for pos in POSITION_SWEEP:
        ic_row = measure_diff_rate(
            model=model,
            tokenizer=tokenizer,
            track=track,
            layer=layer,
            position=str(pos),
            rank=rank,
            mode="interchange",
            pair_limit=pair_limit,
            probe_max_tokens=probe_max_tokens,
            ni_cache=ni_cache,
        )
        summary = _row_behavioral_summary(ic_row)
        rows_out.append({"position": pos, **summary})
    best = max(rows_out, key=lambda r: float(r.get("product_id_rate") or 0.0))
    return {
        "schema_version": "ccer_round8_position_sweep_v1",
        "primary_metric": "product_id_diff",
        "track": track.upper(),
        "layer": layer,
        "rank": rank,
        "rows": rows_out,
        "best_position_product_id": best.get("position"),
        "best_product_id_rate": best.get("product_id_rate"),
    }


def run_das_pilot(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer: int = 32,
    position: str = "commitment",
    rank: int = 16,
    pair_limit: int | None = None,
    probe_max_tokens: int = 96,
) -> dict[str, Any]:
    """DAS subspace interchange pilot (product_id primary)."""
    ni_cache: dict[str, str] = {}
    pca_row = measure_diff_rate(
        model=model,
        tokenizer=tokenizer,
        track=track,
        layer=layer,
        position=position,
        rank=rank,
        mode="interchange",
        pair_limit=pair_limit,
        probe_max_tokens=probe_max_tokens,
        subspace_source="pca",
        ni_cache=ni_cache,
    )
    das_row = measure_diff_rate(
        model=model,
        tokenizer=tokenizer,
        track=track,
        layer=layer,
        position=position,
        rank=rank,
        mode="interchange",
        pair_limit=pair_limit,
        probe_max_tokens=probe_max_tokens,
        subspace_source="das",
        ni_cache=ni_cache,
    )
    return {
        "schema_version": "ccer_round8_das_pilot_v1",
        "primary_metric": "product_id_diff",
        "layer": layer,
        "position": position,
        "rank": rank,
        "pca": _row_behavioral_summary(pca_row),
        "das": _row_behavioral_summary(das_row),
    }


def run_path_patch_pilot(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer: int = 32,
    position: str = "commitment",
    rank: int = 16,
    pair_limit: int = 4,
    probe_max_tokens: int = 96,
) -> dict[str, Any]:
    """Attention vs MLP 4-pair pilot; primary metric = product_id_diff."""
    summaries: dict[str, Any] = {}
    for site in ("attention", "mlp"):
        row = measure_diff_rate(
            model=model,
            tokenizer=tokenizer,
            track=track,
            layer=layer,
            position=position,
            rank=rank,
            mode="interchange",
            pair_limit=pair_limit,
            probe_max_tokens=probe_max_tokens,
            inject_site=site,
        )
        summaries[site] = _row_behavioral_summary(row)
    return {
        "schema_version": "ccer_round8_path_patch_v1",
        "primary_metric": "product_id_diff",
        "layer": layer,
        "position": position,
        "rank": rank,
        "pair_limit": pair_limit,
        "by_inject_site": summaries,
    }
