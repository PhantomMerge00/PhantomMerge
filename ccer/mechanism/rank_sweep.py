"""ROUND5: PCA rank sweep + cross-position causal contrast at fixed layer."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.controls import build_control_specs
from ccer.mechanism.interchange import greedy_generate_with_hook
from ccer.mechanism.das_train import fit_das_subspace
from ccer.mechanism.iia_roi_probe import _outputs_differ, _probe_log
from ccer.mechanism.line_b_protocol import (
    LINE_B_PROTOCOL_VERSION,
    resolve_wrong_owner_donor_vec,
    subspace_projection_stats,
)
from ccer.replay.live_position import (
    ANSWER_REGION_POSITIONS,
    LINE_D_PROTOCOL_VERSION,
    cross_pair_passes_line_d_v3_gate,
    intervention_steering_apply,
    live_token_idx_for_intervention,
    make_live_position_resolver,
    pair_has_symmetric_claim_anchor,
)
from ccer.mechanism.owner_pca import collect_diff_vectors, fit_owner_pca
from ccer.mechanism.pair_select import cem_valid_pair_ids, load_trajectory_index, messages_for_condition
from ccer.mechanism.stats import wilson_ci
from ccer.replay.answer_utils import (
    answers_differ,
    extract_selected_product_id,
    full_response_excerpt_differ,
    selected_product_blocks_differ,
)
from ccer.replay.hf_forward import tokenize_ccer_messages

InterventionMode = Literal["interchange", "full_vector"]
SubspaceSource = Literal["pca", "das", "jspace"]

DEFAULT_RANKS = [16, 32, 44]
POSITION_SWEEP = ("commitment", "claim_onset", "pre_value", "prompt_end")


def _cross_pairs(track: str) -> list[dict[str, Any]]:
    track_u = track.upper()
    if track_u == "CEM":
        return cem_valid_pair_ids(include_matched_clean=True)["pm_clean_cross_pairs"]
    from ccer.mechanism.pair_select import cap_valid_pair_ids

    return cap_valid_pair_ids(include_matched_clean=True)["pm_clean_cross_pairs"]


def _line_d_eligible_cross_pairs(
    cross_pairs: list[dict[str, Any]],
    *,
    position: str,
    track_u: str,
    rows_index: dict[str, Any],
) -> list[dict[str, Any]]:
    """For answer-region positions, keep pairs with bilateral symmetric claim anchors."""
    if position not in ANSWER_REGION_POSITIONS:
        return cross_pairs
    pm_key = "pm_trajectory_id" if track_u == "CEM" else "cap_trajectory_id"
    eligible: list[dict[str, Any]] = []
    for cp in cross_pairs:
        pm_tid = str(cp.get(pm_key) or cp.get("cap_trajectory_id"))
        clean_tid = str(cp["clean_trajectory_id"])
        pm_traj = rows_index.get(pm_tid)
        clean_traj = rows_index.get(clean_tid)
        if not pm_traj or not clean_traj:
            continue
        if pair_has_symmetric_claim_anchor(pm_traj) and pair_has_symmetric_claim_anchor(clean_traj):
            eligible.append(cp)
    return eligible


def max_feasible_rank(layer: int, position: str, track: str = "CEM") -> int:
    bundle = collect_diff_vectors(position=position, layer=layer, track=track)
    n = len(bundle["within_vectors"]) + len(bundle["cross_vectors"])
    return max(1, n)


def _shard_pairs(pairs: list[dict[str, Any]], shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return pairs
    return [p for i, p in enumerate(pairs) if i % num_shards == shard_index]


def merge_measure_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge shard-local measure_diff_rate rows into one aggregate."""
    if not rows:
        return {"n_pairs": 0, "n_output_diff": 0, "output_diff_rate": 0.0, "pair_details": []}
    if len(rows) == 1:
        return rows[0]
    n_diff = sum(int(r.get("n_output_diff") or 0) for r in rows)
    n_pid = sum(int(r.get("n_product_id_diff") or 0) for r in rows)
    n_block = sum(int(r.get("n_selected_product_block_diff") or 0) for r in rows)
    n = sum(int(r.get("n_pairs") or 0) for r in rows)
    skip = {
        "n_output_diff", "n_pairs", "output_diff_rate", "pair_details", "elapsed_s",
        "n_product_id_diff", "product_id_diff_rate", "product_id_ci95",
        "n_selected_product_block_diff", "selected_product_block_diff_rate", "selected_product_block_ci95",
    }
    merged = {k: v for k, v in rows[0].items() if k not in skip}
    merged["n_output_diff"] = n_diff
    merged["n_pairs"] = n
    merged["output_diff_rate"] = n_diff / n if n else 0.0
    merged["n_product_id_diff"] = n_pid
    merged["product_id_diff_rate"] = n_pid / n if n else 0.0
    merged["product_id_ci95"] = wilson_ci(n_pid, n)
    merged["n_selected_product_block_diff"] = n_block
    merged["selected_product_block_diff_rate"] = n_block / n if n else 0.0
    merged["selected_product_block_ci95"] = wilson_ci(n_block, n)
    merged["pair_details"] = [d for r in rows for d in (r.get("pair_details") or [])]
    merged["elapsed_s"] = round(sum(float(r.get("elapsed_s") or 0) for r in rows), 1)
    merged["n_patched"] = sum(int(r.get("n_patched") or 0) for r in rows)
    return merged


def merge_rank_sweep_shards(shard_results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(shard_results) == 1:
        return shard_results[0]
    base = dict(shard_results[0])
    base["full_vector_ceiling"] = merge_measure_rows([s.get("full_vector_ceiling") or {} for s in shard_results])
    ranks = [int(r.get("rank") or 0) for r in (shard_results[0].get("interchange_by_rank") or [])]
    merged_ic: list[dict[str, Any]] = []
    for rank in ranks:
        parts = []
        for s in shard_results:
            for row in s.get("interchange_by_rank") or []:
                if int(row.get("rank") or 0) == rank:
                    parts.append(row)
        merged_ic.append(merge_measure_rows(parts))
    base["interchange_by_rank"] = merged_ic
    best = max(merged_ic, key=lambda r: float(r.get("output_diff_rate") or 0.0))
    base["best_interchange_rank"] = best.get("rank")
    base["best_interchange_diff_rate"] = best.get("output_diff_rate")
    base["num_shards_merged"] = len(shard_results)
    return base


def merge_position_sweep_shards(shard_results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(shard_results) == 1:
        return shard_results[0]
    positions = []
    for s in shard_results:
        for row in s.get("rows") or []:
            pos = row.get("position")
            if pos and pos not in positions:
                positions.append(pos)
    merged_rows: list[dict[str, Any]] = []
    for pos in positions:
        fv_parts = []
        ic_parts = []
        for s in shard_results:
            for row in s.get("rows") or []:
                if row.get("position") == pos:
                    fv_parts.append(
                        {
                            "n_output_diff": int(round(float(row.get("full_vector_diff_rate") or 0) * int(row.get("n_pairs") or 0))),
                            "n_pairs": int(row.get("n_pairs") or 0),
                            "output_diff_rate": row.get("full_vector_diff_rate"),
                        }
                    )
                    ic_parts.append(
                        {
                            "n_output_diff": int(round(float(row.get("interchange_diff_rate") or 0) * int(row.get("n_pairs") or 0))),
                            "n_pairs": int(row.get("n_pairs") or 0),
                            "output_diff_rate": row.get("interchange_diff_rate"),
                            "pc_variance_explained": row.get("pc_variance_explained"),
                            "rank": row.get("rank"),
                        }
                    )
        fv = merge_measure_rows(fv_parts) if fv_parts else {}
        ic = merge_measure_rows(ic_parts) if ic_parts else {}
        merged_rows.append(
            {
                "position": pos,
                "rank": ic.get("rank"),
                "full_vector_diff_rate": fv.get("output_diff_rate"),
                "interchange_diff_rate": ic.get("output_diff_rate"),
                "n_pairs": ic.get("n_pairs"),
                "pc_variance_explained": ic.get("pc_variance_explained"),
            }
        )
    best = max(merged_rows, key=lambda r: float(r.get("interchange_diff_rate") or 0.0))
    out = dict(shard_results[0])
    out["rows"] = merged_rows
    out["best_position_interchange"] = best.get("position")
    out["best_interchange_diff_rate"] = best.get("interchange_diff_rate")
    out["num_shards_merged"] = len(shard_results)
    return out


def expand_cross_pairs_to_n(
    cross_pairs: list[dict[str, Any]],
    clean_ids: list[str],
    target_n: int,
) -> list[dict[str, Any]]:
    """Expand PM×clean pairing to target_n via multi-clean assignment (disclosed in reports)."""
    if target_n <= 0 or len(cross_pairs) >= target_n:
        return cross_pairs[:target_n] if target_n > 0 else cross_pairs
    expanded = list(cross_pairs)
    seen = {(str(cp.get("pm_trajectory_id") or cp.get("cap_trajectory_id")), str(cp["clean_trajectory_id"])) for cp in cross_pairs}
    pm_ids = list(dict.fromkeys(str(cp.get("pm_trajectory_id") or cp.get("cap_trajectory_id")) for cp in cross_pairs))
    if not pm_ids or not clean_ids:
        return expanded
    i = 0
    while len(expanded) < target_n:
        pm_tid = pm_ids[i % len(pm_ids)]
        clean_tid = clean_ids[i % len(clean_ids)]
        key = (pm_tid, clean_tid)
        if key not in seen:
            base = next(cp for cp in cross_pairs if str(cp.get("pm_trajectory_id") or cp.get("cap_trajectory_id")) == pm_tid)
            expanded.append(
                {
                    **base,
                    "clean_trajectory_id": clean_tid,
                    "pairing_method": "multi_clean_expansion",
                }
            )
            seen.add(key)
        i += 1
        if i > target_n * max(len(clean_ids), 1) * 2:
            break
    return expanded


def _count_pair_detail_metrics(pair_details: list[dict[str, Any]]) -> dict[str, int]:
    diff = answer_diff = product_id_diff = selected_block_diff = full_excerpt_diff = patched = 0
    claim_attr_diff = claim_attr_corrected = claim_scorable = 0
    for d in pair_details:
        if d.get("output_diff"):
            diff += 1
        if d.get("answer_diff") or d.get("extract_answer_diff"):
            answer_diff += 1
        if d.get("product_id_diff"):
            product_id_diff += 1
        if d.get("selected_product_block_diff"):
            selected_block_diff += 1
        if d.get("full_response_excerpt_diff"):
            full_excerpt_diff += 1
        if d.get("patched"):
            patched += 1
        if d.get("patched") and d.get("claim_attribution_scorable"):
            claim_scorable += 1
            if d.get("claim_attribution_diff"):
                claim_attr_diff += 1
            if d.get("claim_attribution_corrected"):
                claim_attr_corrected += 1
    return {
        "diff": diff,
        "answer_diff": answer_diff,
        "product_id_diff": product_id_diff,
        "selected_block_diff": selected_block_diff,
        "full_excerpt_diff": full_excerpt_diff,
        "patched": patched,
        "claim_attribution_diff": claim_attr_diff,
        "claim_attribution_corrected": claim_attr_corrected,
        "claim_attribution_scorable": claim_scorable,
        "n": len(pair_details),
    }


def measure_diff_rate(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer: int,
    position: str,
    rank: int,
    mode: InterventionMode = "interchange",
    pair_limit: int | None = None,
    probe_max_tokens: int = 96,
    ni_cache: dict[str, str] | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    save_texts: bool = False,
    subspace_source: SubspaceSource = "pca",
    inject_site: str = "residual",
    evidence_mask: bool = False,
    alpha: float = 1.0,
    cross_pairs_override: list[dict[str, Any]] | None = None,
    control_id: str = "target_interchange",
    wrong_donor_vec: Any = None,
    pool: str = "dev",
    resume_pair_details: list[dict[str, Any]] | None = None,
    on_pair_complete: Callable[[list[dict[str, Any]]], None] | None = None,
    u_override: Any = None,
    das_steps: int = 500,
    use_cache: bool | None = False,
    enrich_claim_attribution: bool = False,
    text_sidecar_path: str | None = None,
) -> dict[str, Any]:
    """Fraction of pairs where target intervention output differs from no_intervention."""
    track_u = track.upper()
    cross_pairs = list(cross_pairs_override) if cross_pairs_override is not None else _cross_pairs(track_u)
    if pair_limit is not None:
        cross_pairs = cross_pairs[:pair_limit]
    cross_pairs = _shard_pairs(cross_pairs, shard_index, num_shards)
    rows_index = load_trajectory_index()
    pm_key = "pm_trajectory_id" if track_u == "CEM" else "cap_trajectory_id"
    cross_pairs = _line_d_eligible_cross_pairs(
        cross_pairs,
        position=position,
        track_u=track_u,
        rows_index=rows_index,
    )
    pm_allowlist = {
        str(cp.get(pm_key) or cp.get("cap_trajectory_id"))
        for cp in cross_pairs
    }
    line_d_v3 = position in ANSWER_REGION_POSITIONS
    do_claim_attribution = enrich_claim_attribution or (
        save_texts and position in ANSWER_REGION_POSITIONS
    )
    ni_cache = ni_cache if ni_cache is not None else {}
    steering_apply = intervention_steering_apply(position)
    live_resolver = (
        make_live_position_resolver(tokenizer, position) if position in ANSWER_REGION_POSITIONS else None
    )

    if u_override is not None:
        u = np.asarray(u_override, dtype=np.float32)
        fit = {"rank": rank, "U_owner": u, "n_vectors": 0}
    else:
        fit = fit_owner_pca(
            layer=layer,
            position=position,
            rank=rank,
            track=track_u,
            pool=pool,
            pm_trajectory_allowlist=pm_allowlist if line_d_v3 else None,
            require_line_d_v3=line_d_v3,
            cross_only=line_d_v3,
        )
        u = fit.get("U_owner")
        if subspace_source == "das":
            bundle = collect_diff_vectors(
                position=position,
                layer=layer,
                track=track_u,
                pool=pool,
                pm_trajectory_allowlist=pm_allowlist if line_d_v3 else None,
                require_line_d_v3=line_d_v3,
            )
            das_vecs = bundle["within_vectors"] + bundle["cross_vectors"]
            das_fit = fit_das_subspace(
                das_vecs,
                rank=rank,
                layer=layer,
                position=position,
                track=track_u,
                pool=pool,
                steps=das_steps,
            )
            u = das_fit.get("U_das")
            if u is None:
                return {
                    "layer": layer,
                    "position": position,
                    "rank": rank,
                    "mode": mode,
                    "subspace_source": subspace_source,
                    "error": das_fit.get("error", "das_fit_failed"),
                    "n_pairs": 0,
                    "output_diff_rate": 0.0,
                }
    if u is None:
        return {
            "layer": layer,
            "position": position,
            "rank": rank,
            "mode": mode,
            "subspace_source": subspace_source,
            "error": fit.get("error", "pca_fit_failed"),
            "n_pairs": 0,
            "output_diff_rate": 0.0,
        }

    roi_stub = {
        "primary_roi": {"layer": layer, "position": position},
        "U_owner": u.tolist(),
        "inject_site": inject_site,
        "steering_apply": steering_apply,
    }
    pair_details: list[dict[str, Any]] = list(resume_pair_details or [])
    done_pm = {str(d["pm_trajectory_id"]) for d in pair_details}
    counts = _count_pair_detail_metrics(pair_details)
    diff = counts["diff"]
    answer_diff = counts["answer_diff"]
    product_id_diff = counts["product_id_diff"]
    selected_block_diff = counts["selected_block_diff"]
    full_excerpt_diff = counts["full_excerpt_diff"]
    patched = counts["patched"]
    claim_attr_diff = counts.get("claim_attribution_diff", 0)
    claim_attr_corrected = counts.get("claim_attribution_corrected", 0)
    claim_scorable = counts.get("claim_attribution_scorable", 0)
    n = counts["n"]
    t0 = time.time()

    for pi, cp in enumerate(cross_pairs):
        pm_tid = str(cp.get(pm_key) or cp.get("cap_trajectory_id"))
        if pm_tid in done_pm:
            continue
        clean_tid = str(cp["clean_trajectory_id"])
        pm_traj = rows_index.get(pm_tid)
        if not pm_traj:
            continue
        pm_npz = load_activation_npz(activation_path(pm_tid, "original"))
        clean_npz = load_activation_npz(activation_path(clean_tid, "original"))
        if line_d_v3 and not cross_pair_passes_line_d_v3_gate(pm_npz, clean_npz, position):
            _probe_log(
                f"rank_sweep skip pair {pm_tid}: stale/missing line_d_v3 symmetric activations"
            )
            continue
        pm_vec = get_vector(pm_npz, position=position, layer=layer)
        clean_vec = get_vector(clean_npz, position=position, layer=layer)
        messages = messages_for_condition(pm_traj, "original", track=track_u)
        mask_meta: dict[str, Any] = {"masked": False}
        if evidence_mask and messages:
            from ccer.replay.evidence_mask import mask_cem_evidence_spans

            messages, mask_meta = mask_cem_evidence_spans(messages, pm_traj)
        tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
        prompt_len = int(tok["prompt_token_count"])
        pos_idx = live_token_idx_for_intervention(
            pm_npz=pm_npz,
            position=position,
            prompt_len=prompt_len,
        )
        if pm_vec is None or clean_vec is None:
            continue
        if position not in ANSWER_REGION_POSITIONS and pos_idx < 0:
            continue

        cache_key = f"{pm_tid}|{clean_tid}|{position}|{probe_max_tokens}|mask={evidence_mask}|a={alpha}"
        if cache_key not in ni_cache:
            spec_ni = build_control_specs(
                control_id="no_intervention",
                roi=roi_stub,
                donor_vec=clean_vec,
                recipient_vec=pm_vec,
                position_token_idx=pos_idx,
                alpha=alpha,
            )
            out_ni = greedy_generate_with_hook(
                model,
                tokenizer,
                messages,
                spec=spec_ni,
                max_new_tokens=probe_max_tokens,
                use_cache=use_cache,
                live_position_resolver=live_resolver,
            )
            ni_cache[cache_key] = out_ni["text"]

        cid = control_id or "target_interchange"
        spec_ti: Any
        wrong_impl: str | None = None
        if cid == "wrong_owner_donor":
            wrong_vec, wrong_impl = resolve_wrong_owner_donor_vec(
                pm_tid=pm_tid,
                paired_clean_tid=clean_tid,
                cross_pairs=cross_pairs,
                layer=layer,
                position=position,
            )
            if wrong_vec is not None and wrong_impl != "orthogonal_complement_fallback":
                spec_ti = build_control_specs(
                    control_id="target_interchange",
                    roi=roi_stub,
                    donor_vec=wrong_vec,
                    recipient_vec=pm_vec,
                    position_token_idx=pos_idx,
                    alpha=alpha,
                )
            else:
                spec_ti = build_control_specs(
                    control_id="orthogonal_complement",
                    roi=roi_stub,
                    donor_vec=clean_vec,
                    recipient_vec=pm_vec,
                    position_token_idx=pos_idx,
                    alpha=alpha,
                )
                wrong_impl = "orthogonal_complement"
        else:
            spec_ti = build_control_specs(
                control_id=cid,
                roi=roi_stub,
                donor_vec=clean_vec,
                recipient_vec=pm_vec,
                position_token_idx=pos_idx,
                alpha=alpha,
            )
        if mode == "full_vector":
            spec_ti.mode = "full_vector"
        spec_ti.inject_site = inject_site
        alpha_applied = float(spec_ti.alpha)
        out_ti = greedy_generate_with_hook(
            model,
            tokenizer,
            messages,
            spec=spec_ti,
            max_new_tokens=probe_max_tokens,
            use_cache=use_cache,
            live_position_resolver=live_resolver,
        )
        n += 1
        if out_ti.get("patched"):
            patched += 1
        ni_text = ni_cache[cache_key]
        ti_text = out_ti["text"]
        differs = _outputs_differ(ni_text, ti_text)
        ans_differs = answers_differ(ni_text, ti_text)
        sel_differs = selected_product_blocks_differ(ni_text, ti_text)
        full_differs = full_response_excerpt_differ(ni_text, ti_text)
        pid_ni = extract_selected_product_id(ni_text)
        pid_ti = extract_selected_product_id(ti_text)
        pid_differs = bool(pid_ni and pid_ti and pid_ni != pid_ti)
        behavioral_differs = pid_differs or sel_differs
        if differs:
            diff += 1
        if ans_differs:
            answer_diff += 1
        if pid_differs:
            product_id_diff += 1
        if sel_differs:
            selected_block_diff += 1
        if full_differs:
            full_excerpt_diff += 1
        proj_stats = subspace_projection_stats(u, pm_vec, clean_vec)
        detail = {
            "pm_trajectory_id": pm_tid,
            "clean_trajectory_id": clean_tid,
            "control_id": cid,
            "wrong_owner_impl": (
                wrong_impl
                if cid == "wrong_owner_donor"
                else None
            ),
            "subspace_proj_fraction": proj_stats.get("proj_fraction"),
            "patched": bool(out_ti.get("patched")),
            "patch_tier": out_ti.get("patch_tier"),
            "live_claim_source": out_ti.get("live_claim_source"),
            "first_patch_step": out_ti.get("first_patch_step"),
            "ni_use_cache": use_cache,
            "ti_use_cache": use_cache,
            "output_diff": differs,
            "answer_diff": ans_differs,
            "extract_answer_diff": ans_differs,
            "selected_product_block_diff": sel_differs,
            "full_response_excerpt_diff": full_differs,
            "product_id_ni": pid_ni,
            "product_id_ti": pid_ti,
            "product_id_diff": pid_differs,
            "behavioral_diff": behavioral_differs,
            "behavioral_metric": "product_id_diff" if pid_differs else ("selected_block_diff" if sel_differs else None),
            "evidence_mask": evidence_mask,
            "evidence_mask_meta": mask_meta if evidence_mask else None,
            "intervention_alpha": alpha_applied,
        }
        if save_texts:
            detail["ni_text"] = ni_text
            detail["ti_text"] = ti_text
        if do_claim_attribution and save_texts:
            from ccer.mechanism.claim_attribution import evaluate_claim_attribution_pair

            clean_traj = rows_index.get(clean_tid) or {}
            detail.update(
                evaluate_claim_attribution_pair(
                    ni_text=ni_text,
                    ti_text=ti_text,
                    pm_trajectory=pm_traj,
                    clean_trajectory=clean_traj,
                )
            )
        pair_details.append(detail)
        done_pm.add(pm_tid)
        if on_pair_complete is not None:
            on_pair_complete(pair_details)
        _probe_log(
            f"rank_sweep L{layer}/{position} rank={rank} mode={mode} "
            f"pair={len(pair_details)}/{len(cross_pairs)} text_diff={differs} pid_diff={pid_differs} "
            f"patched={out_ti.get('patched')}"
        )

    rate = diff / n if n else 0.0
    ans_rate = answer_diff / n if n else 0.0
    out = {
        "layer": layer,
        "position": position,
        "rank": int(fit.get("rank") or rank),
        "mode": mode,
        "subspace_source": subspace_source,
        "inject_site": inject_site,
        "control_id": control_id,
        "evidence_mask": evidence_mask,
        "pc_variance_explained": fit.get("pc_variance_explained"),
        "n_pairs": n,
        "n_output_diff": diff,
        "n_extract_answer_diff": answer_diff,
        "n_product_id_diff": product_id_diff,
        "product_id_diff_rate": product_id_diff / n if n else 0.0,
        "product_id_ci95": wilson_ci(product_id_diff, n),
        "n_selected_product_block_diff": selected_block_diff,
        "selected_product_block_diff_rate": selected_block_diff / n if n else 0.0,
        "selected_product_block_ci95": wilson_ci(selected_block_diff, n),
        "n_full_response_excerpt_diff": full_excerpt_diff,
        "full_response_excerpt_diff_rate": full_excerpt_diff / n if n else 0.0,
        "full_response_excerpt_ci95": wilson_ci(full_excerpt_diff, n),
        "n_patched": patched,
        "n_claim_attribution_scorable": claim_scorable,
        "n_claim_attribution_diff": claim_attr_diff,
        "claim_attribution_diff_rate": claim_attr_diff / claim_scorable if claim_scorable else 0.0,
        "claim_attribution_diff_ci95": wilson_ci(claim_attr_diff, claim_scorable),
        "n_claim_attribution_corrected": claim_attr_corrected,
        "claim_attribution_corrected_rate": claim_attr_corrected / claim_scorable if claim_scorable else 0.0,
        "claim_attribution_corrected_ci95": wilson_ci(claim_attr_corrected, claim_scorable),
        "output_diff_rate": rate,
        "answer_diff_rate": ans_rate,
        "extract_answer_diff_rate": ans_rate,
        "output_diff_ci95": wilson_ci(diff, n),
        "extract_answer_ci95": wilson_ci(answer_diff, n),
        "answer_diff_ci95": wilson_ci(answer_diff, n),
        "probe_max_tokens": probe_max_tokens,
        "evidence_mask": evidence_mask,
        "intervention_alpha": float(alpha),
        "alpha_confirmed_full_strength": float(alpha) == 1.0,
        "protocol_version": LINE_B_PROTOCOL_VERSION,
        "line_d_protocol_version": LINE_D_PROTOCOL_VERSION,
        "steering_apply": steering_apply,
        "use_cache": use_cache,
        "claim_anchor_mode": "symmetric" if line_d_v3 else "metadata_first",
        "line_d_v3_gate": line_d_v3,
        "pca_cross_only": line_d_v3,
        "n_eligible_pairs": len(cross_pairs),
        "pm_allowlist_n": len(pm_allowlist),
        "elapsed_s": round(time.time() - t0, 1),
        "pair_details": pair_details,
    }
    if save_texts and pair_details:
        from pathlib import Path

        from ccer.paths import P3_DIR

        sidecar = (
            Path(text_sidecar_path)
            if text_sidecar_path
            else (
                P3_DIR
                / "task_h"
                / f"texts_L{layer}_{position}_{mode}_r{int(fit.get('rank') or rank)}_{control_id or 'ti'}.jsonl"
            )
        )
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        import json

        with sidecar.open("w", encoding="utf-8") as fh:
            for row in pair_details:
                if row.get("ni_text") is not None:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        out["text_sidecar"] = str(sidecar)
    return out


def run_round5_rank_sweep(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer: int = 32,
    position: str = "commitment",
    ranks: list[int] | None = None,
    pair_limit: int | None = None,
    probe_max_tokens: int = 96,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    """ROUND5 §二.1: sweep PCA rank with interchange; record full_vector ceiling."""
    max_rank = max_feasible_rank(layer, position, track)
    if ranks is None:
        ranks = sorted({r for r in DEFAULT_RANKS + [max_rank] if r <= max_rank})

    ni_cache: dict[str, str] = {}
    _probe_log(
        f"round5 rank_sweep L{layer}/{position} ranks={ranks} max_rank={max_rank} "
        f"shard={shard_index}/{num_shards}"
    )

    ceiling = measure_diff_rate(
        model=model,
        tokenizer=tokenizer,
        track=track,
        layer=layer,
        position=position,
        rank=min(16, max_rank),
        mode="full_vector",
        pair_limit=pair_limit,
        probe_max_tokens=probe_max_tokens,
        ni_cache=ni_cache,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    _probe_log(
        f"full_vector ceiling diff_rate={ceiling['output_diff_rate']:.3f} "
        f"n={ceiling['n_pairs']} elapsed={ceiling['elapsed_s']}s"
    )

    interchange_rows: list[dict[str, Any]] = []
    for rank in ranks:
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
            ni_cache=ni_cache,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        interchange_rows.append(row)
        _probe_log(
            f"interchange rank={rank} diff_rate={row['output_diff_rate']:.3f} "
            f"var={row.get('pc_variance_explained')} elapsed={row['elapsed_s']}s"
        )

    best = max(interchange_rows, key=lambda r: float(r.get("output_diff_rate") or 0.0))
    return {
        "schema_version": "ccer_round5_rank_sweep_v1",
        "track": track.upper(),
        "layer": layer,
        "position": position,
        "max_feasible_rank": max_rank,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "full_vector_ceiling": ceiling,
        "interchange_by_rank": interchange_rows,
        "best_interchange_rank": best.get("rank"),
        "best_interchange_diff_rate": best.get("output_diff_rate"),
    }


def run_round5_position_sweep(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer: int = 32,
    positions: tuple[str, ...] = POSITION_SWEEP,
    rank: int = 64,
    pair_limit: int | None = None,
    probe_max_tokens: int = 96,
    shard_index: int = 0,
    num_shards: int = 1,
    positions_for_shard: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """ROUND5 §二.1: cross-position causal contrast at fixed layer."""
    ni_cache: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    ceilings: list[dict[str, Any]] = []
    pos_list = positions_for_shard if positions_for_shard is not None else positions

    for position in pos_list:
        max_rank = max_feasible_rank(layer, position, track)
        use_rank = min(rank, max_rank)
        ceil_row = measure_diff_rate(
            model=model,
            tokenizer=tokenizer,
            track=track,
            layer=layer,
            position=position,
            rank=use_rank,
            mode="full_vector",
            pair_limit=pair_limit,
            probe_max_tokens=probe_max_tokens,
            ni_cache=ni_cache,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        int_row = measure_diff_rate(
            model=model,
            tokenizer=tokenizer,
            track=track,
            layer=layer,
            position=position,
            rank=use_rank,
            mode="interchange",
            pair_limit=pair_limit,
            probe_max_tokens=probe_max_tokens,
            ni_cache=ni_cache,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        ceilings.append(ceil_row)
        rows.append(
            {
                "position": position,
                "rank": use_rank,
                "full_vector_diff_rate": ceil_row.get("output_diff_rate"),
                "interchange_diff_rate": int_row.get("output_diff_rate"),
                "n_pairs": int_row.get("n_pairs"),
                "pc_variance_explained": int_row.get("pc_variance_explained"),
            }
        )
        _probe_log(
            f"position_sweep L{layer}/{position} rank={use_rank} "
            f"fv={ceil_row['output_diff_rate']:.3f} ic={int_row['output_diff_rate']:.3f}"
        )

    best = max(rows, key=lambda r: float(r.get("interchange_diff_rate") or 0.0))
    return {
        "schema_version": "ccer_round5_position_sweep_v1",
        "track": track.upper(),
        "layer": layer,
        "rank": rank,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "rows": rows,
        "best_position_interchange": best.get("position"),
        "best_interchange_diff_rate": best.get("interchange_diff_rate"),
    }
