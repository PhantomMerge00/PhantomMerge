"""IIA-based ROI selection: pick layer/position by interchange output distinguishability."""
from __future__ import annotations

import re
import sys
import time
from typing import Any

import numpy as np
import torch

from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.controls import build_control_specs
from ccer.mechanism.interchange import InterchangeSpec, _patch_hidden, greedy_generate_with_hook
from ccer.mechanism.owner_pca import (
    MIN_CROSS_FOR_INTERCHANGE,
    collect_diff_vectors,
    fit_owner_pca,
    position_permutation_control,
)
from ccer.paths import P3_BINDING_ROI
from ccer.mechanism.pair_select import cem_valid_pair_ids, load_trajectory_index, messages_for_condition
from ccer.replay.answer_utils import extract_answer
from ccer.replay.hf_forward import _model_layers, tokenize_ccer_messages


def _probe_log(msg: str) -> None:
    print(f"[iia_probe] {msg}", flush=True)


def _outputs_differ(a: str, b: str) -> bool:
    if a != b:
        return True
    if extract_answer(a) != extract_answer(b):
        return True
    pa = re.search(r"Selected product ID:\s*(\d+)", a, re.I)
    pb = re.search(r"Selected product ID:\s*(\d+)", b, re.I)
    if pa and pb and pa.group(1) != pb.group(1):
        return True
    return False


def _live_token_idx(pm_npz: dict, position: str, prompt_len: int) -> int:
    from ccer.replay.live_position import live_token_idx_for_intervention

    return live_token_idx_for_intervention(
        pm_npz=pm_npz,
        position=position,
        prompt_len=prompt_len,
    )


def screen_candidates_by_hidden_delta(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    candidates: list[dict[str, Any]],
    pair_limit: int = 2,
) -> list[dict[str, Any]]:
    """Cheap pre-screen: mean L2(hidden_after - hidden_before) at patch site."""
    track_u = track.upper()
    selection = cem_valid_pair_ids(include_matched_clean=True) if track_u == "CEM" else None
    if selection is None:
        from ccer.mechanism.pair_select import cap_valid_pair_ids

        selection = cap_valid_pair_ids(include_matched_clean=True)
    cross_pairs = selection["pm_clean_cross_pairs"][:pair_limit]
    rows_index = load_trajectory_index()
    pm_key = "pm_trajectory_id" if track_u == "CEM" else "cap_trajectory_id"
    device = next(model.parameters()).device
    scored: list[dict[str, Any]] = []
    _probe_log(f"hidden_screen start n_candidates={len(candidates)} pair_limit={pair_limit}")

    for ci, cand in enumerate(candidates):
        layer = int(cand["layer"])
        position = str(cand["position"])
        deltas: list[float] = []
        for pi, cp in enumerate(cross_pairs):
            _probe_log(f"hidden_screen cand={ci+1}/{len(candidates)} L{layer}/{position} pair={pi+1}/{len(cross_pairs)}")
            pm_tid = str(cp.get(pm_key) or cp.get("cap_trajectory_id"))
            clean_tid = str(cp["clean_trajectory_id"])
            pm_traj = rows_index.get(pm_tid)
            if not pm_traj:
                continue
            pm_npz = load_activation_npz(activation_path(pm_tid, "original"))
            clean_npz = load_activation_npz(activation_path(clean_tid, "original"))
            pm_vec = get_vector(pm_npz, position=position, layer=layer)
            clean_vec = get_vector(clean_npz, position=position, layer=layer)
            if pm_vec is None or clean_vec is None:
                continue
            messages = messages_for_condition(pm_traj, "original", track=track_u)
            tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
            pos_idx = _live_token_idx(pm_npz, position, int(tok["prompt_token_count"]))
            if pos_idx < 0:
                continue
            input_ids = torch.tensor([tok["full_ids"][: tok["prompt_token_count"]]], device=device)
            captured: dict[str, torch.Tensor] = {}

            def _hook(_module, _inp, output):
                out = output.clone()
                h = out[0, pos_idx, :]
                captured["before"] = h.detach().clone()
                spec = InterchangeSpec(
                    control_id="target_interchange",
                    layer=layer,
                    position_token_idx=pos_idx,
                    alpha=1.0,
                    U=np.eye(len(pm_vec), 1, dtype=np.float32),
                    h_donor=clean_vec.astype(np.float32),
                    h_recipient=pm_vec.astype(np.float32),
                    mode="full_vector",
                )
                out[0, pos_idx, :] = _patch_hidden(h, spec)
                captured["after"] = out[0, pos_idx, :].detach().clone()
                return out

            handle = _model_layers(model)[layer].register_forward_hook(_hook)
            try:
                with torch.no_grad():
                    model(input_ids, use_cache=False)
            finally:
                handle.remove()
            if "before" in captured and "after" in captured:
                deltas.append(float(torch.linalg.vector_norm(captured["after"] - captured["before"]).item()))

        scored.append(
            {
                **cand,
                "hidden_delta_l2_mean": float(np.mean(deltas)) if deltas else 0.0,
                "n_hidden_probe_pairs": len(deltas),
            }
        )

    scored.sort(
        key=lambda r: (
            float(r.get("hidden_delta_l2_mean") or 0.0),
            float(r.get("combined_score") or 0.0),
        ),
        reverse=True,
    )
    return scored


def _cached_scan_rows(track: str) -> list[dict[str, Any]]:
    if not P3_BINDING_ROI.is_file():
        return []
    import json

    roi = json.loads(P3_BINDING_ROI.read_text(encoding="utf-8"))
    if str(roi.get("track") or "").upper() != track.upper():
        return []
    scan = roi.get("two_stage_scan") or {}
    rows = list(scan.get("coarse_scan") or []) + list(scan.get("dense_scan") or [])
    rows.extend(roi.get("position_comparison_table") or [])
    # de-dupe by layer+position keeping best combined_score
    best: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["layer"]), str(row["position"]))
        prev = best.get(key)
        if prev is None or float(row.get("combined_score") or 0) > float(prev.get("combined_score") or 0):
            best[key] = row
    return list(best.values())


def _mid_layer_scan(
    *,
    track: str,
    layer_lo: int,
    layer_hi: int,
) -> list[dict[str, Any]]:
    """Lightweight scan: only prompt_end/commitment in [layer_lo, layer_hi]."""
    track_u = track.upper()
    rows: list[dict[str, Any]] = []
    for layer in range(layer_lo, layer_hi + 1):
        for position in ("prompt_end", "commitment"):
            _probe_log(f"mid_scan L{layer}/{position}")
            bundle = collect_diff_vectors(position=position, layer=layer, track=track_u)
            within, labels, cross = bundle["within_vectors"], bundle["within_labels"], bundle["cross_vectors"]
            cross_labels = (
                [0] * (len(cross) // 2) + [1] * (len(cross) - len(cross) // 2) if len(cross) >= 4 else []
            )
            from ccer.mechanism.owner_pca import _separation_score

            row = {
                "layer": layer,
                "position": position,
                "n_within": len(within),
                "n_cross": len(cross),
                "follow_separation": _separation_score(within, labels),
                "cross_separation": _separation_score(cross, cross_labels) if cross_labels else 0.0,
            }
            row["combined_score"] = row["follow_separation"] + 0.5 * row["cross_separation"]
            rows.append(row)
    rows.sort(key=lambda r: float(r.get("combined_score") or 0.0), reverse=True)
    return rows


def mid_layer_candidates(
    *,
    track: str = "CEM",
    layer_lo: int = 28,
    layer_hi: int = 54,
    top_k: int = 6,
) -> list[dict[str, Any]]:
    _probe_log(f"mid_layer_candidates lo={layer_lo} hi={layer_hi}")
    rows = _cached_scan_rows(track)
    if rows:
        _probe_log(f"using cached scan rows n={len(rows)}")
    pool = [
        r
        for r in rows
        if layer_lo <= int(r["layer"]) <= layer_hi
        and int(r.get("n_cross") or 0) >= MIN_CROSS_FOR_INTERCHANGE
        and str(r.get("position")) in ("prompt_end", "commitment")
    ]
    if not pool:
        _probe_log("no mid-layer pool in cache; running lightweight mid-layer scan")
        rows = _mid_layer_scan(track=track, layer_lo=layer_lo, layer_hi=layer_hi)
        pool = [
            r
            for r in rows
            if int(r.get("n_cross") or 0) >= MIN_CROSS_FOR_INTERCHANGE
            and str(r.get("position")) in ("prompt_end", "commitment")
        ]
    pool.sort(key=lambda r: float(r.get("combined_score") or 0.0), reverse=True)
    _probe_log(f"mid_layer_candidates pool={len(pool)} top={[ (int(r['layer']), r['position']) for r in pool[:3] ]}")
    return pool[:top_k]


def probe_interchange_distinguishability(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    candidates: list[dict[str, Any]],
    pair_limit: int = 4,
    rank: int = 8,
    probe_max_tokens: int = 96,
    ni_cache: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Score each candidate by fraction of pairs where target_interchange != no_intervention."""
    track_u = track.upper()
    selection = cem_valid_pair_ids(include_matched_clean=True) if track_u == "CEM" else None
    if selection is None:
        from ccer.mechanism.pair_select import cap_valid_pair_ids

        selection = cap_valid_pair_ids(include_matched_clean=True)
    cross_pairs = selection["pm_clean_cross_pairs"][:pair_limit]
    rows_index = load_trajectory_index()
    pm_key = "pm_trajectory_id" if track_u == "CEM" else "cap_trajectory_id"
    ni_cache = ni_cache if ni_cache is not None else {}

    scored: list[dict[str, Any]] = []
    _probe_log(f"gen_probe start n_candidates={len(candidates)} pair_limit={pair_limit} max_tokens={probe_max_tokens}")
    for ci, cand in enumerate(candidates):
        layer = int(cand["layer"])
        position = str(cand["position"])
        t0 = time.time()
        _probe_log(f"gen_probe cand={ci+1}/{len(candidates)} L{layer}/{position} pca_fit...")
        fit = fit_owner_pca(layer=layer, position=position, rank=rank, track=track_u)
        u = fit.get("U_owner")
        if u is None:
            scored.append({**cand, "probe_error": "pca_fit_failed", "output_diff_rate": 0.0})
            continue

        roi_stub = {"primary_roi": {"layer": layer, "position": position}, "U_owner": u.tolist()}
        diff = 0
        patched = 0
        n = 0
        for pi, cp in enumerate(cross_pairs):
            pm_tid = str(cp.get(pm_key) or cp.get("cap_trajectory_id"))
            clean_tid = str(cp["clean_trajectory_id"])
            pm_traj = rows_index.get(pm_tid)
            if not pm_traj:
                continue
            _probe_log(f"gen_probe L{layer}/{position} pair={pi+1}/{len(cross_pairs)} pm={pm_tid[-8:]}")
            pm_npz = load_activation_npz(activation_path(pm_tid, "original"))
            clean_npz = load_activation_npz(activation_path(clean_tid, "original"))
            pm_vec = get_vector(pm_npz, position=position, layer=layer)
            clean_vec = get_vector(clean_npz, position=position, layer=layer)
            messages = messages_for_condition(pm_traj, "original", track=track_u)
            tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
            pos_idx = _live_token_idx(pm_npz, position, int(tok["prompt_token_count"]))
            if pm_vec is None or clean_vec is None or pos_idx < 0:
                continue
            cache_key = f"{pm_tid}|{position}"
            if cache_key not in ni_cache:
                _probe_log(f"  no_intervention generate (cache miss)")
                spec_ni = build_control_specs(
                    control_id="no_intervention",
                    roi=roi_stub,
                    donor_vec=clean_vec,
                    recipient_vec=pm_vec,
                    position_token_idx=pos_idx,
                )
                out_ni = greedy_generate_with_hook(
                    model, tokenizer, messages, spec=spec_ni, max_new_tokens=probe_max_tokens
                )
                ni_cache[cache_key] = out_ni["text"]
            ni_text = ni_cache[cache_key]
            _probe_log(f"  target_interchange generate full_vector")
            spec_ti = build_control_specs(
                control_id="target_interchange",
                roi=roi_stub,
                donor_vec=clean_vec,
                recipient_vec=pm_vec,
                position_token_idx=pos_idx,
            )
            spec_ti.mode = "full_vector"
            out_ti = greedy_generate_with_hook(
                model, tokenizer, messages, spec=spec_ti, max_new_tokens=probe_max_tokens
            )
            n += 1
            if out_ti.get("patched"):
                patched += 1
            if _outputs_differ(ni_text, out_ti["text"]):
                diff += 1
            _probe_log(f"  done patched={out_ti.get('patched')} diff={_outputs_differ(ni_text, out_ti['text'])}")

        ctrl = position_permutation_control(layer=layer, position=position, track=track_u)
        _probe_log(
            f"gen_probe L{layer}/{position} summary diff_rate={diff/n if n else 0:.2f} "
            f"n={n} elapsed={time.time()-t0:.0f}s"
        )
        scored.append(
            {
                **cand,
                "output_diff_rate": diff / n if n else 0.0,
                "n_probe_pairs": n,
                "n_output_diff": diff,
                "n_patched": patched,
                "passes_position_control": bool(ctrl.get("passes_control")),
                "pc_variance_explained": fit.get("pc_variance_explained"),
                "probe_rank": rank,
            }
        )

    scored.sort(
        key=lambda r: (
            float(r.get("output_diff_rate") or 0.0),
            float(r.get("hidden_delta_l2_mean") or 0.0),
            float(r.get("combined_score") or 0.0),
        ),
        reverse=True,
    )
    return scored


def pick_roi_by_iia_probe(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer_lo: int = 28,
    layer_hi: int = 54,
    top_k: int = 6,
    probe_pairs: int = 4,
    probe_max_tokens: int = 96,
    fast_screen: bool = True,
    screen_pairs: int = 2,
    gen_probe_top_n: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Two-stage: hidden L2 screen → generation probe on top-N only."""
    candidates = mid_layer_candidates(track=track, layer_lo=layer_lo, layer_hi=layer_hi, top_k=top_k)
    _probe_log(f"pick_roi candidates={len(candidates)} layer_lo={layer_lo} layer_hi={layer_hi}")
    ni_cache: dict[str, str] = {}
    if fast_screen and len(candidates) > gen_probe_top_n:
        screened = screen_candidates_by_hidden_delta(
            model=model,
            tokenizer=tokenizer,
            track=track,
            candidates=candidates,
            pair_limit=screen_pairs,
        )
        for row in screened:
            row.setdefault("screen_stage", "hidden_l2")
        gen_pool = screened[:gen_probe_top_n]
    else:
        gen_pool = candidates

    probed = probe_interchange_distinguishability(
        model=model,
        tokenizer=tokenizer,
        track=track,
        candidates=gen_pool,
        pair_limit=probe_pairs,
        probe_max_tokens=probe_max_tokens,
        ni_cache=ni_cache,
    )
    for row in probed:
        row["screen_stage"] = "generation"
    return probed, ni_cache
