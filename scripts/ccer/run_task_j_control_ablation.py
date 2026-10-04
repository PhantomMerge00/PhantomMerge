#!/usr/bin/env python3
"""Task J Step 4: same-code prompt_end vs claim_onset patched control (GPU smoke)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
sys.path.insert(0, str(ROOT))

from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.interchange import build_spec_from_roi, greedy_generate_with_hook
from ccer.mechanism.line_b_protocol import filter_cross_pairs_by_quality
from ccer.mechanism.mechanism_pool import build_cem_mechanism_cross_pairs
from ccer.mechanism.pair_select import load_trajectory_index, messages_for_condition
from ccer.replay.answer_utils import answers_differ
from ccer.replay.hf_forward import load_hf_model, tokenize_ccer_messages
from ccer.replay.live_position import (
    cross_pair_passes_line_d_v3_gate,
    intervention_steering_apply,
    live_token_idx_for_intervention,
    make_live_position_resolver,
)

AUDIT_DIR = ROOT / "results/p3/round12_line_b/task_j_audit"
OUT_PATH = AUDIT_DIR / "step4_control_ablation.json"
PROBE_MAX_TOKENS = 384


def _select_pairs(*, n_pairs: int, pool: str) -> list[dict[str, Any]]:
    cross = build_cem_mechanism_cross_pairs(pool=pool)["pm_clean_cross_pairs"]
    cross_pairs, _ = filter_cross_pairs_by_quality(cross, min_quality=0.35)
    gated: list[dict[str, Any]] = []
    for cp in cross_pairs:
        pm_tid = str(cp["pm_trajectory_id"])
        clean_tid = str(cp["clean_trajectory_id"])
        pm_path = activation_path(pm_tid, "original")
        clean_path = activation_path(clean_tid, "original")
        if not pm_path.is_file() or not clean_path.is_file():
            continue
        pm_npz = load_activation_npz(pm_path)
        clean_npz = load_activation_npz(clean_path)
        if cross_pair_passes_line_d_v3_gate(pm_npz, clean_npz, "claim_onset"):
            gated.append(cp)
        if len(gated) >= n_pairs:
            break
    return gated


def _run_arm(
    *,
    model: Any,
    tokenizer: Any,
    pm_traj: dict[str, Any],
    pm_npz: dict[str, Any],
    clean_vec: Any,
    pm_vec: Any,
    position: str,
    layer: int,
) -> dict[str, Any]:
    messages = messages_for_condition(pm_traj, "original", track="CEM")
    tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
    prompt_len = int(tok["prompt_token_count"])
    pos_idx = live_token_idx_for_intervention(
        pm_npz=pm_npz,
        position=position,
        prompt_len=prompt_len,
    )
    steering_apply = intervention_steering_apply(position)
    live_resolver = (
        make_live_position_resolver(tokenizer, position)
        if position in ("claim_onset", "pre_value")
        else None
    )
    dim = int(pm_vec.shape[0])
    roi = {
        "primary_roi": {"layer": layer, "position": position},
        "U_owner": [[0.0] * dim],
        "steering_apply": steering_apply,
    }
    spec_ni = build_spec_from_roi(
        control_id="no_intervention",
        roi=roi,
        donor_vec=clean_vec,
        recipient_vec=pm_vec,
        position_token_idx=pos_idx,
        alpha=0.0,
        mode="full_vector",
    )
    spec_ti = build_spec_from_roi(
        control_id="target_interchange",
        roi=roi,
        donor_vec=clean_vec,
        recipient_vec=pm_vec,
        position_token_idx=pos_idx,
        alpha=1.0,
        mode="full_vector",
    )
    out_ni = greedy_generate_with_hook(
        model,
        tokenizer,
        messages,
        spec=spec_ni,
        max_new_tokens=PROBE_MAX_TOKENS,
        use_cache=False,
        live_position_resolver=live_resolver,
    )
    out_ti = greedy_generate_with_hook(
        model,
        tokenizer,
        messages,
        spec=spec_ti,
        max_new_tokens=PROBE_MAX_TOKENS,
        use_cache=False,
        live_position_resolver=live_resolver,
    )
    return {
        "position": position,
        "layer": layer,
        "steering_apply": steering_apply,
        "stored_position_token_idx": pos_idx,
        "patched": bool(out_ti.get("patched")),
        "target_position": out_ti.get("target_position"),
        "text_diff": answers_differ(out_ni.get("text") or "", out_ti.get("text") or ""),
        "ni_text_head": (out_ni.get("text") or "")[:120],
        "ti_text_head": (out_ti.get("text") or "")[:120],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task J Step 4 control ablation")
    parser.add_argument("--n-pairs", type=int, default=5)
    parser.add_argument("--pool", default="mechanism_research")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--shared-gpu", action="store_true", default=True)
    args = parser.parse_args()

    pairs = _select_pairs(n_pairs=args.n_pairs, pool=args.pool)
    if not pairs:
        raise SystemExit("no v3-gated pairs found")

    rows_index = load_trajectory_index()
    print(f"[task_j step4] loading model device_map={args.device_map} ...", flush=True)
    model, tokenizer, _ = load_hf_model(device_map=args.device_map, shared_gpu=args.shared_gpu)
    print(f"[task_j step4] running {len(pairs)} pairs x 2 arms", flush=True)

    results: list[dict[str, Any]] = []
    for cp in pairs:
        pm_tid = str(cp["pm_trajectory_id"])
        clean_tid = str(cp["clean_trajectory_id"])
        pm_traj = rows_index[pm_tid]
        pm_npz = load_activation_npz(activation_path(pm_tid, "original"))
        clean_npz = load_activation_npz(activation_path(clean_tid, "original"))
        pm_vec_l32 = get_vector(pm_npz, position="prompt_end", layer=32)
        clean_vec_l32 = get_vector(clean_npz, position="prompt_end", layer=32)
        pm_vec_l49 = get_vector(pm_npz, position="claim_onset", layer=49)
        clean_vec_l49 = get_vector(clean_npz, position="claim_onset", layer=49)
        if any(v is None for v in (pm_vec_l32, clean_vec_l32, pm_vec_l49, clean_vec_l49)):
            print(f"[task_j step4] skip {pm_tid}: missing vectors", flush=True)
            continue
        row = {
            "pm_trajectory_id": pm_tid,
            "clean_trajectory_id": clean_tid,
            "arms": {
                "prompt_end_L32": _run_arm(
                    model=model,
                    tokenizer=tokenizer,
                    pm_traj=pm_traj,
                    pm_npz=pm_npz,
                    clean_vec=clean_vec_l32,
                    pm_vec=pm_vec_l32,
                    position="prompt_end",
                    layer=32,
                ),
                "claim_onset_L49": _run_arm(
                    model=model,
                    tokenizer=tokenizer,
                    pm_traj=pm_traj,
                    pm_npz=pm_npz,
                    clean_vec=clean_vec_l49,
                    pm_vec=pm_vec_l49,
                    position="claim_onset",
                    layer=49,
                ),
            },
        }
        results.append(row)
        pe = row["arms"]["prompt_end_L32"]["patched"]
        co = row["arms"]["claim_onset_L49"]["patched"]
        print(f"[task_j step4] {pm_tid} prompt_end patched={pe} claim_onset patched={co}", flush=True)

    n = len(results)
    pe_ok = sum(1 for r in results if r["arms"]["prompt_end_L32"]["patched"])
    co_ok = sum(1 for r in results if r["arms"]["claim_onset_L49"]["patched"])
    summary = {
        "schema_version": "ccer_task_j_step4_control_ablation_v1",
        "task": "J",
        "step": 4,
        "n_pairs": n,
        "probe_max_tokens": PROBE_MAX_TOKENS,
        "intervention": "full_vector alpha=1.0 target_interchange vs no_intervention",
        "prompt_end_patched_k_n": f"{pe_ok}/{n}",
        "claim_onset_patched_k_n": f"{co_ok}/{n}",
        "verdict": (
            "engineering_bug_confirmed_and_fix_effective"
            if pe_ok == n and co_ok == n and n > 0
            else "needs_further_investigation"
        ),
        "pairs": results,
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "pairs"}, indent=2), flush=True)
    print(f"[task_j step4] wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
