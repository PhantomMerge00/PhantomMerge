"""Line E v2 fast probe: paired CAA + generation_decode steering on selected trajectories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.mechanism.activation_store import activation_path, load_activation_npz
from ccer.mechanism.iia_roi_probe import _live_token_idx
from ccer.mechanism.interchange import (
    build_paired_full_vector_spec,
    build_steering_spec,
    greedy_generate_with_hook,
)
from ccer.mechanism.line_e_eval import _trajectory_track, behavioral_outcome
from ccer.mechanism.pair_select import load_trajectory_index, messages_for_condition
from ccer.mechanism.steering_vector import paired_activation_vectors
from ccer.replay.hf_forward import load_hf_model, tokenize_ccer_messages

P3_LINE_E_PROBE = ROOT / "artifacts" / "ccer" / "p3" / "line_e" / "probe_v2_results.json"
DEFAULT_PROBE_IDS = (
    "shop_D_261a8e80549394e6",
    "shop_D_7d43c76d96f47e18",
)
PROBE_ALPHAS = (1.0, 2.0)  # fast probe; full sweep via run_p3_line_e --steering-v2
PROBE_MAX_NEW_TOKENS = 256


def _resolve_pos(traj_id: str, *, layer: int, position: str, tokenizer: Any, messages: list) -> int:
    npz = load_activation_npz(activation_path(traj_id, "original"))
    tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
    return _live_token_idx(npz, position, int(tok["prompt_token_count"]))


def _run_one(
    model: Any,
    tokenizer: Any,
    traj: dict[str, Any],
    *,
    layer: int,
    position: str,
    pos_idx: int,
    mode: str,
    alpha: float,
    paired: dict[str, Any],
    direction: str = "subtract",
    steering_apply: str = "generation_decode",
) -> dict[str, Any]:
    track = _trajectory_track(traj)
    messages = messages_for_condition(traj, "original", track=track)
    if mode == "paired_full":
        spec = build_paired_full_vector_spec(
            control_id="paired_full",
            layer=layer,
            position_token_idx=pos_idx,
            pm_vec=paired["pm_vec"],
            clean_vec=paired["clean_vec"],
            alpha=alpha,
            steering_apply=steering_apply,
        )
    elif alpha == 0.0:
        spec = build_steering_spec(
            control_id="no_intervention",
            layer=layer,
            position_token_idx=pos_idx,
            steering_vec=paired["paired_caa_vector"],
            alpha=0.0,
            steering_apply="anchor_once",
        )
    else:
        spec = build_steering_spec(
            control_id="paired_caa",
            layer=layer,
            position_token_idx=pos_idx,
            steering_vec=paired["paired_caa_vector"],
            alpha=alpha,
            steering_direction=direction,
            steering_apply=steering_apply,
        )
    out = greedy_generate_with_hook(model, tokenizer, messages, spec=spec, max_new_tokens=PROBE_MAX_NEW_TOKENS)
    beh = behavioral_outcome(out.get("text") or "", traj)
    return {
        "mode": mode,
        "alpha": alpha,
        "direction": direction,
        "steering_apply": steering_apply,
        "patched": bool(out.get("patched")),
        "use_cache": bool(out.get("use_cache")),
        "selected_product_id": beh.get("selected_product_id"),
        "action_anchor": beh.get("action_anchor"),
        "wrong_owner": beh.get("wrong_owner"),
        "product_id_correct": beh.get("product_id_correct"),
        "text_head": (out.get("text") or "")[:160],
    }


def run_probe(
    trajectory_ids: list[str],
    *,
    layer: int = 32,
    position: str = "commitment",
    device_map: str = "auto",
) -> dict[str, Any]:
    rows_index = load_trajectory_index()
    model, tokenizer, _ = load_hf_model(device_map=device_map)
    results: list[dict[str, Any]] = []

    for tid in trajectory_ids:
        traj = rows_index.get(tid)
        if not traj:
            results.append({"trajectory_id": tid, "error": "trajectory_not_found"})
            continue
        paired = paired_activation_vectors(tid, layer=layer, position=position)
        if paired.get("error"):
            results.append({"trajectory_id": tid, "error": paired})
            continue
        track = _trajectory_track(traj)
        messages = messages_for_condition(traj, "original", track=track)
        pos_idx = _resolve_pos(tid, layer=layer, position=position, tokenizer=tokenizer, messages=messages)
        if pos_idx < 0:
            results.append({"trajectory_id": tid, "error": "position_unresolved"})
            continue

        entry: dict[str, Any] = {
            "trajectory_id": tid,
            "layer": layer,
            "position": position,
            "position_token_idx": pos_idx,
            "clean_trajectory_id": paired["clean_trajectory_id"],
            "paired_norm": paired["paired_norm"],
            "runs": [],
        }
        baseline = _run_one(
            model, tokenizer, traj, layer=layer, position=position, pos_idx=pos_idx,
            mode="baseline", alpha=0.0, paired=paired,
        )
        entry["baseline"] = baseline
        entry["runs"].append(baseline)

        for alpha in PROBE_ALPHAS:
            row = _run_one(
                model, tokenizer, traj, layer=layer, position=position, pos_idx=pos_idx,
                mode="paired_caa", alpha=alpha, paired=paired, direction="subtract",
                steering_apply="generation_decode",
            )
            entry["runs"].append(row)
            if row.get("product_id_correct") and baseline.get("wrong_owner"):
                print(f"  PASS paired_caa add alpha={alpha} pid={row.get('selected_product_id')}", flush=True)

        for alpha in PROBE_ALPHAS:
            row = _run_one(
                model, tokenizer, traj, layer=layer, position=position, pos_idx=pos_idx,
                mode="paired_full", alpha=alpha, paired=paired,
                steering_apply="generation_decode",
            )
            entry["runs"].append(row)

        passes = [
            r for r in entry["runs"]
            if float(r.get("alpha") or 0) > 0 and baseline.get("wrong_owner") and not r.get("wrong_owner")
        ]
        entry["probe_pass"] = bool(passes)
        entry["passing_runs"] = passes
        results.append(entry)
        print(
            f"[probe_v2] {tid} baseline_wrong={baseline.get('wrong_owner')} "
            f"pass={entry['probe_pass']} n_pass={len(passes)}",
            flush=True,
        )

    out = {
        "schema_version": "ccer_line_e_probe_v2",
        "layer": layer,
        "position": position,
        "steering_apply": "generation_decode",
        "vector": "paired_pm_minus_clean",
        "results": results,
        "n_pass": sum(1 for r in results if r.get("probe_pass")),
    }
    P3_LINE_E_PROBE.parent.mkdir(parents=True, exist_ok=True)
    P3_LINE_E_PROBE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Line E v2 fast probe (2-case)")
    parser.add_argument("--trajectory-ids", type=str, default=",".join(DEFAULT_PROBE_IDS))
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--position", type=str, default="commitment")
    parser.add_argument(
        "--device-map",
        type=str,
        default="cuda:0",
        help="Maps to the sole visible GPU when CUDA_VISIBLE_DEVICES is set (e.g. cuda:0)",
    )
    args = parser.parse_args()
    tids = [x.strip() for x in args.trajectory_ids.split(",") if x.strip()]
    result = run_probe(tids, layer=args.layer, position=args.position, device_map=args.device_map)
    print(json.dumps({"path": str(P3_LINE_E_PROBE), "n_pass": result["n_pass"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
