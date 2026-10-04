#!/usr/bin/env python3
"""Compare Line E claim_onset steering: use_cache=False vs True (patched + speed)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
sys.path.insert(0, str(ROOT))

from ccer.mechanism.activation_store import activation_path, load_activation_npz
from ccer.mechanism.iia_roi_probe import _live_token_idx
from ccer.mechanism.line_e_eval import _generate_with_steering, behavioral_outcome
from ccer.mechanism.line_e_protocol import (
    LINE_E_V3_PRIMARY_POSITION,
    LINE_E_V3_STEERING_DIRECTION,
    build_trajectory_steering_bundle,
    line_e_steering_apply_for_position,
    line_e_v3_config,
)
from ccer.mechanism.pair_select import load_trajectory_index, messages_for_condition
from ccer.replay.hf_forward import load_hf_model, tokenize_ccer_messages

DEFAULT_TID = "shop_D_261a8e80549394e6"
OUT = ROOT / "results/p3/line_e/use_cache_validation.json"


def _run_once(model, tokenizer, traj, bundle, pos_idx, *, use_cache: bool) -> dict:
    cfg = line_e_v3_config()
    t0 = time.time()
    out = _generate_with_steering(
        model,
        tokenizer,
        traj,
        layer=cfg.layer,
        position=cfg.position,
        position_token_idx=pos_idx,
        steering_vec=bundle["mitigation_vector"],
        control_id="paired_caa",
        alpha=1.0,
        max_new_tokens=384,
        steering_direction=LINE_E_V3_STEERING_DIRECTION,
        steering_apply=line_e_steering_apply_for_position(cfg.position),
        use_live_position_resolver=True,
        use_cache=use_cache,
    )
    beh = behavioral_outcome(out.get("text") or "", traj)
    return {
        "use_cache": use_cache,
        "elapsed_s": round(time.time() - t0, 2),
        "patched": bool(out.get("patched")),
        "patch_tier": out.get("patch_tier"),
        "first_patch_step": out.get("first_patch_step"),
        "n_generated": len(out.get("generated_token_ids") or []),
        "selected_product_id": beh.get("selected_product_id"),
        "wrong_owner": beh.get("wrong_owner"),
        "text_head": (out.get("text") or "")[:200],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-id", default=DEFAULT_TID)
    parser.add_argument("--device-map", default="cuda:0")
    args = parser.parse_args()

    rows_index = load_trajectory_index()
    traj = rows_index.get(args.trajectory_id)
    if not traj:
        raise SystemExit(f"unknown trajectory {args.trajectory_id}")

    bundle = build_trajectory_steering_bundle(
        args.trajectory_id,
        layer=line_e_v3_config().layer,
        position=LINE_E_V3_PRIMARY_POSITION,
    )
    if bundle.get("error"):
        raise SystemExit(f"bundle error: {bundle['error']}")

    messages = messages_for_condition(traj, "original")
    print(f"[validate] loading model {args.device_map}...", flush=True)
    model, tokenizer, _ = load_hf_model(device_map=args.device_map, shared_gpu=False)
    tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
    npz = load_activation_npz(activation_path(args.trajectory_id, "original"))
    pos_idx = _live_token_idx(npz, LINE_E_V3_PRIMARY_POSITION, int(tok["prompt_token_count"]))
    print(f"[validate] tid={args.trajectory_id} pos={pos_idx}", flush=True)

    results = []
    for uc in (False, True):
        print(f"[validate] running use_cache={uc}...", flush=True)
        results.append(_run_once(model, tokenizer, traj, bundle, pos_idx, use_cache=uc))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    speedup = None
    if results[0]["elapsed_s"] > 0:
        speedup = round(results[0]["elapsed_s"] / results[1]["elapsed_s"], 2)
    pid_match = results[0]["selected_product_id"] == results[1]["selected_product_id"]
    report = {
        "trajectory_id": args.trajectory_id,
        "position": LINE_E_V3_PRIMARY_POSITION,
        "control": "paired_caa",
        "alpha": 1.0,
        "runs": results,
        "speedup_false_over_true": speedup,
        "pid_match": pid_match,
        "pass_patched_both": all(r["patched"] for r in results),
        "verdict": "PASS" if all(r["patched"] for r in results) else "FAIL_PATCH",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[validate] verdict={report['verdict']} speedup≈{speedup}x pid_match={pid_match}", flush=True)
    print(f"[validate] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
