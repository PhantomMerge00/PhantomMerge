"""Line E v4 ultimate probe: bilateral-gated claim_onset + multi-donor controls."""
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
from ccer.mechanism.line_e_eval import _generate_with_steering, _trajectory_track, behavioral_outcome
from ccer.mechanism.line_e_protocol import (
    LINE_E_PROBE_ALPHAS,
    LINE_E_V3_LEGACY_POSITION,
    LINE_E_V3_PRIMARY_POSITION,
    LINE_E_V3_PROTOCOL_VERSION,
    LINE_E_V3_STEERING_DIRECTION,
    build_trajectory_steering_bundle,
    line_e_steering_apply_for_position,
    line_e_v3_config,
)
from ccer.mechanism.pair_select import load_trajectory_index, messages_for_condition
from ccer.replay.hf_forward import load_hf_model, tokenize_ccer_messages

P3_LINE_E_PROBE = ROOT / "artifacts" / "ccer" / "p3" / "line_e" / "probe_v4_ultimate_results.json"
DEFAULT_PROBE_IDS = (
    "shop_D_261a8e80549394e6",
    "shop_D_7d43c76d96f47e18",
)
PROBE_MAX_NEW_TOKENS = 384
PROBE_CONTROLS = (
    "slot_consistent_restore",
    "donor_restore",
    "paired_caa",
)


def _resolve_stored_pos(traj_id: str, *, layer: int, position: str, tokenizer: Any, messages: list) -> int:
    npz = load_activation_npz(activation_path(traj_id, "original"))
    tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
    idx = _live_token_idx(npz, position, int(tok["prompt_token_count"]))
    if idx < 0:
        stored = int((npz.get("positions") or {}).get(position) or -1)
        if stored >= 0:
            return stored
    return idx


def _run_variant(
    model: Any,
    tokenizer: Any,
    traj: dict[str, Any],
    bundle: dict[str, Any],
    *,
    layer: int,
    position: str,
    pos_idx: int,
    control_id: str,
    alpha: float,
) -> dict[str, Any]:
    steering_apply = line_e_steering_apply_for_position(position)
    use_live = position in ("claim_onset", "pre_value")
    out = _generate_with_steering(
        model,
        tokenizer,
        traj,
        layer=layer,
        position=position,
        position_token_idx=pos_idx,
        steering_vec=bundle["mitigation_vector"],
        control_id=control_id if alpha > 0 else "no_intervention",
        alpha=alpha,
        max_new_tokens=PROBE_MAX_NEW_TOKENS,
        steering_direction=LINE_E_V3_STEERING_DIRECTION,
        steering_apply=steering_apply,
        donor_restore=bundle.get("donor_restore"),
        slot_consistent_restore=bundle.get("slot_consistent_restore"),
        anchor_aligned_restore=bundle.get("anchor_aligned_restore"),
        use_live_position_resolver=use_live,
        capture_mechanism_obs=alpha > 0 and control_id in PROBE_CONTROLS,
    )
    beh = behavioral_outcome(out.get("text") or "", traj)
    mech = out.get("mechanism_obs") or {}
    return {
        "control_id": control_id if alpha > 0 else "baseline",
        "alpha": alpha,
        "layer": layer,
        "position": position,
        "steering_apply": steering_apply,
        "steering_direction": LINE_E_V3_STEERING_DIRECTION,
        "patched": bool(out.get("patched")),
        "use_cache": bool(out.get("use_cache")),
        "selected_product_id": beh.get("selected_product_id"),
        "action_anchor": beh.get("action_anchor"),
        "wrong_owner": beh.get("wrong_owner"),
        "product_id_correct": beh.get("product_id_correct"),
        "hidden_cosine_to_donor_after": mech.get("hidden_cosine_to_donor_after"),
        "anchor_logit_rank_mean": mech.get("anchor_logit_rank_mean"),
        "text_head": (out.get("text") or "")[:200],
    }


def run_probe(
    trajectory_ids: list[str],
    *,
    device_map: str = "auto",
    shared_gpu: bool = True,
) -> dict[str, Any]:
    rows_index = load_trajectory_index()
    print(f"[probe_v4] loading model device_map={device_map} shared_gpu={shared_gpu} ...", flush=True)
    model, tokenizer, _ = load_hf_model(device_map=device_map, shared_gpu=shared_gpu)
    print("[probe_v4] model ready", flush=True)
    primary_cfg = line_e_v3_config(position=LINE_E_V3_PRIMARY_POSITION)
    legacy_cfg = line_e_v3_config(position=LINE_E_V3_LEGACY_POSITION)
    results: list[dict[str, Any]] = []

    for tid in trajectory_ids:
        traj = rows_index.get(tid)
        if not traj:
            results.append({"trajectory_id": tid, "error": "trajectory_not_found"})
            continue
        bundle = build_trajectory_steering_bundle(
            tid,
            layer=primary_cfg.layer,
            position=primary_cfg.position,
            allow_legacy_fallback=False,
        )
        if bundle.get("error"):
            results.append({"trajectory_id": tid, "error": bundle})
            continue

        track = _trajectory_track(traj)
        messages = messages_for_condition(traj, "original", track=track)
        primary_pos = _resolve_stored_pos(
            tid, layer=primary_cfg.layer, position=primary_cfg.position, tokenizer=tokenizer, messages=messages
        )
        legacy_pos = _resolve_stored_pos(
            tid, layer=legacy_cfg.layer, position=legacy_cfg.position, tokenizer=tokenizer, messages=messages
        )
        if primary_pos < 0:
            results.append({"trajectory_id": tid, "error": "claim_onset_unresolved"})
            continue

        entry: dict[str, Any] = {
            "trajectory_id": tid,
            "clean_trajectory_id": bundle["clean_trajectory_id"],
            "commitment_stratum": bundle.get("commitment_stratum"),
            "paired_norm": bundle.get("paired_norm"),
            "slot_consistent_donor_id": (bundle.get("slot_consistent_restore") or {}).get("donor_trajectory_id"),
            "anchor_aligned_donor_id": (bundle.get("anchor_aligned_restore") or {}).get("donor_trajectory_id"),
            "slot_consistent_error": bundle.get("slot_consistent_error"),
            "anchor_aligned_error": bundle.get("anchor_aligned_error"),
            "primary": {"layer": primary_cfg.layer, "position": primary_cfg.position, "stored_token_idx": primary_pos},
            "legacy_ablation": {"layer": legacy_cfg.layer, "position": legacy_cfg.position, "stored_token_idx": legacy_pos},
            "runs": [],
        }

        baseline = _run_variant(
            model, tokenizer, traj, bundle,
            layer=primary_cfg.layer, position=primary_cfg.position, pos_idx=primary_pos,
            control_id="baseline", alpha=0.0,
        )
        entry["baseline"] = baseline
        entry["runs"].append(baseline)
        baseline_pid = baseline.get("selected_product_id")

        for alpha in LINE_E_PROBE_ALPHAS:
            for control_id in PROBE_CONTROLS:
                if control_id == "slot_consistent_restore" and not bundle.get("slot_consistent_restore"):
                    continue
                if control_id == "anchor_aligned_restore" and not bundle.get("anchor_aligned_restore"):
                    continue
                row = _run_variant(
                    model, tokenizer, traj, bundle,
                    layer=primary_cfg.layer, position=primary_cfg.position, pos_idx=primary_pos,
                    control_id=control_id, alpha=alpha,
                )
                row["product_id_diff"] = bool(
                    baseline_pid and row.get("selected_product_id") and row["selected_product_id"] != baseline_pid
                )
                entry["runs"].append(row)
                if row.get("product_id_correct") and baseline.get("wrong_owner"):
                    print(
                        f"  PASS primary {control_id} alpha={alpha} pid={row.get('selected_product_id')}",
                        flush=True,
                    )

        if legacy_pos >= 0:
            row = _run_variant(
                model, tokenizer, traj, bundle,
                layer=legacy_cfg.layer, position=legacy_cfg.position, pos_idx=legacy_pos,
                control_id="paired_caa", alpha=1.0,
            )
            row["product_id_diff"] = bool(
                baseline_pid and row.get("selected_product_id") and row["selected_product_id"] != baseline_pid
            )
            row["ablation"] = "commitment_legacy"
            entry["runs"].append(row)

        passes = [
            r for r in entry["runs"]
            if r.get("position") == primary_cfg.position
            and float(r.get("alpha") or 0) > 0
            and baseline.get("wrong_owner")
            and r.get("product_id_correct")
        ]
        entry["probe_pass"] = bool(passes)
        entry["passing_runs"] = passes
        results.append(entry)
        print(
            f"[probe_v4] {tid} stratum={entry.get('commitment_stratum')} "
            f"baseline_wrong={baseline.get('wrong_owner')} pass={entry['probe_pass']}",
            flush=True,
        )

    out = {
        "schema_version": "ccer_line_e_probe_v4_ultimate",
        "protocol_version": LINE_E_V3_PROTOCOL_VERSION,
        "primary_roi": {"layer": primary_cfg.layer, "position": primary_cfg.position},
        "steering_direction": LINE_E_V3_STEERING_DIRECTION,
        "strict_bilateral_gate": True,
        "results": results,
        "n_pass": sum(1 for r in results if r.get("probe_pass")),
    }
    P3_LINE_E_PROBE.parent.mkdir(parents=True, exist_ok=True)
    P3_LINE_E_PROBE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    # Backward-compat symlink path for tooling expecting probe_v3_results.json
    legacy_path = P3_LINE_E_PROBE.parent / "probe_v3_results.json"
    legacy_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Line E v4 ultimate fast probe")
    parser.add_argument("--trajectory-ids", type=str, default=",".join(DEFAULT_PROBE_IDS))
    parser.add_argument("--device-map", type=str, default="auto", help="auto recommended; cuda:0 only if GPU is empty")
    parser.add_argument("--shared-gpu", action="store_true", default=True, help="CPU offload when VRAM < 62GiB free")
    parser.add_argument("--no-shared-gpu", action="store_false", dest="shared_gpu")
    args = parser.parse_args()
    tids = [x.strip() for x in args.trajectory_ids.split(",") if x.strip()]
    result = run_probe(tids, device_map=args.device_map, shared_gpu=args.shared_gpu)
    print(json.dumps({"path": str(P3_LINE_E_PROBE), "n_pass": result["n_pass"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
