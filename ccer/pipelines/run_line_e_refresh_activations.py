"""Refresh missing bilateral claim_onset activations for Line E v4 ultimate pool."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json
from ccer.mechanism.activation_store import activation_path, save_activation_npz
from ccer.mechanism.line_e_protocol import LINE_E_V3_PRIMARY_POSITION, audit_v3_pool_coverage, line_e_v3_config
from ccer.mechanism.pair_select import load_trajectory_index, messages_for_condition
from ccer.mechanism.steering_vector import mechanism_research_pm_with_activations, paired_clean_trajectory_id
# load_hf_model imported lazily — avoids slow torch import on --dry-run

OUT_PATH = ROOT / "artifacts" / "ccer" / "p3" / "line_e" / "activation_refresh_report.json"


def _trajectory_track(traj: dict) -> str:
    for claim in traj.get("claims") or []:
        if claim.get("legacy_label") == "cross_object_merge":
            return "CEM"
    return "CEM"


def refresh_missing(
    *,
    device_map: str = "auto",
    shared_gpu: bool = True,
    dry_run: bool = False,
) -> dict:
    print("[line_e refresh] scanning pool (no GPU)...", flush=True)
    cfg = line_e_v3_config(position=LINE_E_V3_PRIMARY_POSITION)
    pm_pool = mechanism_research_pm_with_activations(layer=cfg.layer, position=cfg.position)
    print(f"[line_e refresh] pm_pool n={len(pm_pool)} layer=L{cfg.layer} pos={cfg.position}", flush=True)
    audit = audit_v3_pool_coverage(pm_pool, layer=cfg.layer, position=cfg.position)
    missing = audit["missing_bilateral_pairs"]
    print(f"[line_e refresh] bilateral missing n={len(missing)} eligible={audit['n_eligible']}", flush=True)
    rows = load_trajectory_index()

    to_refresh: list[str] = []
    for item in missing:
        reason = str(item.get("reason") or "")
        pm_tid = str(item.get("pm_trajectory_id") or "")
        clean_tid = str(item.get("clean_trajectory_id") or "")
        if "donor" in reason or "clean" in reason:
            if clean_tid:
                to_refresh.append(clean_tid)
        elif "pm" in reason and pm_tid:
            to_refresh.append(pm_tid)
    to_refresh = sorted(set(t for t in to_refresh if t))

    report = {
        "schema_version": "ccer_line_e_activation_refresh_v1",
        "layer": cfg.layer,
        "position": cfg.position,
        "n_missing_pairs_before": len(missing),
        "n_trajectories_to_refresh": len(to_refresh),
        "trajectory_ids": to_refresh,
        "refreshed": [],
        "failed": [],
    }
    if dry_run or not to_refresh:
        write_json(OUT_PATH, report)
        print(
            json.dumps(
                {
                    "dry_run": dry_run,
                    "path": str(OUT_PATH),
                    "n_trajectories_to_refresh": len(to_refresh),
                    "trajectory_ids": to_refresh[:5],
                    "skip_gpu": dry_run or not to_refresh,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return report

    from ccer.replay.hf_forward import extract_activations, line_d_activation_layer_indices, load_hf_model

    print(f"[line_e refresh] loading model device_map={device_map} shared_gpu={shared_gpu} ...", flush=True)
    model, tokenizer, n_layers = load_hf_model(device_map=device_map, shared_gpu=shared_gpu)
    print(f"[line_e refresh] model ready n_layers={n_layers}", flush=True)
    layer_indices = line_d_activation_layer_indices(n_layers)

    for tid in to_refresh:
        traj = rows.get(tid)
        if not traj:
            report["failed"].append({"trajectory_id": tid, "error": "trajectory_not_found"})
            continue
        track = _trajectory_track(traj)
        messages = messages_for_condition(traj, "original", track=track)
        if not messages:
            report["failed"].append({"trajectory_id": tid, "error": "messages_unavailable"})
            continue
        output_text = str((traj.get("metadata") or {}).get("final_answer") or "")
        act = extract_activations(
            model,
            tokenizer,
            messages=messages,
            trajectory=traj,
            output_text=output_text,
            layer_indices=layer_indices,
            claim_anchor_mode="symmetric",
        )
        npz_path = activation_path(tid, "original")
        save_activation_npz(
            npz_path,
            trajectory_id=tid,
            condition_id="original",
            cohort=track,
            layer_indices=act["layer_indices"],
            positions=act["positions"],
            vectors=act["vectors"],
            metadata={
                **(act.get("metadata") or {}),
                "refreshed_line_d_v3_symmetric": True,
                "claim_anchor_mode": "symmetric",
                "line_e_refresh": True,
            },
        )
        report["refreshed"].append(
            {
                "trajectory_id": tid,
                "position_ok": bool(act.get("position_ok")),
                "claim_onset": (act.get("positions") or {}).get("claim_onset"),
            }
        )
        print(f"[line_e refresh] {tid} claim_onset={(act.get('positions') or {}).get('claim_onset')}", flush=True)

    post_audit = audit_v3_pool_coverage(pm_pool, layer=cfg.layer, position=cfg.position)
    report["n_eligible_after"] = post_audit["n_eligible"]
    report["n_excluded_after"] = post_audit["n_excluded"]
    write_json(OUT_PATH, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Line E bilateral claim_onset activations")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--shared-gpu", action="store_true", default=True)
    parser.add_argument("--no-shared-gpu", action="store_false", dest="shared_gpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print("[line_e refresh] start", "dry_run=" + str(args.dry_run), flush=True)
    report = refresh_missing(
        device_map=args.device_map,
        shared_gpu=args.shared_gpu,
        dry_run=args.dry_run,
    )
    print(json.dumps({"path": str(OUT_PATH), "n_refresh": len(report.get("refreshed") or [])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
