"""P3a: layer/position scan + owner PCA + position control → binding ROI v2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.mechanism.iia_roi_probe import pick_roi_by_iia_probe
from ccer.mechanism.owner_pca import run_owner_pca_analysis, write_binding_roi
from ccer.paths import P3_BINDING_ROI, REPORTS
from ccer.replay.hf_forward import load_hf_model


def run_p3a(
    *,
    seed: int = 42,
    track: str = "CAP",
    iia_pick: bool = False,
    device_map: str = "auto",
    probe_pairs: int = 4,
    probe_max_tokens: int = 96,
    fast_screen: bool = True,
    layer_lo: int = 28,
    layer_hi: int = 54,
) -> dict:
    track_name = "CEM" if track.upper() == "CEM" else "CAP"
    iia_probe_table: list[dict] | None = None
    peak_layer_override: int | None = None
    peak_pos_override: str | None = None

    if iia_pick:
        print("[p3a] loading model for IIA probe...", flush=True)
        model, tokenizer, _ = load_hf_model(device_map=device_map)
        print("[p3a] model loaded, starting IIA probe", flush=True)
        iia_probe_table, _ = pick_roi_by_iia_probe(
            model=model,
            tokenizer=tokenizer,
            track=track_name,
            layer_lo=layer_lo,
            layer_hi=layer_hi,
            probe_pairs=probe_pairs,
            probe_max_tokens=probe_max_tokens,
            fast_screen=fast_screen,
        )
        if iia_probe_table:
            best = iia_probe_table[0]
            peak_layer_override = int(best["layer"])
            peak_pos_override = str(best["position"])

    roi, u_pca, u_das = run_owner_pca_analysis(
        seed=seed,
        track=track_name,
        peak_layer_override=peak_layer_override,
        peak_pos_override=peak_pos_override,
        iia_probe_table=iia_probe_table,
    )
    write_binding_roi(roi, u_pca=u_pca, u_das=u_das)
    REPORTS.mkdir(parents=True, exist_ok=True)
    primary = roi.get("primary_roi") or {}
    pos_control = roi.get("factorial_disentanglement_table") or {}
    pos_table = roi.get("position_comparison_table") or []
    md = [
        "# P3a Binding ROI",
        "",
        f"- track: {track_name}",
        f"- selection_method: {primary.get('selection_method', 'pca_scan')}",
        f"- primary layer: {primary.get('layer')}",
        f"- primary position: {primary.get('position')}",
        f"- iia_probe_output_diff_rate: {primary.get('iia_probe_output_diff_rate')}",
        f"- roi_status: {primary.get('roi_status', 'candidate')}",
        f"- rank: {primary.get('rank')}",
        f"- pc variance explained: {primary.get('pc_variance_explained')}",
        f"- signal_present: {roi.get('signal_present')}",
        "",
        "## IIA probe table (if used)",
        "",
        json.dumps(roi.get("iia_probe_table") or [], indent=2),
        "",
        "## Position-permutation control",
        "",
        json.dumps(pos_control, indent=2),
        "",
        "## Cross-position separation (layer fixed at primary)",
        "",
        "| position | follow_sep | cross_sep | combined | n_within | n_cross |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pos_table:
        md.append(
            f"| {row.get('position')} | {row.get('follow_separation', 0):.3f} | "
            f"{row.get('cross_separation', 0):.3f} | {row.get('combined_score', 0):.3f} | "
            f"{row.get('n_within')} | {row.get('n_cross')} |"
        )
    if pos_control.get("recency_confounded"):
        md.append("")
        md.append("> **WARNING**: primary ROI fails position-permutation control (recency/salience confound).")
    (REPORTS / "p3_binding_roi.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    ctrl_md = [
        "# P3 Position-Permutation Control",
        "",
        json.dumps(pos_control, indent=2),
        "",
        "## Cross-position table",
        "",
        json.dumps(pos_table, indent=2),
    ]
    (REPORTS / "p3_position_control.md").write_text("\n".join(ctrl_md) + "\n", encoding="utf-8")
    return roi


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--track", default="CAP", choices=["CAP", "CEM"])
    ap.add_argument("--iia-pick", action="store_true", help="Pick ROI by interchange output diff (mid-layer probe)")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--probe-pairs", type=int, default=4)
    ap.add_argument("--probe-max-tokens", type=int, default=96)
    ap.add_argument("--fast-screen", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--layer-lo", type=int, default=28)
    ap.add_argument("--layer-hi", type=int, default=54)
    args = ap.parse_args()
    summary = run_p3a(
        seed=args.seed,
        track=args.track,
        iia_pick=args.iia_pick,
        device_map=args.device_map,
        probe_pairs=args.probe_pairs,
        probe_max_tokens=args.probe_max_tokens,
        fast_screen=args.fast_screen,
        layer_lo=args.layer_lo,
        layer_hi=args.layer_hi,
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "schema_version",
                    "track",
                    "primary_roi",
                    "signal_present",
                    "factorial_disentanglement_table",
                    "iia_probe_table",
                )
                if k in summary
            },
            indent=2,
        )
    )
