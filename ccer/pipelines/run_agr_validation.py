"""Run full AGR signal validation (Tracks P/J/C/F/R)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.audit.agr_split_manifest import write_agr_split_manifest
from ccer.io_utils import write_json
from ccer.mechanism.agr.signals import apply_z2_variant, build_agr_signal_rows, train_agr_probe
from ccer.mechanism.agr.track_c import run_track_c
from ccer.mechanism.agr.track_f import run_track_f
from ccer.mechanism.agr.track_j import run_track_j
from ccer.mechanism.agr.track_p import run_track_p
from ccer.mechanism.agr.prism_gate import evaluate_prism_gate
from ccer.mechanism.agr.track_r import default_synthetic_suite, run_oracle_routing_batch
from ccer.paths import AGR_DIR, AGR_REPORT, AGR_SUMMARY


def _render_report(summary: dict) -> str:
    lines = [
        "# AGR Signal Validation Report",
        "",
        f"**Schema**: {summary.get('schema')}",
        f"**Seed**: {summary.get('seed')}",
        "",
        "## §0 Splits",
        "",
        "```json",
        json.dumps(summary.get("splits", {}), indent=2),
        "```",
        "",
        "## Track P — Probe (Eq.1)",
        "",
        "```json",
        json.dumps(summary.get("track_p", {}), indent=2)[:8000],
        "```",
        "",
        "## Track J — Causal (Eq.2)",
        "",
        "```json",
        json.dumps(summary.get("track_j", {}), indent=2),
        "```",
        "",
        "## Track C — Calibration (Eq.3/4)",
        "",
        "```json",
        json.dumps(summary.get("track_c", {}), indent=2)[:6000],
        "```",
        "",
        "## Track F — Fusion (Eq.5/8/9)",
        "",
        "### Ablation (test)",
        "",
        "| Mode | AUROC | F1 | ECE |",
        "|------|-------|-----|-----|",
    ]
    for row in summary.get("track_f", {}).get("ablation_test", []):
        lines.append(
            f"| {row.get('mode')} | {row.get('auroc')} | {row.get('f1')} | {row.get('ece_equal_width')} |"
        )
    lines.extend(
        [
            "",
            "### 2×2 disentangle (test)",
            "",
            "```json",
            json.dumps(summary.get("track_f", {}).get("disentangle_2x2_test", {}), indent=2),
            "```",
            "",
            "### Upgrade decision",
            "",
            "```json",
            json.dumps(summary.get("track_f", {}).get("upgrade_recommendation", {}), indent=2),
            "```",
            "",
            "## Track R — Routing oracle",
            "",
            "```json",
            json.dumps(summary.get("track_r", {}), indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--z2-variant",
        choices=("log_mass", "odds"),
        default="odds",
        help="Primary z2: log_mass (legacy) or odds (Eq.2 log-odds)",
    )
    ap.add_argument(
        "--z2-source",
        choices=("anchor_value", "lexicon"),
        default="anchor_value",
        help="z2 mass source: anchor_value (Eq.2) or lexicon (legacy BindSurprise)",
    )
    args = ap.parse_args()

    split_payload = write_agr_split_manifest(seed=args.seed)
    probe = train_agr_probe(train_split="D_p")
    rows = build_agr_signal_rows(probe, tokenizer=None, z2_source=args.z2_source)
    z2_def = apply_z2_variant(rows, variant=args.z2_variant, z2_source=args.z2_source)

    track_r = run_oracle_routing_batch(default_synthetic_suite())

    track_f = run_track_f(rows)
    summary = {
        "schema": "agr_validation_v1",
        "seed": args.seed,
        "z2_variant": args.z2_variant,
        "z2_source": args.z2_source,
        "z2_definition": z2_def,
        "splits": {
            "trajectory_counts": split_payload.get("trajectory_counts"),
            "instance_counts": split_payload.get("instance_counts"),
            "verdict_counts_by_split": split_payload.get("verdict_counts_by_split"),
        },
        "track_p": run_track_p(rows, probe),
        "track_j": run_track_j(rows, z2_source=args.z2_source),
        "track_c": run_track_c(rows),
        "track_f": track_f,
        "track_r": track_r,
    }
    summary["prism_gate"] = evaluate_prism_gate(summary)

    AGR_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        AGR_DIR / "fusion_weights.json",
        {
            "calibration": track_f.get("calibration"),
            "fusion_weights": track_f.get("fusion_weights"),
            "interaction_model": track_f.get("interaction_model"),
            "upgrade_recommendation": track_f.get("upgrade_recommendation"),
        },
    )
    write_json(AGR_SUMMARY, summary)
    write_json(AGR_DIR / "PRISM_GATE_EVALUATION.json", summary["prism_gate"])
    AGR_REPORT.write_text(_render_report(summary), encoding="utf-8")
    print(f"Wrote {AGR_SUMMARY}")
    print(f"Wrote {AGR_REPORT}")
    up = summary["track_f"]["upgrade_recommendation"]
    print(f"Upgrade recommended: {up.get('recommend_upgrade')}")
    gate = summary["prism_gate"]
    print(f"Prism gate all_pass: {gate.get('all_pass')} — {gate.get('recommendation')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
