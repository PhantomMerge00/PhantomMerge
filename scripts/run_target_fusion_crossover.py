"""Controlled target x fusion on Shopping AGR cohort (same probe, splits, test n=274)."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json
from ccer.mechanism.agr.calibration import fit_threshold_f1
from ccer.mechanism.agr.fusion import (
    enrich_calibrated,
    eval_mode,
    fit_fusion_calibrations,
    fit_mle_fusion,
)
from ccer.mechanism.agr.signals import apply_z2_variant, build_agr_signal_rows, train_agr_probe
from ccer.mechanism.bind_surprise import try_auroc
from ccer.paths import AGR_DIR


def _rows_with_z2_column(rows: list[dict], z2_key: str) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        rr = deepcopy(r)
        z = float(rr.get(z2_key) if rr.get(z2_key) is not None else float("nan"))
        if z2_key == "z2_anchor" and not rr.get("z2_available"):
            continue
        if not (z == z):  # nan
            continue
        rr["z2"] = z
        rr["z2_odds"] = float(rr.get("z2_odds") or 0.0)
        out.append(rr)
    return out


def _build_dual_target_rows() -> list[dict]:
    probe = train_agr_probe(train_split="D_p")
    rows = build_agr_signal_rows(probe, z2_source="anchor_value")
    apply_z2_variant(rows, variant="odds", z2_source="anchor_value")
    for r in rows:
        r["z2_anchor"] = r["z2"]
        r["z2_slot"] = float(r.get("z2_log_mass") or float("nan"))
    return rows


def _eval_cell(rows: list[dict], *, fusion: str) -> dict:
    cal = fit_fusion_calibrations(rows, cal_split="D_c")
    enrich_calibrated(rows, cal)
    weights = fit_mle_fusion(rows, fit_split="D_f", use_calibrated=True, interaction=False)
    test = [r for r in rows if r.get("agr_split") == "test"]
    mode = "calibrated_equal" if fusion == "fixed" else "full_fusion"
    return eval_mode(test, mode=mode, cal=cal, weights=weights, all_rows=rows)


def main() -> int:
    base = _build_dual_target_rows()
    anchor_rows = _rows_with_z2_column(base, "z2_anchor")
    slot_rows = _rows_with_z2_column(base, "z2_slot")

    iak_anchor = {r["instance_audit_key"] for r in anchor_rows}
    iak_slot = {r["instance_audit_key"] for r in slot_rows}
    common = iak_anchor & iak_slot
    common_rows_anchor = [r for r in anchor_rows if r["instance_audit_key"] in common]
    common_rows_slot = [r for r in slot_rows if r["instance_audit_key"] in common]

    grid = {}
    for target, subset in (
        ("anchor_value", anchor_rows),
        ("slot_lexicon", slot_rows),
    ):
        grid[target] = {
            "fixed_equal": _eval_cell(deepcopy(subset), fusion="fixed"),
            "learned_mle": _eval_cell(deepcopy(subset), fusion="learned"),
            "n_rows": len(subset),
        }

    grid_common = {}
    for target, subset in (
        ("anchor_value", common_rows_anchor),
        ("slot_lexicon", common_rows_slot),
    ):
        grid_common[target] = {
            "fixed_equal": _eval_cell(deepcopy(subset), fusion="fixed"),
            "learned_mle": _eval_cell(deepcopy(subset), fusion="learned"),
            "n_rows": len(subset),
        }

    # BindSurprise headline (prism) on full test for reference
    from ccer.io_utils import load_jsonl
    from ccer.mechanism.prism_audit import build_audit_index
    from ccer.paths import PRISM_AUDIT_JSONL

    audit = build_audit_index(list(load_jsonl(PRISM_AUDIT_JSONL)))
    test_keys = {r["instance_audit_key"] for r in base if r.get("agr_split") == "test"}
    bs_scores = []
    y = []
    for iak in test_keys:
        pa = audit.get(iak) or {}
        if "bind_surprise" not in pa:
            continue
        row = next(r for r in base if r["instance_audit_key"] == iak)
        bs_scores.append(float(pa["bind_surprise"]))
        y.append(int(row["y_pm"]))
    import numpy as np

    from sklearn.metrics import roc_auc_score

    bind_auroc = float(roc_auc_score(y, bs_scores)) if len(set(y)) > 1 else None

    out = {
        "schema": "target_fusion_crossover_v1",
        "protocol": (
            "Same Line-A v3 AGR probe (D_p); affine cal D_c; tau from D_f F1 per cell; "
            "eval test only. anchor_value subset requires z2_available."
        ),
        "n_test_full": len([r for r in base if r.get("agr_split") == "test"]),
        "n_test_common_anchor_and_slot": len(common),
        "bind_surprise_auroc_test_reference": bind_auroc,
        "full_cohort": grid,
        "common_subset": grid_common,
    }
    path = AGR_DIR / "TARGET_FUSION_CROSSOVER.json"
    write_json(path, out)

    md = AGR_DIR / "TARGET_FUSION_CROSSOVER.md"
    lines = [
        "# Target x fusion crossover (Shopping test)",
        "",
        f"- Full test n={out['n_test_full']}; common anchor∩slot n={out['n_test_common_anchor_and_slot']}",
        f"- BindSurprise AUROC (reference): {bind_auroc}",
        "",
        "## Full cohort (per-target eligible n)",
        "",
        "| Target | Fusion | n | AUROC | F1 | tau (D_f) |",
        "|--------|--------|--:|------:|---:|----------:|",
    ]
    for target, block in grid.items():
        for fusion_key, label in (("fixed_equal", "fixed"), ("learned_mle", "learned")):
            m = block[fusion_key]
            lines.append(
                f"| {target} | {label} | {block['n_rows']} | {m.get('auroc')} | {m.get('f1')} | {m.get('threshold')} |"
            )
    lines.extend(["", "## Common subset (both targets defined)", ""])
    lines.append("| Target | Fusion | n | AUROC | F1 |")
    lines.append("|--------|--------|--:|------:|---:|")
    for target, block in grid_common.items():
        for fusion_key, label in (("fixed_equal", "fixed"), ("learned_mle", "learned")):
            m = block[fusion_key]
            lines.append(
                f"| {target} | {label} | {block['n_rows']} | {m.get('auroc')} | {m.get('f1')} |"
            )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
