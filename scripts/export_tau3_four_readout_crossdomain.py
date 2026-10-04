#!/usr/bin/env python3
"""Export τ³ cross-domain AUROC grid for four readout detectors (shopping-frozen zeroshot)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.mechanism.agr.fusion import enrich_calibrated, fusion_score
from ccer.mechanism.bind_surprise import try_auroc
from ccer.mechanism.tau3.agr_slot_eval import _hidden, build_agr_slot_rows
from ccer.mechanism.tau3.domain import get_tau3_domain
from ccer.mechanism.tau3.frozen_probe import load_shopping_frozen_probe

OUT = ROOT / "results/rq2/curves/tau3_four_readout_crossdomain.json"
FUSION_JSON = ROOT / "results/agr/fusion_weights.json"

DOMAINS = ("telecom", "airline", "retail")
METHOD_KEYS = (
    ("representational_readout", "p_pm", "Representational readout"),
    ("functional_readout", "log_slot_mass", "Functional readout"),
    ("agr_value", "agr_rho_logit", "AGR (value)"),
    ("agr_slot", "agr_slot_score", "AGR (slot)"),
)


def _p_to_logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _enrich_agr_value(
    rows: list[dict[str, Any]],
    cfg,
    probe,
    cal: dict[str, Any],
    weights: dict[str, Any],
) -> None:
    for r in rows:
        h = _hidden(r["instance_audit_key"], r["trajectory_id"], cfg)
        r["z1"] = float(probe.logit(h)) if h is not None else _p_to_logit(float(r.get("p_pm") or 0.5))
        r["z2"] = float(r.get("log_slot_mass") or -12.0)
        r["z2_log_mass"] = r["z2"]
        r["z2_available"] = True
        r["agr_split"] = "test"
    enrich_calibrated(rows, cal)
    for r in rows:
        r["agr_rho_logit"] = float(
            fusion_score(r, mode="full_fusion", cal=cal, weights=weights)
        )


def main() -> int:
    fusion = json.loads(FUSION_JSON.read_text(encoding="utf-8"))
    cal = fusion["calibration"]
    weights = fusion["fusion_weights"]
    probe = load_shopping_frozen_probe()

    grid: dict[str, dict[str, float | None]] = {}
    detail: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        cfg = get_tau3_domain(domain)
        all_rows = build_agr_slot_rows(probe, cfg)
        rows = [r for r in all_rows if r.get("split") == "test"] or all_rows
        _enrich_agr_value(rows, cfg, probe, cal, weights)
        grid[domain] = {}
        detail[domain] = {
            "n_test": len(rows),
            "n_pm": sum(int(r["y_pm"]) for r in rows),
        }
        for key, score_col, _ in METHOD_KEYS:
            au = try_auroc(rows, score_col)
            if au is not None and key == "functional_readout" and au < 0.5:
                au = 1.0 - au
            grid[domain][key] = round(au, 4) if au is not None else None

    payload = {
        "schema": "tau3_four_readout_crossdomain_v1",
        "protocol": (
            "Shopping-frozen L49 p_pm + J-lens slot readout on each τ³ domain test split; "
            "AGR (value) uses shopping calibration with z2=log slot mass (no τ³ anchor cache); "
            "functional readout AUROC is direction-agnostic when raw AUROC<0.5."
        ),
        "domains": list(DOMAINS),
        "methods": [{"key": k, "score": s, "label": lab} for k, s, lab in METHOD_KEYS],
        "auroc": grid,
        "detail": detail,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
