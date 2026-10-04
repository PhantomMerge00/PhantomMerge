"""Compare recomputed metrics to submission reference JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _close(a: float | None, b: float | None, tol: float) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def run_verify(bundle: Path, results: Path, ref_path: Path) -> dict[str, Any]:
    metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {"comparisons": [], "passed": True}
    readout_ref = ref.get("shopping_detection", {}).get("readouts", {})
    for key, block in metrics.get("readouts", {}).items():
        rref = readout_ref.get(key, {})
        comp = {
            "readout": key,
            "computed_auroc": block.get("auroc"),
            "reference_auroc": rref.get("auroc"),
            "computed_ap": block.get("ap"),
            "reference_ap": rref.get("ap"),
            "auroc_ok": _close(block.get("auroc"), rref.get("auroc"), 0.003),
            "ap_ok": _close(block.get("ap"), rref.get("ap"), 0.005),
        }
        if not (comp["auroc_ok"] and comp["ap_ok"]):
            report["passed"] = False
        report["comparisons"].append(comp)
    (results / "verify_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
