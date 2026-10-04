"""AGR Track C: calibration Eq.3/4."""
from __future__ import annotations

from typing import Any

from ccer.mechanism.agr.calibration import calibration_report


def run_track_c(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "z1": calibration_report(rows, z_key="z1", split="D_c"),
        "z2": calibration_report(rows, z_key="z2", split="D_c"),
    }
