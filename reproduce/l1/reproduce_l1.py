"""Recompute AUROC/AP from frozen per-claim scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.mechanism.bind_surprise import try_auroc


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ap(rows: list[dict], score_key: str) -> float:
    from sklearn.metrics import average_precision_score

    y = np.array([int(r["y_pm"]) for r in rows], dtype=int)
    s = np.array([float(r[score_key]) for r in rows], dtype=float)
    return float(average_precision_score(y, s))


def run_l1(bundle: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    base = bundle if bundle.is_absolute() else ROOT / bundle
    scores_path = base / "rq2/table2_prf_audit/per_claim_scores_test274.jsonl"
    if not scores_path.is_file():
        scores_path = ROOT / "results/rq2/table2_prf_audit/per_claim_scores_test274.jsonl"
    if not scores_path.is_file():
        raise FileNotFoundError(scores_path)

    rows = _load_jsonl(scores_path)
    keys = {
        "representational": "p_pm",
        "jacobian_slot": "log_slot_mass",
        "agr_value": "agr_rho_logit",
        "agr_slot": "bind_surprise",
    }
    metrics: dict[str, Any] = {"n_test": len(rows), "readouts": {}}
    for name, sk in keys.items():
        metrics["readouts"][name] = {
            "score_key": sk,
            "auroc": try_auroc(rows, sk),
            "ap": _ap(rows, sk),
            "n": len(rows),
        }

    for rel in (
        "shopping/template_stratified_auroc.json",
        "rq3/FAC_BRANCH_STATS.json",
    ):
        p = base / rel if (base / rel).is_file() else ROOT / "results" / rel
        if p.is_file():
            metrics[p.stem] = json.loads(p.read_text(encoding="utf-8"))

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
