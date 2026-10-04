#!/usr/bin/env python3
"""Export per-claim scores for RQ2 ROC/PR/distribution figures (test split)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.pipelines.run_detection_baselines import _load_rows

ART = ROOT / "results/rq2/curves"


def _merge_scores(path: Path, field: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        iak = str(r.get("instance_audit_key") or "")
        if iak and field in r:
            out[iak] = float(r[field])
    return out


def _load_rarr_gate() -> dict[str, float]:
    from ccer.mechanism.rarr_vllm_shim import parse_agreement_gate
    from ccer.paths import PRISM_L_PLUS_CACHE

    flags: dict[str, float] = {}
    cache_path = PRISM_L_PLUS_CACHE / "rarr_official.jsonl"
    if not cache_path.is_file():
        return flags
    for line in cache_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = str(rec.get("cache_key") or "")
        if "|gate|" not in key:
            continue
        iak = key.split("|", 4)[1]
        is_open, _, _ = parse_agreement_gate(rec.get("response") or "")
        flags[iak] = max(flags.get(iak, 0.0), 1.0 if is_open else 0.0)
    return flags


def export_scores() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    rows = [r for r in _load_rows() if r.get("split") == "test"]

    vendor_root = ROOT / "results/rq2/p2_vendor"
    vendors = {
        "D-SELFCHECK-NLI": vendor_root / "selfcheck_nli/scores.jsonl",
        "D-FACTOOL-KBQA": vendor_root / "factool_kbqa/scores.jsonl",
        "D-FACTSCORE-OBS": vendor_root / "factscore/scores.jsonl",
    }
    vendor_fields = {
        "D-SELFCHECK-NLI": "p_selfcheck_nli",
        "D-FACTOOL-KBQA": "p_factool_kbqa",
        "D-FACTSCORE-OBS": "p_factscore",
    }
    merged: dict[str, dict[str, float]] = {}
    for mid, path in vendors.items():
        merged[mid] = _merge_scores(path, vendor_fields[mid])

    rarr = _load_rarr_gate()

    specs: list[tuple[str, str, str | None]] = [
        ("D-RARR-G", "rarr_gate", "rarr"),
        ("D-FACTSCORE-OBS", "p_factscore", "vendor"),
        ("D-FACTOOL-KBQA", "p_factool_kbqa", "vendor"),
        ("D-MBERT-Q", "p_modernbert_quote", "inline"),
        ("D-TFIDF", "p_bow", "inline"),
        ("D-SELFCHECK-NLI", "p_selfcheck_nli", "vendor"),
        ("D-AGR-SLOT", "bind_surprise", "inline"),
    ]

    for mid, key, src in specs:
        records = []
        for r in rows:
            iak = str(r["instance_audit_key"])
            if src == "rarr":
                score = float(rarr.get(iak, 0.0))
            elif src == "vendor":
                score = float(merged[mid].get(iak, 0.0))
            else:
                score = float(r.get(key) or 0.0)
            records.append(
                {
                    "instance_audit_key": iak,
                    "trajectory_id": r.get("trajectory_id"),
                    "score": score,
                    "gold_label": int(r.get("y_pm") or 0),
                }
            )
        out_path = ART / f"{mid}_scores.json"
        payload = {
            "method_id": mid,
            "split": "test",
            "n_claims": len(records),
            "n_pm": sum(x["gold_label"] for x in records),
            "records": records,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


def export_dev_threshold_sweep() -> None:
    """Dev-set τ sweep for representational readout p_pm (mitigation τ=0.05 protocol)."""
    from ccer.mechanism.bind_surprise import detection_metrics

    dev = [r for r in _load_rows() if r.get("split") == "dev"]
    taus = [round(x, 2) for x in __import__("numpy").linspace(0.01, 0.99, 50)]
    rows_out = []
    for tau in taus:
        m = detection_metrics(dev, score_key="p_pm", threshold=float(tau))
        rows_out.append(
            {
                "tau": tau,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
            }
        )
    out = ART / "dev_p_pm_threshold_sweep.json"
    out.write_text(
        json.dumps(
            {
                "score_key": "p_pm",
                "split": "dev",
                "protocol_note": "Representational readout; τ=0.05 is frozen mitigation operating point",
                "n_claims": len(dev),
                "points": rows_out,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")


def main() -> int:
    export_scores()
    export_dev_threshold_sweep()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
