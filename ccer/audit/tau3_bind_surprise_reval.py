"""Tau3 BindSurprise re-eval after slot_norm fix + slot readout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# json used in run_domain for manifest read
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from ccer.io_utils import write_json
from ccer.mechanism.bind_surprise import compute_bind_surprise, try_auroc
from ccer.mechanism.tau3.agr_slot_eval import (
    _load_split_maps,
    _shopping_tau_b,
    _train_domain_probe,
    build_agr_slot_rows,
)
from ccer.mechanism.tau3.domain import Tau3DomainConfig, get_tau3_domain
from ccer.mechanism.tau3.frozen_probe import load_shopping_frozen_probe
from ccer.paths import TAU3_DIR

from ccer.audit.tau3_agr_red_flag_audit import _fit_probe, _rows_from_probe, _sign_flip_auroc, _XY
from ccer.audit.tau3_quote_template_split import (
    _collect_xy_with_manifest,
    build_quote_template_split_manifest,
    evaluate_probe_on_quote_template_split,
)


def _bind_surprise_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    floor = sum(1 for r in rows if float(r.get("log_slot_mass") or 0) <= -11.99)
    n = max(len(rows), 1)
    masses = [float(r.get("log_slot_mass") or -99) for r in rows]
    finite = [m for m in masses if m > -11.99]
    return {
        "floor_rate": floor / n,
        "n_with_finite_log_slot_mass": len(finite),
        "log_slot_mass_mean_finite": float(np.mean(finite)) if finite else None,
        "effective_method": "bind_surprise" if floor / n < 0.99 else "probe_only",
    }


def _score_rows(rows: list[dict[str, Any]], *, score_key: str) -> list[dict[str, Any]]:
    return [{**r, "p_score": float(r.get(score_key) or 0.0)} for r in rows]


def run_domain(cfg: Tau3DomainConfig, *, test_only: bool = False) -> dict[str, Any]:
    if cfg.split_manifest.is_file():
        manifest_payload = json.loads(cfg.split_manifest.read_text(encoding="utf-8"))
        test_only = test_only or not manifest_payload.get("indomain_eligible", True)
    else:
        manifest_payload = {}

    if test_only or cfg.domain == "retail":
        manifest = manifest_payload if manifest_payload else build_quote_template_split_manifest(cfg)
        zs_probe = load_shopping_frozen_probe()
        zs_rows = build_agr_slot_rows(zs_probe, cfg)
        zs_test = [r for r in zs_rows if r.get("split") == "test"]
        if not zs_test:
            zs_test = zs_rows
        bs_stats = _bind_surprise_stats(zs_test)
        zf = _sign_flip_auroc(zs_test, "p_pm")
        zf_b = _sign_flip_auroc(zs_test, "agr_slot_score")
        return {
            "schema": "tau3_bind_surprise_reval_v1",
            "domain": cfg.domain,
            "eval_mode": "zeroshot_test_only",
            "indomain_eligible": False,
            "bind_surprise_health": bs_stats,
            "original_split_test": {
                "zeroshot": {
                    "probe_p_pm_auroc": try_auroc(zs_test, "p_pm"),
                    "bind_surprise_auroc": try_auroc(zs_test, "agr_slot_score"),
                    "sign_flip_probe": zf,
                    "sign_flip_bind": zf_b,
                    "n_test": len(zs_test),
                },
                "indomain_retrained": {
                    "skipped": True,
                    "reason": manifest_payload.get("indomain_skip_reason", "test_only cohort"),
                },
            },
            "quote_template_zero_overlap_split": {"skipped": True, "reason": "insufficient PM for quote-template split"},
        }

    _, inst_split = _load_split_maps(cfg)
    manifest = build_quote_template_split_manifest(cfg)

    zs_probe = load_shopping_frozen_probe()
    id_probe = _train_domain_probe(cfg, train_split="D_p")

    zs_rows = build_agr_slot_rows(zs_probe, cfg)
    id_rows = build_agr_slot_rows(id_probe, cfg)

    def _test(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if r.get("split") == "test"]

    zs_test = _test(zs_rows)
    id_test = _test(id_rows)
    id_train = [r for r in id_rows if r.get("split") == "D_p"]

    bs_stats = _bind_surprise_stats(id_test)

    def _metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        scored = _score_rows(rows, score_key=key)
        preds = [float(r["p_score"]) >= 0.0 for r in scored]  # log-domain threshold varies
        y = [int(r["y_pm"]) for r in rows]
        # AUROC only needs ranking
        au = try_auroc([{"y_pm": y[i], "p_pm": scored[i]["p_score"]} for i in range(len(rows))], "p_pm")
        return {"auroc": au, "n": len(rows)}

    probe_eval = evaluate_probe_on_quote_template_split(cfg, manifest)
    X_te, y_te, _, meta_te = _collect_xy_with_manifest(cfg, manifest, splits={"test"})
    zs_probe_rows = _rows_from_probe(zs_probe, _XY(X_te, y_te, np.array([]), meta_te))

    out = {
        "schema": "tau3_bind_surprise_reval_v1",
        "domain": cfg.domain,
        "slot_norm_fix": "adjudication reads normalized slot_norm; clean inferred from tool-JSON quote",
        "bind_surprise_health": bs_stats,
        "original_split_test": {
            "zeroshot": {
                "probe_p_pm_auroc": try_auroc(zs_test, "p_pm"),
                "bind_surprise_auroc": try_auroc(zs_test, "agr_slot_score"),
                "sign_flip_probe": _sign_flip_auroc(zs_test, "p_pm"),
                "sign_flip_bind": _sign_flip_auroc(zs_test, "agr_slot_score"),
                "n_test": len(zs_test),
            },
            "indomain_retrained": {
                "train_D_p_probe_auroc": try_auroc(id_train, "p_pm"),
                "test_probe_auroc": try_auroc(id_test, "p_pm"),
                "test_bind_surprise_auroc": try_auroc(id_test, "agr_slot_score"),
                "probe_vs_bind_auroc_delta": None,
                "n_test": len(id_test),
            },
        },
        "quote_template_zero_overlap_split": probe_eval,
        "quote_template_zeroshot_probe_auroc": try_auroc(zs_probe_rows, "p_pm"),
    }
    pt = out["original_split_test"]["indomain_retrained"]["test_probe_auroc"]
    bt = out["original_split_test"]["indomain_retrained"]["test_bind_surprise_auroc"]
    if pt is not None and bt is not None:
        out["original_split_test"]["indomain_retrained"]["probe_vs_bind_auroc_delta"] = round(bt - pt, 4)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="telecom", choices=["telecom", "airline", "both"])
    args = ap.parse_args(argv)
    domains = ["telecom", "airline"] if args.domain == "both" else [args.domain]
    results: dict[str, Any] = {}
    for d in domains:
        cfg = get_tau3_domain(d)
        r = run_domain(cfg)
        out = cfg.report.parent / "bind_surprise_reval.json"
        write_json(out, r)
        results[d] = r
        print(
            json.dumps(
                {
                    d: {
                        "floor_rate": r["bind_surprise_health"]["floor_rate"],
                        "effective": r["bind_surprise_health"]["effective_method"],
                        "test_probe_auroc": r["original_split_test"]["indomain_retrained"]["test_probe_auroc"],
                        "test_bind_auroc": r["original_split_test"]["indomain_retrained"]["test_bind_surprise_auroc"],
                        "delta": r["original_split_test"]["indomain_retrained"]["probe_vs_bind_auroc_delta"],
                    }
                },
                indent=2,
            )
        )

    report = TAU3_DIR / "TAU3_BIND_SURPRISE_REVAL_REPORT.md"
    lines = [
        "# Tau3 BindSurprise Re-evaluation (slot_norm fix)",
        "",
        "BindSurprise: B = ℓ_pm + log π_J(T_slot). Requires non-empty `slot_norm` + J-lens readout.",
        "",
    ]
    for d, r in results.items():
        h = r["bind_surprise_health"]
        z = r["original_split_test"]["zeroshot"]
        i = r["original_split_test"]["indomain_retrained"]
        lines += [
            f"## {d.capitalize()}",
            "",
            f"- **floor_rate**: {h['floor_rate']:.1%} → `{h['effective_method']}`",
            f"- **finite log_slot_mass**: {h['n_with_finite_log_slot_mass']} / {i['n_test']} test claims",
            f"- **in-domain test AUROC**: probe={i['test_probe_auroc']}, BindSurprise={i['test_bind_surprise_auroc']}, Δ={i['probe_vs_bind_auroc_delta']}",
            f"- **zero-shot**: probe={z['probe_p_pm_auroc']}, BindSurprise={z['bind_surprise_auroc']}",
            "",
        ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
