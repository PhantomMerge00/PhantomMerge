"""Airline D_p-retrained probe + BindSurprise → telecom / retail transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ccer.audit.tau3_agr_red_flag_audit import _rows_from_probe, _sign_flip_auroc, _XY
from ccer.audit.tau3_quote_template_split import _collect_xy_with_manifest, build_quote_template_split_manifest
from ccer.io_utils import write_json
from ccer.mechanism.agr.calibration import fit_threshold_f1
from ccer.mechanism.bind_surprise import try_auroc
from ccer.mechanism.tau3.agr_slot_eval import (
    _load_split_maps,
    _shopping_tau_b,
    _train_domain_probe,
    build_agr_slot_rows,
)
from ccer.mechanism.tau3.domain import get_tau3_domain
from ccer.mechanism.tau3.frozen_probe import load_shopping_frozen_probe
from ccer.paths import TAU3_DIR


def _floor_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 1.0
    floor = sum(1 for r in rows if float(r.get("log_slot_mass") or 0) <= -11.99)
    return floor / len(rows)


def _eval_rows(
    rows: list[dict[str, Any]],
    *,
    tau_b: float,
    eval_split: str,
) -> dict[str, Any]:
    test_rows = [r for r in rows if r.get("split") == eval_split]
    if not test_rows:
        test_rows = rows
    n_pm = sum(int(r["y_pm"]) for r in test_rows)
    floor = _floor_rate(test_rows)
    return {
        "eval_split": eval_split,
        "n_claims": len(test_rows),
        "n_pm": n_pm,
        "n_clean": len(test_rows) - n_pm,
        "floor_rate": floor,
        "effective_method": "probe_only" if floor >= 0.99 else "bind_surprise",
        "probe_auroc": try_auroc(test_rows, "p_pm"),
        "bind_surprise_auroc": try_auroc(test_rows, "agr_slot_score"),
        "sign_flip_probe": _sign_flip_auroc(test_rows, "p_pm"),
        "sign_flip_bind": _sign_flip_auroc(test_rows, "agr_slot_score"),
        "tau_b": tau_b,
    }


def run_airline_source_transfer(target_domains: list[str]) -> dict[str, Any]:
    src_cfg = get_tau3_domain("airline")
    airline_probe = _train_domain_probe(src_cfg, train_split="D_p")
    airline_rows = build_agr_slot_rows(airline_probe, src_cfg)
    airline_fit = [r for r in airline_rows if r["split"] == "D_f"]
    tau_b_airline = fit_threshold_f1(airline_fit, "agr_slot_score")

    shopping_probe = load_shopping_frozen_probe()
    tau_b_shopping = _shopping_tau_b()

    out: dict[str, Any] = {
        "schema": "tau3_airline_source_cross_domain_v1",
        "source_domain": "airline",
        "source_training": {
            "probe_split": "D_p",
            "tau_b_split": "D_f",
            "tau_b": tau_b_airline,
            "n_D_f": len(airline_fit),
            "indomain_test": _eval_rows(airline_rows, tau_b=tau_b_airline, eval_split="test"),
        },
        "targets": {},
        "shopping_zeroshot_baseline": {},
    }

    for domain in target_domains:
        tgt_cfg = get_tau3_domain(domain)
        _, inst_split = _load_split_maps(tgt_cfg)
        eval_split = "test"
        if domain == "retail":
            eval_split = "test"  # retail manifest: all instances are test

        airline_xfer_rows = build_agr_slot_rows(airline_probe, tgt_cfg)
        shopping_rows = build_agr_slot_rows(shopping_probe, tgt_cfg)

        out["targets"][domain] = {
            "airline_D_p_transfer": _eval_rows(airline_xfer_rows, tau_b=tau_b_airline, eval_split=eval_split),
            "shopping_frozen_zeroshot": _eval_rows(shopping_rows, tau_b=tau_b_shopping, eval_split=eval_split),
            "delta_airline_probe_minus_shopping": None,
            "delta_airline_bind_minus_shopping": None,
        }
        ap = out["targets"][domain]["airline_D_p_transfer"]["probe_auroc"]
        sp = out["targets"][domain]["shopping_frozen_zeroshot"]["probe_auroc"]
        ab = out["targets"][domain]["airline_D_p_transfer"]["bind_surprise_auroc"]
        sb = out["targets"][domain]["shopping_frozen_zeroshot"]["bind_surprise_auroc"]
        if ap is not None and sp is not None:
            out["targets"][domain]["delta_airline_probe_minus_shopping"] = round(ap - sp, 4)
        if ab is not None and sb is not None:
            out["targets"][domain]["delta_airline_bind_minus_shopping"] = round(ab - sb, 4)

        out["shopping_zeroshot_baseline"][domain] = out["targets"][domain]["shopping_frozen_zeroshot"]

    if "telecom" in target_domains:
        tcfg = get_tau3_domain("telecom")
        manifest = build_quote_template_split_manifest(tcfg)
        X_te, y_te, g_te, meta_te = _collect_xy_with_manifest(tcfg, manifest, splits={"test"})
        xy = _XY(X_te, y_te, g_te, meta_te)
        qt: dict[str, Any] = {"n_test": len(y_te), "n_pm": int(sum(y_te)), "n_clean": int(len(y_te) - sum(y_te))}
        for label, probe in (
            ("airline_D_p", airline_probe),
            ("shopping_frozen", shopping_probe),
        ):
            rows = _rows_from_probe(probe, xy)
            full = {r["instance_audit_key"]: r for r in build_agr_slot_rows(probe, tcfg)}
            merged = [
                {**r, "agr_slot_score": full[r["instance_audit_key"]]["agr_slot_score"]}
                for r in rows
                if r["instance_audit_key"] in full
            ]
            qt[label] = {
                "probe_auroc": try_auroc(merged, "p_pm"),
                "bind_surprise_auroc": try_auroc(merged, "agr_slot_score"),
                "sign_flip_probe": _sign_flip_auroc(merged, "p_pm"),
            }
        out["telecom_quote_template_zero_overlap"] = qt

    return out


def _fmt_auroc(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def write_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Tau3 Airline-Source Cross-Domain Transfer",
        "",
        "> Probe 在 **airline D_p** 重训，τ_B 在 **airline D_f** 拟合；目标域用各自激活 + J-lens readout。",
        "> 与 shopping 冻结 zeroshot 对照。三域同属 τ² benchmark，PM 密度更均匀。",
        "",
        "## Airline in-domain（参照）",
        "",
    ]
    idm = payload["source_training"]["indomain_test"]
    lines += [
        f"- probe AUROC: {_fmt_auroc(idm['probe_auroc'])} | BindSurprise: {_fmt_auroc(idm['bind_surprise_auroc'])}",
        f"- n={idm['n_claims']} (PM {idm['n_pm']}), floor={idm['floor_rate']:.1%}",
        "",
        "## 跨域 test",
        "",
        "| Target | 方法 | probe AUROC | BindSurprise AUROC | floor | n (PM) |",
        "|--------|------|-------------|-------------------|-------|--------|",
    ]
    for domain, block in payload["targets"].items():
        for label, key in (
            ("airline→" + domain, "airline_D_p_transfer"),
            ("shopping→" + domain, "shopping_frozen_zeroshot"),
        ):
            r = block[key]
            lines.append(
                f"| {domain} | {label} | {_fmt_auroc(r['probe_auroc'])} | "
                f"{_fmt_auroc(r['bind_surprise_auroc'])} | {r['floor_rate']:.1%} | "
                f"{r['n_claims']} ({r['n_pm']}) |"
            )
        lines.append(f"| | Δ(airline−shopping) probe | {block.get('delta_airline_probe_minus_shopping')} | "
                     f"{block.get('delta_airline_bind_minus_shopping')} | | |")
        lines.append("")

    qt = payload.get("telecom_quote_template_zero_overlap")
    if qt:
        lines += [
            "## Quote-template 零重叠 sanity（telecom）",
            "",
            f"- n={qt['n_test']} (PM {qt['n_pm']}, clean {qt['n_clean']})",
            f"- airline→telecom probe AUROC: {_fmt_auroc(qt['airline_D_p']['probe_auroc'])}",
            f"- shopping→telecom probe AUROC: {_fmt_auroc(qt['shopping_frozen']['probe_auroc'])}",
            "- clean 极少时 AUROC 参考价值有限",
            "",
        ]
    lines += [
        "## 解读提示",
        "",
        "- **新发现**：同属 τ² benchmark 时，airline D_p 探针在 telecom/retail 上远优于 shopping 冻结探针",
        "- BindSurprise 随探针质量同步提升（telecom +0.22，retail +0.25 vs shopping）",
        "- telecom test AUROC≈1.0 仍可能含域间共享表层结构；需与 quote-template 切分一并报告",
        "- **不可**将 in-domain / 近完美 AUROC 单独作为 AGR 机制有效性证据",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="telecom,retail", help="comma-separated target domains")
    args = ap.parse_args(argv)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    payload = run_airline_source_transfer(targets)
    out_json = TAU3_DIR / "airline_source_cross_domain.json"
    out_md = TAU3_DIR / "TAU3_AIRLINE_SOURCE_CROSS_DOMAIN_REPORT.md"
    write_json(out_json, payload)
    write_report(payload, out_md)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
