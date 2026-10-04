"""τ_B calibration ablation: pooled probe fixed, vary threshold fitting protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ccer.audit.tau3_pooled_probe_sweep import (
    POOL_BUILDERS,
    _calibration_rows,
    _collect_shopping,
    _collect_tau3,
    _fit_probe,
    _register_pools,
    _shopping_eval_rows,
    _tau3_eval_rows,
)
from ccer.io_utils import write_json
from ccer.mechanism.agr.calibration import fit_threshold_f1
from ccer.mechanism.bind_surprise import detection_metrics, try_auroc
from ccer.mechanism.claim_filter_audit import audit_gating, instances_to_dataframe
from ccer.mechanism.tau3.agr_slot_eval import _shopping_tau_b, build_agr_slot_rows
from ccer.mechanism.tau3.domain import get_tau3_domain
from ccer.mechanism.tau3.frozen_probe import load_shopping_frozen_probe
from ccer.paths import TAU3_DIR

EVAL_DOMAINS = ("shopping", "telecom", "airline", "retail")


def _floor_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 1.0
    floor = sum(1 for r in rows if float(r.get("log_slot_mass") or 0) <= -11.99)
    return floor / len(rows)


def _gating_metrics(rows: list[dict[str, Any]], tau_b: float) -> dict[str, Any]:
    instances = [
        {
            "instance_audit_key": r["instance_audit_key"],
            "trajectory_id": r.get("trajectory_id", r["instance_audit_key"]),
            "split": r.get("split", "test"),
            "y": r["y_pm"],
            "y_pm": r["y_pm"],
            "gold_verdict": r.get("gold_verdict", ""),
            "response_quote": r.get("response_quote", ""),
        }
        for r in rows
    ]
    probe_map = {str(r["instance_audit_key"]): float(r["agr_slot_score"]) for r in rows}
    audit = audit_gating(df := instances_to_dataframe(instances, probe_map), lambda rec, t=tau_b: float(rec.get("p_pm", 0.0)) > t)
    del df
    return audit.to_dict()


def _enrich_shopping_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add trajectory_id / response_quote for gating."""
    by_key = {
        s.instance_audit_key: s
        for s in _collect_shopping(splits={"test", "dev", "train"})
    }
    out: list[dict[str, Any]] = []
    for r in rows:
        s = by_key.get(r["instance_audit_key"])
        out.append(
            {
                **r,
                "trajectory_id": s.trajectory_id if s else r["instance_audit_key"],
                "response_quote": "",
                "split": s.split if s else "test",
            }
        )
    return out


def _test_rows(probe, domain: str) -> list[dict[str, Any]]:
    if domain == "shopping":
        return _enrich_shopping_rows(_shopping_eval_rows(probe, split="test"))
    cfg = get_tau3_domain(domain)
    return _tau3_eval_rows(probe, cfg, eval_split="test")


def _cal_rows(probe, domain: str) -> list[dict[str, Any]]:
    if domain == "shopping":
        return _enrich_shopping_rows(_shopping_eval_rows(probe, split="dev"))
    cfg = get_tau3_domain(domain)
    return _tau3_eval_rows(probe, cfg, eval_split="D_f")


def _fit_tau(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return fit_threshold_f1(rows, "agr_slot_score")


def _eval_at_tau(test_rows: list[dict[str, Any]], tau_b: float) -> dict[str, Any]:
    det = detection_metrics(test_rows, score_key="agr_slot_score", threshold=tau_b)
    gate = _gating_metrics(test_rows, tau_b)
    return {
        "tau_b": tau_b,
        "probe_auroc": try_auroc(test_rows, "p_pm"),
        "bind_surprise_auroc": try_auroc(test_rows, "agr_slot_score"),
        "detection_f1": det.get("f1"),
        "detection_precision": det.get("precision"),
        "detection_recall": det.get("recall"),
        "gating_pm_reduction": gate.get("pm_reduction"),
        "gating_cb_retention": gate.get("cb_retention_rate"),
    }


def _tau_protocols(probe, *, pooled_cal: list[dict[str, Any]]) -> dict[str, Any]:
    shopping_frozen = _shopping_tau_b()
    pooled_tau = _fit_tau(pooled_cal)
    per_domain: dict[str, float | None] = {}
    for dom in EVAL_DOMAINS:
        per_domain[dom] = _fit_tau(_cal_rows(probe, dom))
    return {
        "shopping_prism_frozen": {"global": shopping_frozen, "per_domain": {d: shopping_frozen for d in EVAL_DOMAINS}},
        "pooled_D_f": {"global": pooled_tau, "per_domain": {d: pooled_tau for d in EVAL_DOMAINS}},
        "per_domain_D_f": {"global": None, "per_domain": per_domain},
    }


def run_ablation() -> dict[str, Any]:
    _register_pools()
    probes = {
        "shopping_frozen": load_shopping_frozen_probe(),
        "pooled_shop_air_tel": _fit_probe(POOL_BUILDERS["shop_plus_airline_plus_telecom"]()),
    }
    pooled_cal = (
        _enrich_shopping_rows(_shopping_eval_rows(probes["pooled_shop_air_tel"], split="dev"))
        + _tau3_eval_rows(probes["pooled_shop_air_tel"], get_tau3_domain("telecom"), eval_split="D_f")
        + _tau3_eval_rows(probes["pooled_shop_air_tel"], get_tau3_domain("airline"), eval_split="D_f")
    )

    out: dict[str, Any] = {
        "schema": "tau3_pooled_tau_ablation_v1",
        "note": "AUROC invariant to τ_B; ablation isolates operational threshold / gating.",
        "test_sets_fixed": {d: len(_test_rows(probes["pooled_shop_air_tel"], d)) for d in EVAL_DOMAINS},
        "probes": {},
    }

    for probe_name, probe in probes.items():
        protocols = _tau_protocols(probe, pooled_cal=pooled_cal if probe_name == "pooled_shop_air_tel" else _calibration_rows(probe, {"shopping"}))
        if probe_name == "shopping_frozen":
            # shopping frozen: pooled cal = shopping dev only for pooled_D_f variant
            sf_cal = _enrich_shopping_rows(_shopping_eval_rows(probe, split="dev"))
            protocols["pooled_D_f"]["global"] = _fit_tau(sf_cal)
            protocols["pooled_D_f"]["per_domain"] = {d: protocols["pooled_D_f"]["global"] for d in EVAL_DOMAINS}

        probe_block: dict[str, Any] = {"tau_protocols": protocols, "eval_by_protocol": {}}
        for proto_name, proto in protocols.items():
            probe_block["eval_by_protocol"][proto_name] = {}
            for dom in EVAL_DOMAINS:
                test_rows = _test_rows(probe, dom)
                if proto_name == "per_domain_D_f":
                    tau = proto["per_domain"].get(dom)
                else:
                    tau = proto["global"]
                if tau is None:
                    # retail has no D_f: fall back to pooled
                    tau = protocols["pooled_D_f"]["global"] or _shopping_tau_b()
                probe_block["eval_by_protocol"][proto_name][dom] = {
                    "n_claims": len(test_rows),
                    "floor_rate": _floor_rate(test_rows),
                    **_eval_at_tau(test_rows, tau),
                }
        out["probes"][probe_name] = probe_block

    # summary deltas: pooled probe, per_domain vs shopping_frozen tau
    p = out["probes"]["pooled_shop_air_tel"]["eval_by_protocol"]
    sf = out["probes"]["shopping_frozen"]["eval_by_protocol"]
    out["summary"] = {
        "auroc_unchanged_by_tau": True,
        "pooled_probe_bind_auroc": {
            d: p["per_domain_D_f"][d]["bind_surprise_auroc"] for d in EVAL_DOMAINS
        },
        "f1_gain_per_domain_tau_vs_shopping_frozen_tau": {
            d: (p["per_domain_D_f"][d]["detection_f1"] or 0) - (p["shopping_prism_frozen"][d]["detection_f1"] or 0)
            for d in EVAL_DOMAINS
        },
        "interpretation": (
            "High AUROC with poor F1 under shopping τ_B ⇒ need domain τ calibration; "
            "similar AUROC across τ protocols confirms ranking is probe-driven."
        ),
    }
    return out


def write_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Pooled Probe τ_B Calibration Ablation",
        "",
        "> 固定 test；固定探针；只变 τ_B 拟合协议。AUROC 与 τ_B 无关。",
        "",
        "## Pooled probe (shop+air+tel) — BindSurprise AUROC（各 τ 协议相同）",
        "",
    ]
    p = payload["probes"]["pooled_shop_air_tel"]["eval_by_protocol"]["per_domain_D_f"]
    lines.append("| Domain | probe AUROC | bind AUROC | floor |")
    lines.append("|--------|-------------|------------|-------|")
    for d in EVAL_DOMAINS:
        lines.append(f"| {d} | {p[d]['probe_auroc']:.3f} | {p[d]['bind_surprise_auroc']:.3f} | {p[d]['floor_rate']:.1%} |")

    lines += ["", "## F1 @ τ_B（pooled probe，不同校准）", "", "| Domain | shopping frozen τ | pooled τ | per-domain τ |", "|--------|--------------------:|---------:|-------------:|"]
    pp = payload["probes"]["pooled_shop_air_tel"]["eval_by_protocol"]
    for d in EVAL_DOMAINS:
        f_sf = pp["shopping_prism_frozen"][d]["detection_f1"]
        f_po = pp["pooled_D_f"][d]["detection_f1"]
        f_pd = pp["per_domain_D_f"][d]["detection_f1"]
        lines.append(f"| {d} | {f_sf:.3f} | {f_po:.3f} | {f_pd:.3f} |")

    lines += ["", "## Shopping frozen probe — F1 对照", ""]
    sf = payload["probes"]["shopping_frozen"]["eval_by_protocol"]
    lines.append("| Domain | shopping τ | per-domain τ |")
    lines.append("|--------|------------:|-------------:|")
    for d in EVAL_DOMAINS:
        lines.append(
            f"| {d} | {sf['shopping_prism_frozen'][d]['detection_f1']:.3f} | "
            f"{sf['per_domain_D_f'][d]['detection_f1']:.3f} |"
        )

    lines += [
        "",
        "## 结论模板",
        "",
        "- **排序（AUROC）**：由探针 + slot readout 决定；pooled 探针解决跨域排序",
        "- **操作点（F1/门控）**：需域内 τ_B；shopping 冻结 τ 在 τ³ 上常欠拟合",
        "- **零售**：无 D_f，per-domain 回退 pooled τ",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    payload = run_ablation()
    out_json = TAU3_DIR / "pooled_tau_ablation.json"
    out_md = TAU3_DIR / "TAU3_POOLED_TAU_ABLATION_REPORT.md"
    write_json(out_json, payload)
    write_report(payload, out_md)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
