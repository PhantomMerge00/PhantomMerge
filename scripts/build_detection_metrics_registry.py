#!/usr/bin/env python3
"""Full Shopping PM detection metrics registry (AUROC/AP/P/R/F1 + bootstrap CI + DeLong vs Bind)."""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.mechanism.agr.calibration import expected_calibration_error, fit_threshold_f1, sigmoid
from ccer.mechanism.agr.detector_comparison import build_unified_detector_rows
from ccer.mechanism.agr.significance import (
    cluster_bootstrap_ap,
    cluster_bootstrap_auroc,
    cluster_bootstrap_auroc_difference,
    cluster_bootstrap_detection_at_threshold,
    compare_scores_on_rows,
    delong_test,
)
from ccer.mechanism.bind_surprise import (
    detection_metrics,
    fit_log_slot_threshold,
    fit_tau_b,
    precision_at_recall,
    try_auroc,
)
from ccer.mechanism.rarr_vllm_shim import parse_agreement_gate
from ccer.paths import PRISM_L_PLUS_CACHE
from ccer.pipelines.run_detection_baselines import TAU_PROBE, _load_rows

ART = ROOT / "results"
OUT_JSON = ART / "DETECTION_METRICS_REGISTRY.json"
OUT_MD = ART / "DETECTION_METRICS_REGISTRY.md"
N_BOOT = 2000
SEED = 42
RECALL_GRID = (0.80, 0.85, 0.90, 0.917, 0.95)
BIND_ID = "D-BIND"
BIND_KEY = "bind_surprise"


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


def _load_rarr_gate_flags() -> dict[str, float]:
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
        parts = key.split("|", 4)
        if len(parts) < 2:
            continue
        iak = parts[1]
        is_open, _, _ = parse_agreement_gate(rec.get("response") or "")
        flags[iak] = max(flags.get(iak, 0.0), 1.0 if is_open else 0.0)
    return flags


def _enrich_test_rows(test: list[dict], dev: list[dict], train: list[dict]) -> None:
    vendors = [
        ("p_selfcheck_nli", ART / "rq2/p2_vendor/selfcheck_nli/scores.jsonl", "p_selfcheck_nli"),
        ("p_selfcheck_nli_multi", ART / "rq2/p2_vendor/selfcheck_nli_multi/scores.jsonl", "p_selfcheck_nli"),
        ("p_factool_kbqa", ART / "rq2/p2_vendor/factool_kbqa/scores.jsonl", "p_factool_kbqa"),
        ("p_factscore", ART / "rq2/p2_vendor/factscore/scores.jsonl", "p_factscore"),
    ]
    # multi arm uses separate row key (same jsonl field name)
    for key, path, field in vendors:
        m = _merge_scores(path, field)
        for block in (test, dev, train):
            for r in block:
                r[key] = m.get(str(r["instance_audit_key"]), 0.0)

    rarr = _load_rarr_gate_flags()
    for block in (test, dev, train):
        for r in block:
            r["rarr_gate"] = float(rarr.get(str(r["instance_audit_key"]), 0.0))

    pos_rate = sum(int(r.get("y_pm") or 0) for r in train) / max(len(train), 1)
    rng = random.Random(SEED)
    for block in (test, dev):
        for r in block:
            r["_rand_uniform"] = rng.random()
            r["_rand_strat"] = 1.0 if rng.random() < pos_rate else 0.0


def _brier(rows: list[dict], score_key: str, *, squash: Callable[[float], float] | None = None) -> float:
    err = 0.0
    n = 0
    for r in rows:
        y = int(r.get("y_pm") or 0)
        s = float(r.get(score_key) or 0.0)
        if squash:
            s = squash(s)
        s = min(1.0, max(0.0, s))
        err += (s - y) ** 2
        n += 1
    return err / n if n else float("nan")


def _audit_degeneration(op: dict[str, Any], *, n_test: int) -> dict[str, Any]:
    """Flag threshold-degenerate confusion patterns (perfect P/R often meaningless)."""
    tp = int(op.get("tp") or 0)
    fp = int(op.get("fp") or 0)
    fn = int(op.get("fn") or 0)
    tn = int(op.get("tn") or 0)
    n = n_test
    flags: list[str] = []
    n_pos_pred = tp + fp
    prec = op.get("precision")
    rec = op.get("recall")
    if n_pos_pred >= n and fn == 0:
        flags.append("all_claims_flagged_positive")
    if rec is not None and float(rec) >= 0.999 and fp > 0:
        flags.append("recall_saturated_with_many_fp")
    if fp == 0 and tp > 0:
        flags.append("zero_false_positives")
    if n_pos_pred == 0:
        flags.append("no_positive_predictions")
    if prec is not None and float(prec) >= 0.999 and "zero_false_positives" in flags:
        flags.append("precision_saturated_due_to_zero_fp")
    p_ci = op.get("precision_ci") or [None, None]
    r_ci = op.get("recall_ci") or [None, None]
    if (
        prec is not None
        and float(prec) >= 0.999
        and p_ci[0] is not None
        and abs(float(p_ci[1]) - float(p_ci[0])) < 1e-9
    ):
        flags.append("bootstrap_ci_collapsed_at_upper_bound")
    unreliable = bool(
        flags
        and (
            "all_claims_flagged_positive" in flags
            or "precision_saturated_due_to_zero_fp" in flags
            or "no_positive_predictions" in flags
        )
    )
    return {
        "flags": flags,
        "unreliable_for_paper_prf": unreliable,
        "note": (
            "Do not report P/R/F1 in main paper without footnote or use ranking metrics / fair recall-matched op."
            if unreliable
            else None
        ),
    }


def _mcc_from_det(m: dict[str, Any]) -> float:
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom else 0.0


def _fmt_ci(point: float | None, lo: float | None, hi: float | None, digits: int = 3) -> str:
    if point is None:
        return "—"
    if lo is None or hi is None:
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _fmt_pct_ci(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return ""
    return f" [{100*lo:.1f}, {100*hi:.1f}]"


def _eval_method(
    method_id: str,
    label: str,
    test: list[dict],
    *,
    score_key: str,
    paper_tier: str,
    thresholds: dict[str, float],
    prob_like: bool = False,
    squash: Callable[[float], float] | None = None,
    gating_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not test or score_key not in test[0]:
        return {"method_id": method_id, "error": f"missing score_key {score_key}"}

    auroc_boot = cluster_bootstrap_auroc(test, score_key, n_boot=N_BOOT, seed=SEED)
    ap_boot = cluster_bootstrap_ap(test, score_key, n_boot=N_BOOT, seed=SEED)

    y = np.array([int(r["y_pm"]) for r in test], dtype=int)
    sa = np.array([float(r[score_key]) for r in test], dtype=float)
    delong_vs_bind = None
    if method_id != BIND_ID:
        delong_vs_bind = delong_test(y, np.array([float(r[BIND_KEY]) for r in test]), sa)
        boot_vs_bind = cluster_bootstrap_auroc_difference(
            test, BIND_KEY, score_key, n_boot=N_BOOT, seed=SEED
        )
    else:
        boot_vs_bind = None

    par: dict[str, Any] = {}
    for tname, thr in thresholds.items():
        m = cluster_bootstrap_detection_at_threshold(test, score_key, thr, n_boot=N_BOOT, seed=SEED)
        m["mcc"] = _mcc_from_det(m)
        m["threshold_value"] = float(thr)
        m["degeneration"] = _audit_degeneration(m, n_test=len(test))
        par[tname] = m

    p_at_r = {
        str(t): precision_at_recall(test, score_key=score_key, target_recall=float(t)) for t in RECALL_GRID
    }

    brier = None
    ece = None
    if prob_like:
        probs = np.array([min(1.0, max(0.0, float(r[score_key]))) for r in test])
        brier = float(np.mean((probs - y) ** 2))
        ece = expected_calibration_error(y, probs, strategy="equal_width")
    elif squash is not None:
        probs = np.array([min(1.0, max(0.0, squash(float(r[score_key])))) for r in test])
        brier = float(np.mean((probs - y) ** 2))
        ece = expected_calibration_error(y, probs, strategy="equal_width")

    return {
        "method_id": method_id,
        "label": label,
        "paper_tier": paper_tier,
        "prob_like": prob_like,
        "score_key": score_key,
        "n_test": len(test),
        "n_pm_test": int(sum(int(r["y_pm"]) for r in test)),
        "ranking": {
            "auroc": auroc_boot,
            "ap": ap_boot,
            "brier": brier,
            "ece_equal_width": ece.get("ece") if isinstance(ece, dict) else None,
        },
        "vs_bind": {
            "bootstrap_auroc_diff": boot_vs_bind,
            "delong": delong_vs_bind,
        },
        "operating_points": par,
        "precision_at_recall": p_at_r,
        "gating_delete_policy_test": gating_extra,
    }


def _thresholds_line_a(
    dev: list[dict],
    score_key: str,
    *,
    extra: dict[str, float] | None = None,
    include_tau_mitigation: bool = True,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if include_tau_mitigation:
        out["tau_mitigation_0.05"] = TAU_PROBE
    if extra:
        out.update(extra)
    if dev:
        out["tau_dev_f1"] = fit_threshold_f1(dev, score_key)
    return out


def build_registry() -> dict[str, Any]:
    all_rows = _load_rows()
    train = [r for r in all_rows if r.get("split") == "train"]
    dev = [r for r in all_rows if r.get("split") == "dev"]
    test = [r for r in all_rows if r.get("split") == "test"]
    _enrich_test_rows(test, dev, train)

    slot_thr = fit_log_slot_threshold(dev) if dev else -3.4
    tau_b_dev = fit_tau_b(dev) if dev else {}
    tau_b_f1 = float(tau_b_dev.get("tau_b") or 0.0) if tau_b_dev else 0.0

    agr_rows, _agr_meta = build_unified_detector_rows()
    agr_test = [r for r in agr_rows if r.get("agr_split") == "test"]
    agr_df = [r for r in agr_rows if r.get("agr_split") == "D_f"]
    dc_path = ART / "agr/DETECTOR_COMPARISON_SUMMARY.json"
    dc_gating: dict[str, Any] = {}
    if dc_path.is_file():
        for d in json.loads(dc_path.read_text()).get("gating_comparison", {}).get("detectors", []):
            dc_gating[str(d.get("detector_id"))] = d

    tau_df: dict[str, float] = {}
    for det_id, sk in (
        ("frozen_probe_p_pm", "frozen_p_pm"),
        ("bind_surprise", "bind_surprise"),
        ("agr_rho", "agr_rho_logit"),
        ("agr_probe_only", "agr_probe_logit"),
    ):
        if agr_df:
            tau_df[det_id] = fit_threshold_f1(agr_df, sk)

    methods: list[dict[str, Any]] = []

    line_a_specs: list[tuple[str, str, str, str, bool, dict[str, float] | None]] = [
        (BIND_ID, "BindSurprise", BIND_KEY, "MAIN", False, {"tau_dev_f1_bind": tau_b_f1}),
        ("D-L49", "Line-A L49 probe p_pm", "p_pm", "MAIN", True, None),
        ("D-TFIDF", "TF-IDF quote BoW + LR", "p_bow", "MAIN", True, {"tau_balanced_0.6": 0.6}),
        ("D-MBERT-Q", "ModernBERT quote emb + LR", "p_modernbert_quote", "MAIN", True, None),
        ("D-MBERT-STRUCT-FAIR", "ModernBERT bcp fair", "p_modernbert_bcp_fair", "MAIN", True, None),
        (
            "D-MBERT-STRUCT-ORACLE",
            "ModernBERT bcp oracle (leakage)",
            "p_modernbert_bcp_oracle",
            "DIAGNOSTIC",
            True,
            None,
        ),
        ("D-JSL", "J-slot log mass", "log_slot_mass", "MAIN", False, {"tau_dev_median_pos": slot_thr}),
        ("D-RARR-G", "RARR agreement gate", "rarr_gate", "MAIN", True, {"tau_gate_0.5": 0.5}),
        (
            "D-SELFCHECK-NLI",
            "SelfCheck-NLI Obs_tau K=1",
            "p_selfcheck_nli",
            "MAIN",
            True,
            None,
        ),
        (
            "D-SELFCHECK-NLI-MULTI",
            "SelfCheck-NLI K=5",
            "p_selfcheck_nli_multi",
            "MAIN",
            True,
            None,
        ),
        ("D-FACTOOL-KBQA", "FacTool KBQA Obs_tau", "p_factool_kbqa", "MAIN", True, None),
        ("D-FACTSCORE-OBS", "FActScore Obs_tau", "p_factscore", "MAIN", True, None),
        ("D-RAND-UNI", "Random uniform", "_rand_uniform", "HYGIENE", True, {"tau_0.5": 0.5}),
        ("D-RAND-STRAT", "Random stratified", "_rand_strat", "HYGIENE", True, {"tau_0.5": 0.5}),
    ]

    for mid, label, sk, tier, prob, extra_thr in line_a_specs:
        use_mit = sk not in ("log_slot_mass", "_rand_uniform", "_rand_strat")
        thr = _thresholds_line_a(dev, sk, extra=extra_thr, include_tau_mitigation=use_mit)
        if mid == "D-BIND" and tau_b_f1:
            thr["tau_dev_f1_bind"] = tau_b_f1
        methods.append(
            _eval_method(
                mid,
                label,
                test,
                score_key=sk,
                paper_tier=tier,
                thresholds=thr,
                prob_like=prob,
            )
        )

    agr_specs = [
        ("D-AGR", "AGR rho(c) fusion", "agr_rho_logit", "agr_rho", True),
        ("D-AGR-PROBE", "AGR z_rep probe-only", "agr_probe_logit", "agr_probe_only", True),
        ("D-L49-FROZEN", "Frozen prism_audit p_pm (D_f tau)", "frozen_p_pm", "frozen_probe_p_pm", True),
    ]
    for mid, label, sk, dc_id, squash_logits in agr_specs:
        thr: dict[str, float] = {}
        if agr_df and sk in (agr_test[0] if agr_test else {}):
            thr["tau_D_f_f1"] = fit_threshold_f1(agr_df, sk)
        gating = None
        if dc_id in dc_gating:
            gating = dc_gating[dc_id].get("gating_test_delete_policy")
            det = dc_gating[dc_id].get("detection_test") or {}
            thr["tau_D_f_f1"] = float(dc_gating[dc_id].get("tau_D_f") or thr.get("tau_D_f_f1", 0.0))
        methods.append(
            _eval_method(
                mid,
                label,
                agr_test,
                score_key=sk,
                paper_tier="MAIN" if mid != "D-L49-FROZEN" else "APPENDIX",
                thresholds=thr,
                prob_like=False,
                squash=(lambda z: float(sigmoid(np.array([z]))[0])) if squash_logits else None,
                gating_extra=gating,
            )
        )

    bind_gating = dc_gating.get("bind_surprise", {}).get("gating_test_delete_policy")
    for m in methods:
        if m.get("method_id") == BIND_ID and bind_gating:
            m["gating_delete_policy_test"] = bind_gating

    agr_bind_sig = None
    if dc_path.is_file():
        dc_root = json.loads(dc_path.read_text())
        agr_bind_sig = (dc_root.get("gating_comparison") or {}).get(
            "significance_agr_rho_vs_bind_surprise"
        ) or dc_root.get("significance_agr_rho_vs_bind_surprise")

    p2_cal = json.loads((ART / "rq2/p2/P2_SUMMARY.json").read_text()).get("calibration") if (ART / "rq2/p2/P2_SUMMARY.json").is_file() else {}

    probe_at_05 = detection_metrics(test, score_key="p_pm", threshold=TAU_PROBE)
    probe_recall_ref = float(probe_at_05["recall"])

    recall_matched: dict[str, Any] = {
        "reference": "D-L49 probe @ tau=0.05 on test",
        "target_recall": probe_recall_ref,
        "probe_at_0.05": probe_at_05,
        "per_method": {},
    }

    unified_tau_005: list[dict[str, Any]] = []

    agr_ids = {"D-AGR", "D-AGR-PROBE", "D-L49-FROZEN"}

    for m in methods:
        if m.get("error"):
            continue
        mid = m["method_id"]
        sk = m["score_key"]
        cohort_rows = agr_test if mid in agr_ids else test
        rm = precision_at_recall(cohort_rows, score_key=sk, target_recall=probe_recall_ref)
        recall_matched["per_method"][mid] = rm

        op = (m.get("operating_points") or {}).get("tau_mitigation_0.05")
        entry: dict[str, Any] = {
            "method_id": mid,
            "tau": TAU_PROBE,
            "applicable": bool(m.get("prob_like")),
            "metrics": op,
        }
        if not m.get("prob_like"):
            entry["applicable"] = False
            entry["reason"] = "score is not a calibrated probability; tau=0.05 is not protocol-valid (use recall-matched or method-specific tau in appendix)"
        elif op is None:
            entry["applicable"] = False
            entry["reason"] = "tau=0.05 not evaluated for this score"
        elif op.get("degeneration", {}).get("unreliable_for_paper_prf"):
            entry["degenerate"] = True
            entry["degeneration_flags"] = op["degeneration"]["flags"]
        unified_tau_005.append(entry)

    probe_bind_par = precision_at_recall(test, score_key="p_pm", target_recall=probe_recall_ref)
    bind_par = precision_at_recall(test, score_key=BIND_KEY, target_recall=probe_recall_ref)

    known_degeneracy_cases = [
        {
            "method_id": "D-TFIDF",
            "operating_point": "tau_mitigation_0.05",
            "verified": True,
            "explanation": (
                "At tau=0.05 every test claim is flagged (274/274); recall=1 is tautological, not discriminative."
            ),
        },
        {
            "method_id": "D-JSL",
            "operating_point": "tau_dev_f1",
            "verified": True,
            "explanation": (
                "Dev-F1 threshold yields FP=0 on test (120 TP, 0 FP); precision=1 is saturation from sparse flags, not calibration."
            ),
        },
    ]

    return {
        "schema_version": "detection_metrics_registry_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "domain": "shopping_line_a",
            "split": "test",
            "n_claims": len(test),
            "n_pm_claims": int(sum(int(r["y_pm"]) for r in test)),
            "n_trajectories": len({r["trajectory_id"] for r in test}),
            "tau_probe_mitigation": TAU_PROBE,
        },
        "protocol": {
            "paper_main_table_columns": ["AUROC", "AP", "Precision", "Recall", "F1"],
            "paper_main_prf_rule": (
                "Use unified tau=0.05 block for P/R/F1 in main text; exclude or footnote degenerate rows. "
                "AUROC/AP are threshold-free. MCC/ECE/Brier live in JSON supplementary only."
            ),
            "ranking_ci": f"trajectory cluster bootstrap B={N_BOOT}",
            "threshold_tau_mitigation": "score >= 0.05 (prob-like detectors only; aligns with probe gating)",
            "threshold_tau_dev_f1": "appendix: F1-optimal on line_a dev (not cross-method comparable)",
            "threshold_tau_D_f_f1": "appendix: F1-optimal on AGR D_f (Bind/AGR)",
            "recall_matched_fair_op": f"ranking: precision at recall={probe_recall_ref:.4f} (probe @0.05 on test)",
            "significance_vs_bind": "DeLong paired test (claim-level) + bootstrap CI on AUROC(Bind)-AUROC(method)",
        },
        "known_degeneracy_cases": known_degeneracy_cases,
        "unified_tau_0.05": unified_tau_005,
        "recall_matched_to_probe_at_0.05": recall_matched,
        "global_calibration_p2": p2_cal,
        "significance_bind_vs_agr_rho": agr_bind_sig,
        "headline_precision_at_recall_0.917": {
            "probe": probe_bind_par,
            "bind_surprise": bind_par,
            "precision_delta_bind_minus_probe": (
                float(bind_par["precision"]) - float(probe_bind_par["precision"])
                if bind_par.get("precision") and probe_bind_par.get("precision")
                else None
            ),
        },
        "methods": methods,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# PM detection metrics registry (Shopping test)",
        "",
        f"**Cohort**: {payload['cohort']['n_claims']} claims, "
        f"{payload['cohort']['n_pm_claims']} PM+, "
        f"{payload['cohort']['n_trajectories']} trajectories.",
        "",
        "**Paper main text (recommended)**: five columns **AUROC / AP / P / R / F1** with **P/R/F1 @ unified τ=0.05** "
        "(prob-like scores only). **AUROC/AP** are threshold-free; bootstrap CI = trajectory-cluster B=2000.",
        "**Appendix**: per-method optimal τ (dev/D_f); **recall-matched** fair op for Bind/AGR/JSL. "
        "**MCC / ECE / Brier**: JSON `ranking` / `operating_points` only — not paper main table columns.",
        "",
        "## Degeneracy audit (do not ship silent perfect numbers)",
        "",
    ]
    for case in payload.get("known_degeneracy_cases") or []:
        lines.append(f"- **{case['method_id']}** @ `{case['operating_point']}`: {case['explanation']}")
    lines.extend(["", "## Paper Table A — ranking (threshold-free)", ""])
    lines.append("| ID | AUROC [95% CI] | AP [95% CI] | DeLong *p* vs Bind |")
    lines.append("|----|----------------|-------------|-------------------|")
    for m in payload["methods"]:
        if m.get("paper_tier") != "MAIN" or m.get("error"):
            continue
        rk = m["ranking"]
        delong_p = (m.get("vs_bind") or {}).get("delong") or {}
        pval = delong_p.get("pvalue")
        pval_s = f"{pval:.2e}" if pval is not None and m["method_id"] != BIND_ID else "—"
        lines.append(
            f"| {m['method_id']} | "
            f"{_fmt_ci(rk['auroc'].get('auroc'), rk['auroc'].get('ci_low'), rk['auroc'].get('ci_high'))} | "
            f"{_fmt_ci(rk['ap'].get('ap'), rk['ap'].get('ci_low'), rk['ap'].get('ci_high'))} | {pval_s} |"
        )

    lines.extend(
        [
            "",
            "## Paper Table B — P/R/F1 @ **unified τ=0.05** (mitigation-aligned; fair across prob-like methods)",
            "",
            "† = degenerate threshold (see audit). Bind/AGR/JSL: τ=0.05 invalid — use Table C.",
            "",
            "| ID | P [CI] | R [CI] | F1 [CI] | notes |",
            "|----|--------|--------|---------|-------|",
        ]
    )
    for row in payload.get("unified_tau_0.05") or []:
        mid = row["method_id"]
        if mid not in {m["method_id"] for m in payload["methods"] if m.get("paper_tier") == "MAIN"}:
            continue
        if not row.get("applicable"):
            lines.append(f"| {mid} | — | — | — | {row.get('reason', 'N/A')} |")
            continue
        op = row.get("metrics") or {}
        note = ""
        if row.get("degenerate"):
            note = "† " + ", ".join(row.get("degeneration_flags") or [])
        lines.append(
            f"| {mid} | "
            f"{_fmt_ci(op.get('precision'), (op.get('precision_ci') or [None])[0], (op.get('precision_ci') or [None, None])[1])} | "
            f"{_fmt_ci(op.get('recall'), (op.get('recall_ci') or [None])[0], (op.get('recall_ci') or [None, None])[1])} | "
            f"{_fmt_ci(op.get('f1'), (op.get('f1_ci') or [None])[0], (op.get('f1_ci') or [None, None])[1])} | {note} |"
        )

    rm = payload.get("recall_matched_to_probe_at_0.05") or {}
    tr = rm.get("target_recall")
    lines.extend(
        [
            "",
            f"## Paper Table C — precision @ recall≈{tr:.3f} (probe @0.05; **fair across all scores**)",
            "",
            "| ID | P | R | n_flagged |",
            "|----|---:|---:|----------:|",
        ]
    )
    for mid, row in sorted((rm.get("per_method") or {}).items()):
        if not mid.startswith("D-") or mid == "D-L49-FROZEN":
            continue
        if mid in ("D-RAND-UNI", "D-RAND-STRAT", "D-MBERT-STRUCT-ORACLE"):
            continue
        lines.append(
            f"| {mid} | {row.get('precision', 0):.3f} | {row.get('recall', 0):.3f} | {row.get('n_flagged', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Appendix — per-method optimal τ (NOT cross-comparable; may degenerate)",
            "",
            "| ID | τ key | P | R | F1 | TP/FP/FN/TN | degenerate? |",
            "|----|-------|---:|---:|---:|-------------:|:-------------:|",
        ]
    )
    for m in payload["methods"]:
        if m.get("paper_tier") != "MAIN" or m.get("error"):
            continue
        ops = m.get("operating_points") or {}
        for pref in ("tau_dev_f1_bind", "tau_D_f_f1", "tau_dev_f1", "tau_dev_median_pos"):
            if pref not in ops:
                continue
            op = ops[pref]
            deg = "Y" if op.get("degeneration", {}).get("unreliable_for_paper_prf") else ""
            lines.append(
                f"| {m['method_id']} | {pref} | {op.get('precision', 0):.3f} | {op.get('recall', 0):.3f} | "
                f"{op.get('f1', 0):.3f} | {op['tp']}/{op['fp']}/{op['fn']}/{op['tn']} | {deg} |"
            )
            break

    lines.extend(["", "## Gating consequence (delete policy, D_f τ)", ""])
    lines.append("| detector | CB retention | gated PM traj rate | claims retained mean |")
    lines.append("|----------|-------------:|-------------------:|---------------------:|")
    for m in payload["methods"]:
        g = m.get("gating_delete_policy_test")
        if not g:
            continue
        lines.append(
            f"| {m['method_id']} | {g.get('cb_retention_rate', 0):.3f} | "
            f"{g.get('gated_pm_rate', 0):.3f} | {g.get('claims_retained_mean', 0):.3f} |"
        )

    hr = payload.get("headline_precision_at_recall_0.917") or {}
    lines.extend(
        [
            "",
            f"## Precision @ recall≈0.917 (ranking; probe R={hr.get('probe', {}).get('recall', 0):.3f})",
            "",
            f"- Probe P={hr.get('probe', {}).get('precision', 0):.3f}",
            f"- Bind P={hr.get('bind_surprise', {}).get('precision', 0):.3f}",
            f"- ΔP (Bind−probe)={hr.get('precision_delta_bind_minus_probe', 0):.3f}",
            "",
            "Full JSON: `DETECTION_METRICS_REGISTRY.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = build_registry()
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
