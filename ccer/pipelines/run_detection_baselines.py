"""Compute full PM detection baseline matrix (no filtering) and write JSON + markdown."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.mechanism.bind_surprise import (
    detection_metrics,
    fit_log_slot_threshold,
    fit_tau_b,
    fit_tau_b_at_recall,
    try_auroc,
)
from ccer.mechanism.rarr_vllm_shim import parse_agreement_gate
from ccer.paths import LINE_K_SUMMARY, PRISM_AUDIT_JSONL, PRISM_DIR, PRISM_L_PLUS_CACHE

TAU_PROBE = 0.05


def _load_rows() -> list[dict[str, Any]]:
    rows = [json.loads(l) for l in PRISM_AUDIT_JSONL.read_text().splitlines() if l.strip()]
    bow: dict[str, float] = {}
    probe_path = ROOT / "results/line_k/probe_scores.jsonl"
    for line in probe_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        bow[r["instance_audit_key"]] = float(r.get("p_bow") or 0.0)
    mbert_q: dict[str, float] = {}
    mbert_fair: dict[str, float] = {}
    mbert_oracle: dict[str, float] = {}
    for variant, target in (
        ("quote", mbert_q),
        ("bcp_fair", mbert_fair),
        ("bcp_oracle", mbert_oracle),
    ):
        mpath = ROOT / f"results/rq2/modernbert_scores_{variant}.jsonl"
        if not mpath.is_file():
            continue
        for line in mpath.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            mbert = target
            mbert[str(rec["instance_audit_key"])] = float(rec.get("p_modernbert") or 0.0)
    for r in rows:
        iak = r["instance_audit_key"]
        r["p_bow"] = bow.get(iak, 0.0)
        r["p_modernbert_quote"] = mbert_q.get(iak, 0.0)
        r["p_modernbert_bcp_fair"] = mbert_fair.get(iak, 0.0)
        r["p_modernbert_bcp_oracle"] = mbert_oracle.get(iak, 0.0)
    return rows


def _metrics(rows: list[dict], pred_fn: Callable[[dict], bool], *, threshold: float | str = "—") -> dict:
    m = detection_metrics(rows, score_key="__unused", threshold=0.0)
    tp = fp = fn = tn = 0
    for r in rows:
        y = int(r.get("y_pm") or 0)
        pred = bool(pred_fn(r))
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif not pred and y:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "threshold": threshold,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _score_metrics(rows: list[dict], score_key: str, threshold: float) -> dict:
    return detection_metrics(rows, score_key=score_key, threshold=threshold)


def _load_rarr_gate_flags(cache_path: Path) -> dict[str, bool]:
    flags: dict[str, bool] = {}
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
        flags[iak] = flags.get(iak, False) or bool(is_open)
    return flags


def _load_rewrite_flags(
    log_path: Path,
    *,
    split: str = "test",
    tau: float = TAU_PROBE,
    arm: str,
    flagged_if: Callable[[dict], bool],
) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if not log_path.is_file():
        return out
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") != split or float(r.get("tau") or 0) != float(tau):
            continue
        if r.get("arm") != arm:
            continue
        iak = str(r.get("instance_audit_key") or "")
        out[iak] = flagged_if(r)
    return out


def _enrich_rows(rows: list[dict], extra: dict[str, bool], key: str) -> None:
    for r in rows:
        r[key] = 1.0 if extra.get(str(r.get("instance_audit_key") or ""), False) else 0.0


def build_baseline_matrix(rows: list[dict], *, seed: int = 42) -> dict[str, Any]:
    dev = [r for r in rows if r.get("split") == "dev"]
    test = [r for r in rows if r.get("split") == "test"]
    train = [r for r in rows if r.get("split") == "train"]

    tau_b = fit_tau_b(dev)
    tau_b_rec = fit_tau_b_at_recall(dev, target_recall=detection_metrics(dev, score_key="p_pm", threshold=TAU_PROBE)["recall"])
    slot_thr = fit_log_slot_threshold(dev)

    lk = json.loads(LINE_K_SUMMARY.read_text()) if LINE_K_SUMMARY.is_file() else {}
    bow_tau_agg = float((lk.get("dev_thresholds") or {}).get("quote_bow_aggressive") or TAU_PROBE)
    bow_tau_bal = float((lk.get("dev_thresholds") or {}).get("quote_bow_balanced") or 0.6)

    rng = random.Random(seed)
    pos_rate = sum(int(r.get("y_pm") or 0) for r in train) / max(len(train), 1)
    for r in test:
        r["_rand_uniform"] = rng.random()
        r["_rand_strat"] = rng.random() < pos_rate

    rarr_gate = _load_rarr_gate_flags(PRISM_L_PLUS_CACHE / "rarr_official.jsonl")
    _enrich_rows(test, rarr_gate, "rarr_gate_open")

    e2e_log = ROOT / "results/e2e_mitigation/rewrite_log.jsonl"
    pg_log = ROOT / "results/prism_l_plus/rewrite_log.jsonl"

    def _not_keep(rec: dict) -> bool:
        return str(rec.get("action") or "") != "keep"

    mit_flags = {
        "e2e_rarr_intervene": _load_rewrite_flags(e2e_log, arm="B1_RARR_E2E_native", flagged_if=_not_keep),
        "e2e_cove_intervene": _load_rewrite_flags(e2e_log, arm="B2_CoVe_E2E_native", flagged_if=_not_keep),
        "e2e_copy_intervene": _load_rewrite_flags(e2e_log, arm="B3_Copy_E2E_all_claims", flagged_if=_not_keep),
        "e2e_prism_a_intervene": _load_rewrite_flags(e2e_log, arm="PRISM_L_plus_A_E2E", flagged_if=_not_keep),
        "e2e_prism_b_intervene": _load_rewrite_flags(e2e_log, arm="PRISM_L_plus_B_E2E", flagged_if=_not_keep),
        "pg_b1_intervene": _load_rewrite_flags(pg_log, arm="B1_RARR_official", flagged_if=_not_keep),
        "pg_b2_intervene": _load_rewrite_flags(pg_log, arm="B2_CoVe_official", flagged_if=_not_keep),
        "pg_b3_intervene": _load_rewrite_flags(pg_log, arm="B3_Copy_constrained", flagged_if=_not_keep),
    }
    for name, mp in mit_flags.items():
        _enrich_rows(test, mp, name)

    entries: list[dict[str, Any]] = []

    def add(
        name: str,
        category: str,
        *,
        score_key: str | None = None,
        threshold: float | str = "—",
        pred_fn: Callable[[dict], bool] | None = None,
        auroc_key: str | None = None,
        note: str = "",
    ) -> None:
        if score_key is not None:
            m = _score_metrics(test, score_key, float(threshold))
            auroc = try_auroc(test, auroc_key or score_key)
        else:
            m = _metrics(test, pred_fn or (lambda _r: False), threshold=threshold)
            auroc = None
        entries.append(
            {
                "name": name,
                "category": category,
                "auroc": auroc,
                "threshold": threshold,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
                "note": note,
            }
        )

    # --- Trivial ---
    add("predict_all_positive", "trivial", pred_fn=lambda _r: True, note="上界 recall")
    add("predict_all_negative", "trivial", pred_fn=lambda _r: False, note="下界 recall=div=0")

    # --- Random ---
    add("random_uniform@0.5", "random", score_key="_rand_uniform", threshold=0.5, note=f"seed={seed}")
    add("random_stratified@train_prev", "random", score_key="_rand_strat", threshold=0.5, note=f"seed={seed}, p≈{pos_rate:.3f}")

    # --- Scalar detectors (Line A / BindSurprise / ablations) ---
    add("line_a_probe", "strong", score_key="p_pm", threshold=TAU_PROBE, note="主强基线")
    add("bind_surprise@tau_b^F1", "ours", score_key="bind_surprise", threshold=tau_b["tau_b"], note="dev-F1 冻结")
    add(
        "bind_surprise@tau_b^recall_matched",
        "ours",
        score_key="bind_surprise",
        threshold=tau_b_rec["tau_b"],
        note="dev 对齐 probe recall",
    )
    add("quote_bow@tau_agg", "weak_text", score_key="p_bow", threshold=bow_tau_agg, auroc_key="p_bow")
    add("quote_bow@tau_bal", "weak_text", score_key="p_bow", threshold=bow_tau_bal, auroc_key="p_bow")
    if any(float(r.get("p_modernbert_quote") or 0) for r in test):
        add(
            "modernbert_quote@tau_agg",
            "weak_text",
            score_key="p_modernbert_quote",
            threshold=bow_tau_agg,
            auroc_key="p_modernbert_quote",
            note="HF ModernBERT-large emb + LR; quote-only",
        )
    if any(float(r.get("p_modernbert_bcp_fair") or 0) for r in test):
        add(
            "modernbert_struct_fair@tau_agg",
            "weak_text",
            score_key="p_modernbert_bcp_fair",
            threshold=bow_tau_agg,
            auroc_key="p_modernbert_bcp_fair",
            note="bcp_input_v1 fair; heuristic anchor+rollout only",
        )
    if any(float(r.get("p_modernbert_bcp_oracle") or 0) for r in test):
        add(
            "modernbert_struct_oracle_leak@tau_agg",
            "diagnostic_leakage",
            score_key="p_modernbert_bcp_oracle",
            threshold=bow_tau_agg,
            auroc_key="p_modernbert_bcp_oracle",
            note="oracle adjudication anchor; leakage audit only",
        )
    add("j_slot_log_mass@dev_median_pos", "weak_jlens", score_key="log_slot_mass", threshold=slot_thr, auroc_key="log_slot_mass")
    add(
        "prism_v1_hard_AND_probe∧slot",
        "ablation",
        pred_fn=lambda r: float(r["p_pm"]) > TAU_PROBE and bool(r.get("slot_type_aligned")),
        note="非 BindSurprise;硬交集",
    )

    # --- Literature native gates (extracted from mitigation pipeline) ---
    add(
        "rarr_agreement_gate_any_open",
        "lit_native_detect",
        score_key="rarr_gate_open",
        threshold=0.5,
        note="vendor RARR agreement gate;来自 cache",
    )
    for key, label in [
        ("e2e_rarr_intervene", "e2e_B1_RARR_intervene≠keep"),
        ("e2e_cove_intervene", "e2e_B2_CoVe_intervene≠keep"),
        ("e2e_copy_intervene", "e2e_B3_Copy_intervene≠keep"),
        ("e2e_prism_a_intervene", "e2e_BindRepair-A_intervene≠keep"),
        ("e2e_prism_b_intervene", "e2e_BindRepair-B_intervene≠keep"),
        ("pg_b1_intervene", "probe_gated_B1_intervene≠keep"),
        ("pg_b2_intervene", "probe_gated_B2_intervene≠keep"),
        ("pg_b3_intervene", "probe_gated_B3_intervene≠keep"),
    ]:
        add(
            label,
            "mitigation_extracted_detect",
            score_key=key,
            threshold=0.5,
            note="从缓解 rewrite_log 提取;intervene=非 keep",
        )

    # Route labels as detectors
    add(
        "route_confirmed_risk_only",
        "routing",
        pred_fn=lambda r: str(r.get("audit_state") or "") == "confirmed_risk",
    )
    add(
        "route_probe_only_only",
        "routing",
        pred_fn=lambda r: str(r.get("audit_state") or "") == "probe_only",
    )
    add(
        "route_any_probe_high",
        "routing",
        pred_fn=lambda r: float(r["p_pm"]) > TAU_PROBE,
        note="等价 line_a_probe@0.05",
    )

    return {
        "schema": "detection_baseline_matrix_v1",
        "split": "test",
        "n_claims": len(test),
        "n_positive": sum(int(r.get("y_pm") or 0) for r in test),
        "tau_probe": TAU_PROBE,
        "tau_b_f1": tau_b,
        "tau_b_recall_matched": tau_b_rec,
        "bow_thresholds": {"aggressive": bow_tau_agg, "balanced": bow_tau_bal},
        "slot_log_threshold": slot_thr,
        "random_seed": seed,
        "train_positive_rate": pos_rate,
        "baselines": entries,
    }


def _render_md(summary: dict[str, Any]) -> str:
    lines = [
        "# PM Detection Baseline Matrix (test, unfiltered)",
        "",
        f"**Claims**: {summary['n_claims']} ({summary['n_positive']} positive) | τ_probe={summary['tau_probe']}",
        "",
        "| Category | Detector | AUROC | τ | P | R | F1 | TP/FP/FN/TN | Note |",
        "|----------|----------|-------|---|---|---|-----|-------------|------|",
    ]
    for b in summary["baselines"]:
        auroc = f"{b['auroc']:.3f}" if b.get("auroc") is not None else "—"
        thr = b.get("threshold")
        thr_s = f"{thr:.3f}" if isinstance(thr, (int, float)) else str(thr)
        lines.append(
            f"| {b['category']} | {b['name']} | {auroc} | {thr_s} | "
            f"{b['precision']:.3f} | {b['recall']:.3f} | {b['f1']:.3f} | "
            f"{b['tp']}/{b['fp']}/{b['fn']}/{b['tn']} | {b.get('note','')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = _load_rows()
    summary = build_baseline_matrix(rows, seed=args.seed)
    out_json = PRISM_DIR / "detection_baseline_matrix.json"
    out_md = PRISM_DIR / "DETECTION_BASELINE_MATRIX.md"
    PRISM_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(summary), encoding="utf-8")
    print(f"Wrote {out_json} ({len(summary['baselines'])} baselines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
