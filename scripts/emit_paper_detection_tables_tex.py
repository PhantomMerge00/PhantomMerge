#!/usr/bin/env python3
"""Emit LaTeX tables for RQ2 detection (tab:rq2) and RQ3 figD anchor verification."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.mechanism.agr.significance import cluster_bootstrap_ap, cluster_bootstrap_auroc

REG_PATH = ROOT / "results/DETECTION_METRICS_REGISTRY.json"
ANCHOR_PATH = ROOT / "results/rq3/ANCHOR_BASELINE_MATRIX.json"
ANCHOR_LLM_DIR = ROOT / "results/rq3/anchor_line_a_test_llm"
OUT_RQ2 = ROOT / "results/rq2/tables/tab_rq2_detection.tex"
OUT_FIGD = ROOT / "results/rq3/tables/tab_figD_anchor_verification.tex"
OUT_FIGD_JSON = ROOT / "results/rq3/ANCHOR_DETECTION_TABLE_METRICS.json"
N_BOOT = 2000
SEED = 42

RQ2_ROWS = [
    ("D-RARR-G", "RARR", False),
    ("D-FACTSCORE-OBS", "FActScore", False),
    ("D-FACTOOL-KBQA", "FacTool", False),
    ("D-MBERT-STRUCT-FAIR", "ModernBERT", False),
    ("D-TFIDF", "TF-IDF", False),
    ("D-SELFCHECK-NLI", "SelfCheck-NLI", False),
    ("D-AGR", "AGR-value", True),
    ("D-BIND", "AGR-slot", True),
]

ANCHOR_ARMS: list[tuple[str, str, Path, Callable[[dict[str, Any]], float]]] = [
    (
        "B1_anchor_only",
        "Anchor-only LLM",
        ANCHOR_LLM_DIR / "shopping_b1_predictions.jsonl",
        lambda r: 1.0 if r.get("auto_pm") else 0.0,
    ),
    (
        "B2_source_agnostic",
        "Source-agnostic LLM",
        ANCHOR_LLM_DIR / "shopping_b2_predictions.jsonl",
        lambda r: 1.0 if r.get("auto_pm") else 0.0,
    ),
    (
        "cite_or_drop_anchor",
        "Cite-or-drop LLM",
        ANCHOR_LLM_DIR / "shopping_cite_predictions.jsonl",
        lambda r: 1.0 if r.get("auto_pm") else 0.0,
    ),
    (
        "NLI_deberta_anchor",
        "DeBERTa-NLI (anchor)",
        ANCHOR_LLM_DIR / "shopping_nli_predictions.jsonl",
        lambda r: 1.0 - float(r["nli_probs"]["entailment"]),
    ),
]

PROBE_GATE_ID = "D-L49"
PROBE_GATE_LABEL = "AGR-slot gate @ $\\tau{=}0.05$"


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}"


def _rank_cell(block: dict) -> str:
    p = block["auroc"] * 100
    lo, hi = block["ci_low"] * 100, block["ci_high"] * 100
    return f"{p:.1f} [{lo:.1f}, {hi:.1f}]"


def _ap_cell(block: dict) -> str:
    p = block["ap"] * 100
    lo, hi = block["ci_low"] * 100, block["ci_high"] * 100
    return f"{p:.1f} [{lo:.1f}, {hi:.1f}]"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _anchor_detection_rows(
    path: Path, score_fn: Callable[[dict[str, Any]], float]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in _load_jsonl(path):
        out.append(
            {
                "trajectory_id": rec["trajectory_id"],
                "claim_id": rec["claim_id"],
                "y_pm": int(rec["gold_pm"]),
                "score": float(score_fn(rec)),
            }
        )
    return out


def _ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    auroc = cluster_bootstrap_auroc(rows, "score", n_boot=N_BOOT, seed=SEED)
    ap = cluster_bootstrap_ap(rows, "score", n_boot=N_BOOT, seed=SEED)
    return {"auroc": auroc, "ap": ap}


def _prf_from_summary(arm: dict[str, Any]) -> tuple[float, float, float]:
    p = float(arm["pm_precision"])
    r = float(arm["pm_recall"])
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    tp, fp, fn = int(arm["tp"]), int(arm["fp"]), int(arm["fn"])
    if tp + fp + fn > 0:
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return prec, rec, f1
    return p, r, f1


def _collect_anchor_table_rows() -> list[dict[str, Any]]:
    anchor = json.loads(ANCHOR_PATH.read_text())
    llm = anchor["line_a_test_llm"]
    reg = json.loads(REG_PATH.read_text())
    methods = {m["method_id"]: m for m in reg["methods"]}
    tau = {
        r["method_id"]: r
        for r in reg["unified_tau_0.05"]
        if r.get("applicable") and r.get("metrics")
    }

    collected: list[dict[str, Any]] = []
    for key, label, path, score_fn in ANCHOR_ARMS:
        det_rows = _anchor_detection_rows(path, score_fn)
        rank = _ranking_metrics(det_rows)
        arm = llm[key]
        prec, rec, f1 = _prf_from_summary(arm)
        collected.append(
            {
                "method_id": key,
                "label": label,
                "ranking": rank,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "bold": False,
            }
        )

    probe_m = methods[PROBE_GATE_ID]
    probe_t = tau[PROBE_GATE_ID]["metrics"]
    collected.append(
        {
            "method_id": PROBE_GATE_ID,
            "label": PROBE_GATE_LABEL,
            "ranking": {
                "auroc": probe_m["ranking"]["auroc"],
                "ap": probe_m["ranking"]["ap"],
            },
            "precision": float(probe_t["precision"]),
            "recall": float(probe_t["recall"]),
            "f1": float(probe_t["f1"]),
            "bold": True,
        }
    )
    collected.sort(key=lambda r: r["ranking"]["auroc"]["auroc"])
    return collected


def emit_rq2() -> None:
    reg = json.loads(REG_PATH.read_text())
    methods = {m["method_id"]: m for m in reg["methods"]}
    tau = {
        r["method_id"]: r
        for r in reg["unified_tau_0.05"]
        if r.get("applicable") and r.get("metrics")
    }
    body: list[str] = []
    for mid, label, agr in RQ2_ROWS:
        m = methods[mid]
        auroc = _rank_cell(m["ranking"]["auroc"])
        ap = _ap_cell(m["ranking"]["ap"])
        if agr:
            body.append(
                f"\\textbf{{{label}}} & \\textbf{{{auroc}}} & \\textbf{{{ap}}} & -- & -- & -- \\\\"
            )
        else:
            t = tau[mid]["metrics"]
            body.append(
                f"{label:<14} & {auroc} & {ap} & "
                f"{_pct(t['precision'])} & {_pct(t['recall'])} & {_pct(t['f1'])} \\\\"
            )
    tex = f"""% Auto-synced from {REG_PATH.relative_to(ROOT)}
% Regenerate: PYTHONPATH=active_code python scripts/emit_paper_detection_tables_tex.py
\\begin{{table*}}[t]
\\centering
\\caption{{Phantom Merge detection on the Shopping test cohort ($n{{=}}274$ claims, eval-only gold labels). AUROC/AUPRC report the point estimate with the 95\\% bootstrap CI bracket; Precision/Recall/F1 report the point estimate with CI offsets, at $\\tau{{=}}0.05$. Methods are ordered by AUROC. AGR is not a probability score and is not evaluated at this threshold.}}
\\label{{tab:rq2}}
\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} l c c c c c}}
\\toprule
Method & AUROC & AUPRC & Precision & Recall & F1 \\\\
\\midrule
{chr(10).join(body[:6])}
\\midrule
{chr(10).join(body[6:])}
\\bottomrule
\\end{{tabular*}}
\\end{{table*}}
"""
    OUT_RQ2.parent.mkdir(parents=True, exist_ok=True)
    OUT_RQ2.write_text(tex)


def _latex_row(entry: dict[str, Any]) -> str:
    auroc = _rank_cell(entry["ranking"]["auroc"])
    ap = _ap_cell(entry["ranking"]["ap"])
    p, r, f = _pct(entry["precision"]), _pct(entry["recall"]), _pct(entry["f1"])
    label = entry["label"]
    if entry.get("bold"):
        return (
            f"\\textbf{{{label}}} & \\textbf{{{auroc}}} & \\textbf{{{ap}}} & "
            f"\\textbf{{{p}}} & \\textbf{{{r}}} & \\textbf{{{f}}} \\\\"
        )
    return f"{label} & {auroc} & {ap} & {p} & {r} & {f} \\\\"


def emit_figd() -> None:
    rows = _collect_anchor_table_rows()
    OUT_FIGD_JSON.write_text(json.dumps({"rows": rows}, indent=2) + "\n")

    body = [_latex_row(e) for e in rows]
    mid = len(body) - 1
    tex = f"""% Data: {ANCHOR_PATH.relative_to(ROOT)} + anchor_line_a_test_llm/*.jsonl; probe from {REG_PATH.name}
% Figure: results/rq3/figures/figD_anchor_validation_recall_falsedrop.{{svg,pdf}}
% Regenerate: PYTHONPATH=active_code python scripts/emit_paper_detection_tables_tex.py
\\begin{{table*}}[t]
\\centering
\\caption{{Anchor-aware PM verification on the Shopping test cohort ($n{{=}}274$ claims, eval-only gold labels). AUROC/AUPRC use claim-level scores with trajectory-cluster bootstrap CIs ($B{{=}}2000$): LLM verifiers use $\\mathbb{{1}}[\\text{{auto\\_pm}}]$; DeBERTa-NLI uses $1-P(\\text{{entail}})$. Precision/Recall/F1 are at each verifier's native decision rule (same operating points as Fig.~\\ref{{fig:anchor_recall_falsedrop}}). The AGR-slot row is Line-A probe gating with P/R/F1 at $\\tau{{=}}0.05$. Methods ordered by AUROC.}}
\\label{{tab:anchor_figD}}
\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} l c c c c c}}
\\toprule
Method & AUROC & AUPRC & Precision & Recall & F1 \\\\
\\midrule
{chr(10).join(body[:mid])}
\\midrule
{body[mid]}
\\bottomrule
\\end{{tabular*}}
\\end{{table*}}
"""
    OUT_FIGD.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIGD.write_text(tex)


def main() -> None:
    emit_rq2()
    emit_figd()
    print(f"Wrote {OUT_RQ2}")
    print(f"Wrote {OUT_FIGD}")
    print(f"Wrote {OUT_FIGD_JSON}")


if __name__ == "__main__":
    main()
