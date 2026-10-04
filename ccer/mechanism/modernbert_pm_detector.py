"""ModernBERT-large + LR PM detection on Line-A v3 cohort (quote and BCP text)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ccer.io_utils import load_jsonl
from ccer.mechanism.anchor_resolve_heuristic import (
    resolve_heuristic_anchor_pid,
    resolve_heuristic_donor_pids,
)
from ccer.mechanism.bind_surprise import detection_metrics, try_auroc
from ccer.mechanism.line_a_cohort_v3 import build_probe_dataset_v3
from ccer.mechanism.line_l_rewrite_quality import build_evidence_corpus
from ccer.mechanism.supervised_probe import PM_VERDICTS, normalize_quote_for_bow
from ccer.paths import INCREMENTAL_ADJUDICATION_JSONL, NORMALIZED_SHOPPING

ROOT = Path("${PHANTOM_MERGE_ROOT}")
PROBE_ROOT = ROOT / "benchmarks" / "probe"
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from lib.templates import build_bcp_prompt, format_other_evidence  # noqa: E402
from run_validity_controls import _encode_modernbert  # noqa: E402

MODEL_ID = "answerdotai/ModernBERT-large"
PIN_DIR = ROOT / "third_party" / "answerdotai-ModernBERT-large"
ROLLOUT = (
    ROOT
    / "data/rollouts/shopping/shopping_qwen3-32b_rollout/rollout.jsonl"
)

TextVariant = Literal["quote", "bcp_fair", "bcp_oracle"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_adjudication_by_key() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(INCREMENTAL_ADJUDICATION_JSONL):
        key = str(row.get("instance_audit_key") or "")
        if key:
            out[key] = row
    return out


def _load_rollout_obs() -> dict[str, list[dict]]:
    by_tid: dict[str, list[dict]] = {}
    if not ROLLOUT.is_file():
        return by_tid
    for line in ROLLOUT.open(encoding="utf-8"):
        steps = json.loads(line)
        if not isinstance(steps, list):
            continue
        tid = None
        obs_all: list[dict] = []
        for step in steps:
            ei = step.get("extra_info") or {}
            tid = ei.get("seal_task_id") or ei.get("trajectory_id") or tid
            for obs in ei.get("obs_full") or []:
                if isinstance(obs, dict):
                    obs_all.append(obs)
        if tid:
            by_tid[str(tid)] = obs_all
    return by_tid


def _donor_ids_from_adj(adj: dict[str, Any] | None) -> list[str]:
    if not adj:
        return []
    ids: list[str] = []
    for d in adj.get("donor_owners") or []:
        if isinstance(d, dict):
            pid = d.get("pid") or d.get("product_id")
            if pid:
                ids.append(str(pid))
        elif d:
            ids.append(str(d))
    return ids


def _query_for_instance(
    inst: dict[str, Any],
    traj: dict[str, Any],
    adj: dict[str, Any] | None,
) -> str:
    if adj and adj.get("query_quote"):
        return str(adj["query_quote"])
    meta = traj.get("metadata") or {}
    return str(meta.get("query") or meta.get("user_query") or "")


def _anchor_evidence_for_instance(
    inst: dict[str, Any],
    traj: dict[str, Any],
    adj: dict[str, Any] | None,
    anchor_pid: str,
) -> str:
    if adj and adj.get("anchor_evidence_quote"):
        return str(adj["anchor_evidence_quote"])
    if anchor_pid:
        return build_evidence_corpus(traj, anchor_pid)
    return ""


def _other_evidence_for_instance(
    inst: dict[str, Any],
    traj: dict[str, Any],
    adj: dict[str, Any] | None,
    anchor_pid: str,
    anchor_evidence: str,
) -> str:
    donor_ids = _donor_ids_from_adj(adj)
    if not donor_ids and anchor_pid:
        donor_ids = resolve_heuristic_donor_pids(traj, anchor_pid=anchor_pid)
    donor_ids = [d for d in donor_ids if d and d != anchor_pid][:3]
    return format_other_evidence(donor_ids, anchor_evidence, "")


def _resolve_anchor_pid(
    inst: dict[str, Any],
    traj: dict[str, Any],
    adj: dict[str, Any] | None,
    *,
    use_oracle_pid: bool,
) -> str:
    from ccer.replay.answer_utils import extract_selected_product_id, normalize_final_synthesis_text

    anchor_pid = ""
    if use_oracle_pid and adj:
        anchor_pid = str(adj.get("committed_anchor_pid") or "")
    if not anchor_pid:
        anchor_pid = resolve_heuristic_anchor_pid(traj) or ""
    if not anchor_pid:
        meta = traj.get("metadata") or {}
        anchor_pid = str(
            meta.get("selected_product_id")
            or meta.get("anchor_pid")
            or extract_selected_product_id(
                normalize_final_synthesis_text(str(inst.get("final_answer") or ""))
            )
            or ""
        )
    return anchor_pid


def build_detection_text_rows(
    *,
    require_activation: bool = True,
) -> list[dict[str, Any]]:
    """One row per Line-A v3 instance with quote + fair/oracle structured text (n=1244, test=274)."""
    ds = build_probe_dataset_v3(require_activation=require_activation)
    adj_by_key = _load_adjudication_by_key()
    norm = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}

    rows: list[dict[str, Any]] = []
    for inst in ds.instances:
        tid = str(inst["trajectory_id"])
        iak = str(inst["instance_audit_key"])
        traj = norm.get(tid) or {}
        adj = adj_by_key.get(iak)
        quote = str(inst.get("response_quote") or "")

        fair_pid = _resolve_anchor_pid(inst, traj, adj, use_oracle_pid=False)
        fair_query = _query_for_instance(inst, traj, None)
        fair_anchor = _anchor_evidence_for_instance(inst, traj, None, fair_pid)
        fair_other = _other_evidence_for_instance(inst, traj, None, fair_pid, fair_anchor)
        bcp_fair = build_bcp_prompt(
            query=fair_query,
            anchor_evidence=fair_anchor,
            other_evidence=fair_other,
            claim_text=quote,
        )

        oracle_pid = _resolve_anchor_pid(inst, traj, adj, use_oracle_pid=True)
        oracle_query = _query_for_instance(inst, traj, adj)
        oracle_anchor = _anchor_evidence_for_instance(inst, traj, adj, oracle_pid)
        oracle_other = _other_evidence_for_instance(inst, traj, adj, oracle_pid, oracle_anchor)
        bcp_oracle = build_bcp_prompt(
            query=oracle_query,
            anchor_evidence=oracle_anchor,
            other_evidence=oracle_other,
            claim_text=quote,
        )

        rows.append(
            {
                "instance_audit_key": iak,
                "trajectory_id": tid,
                "split": str(inst["split"]),
                "y": int(inst["y"]),
                "gold_verdict": str(inst.get("gold_verdict") or ""),
                "response_quote": quote,
                "text_quote": normalize_quote_for_bow(quote),
                "text_bcp_fair": bcp_fair,
                "text_bcp_oracle": bcp_oracle,
            }
        )
    return rows


def _fit_lr_scores(X: np.ndarray, y: np.ndarray, splits: list[str]) -> np.ndarray:
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(X[train_idx], y[train_idx])
    return clf.predict_proba(X)[:, 1].astype(np.float64)


def _eval_split(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    split: str,
    *,
    bow_tau_agg: float,
    bow_tau_bal: float,
) -> dict[str, Any]:
    idx = [i for i, r in enumerate(rows) if r["split"] == split]
    if not idx:
        return {"n": 0}
    sub = [rows[i] for i in idx]
    sc = scores[idx]
    y = np.array([r["y"] for r in sub], dtype=int)
    scored_rows = [
        {"y_pm": int(r["y"]), "p_score": float(sc[j])} for j, r in enumerate(sub)
    ]
    auroc = try_auroc(scored_rows, "p_score")
    m_agg = detection_metrics(scored_rows, score_key="p_score", threshold=bow_tau_agg)
    m_bal = detection_metrics(scored_rows, score_key="p_score", threshold=bow_tau_bal)
    return {
        "n": len(sub),
        "n_pm": int(y.sum()),
        "n_clean": int((1 - y).sum()),
        "auroc": auroc,
        "f1_at_tau_agg": m_agg.get("f1"),
        "f1_at_tau_bal": m_bal.get("f1"),
        "tau_agg": bow_tau_agg,
        "tau_bal": bow_tau_bal,
    }


def write_pinned_revision() -> None:
    PIN_DIR.mkdir(parents=True, exist_ok=True)
    revision = "main"
    try:
        from huggingface_hub import model_info

        info = model_info(MODEL_ID)
        revision = str(info.sha or "main")
    except Exception:
        pass
    pin_path = PIN_DIR / "PINNED_REVISION"
    pin_path.write_text(
        f"model_id: {MODEL_ID}\n"
        f"revision: {revision}\n"
        f"pinned_at: {_now()}\n"
        f"upstream: https://huggingface.co/{MODEL_ID}\n"
        f"integration: mean-pool last_hidden_state + sklearn LogisticRegression (CCER PM detect)\n",
        encoding="utf-8",
    )


def run_modernbert_detection(
    *,
    out_dir: Path,
    device: str = "cuda",
    batch_size: int = 8,
    max_length: int = 512,
    skip_encode: bool = False,
    variants: tuple[TextVariant, ...] = ("quote", "bcp_fair", "bcp_oracle"),
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pinned_revision()

    text_rows = build_detection_text_rows(require_activation=True)
    y = np.array([r["y"] for r in text_rows], dtype=int)
    splits = [r["split"] for r in text_rows]

    dev_thresh_path = ROOT / "results/line_k/dev_thresholds.json"
    bow_tau_agg = 0.05
    bow_tau_bal = 0.75
    if dev_thresh_path.is_file():
        lk = json.loads(dev_thresh_path.read_text(encoding="utf-8"))
        bow_tau_agg = float(lk.get("quote_bow_aggressive") or bow_tau_agg)
        bow_tau_bal = float(lk.get("quote_bow_balanced") or bow_tau_bal)

    payload: dict[str, Any] = {
        "schema": "ccer_modernbert_detection_v1",
        "generated_at": _now(),
        "model_id": MODEL_ID,
        "pinned_revision_path": str(PIN_DIR / "PINNED_REVISION"),
        "cohort": "line_a_v3_expanded_clean",
        "n_instances": len(text_rows),
        "split_counts": {
            sp: sum(1 for r in text_rows if r["split"] == sp) for sp in ("train", "dev", "test")
        },
        "vendor_label": "HuggingFace checkpoint + in-house LR head (not paper reproduction)",
        "variants": {},
    }

    for variant in variants:
        col = f"text_{variant}"
        texts = [str(r[col]) for r in text_rows]
        cache = out_dir / f"modernbert_embeddings_{variant}.npy"
        if skip_encode and cache.is_file():
            X = np.load(cache).astype(np.float64)
        else:
            X = _encode_modernbert(
                texts,
                model_name=MODEL_ID,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                cache_path=cache,
            )
        if X.shape[0] != len(texts):
            raise RuntimeError(f"embedding shape mismatch for {variant}")
        scores = _fit_lr_scores(X, y, splits)
        test_eval = _eval_split(
            text_rows, scores, "test", bow_tau_agg=bow_tau_agg, bow_tau_bal=bow_tau_bal
        )
        dev_eval = _eval_split(
            text_rows, scores, "dev", bow_tau_agg=bow_tau_agg, bow_tau_bal=bow_tau_bal
        )
        train_idx = [i for i, s in enumerate(splits) if s == "train"]
        train_auroc = None
        if len(set(y[train_idx])) > 1:
            train_auroc = float(roc_auc_score(y[train_idx], scores[train_idx]))

        if variant == "quote":
            det_id, paper_role = "D-MBERT-Q", "text_baseline"
            protocol = "normalize_quote_for_bow(response_quote)"
            leakage_warning = None
        elif variant == "bcp_fair":
            det_id, paper_role = "D-MBERT-STRUCT-FAIR", "text_baseline"
            protocol = (
                "bcp_input_v1; heuristic anchor PID + rollout/traj corpus only "
                "(no adjudication oracle fields; PM/clean symmetric)"
            )
            leakage_warning = None
        else:
            det_id, paper_role = "D-MBERT-STRUCT-ORACLE", "diagnostic_leakage_audit"
            protocol = "bcp_input_v1 + adjudication anchor_evidence_quote on PM rows (leakage audit)"
            leakage_warning = "Do not compare to BindSurprise or fair baselines"
        payload["variants"][variant] = {
            "detector_id": det_id,
            "paper_role": paper_role,
            "feature_protocol": protocol,
            "leakage_warning": leakage_warning,
            "embedding_cache": str(cache),
            "train_auroc": train_auroc,
            "dev": dev_eval,
            "test": test_eval,
        }
        score_path = out_dir / f"modernbert_scores_{variant}.jsonl"
        with score_path.open("w", encoding="utf-8") as fh:
            for r, sc in zip(text_rows, scores):
                fh.write(
                    json.dumps(
                        {
                            "instance_audit_key": r["instance_audit_key"],
                            "split": r["split"],
                            "y": r["y"],
                            "p_modernbert": float(sc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary_path = out_dir / "modernbert_detection_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary_md(out_dir, payload)
    return payload


def _write_summary_md(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ModernBERT-large PM Detection (Line-A v3)",
        "",
        f"Generated: {payload.get('generated_at')}",
        f"Model: `{payload.get('model_id')}`",
        f"Cohort: {payload.get('cohort')} (n={payload.get('n_instances')})",
        "",
        "**主表**：D-MBERT-Q only. **D-MBERT-STRUCT**（`bcp_input_v1` 模板）为泄漏审计，"
        "≠ 激活探针 BCP（Binding-Compatibility Probe）。",
        "",
        "## 主表（text baseline）",
        "",
        "| ID | Test AUROC | F1 @τ_agg | n_test |",
        "|----|----------:|----------:|-------:|",
    ]
    for var, block in (payload.get("variants") or {}).items():
        if block.get("paper_role") == "diagnostic_leakage_audit":
            continue
        te = block.get("test") or {}
        lines.append(
            f"| {block.get('detector_id')} | {te.get('auroc', 0):.4f} | "
            f"{te.get('f1_at_tau_agg', 0):.4f} | {te.get('n', 0)} |"
        )
    lines += [
        "",
        "## 泄漏审计（禁止入主表叙事）",
        "",
        "| ID | Test AUROC | F1 @τ_agg | Warning |",
        "|----|----------:|----------:|---------|",
    ]
    for var, block in (payload.get("variants") or {}).items():
        if block.get("paper_role") != "diagnostic_leakage_audit":
            continue
        te = block.get("test") or {}
        lines.append(
            f"| {block.get('detector_id')} | {te.get('auroc', 0):.4f} | "
            f"{te.get('f1_at_tau_agg', 0):.4f} | {block.get('leakage_warning', '')} |"
        )
    lines.extend(
        [
            "",
            "Script: `python -m ccer.pipelines.run_modernbert_detection_baseline`",
            "",
            payload.get("vendor_label", ""),
        ]
    )
    (out_dir / "MODERNBERT_DETECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
