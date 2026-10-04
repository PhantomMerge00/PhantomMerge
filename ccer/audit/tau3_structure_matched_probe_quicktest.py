"""Quick test: is probe AUROC driven by quote-structure mismatch (JSON PM vs prose clean)?"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ccer.io_utils import load_jsonl, write_json
from ccer.mechanism.bind_surprise import try_auroc
from ccer.mechanism.tau3.agr_slot_eval import _hidden, _load_split_maps
from ccer.mechanism.tau3.cohort import tau3_instance_rows
from ccer.mechanism.tau3.domain import get_tau3_domain
from ccer.mechanism.tau3.replay import find_quote_span, tau2_to_chat_messages, tokenize_tau2_trajectory
from ccer.paths import TAU3_DIR
from ccer.replay.hf_forward import load_hf_tokenizer

_JSON_LIKE = re.compile(r"^\s*[\{\[]")


def quote_structure(quote: str) -> str:
    q = str(quote or "").strip()
    if _JSON_LIKE.match(q):
        return "json_like"
    if q.startswith("###") or q.startswith("1. **"):
        return "markdown_list"
    return "prose"


def _rows_with_hidden(cfg) -> list[dict[str, Any]]:
    _, inst_split = _load_split_maps(cfg)
    out: list[dict[str, Any]] = []
    for row in tau3_instance_rows(require_activation=True, cfg=cfg):
        h = _hidden(row.instance_audit_key, row.trajectory_id, cfg)
        if h is None:
            continue
        out.append(
            {
                "instance_audit_key": row.instance_audit_key,
                "trajectory_id": row.trajectory_id,
                "split": inst_split.get(row.instance_audit_key, "?"),
                "y_pm": int(row.y),
                "gold_verdict": row.gold_verdict,
                "response_quote": row.response_quote,
                "quote_structure": quote_structure(row.response_quote),
                "hidden": h.astype(np.float32),
            }
        )
    return out


def _fit_linear_probe(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(Xs, y)
    return scaler, clf


def _probe_auroc(rows: list[dict[str, Any]], *, train_split: str = "D_p", test_split: str = "test") -> dict[str, Any]:
    train = [r for r in rows if r["split"] == train_split]
    test = [r for r in rows if r["split"] == test_split]
    if not train or not test:
        return {"error": "empty_split", "n_train": len(train), "n_test": len(test)}
    y_tr = np.array([r["y_pm"] for r in train])
    y_te = np.array([r["y_pm"] for r in test])
    if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        return {"error": "single_class", "n_train": len(train), "n_test": len(test)}
    X_tr = np.stack([r["hidden"] for r in train])
    X_te = np.stack([r["hidden"] for r in test])
    scaler, clf = _fit_linear_probe(X_tr, y_tr)
    probs = clf.predict_proba(scaler.transform(X_te))[:, 1]
    scored = [{"y_pm": int(y), "p_pm": float(p)} for y, p in zip(y_te, probs)]
    return {
        "n_train": len(train),
        "n_test": len(test),
        "n_pm_test": int(y_te.sum()),
        "n_clean_test": int((1 - y_te).sum()),
        "auroc": try_auroc(scored, "p_pm"),
    }


def _tfidf_auroc(rows: list[dict[str, Any]], *, train_split: str = "D_p", test_split: str = "test") -> dict[str, Any]:
    train = [r for r in rows if r["split"] == train_split]
    test = [r for r in rows if r["split"] == test_split]
    if not train or not test:
        return {"error": "empty_split"}
    y_tr = [r["y_pm"] for r in train]
    y_te = [r["y_pm"] for r in test]
    if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        return {"error": "single_class"}
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tr = vec.fit_transform([r["response_quote"] for r in train])
    X_te = vec.transform([r["response_quote"] for r in test])
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    scored = [{"y_pm": int(y), "p_pm": float(p)} for y, p in zip(y_te, probs)]
    return {
        "n_train": len(train),
        "n_test": len(test),
        "auroc": try_auroc(scored, "p_pm"),
        "interpretation": "quote-text-only; if ~1.0 then separability is mostly surface form",
    }


def _extract_clean_tool_json_quotes(traj_row: dict[str, Any]) -> list[str]:
    """Candidate structure-matched clean quotes: tool_call JSON args from clean trajectories."""
    quotes: list[str] = []
    for msg in traj_row.get("messages") or []:
        if str(msg.get("role")) != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments") or tc.get("arguments") or ""
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            args = str(args).strip()
            if len(args) >= 10 and _JSON_LIKE.match(args):
                quotes.append(args)
    return quotes


def _count_matched_clean_availability(cfg) -> dict[str, Any]:
    norm = {r["trajectory_id"]: r for r in load_jsonl(cfg.normalized)}
    clean_trajs = [r for r in norm.values() if r.get("trajectory_outcome") == "clean"]
    n_with_json = 0
    n_json_quotes = 0
    examples: list[str] = []
    for traj in clean_trajs:
        qs = _extract_clean_tool_json_quotes(traj)
        if qs:
            n_with_json += 1
            n_json_quotes += len(qs)
            if len(examples) < 3:
                examples.append(qs[0][:120])
    return {
        "n_clean_trajectories": len(clean_trajs),
        "n_with_tool_json_quotes": n_with_json,
        "n_tool_json_quotes_total": n_json_quotes,
        "examples": examples,
    }


def _quick_matched_activation_smoke(cfg, limit: int = 40) -> dict[str, Any]:
    """Smoke: extract L49 at tool-JSON span vs pseudo-clean span on same clean trajectories."""
    from ccer.mechanism.tau3.extract import extract_tau3_instance_trajectory
    from ccer.mechanism.tau3.cohort import Tau3InstanceRow
    from ccer.replay.hf_forward import load_hf_model, probe_layer_indices

    norm = {r["trajectory_id"]: r for r in load_jsonl(cfg.normalized)}
    clean_trajs = [r for r in norm.values() if r.get("trajectory_outcome") == "clean"][:limit]
    tokenizer = load_hf_tokenizer()

    paired: list[dict[str, Any]] = []
    for traj in clean_trajs:
        tid = str(traj["trajectory_id"])
        final_answer = str((traj.get("metadata") or {}).get("final_answer") or "")
        from ccer.mechanism.tau3.replay import pseudo_clean_quote

        pseudo = pseudo_clean_quote(final_answer)
        tool_jsons = _extract_clean_tool_json_quotes(traj)
        if not pseudo or not tool_jsons:
            continue
        pseudo_q, _, _ = pseudo
        messages = tau2_to_chat_messages(traj.get("messages") or [])
        tok_pack = tokenize_tau2_trajectory(messages, tokenizer)
        serialized = tok_pack["serialized_text"]
        json_q = tool_jsons[0]
        pseudo_span = find_quote_span(serialized, pseudo_q)
        json_span = find_quote_span(serialized, json_q)
        if pseudo_span is None or json_span is None:
            continue
        paired.append(
            {
                "trajectory_id": tid,
                "pseudo_quote": pseudo_q[:80],
                "tool_json_quote": json_q[:80],
                "pseudo_span": pseudo_span,
                "json_span": json_span,
            }
        )

    if not paired:
        return {"error": "no_paired_clean_trajectories", "n_attempted": len(clean_trajs)}

    # one forward per trajectory, compare hidden cosine at two spans
    model, tokenizer, n_layers = load_hf_model(device_map="cuda:0", shared_gpu=True, gpu_reserve_gib=18)
    layer_indices = probe_layer_indices(n_layers)
    L = 49

    cosines: list[float] = []
    for item in paired[: min(20, len(paired))]:
        traj = norm[item["trajectory_id"]]
        # build two fake instance rows at different spans
        def _row(iak: str, quote: str, span: tuple[int, int]) -> Tau3InstanceRow:
            return Tau3InstanceRow(
                instance_audit_key=iak,
                trajectory_id=item["trajectory_id"],
                split="smoke",
                y=0,
                gold_verdict="correct_binding",
                response_quote=quote,
                final_answer="",
                quote_start=span[0],
                quote_end=span[1],
                slot_norm="",
                claim_value="",
            )

        pseudo_inst = _row("smoke:pseudo", item["pseudo_quote"], item["pseudo_span"])
        json_inst = _row("smoke:json", item["tool_json_quote"], item["json_span"])
        out = extract_tau3_instance_trajectory(
            model=model,
            tokenizer=tokenizer,
            trajectory=traj,
            instances=[pseudo_inst, json_inst],
            layer_indices=layer_indices,
            cfg=cfg,
        )
        from ccer.mechanism.activation_store import get_vector, load_activation_npz
        from ccer.mechanism.tau3.cohort import instance_npz_path

        p_path = instance_npz_path("smoke:pseudo", item["trajectory_id"], cfg=cfg)
        j_path = instance_npz_path("smoke:json", item["trajectory_id"], cfg=cfg)
        if p_path.is_file() and j_path.is_file():
            hp = get_vector(load_activation_npz(p_path), position="claim_onset", layer=L)
            hj = get_vector(load_activation_npz(j_path), position="claim_onset", layer=L)
            if hp is not None and hj is not None:
                cos = float(np.dot(hp, hj) / (np.linalg.norm(hp) * np.linalg.norm(hj) + 1e-9))
                cosines.append(cos)
        # cleanup smoke npz
        for p in (p_path, j_path):
            if p.is_file():
                p.unlink()

    return {
        "n_paired_clean_trajectories": len(paired),
        "n_activation_pairs": len(cosines),
        "pseudo_vs_tooljson_cosine_mean": float(np.mean(cosines)) if cosines else None,
        "pseudo_vs_tooljson_cosine_std": float(np.std(cosines)) if cosines else None,
        "note": "same clean traj, two quote positions; low cosine => pseudo-clean anchor differs from tool-json anchor",
    }


def run_quicktest(domain: str, *, smoke: bool = False) -> dict[str, Any]:
    cfg = get_tau3_domain(domain)
    rows = _rows_with_hidden(cfg)

    structure_counts = {
        "pm": dict(Counter(r["quote_structure"] for r in rows if r["y_pm"] == 1)),
        "clean": dict(Counter(r["quote_structure"] for r in rows if r["y_pm"] == 0)),
    }

    # Subset: prose PM vs prose clean (structure-matched at text level)
    prose_rows = [r for r in rows if r["quote_structure"] == "prose"]
    json_pm_rows = [r for r in rows if r["y_pm"] == 1 and r["quote_structure"] == "json_like"]

    result: dict[str, Any] = {
        "domain": domain,
        "structure_counts": structure_counts,
        "pm_verdict_mix": dict(Counter(r["gold_verdict"] for r in rows if r["y_pm"] == 1)),
        "baseline_L49_probe": _probe_auroc(rows),
        "baseline_tfidf_quote_only": _tfidf_auroc(rows),
        "prose_only_L49_probe": _probe_auroc(prose_rows),
        "prose_only_tfidf": _tfidf_auroc(prose_rows),
        "json_pm_only_n": len(json_pm_rows),
        "clean_tool_json_availability": _count_matched_clean_availability(cfg),
    }

    if smoke and domain == "telecom":
        try:
            result["matched_clean_smoke"] = _quick_matched_activation_smoke(cfg, limit=40)
        except Exception as exc:  # noqa: BLE001
            result["matched_clean_smoke"] = {"error": str(exc)}

    out = cfg.report.parent / "structure_matched_probe_quicktest.json"
    write_json(out, result)
    return result


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="both", choices=["telecom", "airline", "both"])
    ap.add_argument("--smoke", action="store_true", help="GPU smoke: pseudo vs tool-json on same clean traj")
    args = ap.parse_args()
    domains = ["telecom", "airline"] if args.domain == "both" else [args.domain]
    all_r: dict[str, Any] = {}
    for d in domains:
        all_r[d] = run_quicktest(d, smoke=args.smoke and d == "telecom")
        print(json.dumps({d: all_r[d]}, indent=2, default=str))

    lines = ["# Structure-Matched Probe Quicktest", ""]
    for d, r in all_r.items():
        lines += [
            f"## {d}",
            f"- PM structure: {r['structure_counts']['pm']}",
            f"- Clean structure: {r['structure_counts']['clean']}",
            f"- Baseline L49 AUROC: {r['baseline_L49_probe'].get('auroc')} (TF-IDF quote: {r['baseline_tfidf_quote_only'].get('auroc')})",
            f"- **Prose-only L49 AUROC**: {r['prose_only_L49_probe'].get('auroc')} (TF-IDF: {r['prose_only_tfidf'].get('auroc')})",
            f"- Clean trajs with tool-JSON quotes: {r['clean_tool_json_availability']}",
            "",
        ]
        if "matched_clean_smoke" in r:
            lines.append(f"- Matched smoke: {r['matched_clean_smoke']}")
            lines.append("")
    (TAU3_DIR / "TAU3_STRUCTURE_MATCHED_QUICKTEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
