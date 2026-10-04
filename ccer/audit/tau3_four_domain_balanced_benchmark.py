"""Four-domain trajectory-grouped split + balanced PM subtype / clean test benchmark."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from ccer.audit.tau3_pooled_probe_sweep import (
    LINE_A_LAYER,
    LINE_A_POSITION,
    VERDICT_SHORT,
    _prism_index,
    _shopping_hidden,
    _tau3_hidden,
)
from ccer.io_utils import load_jsonl, write_json
from ccer.mechanism.agr.calibration import fit_threshold_f1
from ccer.mechanism.agr.signals import AgrProbeModel
from ccer.mechanism.bind_surprise import compute_bind_surprise, detection_metrics, try_auroc
from ccer.mechanism.line_a_cohort_v3 import build_probe_dataset_v3
from ccer.mechanism.supervised_probe import normalize_quote_for_bow
from ccer.mechanism.tau3.cohort import tau3_instance_rows
from ccer.mechanism.tau3.domain import Tau3DomainConfig, get_tau3_domain
from ccer.paths import TAU3_DIR

DOMAINS = ("shopping", "telecom", "airline", "retail")
SEED = 42
TEST_TRAJ_FRAC = 0.2


@dataclass
class Inst:
    domain: str
    trajectory_id: str
    group_key: str
    instance_audit_key: str
    y: int
    verdict: str
    vshort: str
    h: np.ndarray
    text: str
    claim_value: str
    log_slot_mass: float
    orig_split: str


def _vshort(verdict: str, y: int) -> str:
    if y == 0:
        return "clean"
    return VERDICT_SHORT.get(verdict, verdict)


def _slot_index(cfg: Tau3DomainConfig) -> dict[str, dict[str, Any]]:
    if not cfg.slot_readout.is_file():
        return {}
    return {str(r["instance_audit_key"]): r for r in load_jsonl(cfg.slot_readout)}


def _collect_shopping_all() -> list[Inst]:
    prism = _prism_index()
    ds = build_probe_dataset_v3(require_activation=True)
    out: list[Inst] = []
    for row in ds.instances:
        iak = str(row["instance_audit_key"])
        tid = str(row["trajectory_id"])
        h = _shopping_hidden(iak, tid)
        if h is None:
            continue
        y = int(row["y"])
        verdict = str(row.get("gold_verdict") or "")
        pr = prism.get(iak) or {}
        lsm = float(pr.get("log_slot_mass") if pr.get("log_slot_mass") is not None else float("-inf"))
        text = str(row.get("response_quote") or "")
        out.append(
            Inst(
                domain="shopping",
                trajectory_id=tid,
                group_key=f"shopping:{tid}",
                instance_audit_key=iak,
                y=y,
                verdict=verdict,
                vshort=_vshort(verdict, y),
                h=h,
                text=text,
                claim_value=str(row.get("claim_value") or text),
                log_slot_mass=lsm,
                orig_split=str(row.get("split") or ""),
            )
        )
    return out


def _collect_tau3_all(domain: str) -> list[Inst]:
    cfg = get_tau3_domain(domain)
    ro = _slot_index(cfg)
    out: list[Inst] = []
    for row in tau3_instance_rows(require_activation=True, cfg=cfg):
        h = _tau3_hidden(row.instance_audit_key, row.trajectory_id, cfg)
        if h is None:
            continue
        ro_row = ro.get(row.instance_audit_key) or {}
        lsm = float(ro_row.get("log_slot_mass") if ro_row.get("log_slot_mass") is not None else float("-inf"))
        out.append(
            Inst(
                domain=domain,
                trajectory_id=row.trajectory_id,
                group_key=f"{domain}:{row.trajectory_id}",
                instance_audit_key=row.instance_audit_key,
                y=row.y,
                verdict=row.gold_verdict,
                vshort=_vshort(row.gold_verdict, row.y),
                h=h,
                text=row.response_quote,
                claim_value=row.claim_value or row.response_quote,
                log_slot_mass=lsm,
                orig_split="tau3",
            )
        )
    return out


def collect_all() -> list[Inst]:
    rows = _collect_shopping_all()
    for d in ("telecom", "airline", "retail"):
        rows.extend(_collect_tau3_all(d))
    return rows


def trajectory_split(groups: list[str], *, test_frac: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    uniq = sorted(set(groups))
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_set = set(uniq[:n_test])
    return {g: ("test" if g in test_set else "train") for g in uniq}


def balance_pool(
    pool: list[Inst],
    *,
    balance_pm_verdicts: bool,
    clean_to_pm_ratio: float,
    seed: int,
) -> tuple[list[Inst], dict[str, Any]]:
    rng = random.Random(seed)
    clean = [x for x in pool if x.y == 0]
    pm = [x for x in pool if x.y == 1]
    meta: dict[str, Any] = {
        "n_in": len(pool),
        "n_pm_in": len(pm),
        "n_clean_in": len(clean),
        "verdict_in": dict(Counter(x.vshort for x in pm)),
    }
    by_v: dict[str, list[Inst]] = defaultdict(list)
    for x in pm:
        if x.vshort in ("CEM", "CAP", "AH"):
            by_v[x.vshort].append(x)
    pm_pick = pm
    if balance_pm_verdicts and by_v:
        k = min(len(v) for v in by_v.values())
        meta["pm_per_verdict_cap"] = k
        pm_pick = []
        for v in ("CEM", "CAP", "AH"):
            sub = by_v.get(v, [])
            rng.shuffle(sub)
            pm_pick.extend(sub[:k])
    n_clean_target = int(round(len(pm_pick) * clean_to_pm_ratio))
    rng.shuffle(clean)
    clean_pick = clean[: min(n_clean_target, len(clean))]
    out = clean_pick + pm_pick
    meta.update(
        {
            "n_out": len(out),
            "n_pm_out": len(pm_pick),
            "n_clean_out": len(clean_pick),
            "verdict_out": dict(Counter(x.vshort for x in pm_pick)),
            "domain_out": dict(Counter(x.domain for x in out)),
        }
    )
    return out, meta


def _fit_probe(train: list[Inst]) -> AgrProbeModel:
    X = np.stack([x.h for x in train], axis=0)
    y = np.asarray([x.y for x in train], dtype=int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(Xs, y)
    return AgrProbeModel(scaler=scaler, clf=clf, layer=LINE_A_LAYER, position=LINE_A_POSITION)


def _score_probe(probe: AgrProbeModel, pool: list[Inst]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for x in pool:
        p = probe.prob(x.h)
        bs = compute_bind_surprise(p, x.log_slot_mass)
        rows.append(
            {
                "y_pm": x.y,
                "p_pm": p,
                "log_slot_mass": bs["log_slot_mass"],
                "agr_slot_score": bs["bind_surprise"],
                "domain": x.domain,
                "vshort": x.vshort,
            }
        )
    return rows


def _fit_eval_tfidf(
    train: list[Inst],
    test: list[Inst],
    *,
    text_fn,
) -> dict[str, Any]:
    y_tr = [x.y for x in train]
    y_te = [x.y for x in test]
    if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        return {"error": "single_class"}
    texts_tr = [text_fn(x) for x in train]
    texts_te = [text_fn(x) for x in test]
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tr = vec.fit_transform(texts_tr)
    X_te = vec.transform(texts_te)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    scored = [{"y_pm": int(y), "p_pm": float(p)} for y, p in zip(y_te, probs)]
    return {
        "n_train": len(train),
        "n_test": len(test),
        "test_auroc": try_auroc(scored, "p_pm"),
        "test_f1": float(f1_score(y_te, preds, zero_division=0)),
    }


def _eval_rows(rows: list[dict[str, Any]], tau_b: float) -> dict[str, Any]:
    det = detection_metrics(rows, score_key="agr_slot_score", threshold=tau_b)
    return {
        "n": len(rows),
        "n_pm": sum(int(r["y_pm"]) for r in rows),
        "probe_auroc": try_auroc(rows, "p_pm"),
        "bind_surprise_auroc": try_auroc(rows, "agr_slot_score"),
        "detection_f1": det.get("f1"),
        "tau_b": tau_b,
        "verdict_counts": dict(Counter(r["vshort"] for r in rows if r["y_pm"])),
        "domain_counts": dict(Counter(r["domain"] for r in rows)),
    }


def run_benchmark() -> dict[str, Any]:
    all_inst = collect_all()
    groups = [x.group_key for x in all_inst]
    traj_map = trajectory_split(groups, test_frac=TEST_TRAJ_FRAC, seed=SEED)

    train_raw = [x for x in all_inst if traj_map[x.group_key] == "train"]
    test_raw = [x for x in all_inst if traj_map[x.group_key] == "test"]

    train_bal, train_meta = balance_pool(
        train_raw, balance_pm_verdicts=True, clean_to_pm_ratio=1.0, seed=SEED
    )
    test_bal, test_meta = balance_pool(
        test_raw, balance_pm_verdicts=True, clean_to_pm_ratio=1.0, seed=SEED + 1
    )

    # Shopping canonical test (original split), unbalanced
    shop_canon = [x for x in all_inst if x.domain == "shopping" and x.orig_split == "test"]

    probe = _fit_probe(train_bal)
    tr_rows = _score_probe(probe, train_bal)
    te_rows = _score_probe(probe, test_bal)
    shop_rows = _score_probe(probe, shop_canon)

    tau_b = fit_threshold_f1(tr_rows, "agr_slot_score")

    tfidf_quote = _fit_eval_tfidf(
        train_bal, test_bal, text_fn=lambda x: normalize_quote_for_bow(x.text)
    )
    tfidf_claim = _fit_eval_tfidf(
        train_bal, test_bal, text_fn=lambda x: normalize_quote_for_bow(x.claim_value)
    )
    tfidf_shop = _fit_eval_tfidf(
        train_bal, shop_canon, text_fn=lambda x: normalize_quote_for_bow(x.text)
    )

    unbal_test_rows = _score_probe(probe, test_raw)

    return {
        "schema": "tau3_four_domain_balanced_benchmark_v1",
        "gold_note": "telecom/airline adjudication post GOLD_LABEL_FINAL_1k (2026-09-24)",
        "protocol": {
            "group_key": "domain:trajectory_id",
            "test_traj_frac": TEST_TRAJ_FRAC,
            "train_balance": "CEM=CAP=AH min-count; clean:pm=1:1",
            "test_balance": "same on held-out trajectories",
        },
        "pool_sizes": {
            "all_instances_with_activation": len(all_inst),
            "unique_trajectories": len(set(groups)),
            "train_traj": sum(1 for v in traj_map.values() if v == "train"),
            "test_traj": sum(1 for v in traj_map.values() if v == "test"),
        },
        "train_balance_meta": train_meta,
        "test_balance_meta": test_meta,
        "eval_balanced_test": _eval_rows(te_rows, tau_b),
        "eval_unbalanced_test_traj_holdout": _eval_rows(unbal_test_rows, tau_b),
        "eval_shopping_canonical_test": _eval_rows(shop_rows, tau_b),
        "tfidf_balanced_test_quote": tfidf_quote,
        "tfidf_balanced_test_claim_value": tfidf_claim,
        "tfidf_shopping_canonical_test": tfidf_shop,
        "gaps": {
            "l49_minus_tfidf_quote_balanced": None
            if te_rows and tfidf_quote.get("test_auroc") is not None
            else None,
        },
    }


def _fill_gaps(payload: dict[str, Any]) -> None:
    e = payload["eval_balanced_test"]
    tq = payload["tfidf_balanced_test_quote"]
    if e.get("probe_auroc") is not None and tq.get("test_auroc") is not None:
        payload["gaps"]["l49_probe_minus_tfidf_quote"] = round(e["probe_auroc"] - tq["test_auroc"], 4)
        payload["gaps"]["bind_minus_tfidf_quote"] = round(
            e["bind_surprise_auroc"] - tq["test_auroc"], 4
        )


def write_report(payload: dict[str, Any], path: Path) -> None:
    e = payload["eval_balanced_test"]
    u = payload["eval_unbalanced_test_traj_holdout"]
    s = payload["eval_shopping_canonical_test"]
    tq = payload["tfidf_balanced_test_quote"]
    tc = payload["tfidf_balanced_test_claim_value"]
    tm = payload["test_balance_meta"]
    lines = [
        "# Four-Domain Balanced Benchmark",
        "",
        "> 四域合并；轨迹级 80/20 分组；test 上 CEM=CAP=AH 均衡 + clean:PM=1:1。",
        "",
        "## 均衡 test 构成",
        "",
        f"- n={tm.get('n_out')} (PM {tm.get('n_pm_out')}, clean {tm.get('n_clean_out')})",
        f"- PM verdict: {tm.get('verdict_out')}",
        f"- domains: {tm.get('domain_out')}",
        "",
        "## 指标（L49 pooled 探针，train 同协议均衡）",
        "",
        "| 测试集 | n (PM) | L49 probe | BindSurprise | TF-IDF quote | TF-IDF claim |",
        "|--------|--------|----------:|-------------:|-------------:|-------------:|",
        f"| **均衡 test** | {e['n']} ({e['n_pm']}) | {e['probe_auroc']:.3f} | {e['bind_surprise_auroc']:.3f} | "
        f"{tq.get('test_auroc', 0):.3f} | {tc.get('test_auroc', 0):.3f} |",
        f"| 轨迹 holdout（未均衡） | {u['n']} ({u['n_pm']}) | {u['probe_auroc']:.3f} | {u['bind_surprise_auroc']:.3f} | — | — |",
        f"| shopping 原 test（对照） | {s['n']} ({s['n_pm']}) | {s['probe_auroc']:.3f} | {s['bind_surprise_auroc']:.3f} | "
        f"{payload['tfidf_shopping_canonical_test'].get('test_auroc', 0):.3f} | — |",
        "",
        "## 解读",
        "",
        "- 均衡 test 用于公平看三子类；shopping 原 test 仍为主文对照",
        "- 若均衡 test 上 L49 ≫ TF-IDF → 四域联合仍有表征间隙",
        "- 若 TF-IDF 仍接近 L49 → 词面/域混杂仍主导",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    payload = run_benchmark()
    _fill_gaps(payload)
    out_json = TAU3_DIR / "four_domain_balanced_benchmark.json"
    out_md = TAU3_DIR / "TAU3_FOUR_DOMAIN_BALANCED_BENCHMARK_REPORT.md"
    write_json(out_json, payload)
    write_report(payload, out_md)
    print(json.dumps({"eval_balanced_test": payload["eval_balanced_test"], "gaps": payload["gaps"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
