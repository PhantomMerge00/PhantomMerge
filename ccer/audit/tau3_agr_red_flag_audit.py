"""Red-flag audit for tau3 cross-domain AGR(slot) / probe-only evaluation."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ccer.io_utils import load_jsonl, write_json
from ccer.mechanism.activation_store import get_vector, load_activation_npz
from ccer.mechanism.agr.signals import AgrProbeModel
from ccer.mechanism.bind_surprise import compute_bind_surprise, try_auroc
from ccer.mechanism.tau3.agr_slot_eval import (
    _hidden,
    _load_split_maps,
    _train_domain_probe,
    build_agr_slot_rows,
)
from ccer.mechanism.tau3.cohort import instance_npz_path, tau3_instance_rows
from ccer.mechanism.tau3.domain import Tau3DomainConfig, get_tau3_domain
from ccer.mechanism.tau3.frozen_probe import load_shopping_frozen_probe
from ccer.paths import TAU3_DIR

HIDDEN_DIM = 5120
C_SWEEP = [0.01, 0.1, 1.0, 10.0]
PCA_DIMS = [50, 100]


def _effective_method(floor_rate: float) -> str:
    return "probe_only" if floor_rate >= 0.99 else "agr_slot"


@dataclass
class _XY:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    meta: list[dict[str, Any]]


def _collect_xy(
    cfg: Tau3DomainConfig,
    *,
    splits: set[str],
) -> _XY:
    _, inst_split = _load_split_maps(cfg)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    groups: list[str] = []
    meta: list[dict[str, Any]] = []
    for row in tau3_instance_rows(require_activation=True, cfg=cfg):
        sp = inst_split.get(row.instance_audit_key, "unassigned")
        if sp not in splits:
            continue
        h = _hidden(row.instance_audit_key, row.trajectory_id, cfg)
        if h is None:
            continue
        X_list.append(h.astype(np.float32))
        y_list.append(int(row.y))
        groups.append(row.trajectory_id)
        meta.append(
            {
                "instance_audit_key": row.instance_audit_key,
                "trajectory_id": row.trajectory_id,
                "split": sp,
                "y_pm": int(row.y),
                "gold_verdict": row.gold_verdict,
                "response_quote": row.response_quote,
                "slot_norm": row.slot_norm,
                "claim_value": row.claim_value,
            }
        )
    if not X_list:
        return _XY(np.empty((0, HIDDEN_DIM)), np.array([]), np.array([]), [])
    return _XY(
        np.stack(X_list, axis=0),
        np.asarray(y_list, dtype=int),
        np.asarray(groups),
        meta,
    )


@dataclass
class _PcaProbe:
    scaler: StandardScaler
    pca: PCA
    clf: LogisticRegression

    def prob(self, h: np.ndarray) -> float:
        x = self.pca.transform(self.scaler.transform(h.reshape(1, -1)))
        return float(self.clf.predict_proba(x)[0, 1])


def _fit_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    pca_dim: int | None = None,
) -> AgrProbeModel | _PcaProbe:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if pca_dim is not None and pca_dim < Xs.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=42)
        Xp = pca.fit_transform(Xs)
        clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs", C=C)
        clf.fit(Xp, y)
        return _PcaProbe(scaler=scaler, pca=pca, clf=clf)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs", C=C)
    clf.fit(Xs, y)
    return AgrProbeModel(scaler=scaler, clf=clf)


def _rows_from_probe(probe: AgrProbeModel | _PcaProbe, xy: _XY) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, m in enumerate(xy.meta):
        h = xy.X[i]
        p_pm = probe.prob(h)
        bs = compute_bind_surprise(p_pm, -12.0)
        rows.append(
            {
                **m,
                "p_pm": p_pm,
                "log_slot_mass": bs["log_slot_mass"],
                "agr_slot_score": bs["bind_surprise"],
            }
        )
    return rows


def _sign_flip_auroc(rows: list[dict[str, Any]], score_key: str = "p_pm") -> dict[str, float | None]:
    base = try_auroc(rows, score_key)
    flipped = try_auroc(
        [{**r, "_flip": 1.0 - float(r.get(score_key) or 0.0)} for r in rows],
        "_flip",
    )
    direction_agnostic = None
    if base is not None:
        direction_agnostic = max(base, 1.0 - base)
    return {
        "auroc": base,
        "auroc_sign_flipped": flipped,
        "auroc_direction_agnostic": direction_agnostic,
    }


def _probe_only_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"floor_rate": 0.0, "corr_agr_vs_p_pm": None, "agr_minus_probe_logit_std": None}
    p = np.array([float(r["p_pm"]) for r in rows], dtype=np.float64)
    agr = np.array([float(r["agr_slot_score"]) for r in rows], dtype=np.float64)
    floor_rate = float(sum(1 for r in rows if float(r.get("log_slot_mass", 0) or 0) <= -11.99) / len(rows))
    if len(p) < 2 or np.std(p) < 1e-12:
        corr = None
    else:
        corr = float(np.corrcoef(p, agr)[0, 1])
    # when floor constant, agr = logit(p) + const → diff should have ~0 variance
    from ccer.mechanism.bind_surprise import probe_logit

    diffs = agr - np.array([probe_logit(float(x)) for x in p])
    return {
        "floor_rate": floor_rate,
        "corr_agr_vs_p_pm": corr,
        "agr_minus_probe_logit_std": float(np.std(diffs)) if len(diffs) else None,
        "effective_method": _effective_method(floor_rate),
    }


def _scenario_family(trajectory_id: str) -> str:
    tid = str(trajectory_id)
    m = re.match(r"^\[([^\]]+)\]", tid)
    if m:
        return m.group(1)
    if tid.startswith("telecom_clean:"):
        inner = tid.split(":", 1)[-1]
        m2 = re.match(r"^\[([^\]]+)\]", inner)
        return m2.group(1) if m2 else "telecom_clean"
    if tid.startswith("airline_clean:"):
        return "airline_clean"
    if tid.startswith("pm_"):
        return "airline_pm"
    return "other"


def _quote_fingerprint(quote: str) -> str:
    q = re.sub(r"\s+", " ", str(quote or "").strip().lower())
    q = re.sub(r'"[^"]{1,32}"', '"<ID>"', q)
    q = re.sub(r"\b[A-Z]\d{3,5}\b", "<ID>", q)
    return q


def _slot_distribution(rows: list[dict[str, Any]]) -> Counter:
    return Counter(str(r.get("slot_norm") or "") for r in rows)


def _chi2_uniform(c1: Counter, c2: Counter) -> dict[str, Any]:
    keys = sorted(set(c1) | set(c2))
    if not keys:
        return {"chi2": None, "n_keys": 0}
    n1 = sum(c1.values()) or 1
    n2 = sum(c2.values()) or 1
    chi2 = 0.0
    for k in keys:
        e1 = n1 * (c1.get(k, 0) + c2.get(k, 0)) / (n1 + n2)
        e2 = n2 * (c1.get(k, 0) + c2.get(k, 0)) / (n1 + n2)
        o1, o2 = c1.get(k, 0), c2.get(k, 0)
        if e1 > 0:
            chi2 += (o1 - e1) ** 2 / e1
        if e2 > 0:
            chi2 += (o2 - e2) ** 2 / e2
    return {"chi2": chi2, "n_keys": len(keys)}


def _template_near_duplicate_audit(
    test_meta: list[dict[str, Any]],
    train_meta: list[dict[str, Any]],
    *,
    domain: str,
) -> dict[str, Any]:
    train_quotes = {_quote_fingerprint(r.get("response_quote", "")) for r in train_meta}
    test_quotes = [_quote_fingerprint(r.get("response_quote", "")) for r in test_meta]
    exact_overlap = sum(1 for q in test_quotes if q in train_quotes)
    train_slots = _slot_distribution(train_meta)
    test_slots = _slot_distribution(test_meta)
    train_fams = Counter(_scenario_family(r["trajectory_id"]) for r in train_meta)
    test_fams = Counter(_scenario_family(r["trajectory_id"]) for r in test_meta)
    return {
        "test_quote_exact_template_overlap": exact_overlap,
        "test_quote_exact_template_overlap_rate": exact_overlap / max(len(test_meta), 1),
        "slot_norm_chi2_train_vs_test": _chi2_uniform(train_slots, test_slots),
        "scenario_family_train": dict(train_fams),
        "scenario_family_test": dict(test_fams),
        "domain": domain,
    }


def _regularization_sweep(xy_train: _XY, xy_test: _XY) -> dict[str, Any]:
    out: dict[str, Any] = {"C_sweep": {}, "pca_sweep": {}}
    for C in C_SWEEP:
        probe = _fit_probe(xy_train.X, xy_train.y, C=C)
        test_rows = _rows_from_probe(probe, xy_test)
        train_rows = _rows_from_probe(probe, xy_train)
        out["C_sweep"][str(C)] = {
            "train_auroc": try_auroc(train_rows, "p_pm"),
            "test_auroc": try_auroc(test_rows, "p_pm"),
        }
    for dim in PCA_DIMS:
        if xy_train.X.shape[0] < dim + 1:
            continue
        probe = _fit_probe(xy_train.X, xy_train.y, C=1.0, pca_dim=dim)
        test_rows = _rows_from_probe(probe, xy_test)
        train_rows = _rows_from_probe(probe, xy_train)
        out["pca_sweep"][str(dim)] = {
            "train_auroc": try_auroc(train_rows, "p_pm"),
            "test_auroc": try_auroc(test_rows, "p_pm"),
        }
    return out


def _grouped_cv(xy_fit: _XY, *, n_splits: int = 5) -> dict[str, Any]:
    if len(set(xy_fit.y.tolist())) < 2:
        return {"error": "single_class", "fold_aurocs": []}
    unique_groups = np.unique(xy_fit.groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        return {"error": "too_few_groups", "fold_aurocs": []}
    gkf = GroupKFold(n_splits=n_splits)
    fold_aurocs: list[float] = []
    for train_idx, test_idx in gkf.split(xy_fit.X, xy_fit.y, groups=xy_fit.groups):
        probe = _fit_probe(xy_fit.X[train_idx], xy_fit.y[train_idx], C=1.0)
        test_rows = _rows_from_probe(probe, _XY(xy_fit.X[test_idx], xy_fit.y[test_idx], xy_fit.groups[test_idx], [xy_fit.meta[i] for i in test_idx]))
        a = try_auroc(test_rows, "p_pm")
        if a is not None:
            fold_aurocs.append(a)
    return {
        "n_splits": n_splits,
        "fold_aurocs": fold_aurocs,
        "mean_auroc": float(np.mean(fold_aurocs)) if fold_aurocs else None,
        "std_auroc": float(np.std(fold_aurocs)) if fold_aurocs else None,
    }


def _template_blocked_cv(cfg: Tau3DomainConfig, xy_fit: _XY) -> dict[str, Any]:
    """Group by scenario family (COARSER than quote_template — does NOT block 87.9% overlap)."""
    if cfg.domain != "telecom":
        return {"skipped": True, "reason": "telecom_only"}
    fam_groups = np.array([_scenario_family(m["trajectory_id"]) for m in xy_fit.meta])
    if len(set(fam_groups.tolist())) < 2:
        return {"error": "too_few_families"}
    n_splits = min(5, len(set(fam_groups.tolist())))
    gkf = GroupKFold(n_splits=n_splits)
    fold_aurocs: list[float] = []
    for train_idx, test_idx in gkf.split(xy_fit.X, xy_fit.y, groups=fam_groups):
        probe = _fit_probe(xy_fit.X[train_idx], xy_fit.y[train_idx], C=1.0)
        test_rows = _rows_from_probe(
            probe,
            _XY(xy_fit.X[test_idx], xy_fit.y[test_idx], fam_groups[test_idx], [xy_fit.meta[i] for i in test_idx]),
        )
        a = try_auroc(test_rows, "p_pm")
        if a is not None:
            fold_aurocs.append(a)
    return {
        "n_splits": n_splits,
        "grouping": "scenario_family",
        "grouping_equals_quote_template": False,
        "grouping_note": "Coarser than quote_template; cannot refute 87.9% overlap. See quote_template_audit.json.",
        "fold_aurocs": fold_aurocs,
        "mean_auroc": float(np.mean(fold_aurocs)) if fold_aurocs else None,
        "std_auroc": float(np.std(fold_aurocs)) if fold_aurocs else None,
    }


def run_red_flag_audit(cfg: Tau3DomainConfig) -> dict[str, Any]:
    manifest = json.loads(cfg.split_manifest.read_text(encoding="utf-8"))
    inst_counts = manifest.get("instance_counts") or {}
    verdict_by_split = manifest.get("verdict_counts_by_split") or {}

    xy_dp = _collect_xy(cfg, splits={"D_p"})
    xy_test = _collect_xy(cfg, splits={"test"})
    xy_fit = _collect_xy(cfg, splits={"D_p", "D_c", "D_f"})

    zs_probe = load_shopping_frozen_probe()
    id_probe = _train_domain_probe(cfg, train_split="D_p")

    zs_rows = build_agr_slot_rows(zs_probe, cfg)
    id_rows = build_agr_slot_rows(id_probe, cfg)
    zs_test = [r for r in zs_rows if r["split"] == "test"]
    id_test = [r for r in id_rows if r["split"] == "test"]
    id_train = [r for r in id_rows if r["split"] == "D_p"]

    probe_only = _probe_only_check(zs_test)

    audit: dict[str, Any] = {
        "schema": "tau3_agr_red_flag_audit_v1",
        "domain": cfg.domain,
        "hidden_dim": HIDDEN_DIM,
        "split_counts": {
            "D_p": inst_counts.get("D_p"),
            "D_c": inst_counts.get("D_c"),
            "D_f": inst_counts.get("D_f"),
            "test": inst_counts.get("test"),
        },
        "n_d_ratio": {
            "D_p": round(len(xy_dp.y) / HIDDEN_DIM, 4),
            "fit_pool": round(len(xy_fit.y) / HIDDEN_DIM, 4),
        },
        "test_class_balance": {
            "n": len(xy_test.y),
            "n_pm": int(xy_test.y.sum()) if len(xy_test.y) else 0,
            "n_clean": int((1 - xy_test.y).sum()) if len(xy_test.y) else 0,
            "verdict_counts": verdict_by_split.get("test", {}),
        },
        "leak_check": manifest.get("leak_check"),
        "probe_only_confirmation": probe_only,
        "zeroshot": {
            "sign_flip": _sign_flip_auroc(zs_test),
            "probe_only": _probe_only_check(zs_test),
        },
        "indomain": {
            "train_D_p_auroc": try_auroc(id_train, "p_pm"),
            "test_auroc": try_auroc(id_test, "p_pm"),
            "train_test_gap": None,
            "sign_flip_test": _sign_flip_auroc(id_test),
            "probe_only": _probe_only_check(id_test),
        },
        "regularization_sweep": _regularization_sweep(xy_dp, xy_test),
        "grouped_cv_trajectory": _grouped_cv(xy_fit),
        "template_blocked_cv": _template_blocked_cv(cfg, xy_fit),
        "template_near_duplicate": _template_near_duplicate_audit(
            xy_test.meta,
            xy_dp.meta,
            domain=cfg.domain,
        ),
        "regularization_used": {
            "model": "LogisticRegression",
            "C_default": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "pca": None,
            "cross_validation": False,
        },
    }
    tr = audit["indomain"]["train_D_p_auroc"]
    te = audit["indomain"]["test_auroc"]
    if tr is not None and te is not None:
        audit["indomain"]["train_test_gap"] = round(te - tr, 4)
    return audit


def write_audit_report_md(audits: dict[str, dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Tau3 Cross-Domain AGR(slot) Red-Flag Audit Report",
        "",
        "> **RQ2 门禁**：在本文档结论确认前，telecom/airline in-domain「完美」指标**不得**作为 AGR 融合有效性证据写入论文。",
        "",
    ]
    for domain, a in audits.items():
        po = a.get("probe_only_confirmation") or {}
        ind = a.get("indomain") or {}
        zs = a.get("zeroshot") or {}
        zsf = zs.get("sign_flip") or {}
        isf = ind.get("sign_flip_test") or {}
        gcv = a.get("grouped_cv_trajectory") or {}
        tcv = a.get("template_blocked_cv") or {}
        tnd = a.get("template_near_duplicate") or {}
        tb = a.get("test_class_balance") or {}
        reg = a.get("regularization_sweep") or {}
        c001 = (reg.get("C_sweep") or {}).get("0.01") or {}
        pca50 = (reg.get("pca_sweep") or {}).get("50") or {}
        lines += [
            f"## {domain.capitalize()}",
            "",
            f"- **effective_method**: `{po.get('effective_method')}` (floor_rate={po.get('floor_rate'):.1%})",
            f"- **D_p claims / d**: {a['split_counts'].get('D_p')} / {a['hidden_dim']} (n/d={a['n_d_ratio']['D_p']})",
            f"- **test PM/clean**: {tb.get('n_pm')} / {tb.get('n_clean')} (n={tb.get('n')})",
            f"- **train AUROC (D_p)**: {ind.get('train_D_p_auroc')}",
            f"- **test AUROC (in-domain probe)**: {ind.get('test_auroc')}",
            f"- **train-test gap**: {ind.get('train_test_gap')}",
            f"- **zero-shot AUROC**: {zsf.get('auroc')} → sign-flip: {zsf.get('auroc_sign_flipped')} (dir-agnostic: {zsf.get('auroc_direction_agnostic')})",
            f"- **in-domain sign-flip test**: {isf.get('auroc')} → {isf.get('auroc_sign_flipped')}",
            f"- **grouped CV (trajectory)**: mean={gcv.get('mean_auroc')} ± {gcv.get('std_auroc')} folds={gcv.get('fold_aurocs')}",
            f"- **scenario-family blocked CV** (≠ quote template): mean={tcv.get('mean_auroc')} ± {tcv.get('std_auroc')}",
            f"- **quote template overlap test↔D_p**: {tnd.get('test_quote_exact_template_overlap')} ({tnd.get('test_quote_exact_template_overlap_rate', 0):.1%})",
            f"- **see also**: `quote_template_audit.json` for quote-template-blocked CV + zero-overlap split",
            f"- **C=0.01 test AUROC**: {c001.get('test_auroc')} (train={c001.get('train_auroc')})",
            f"- **PCA-50 test AUROC**: {pca50.get('test_auroc')} (train={pca50.get('train_auroc')})",
            "",
        ]
    lines += [
        "## RQ2 门禁结论",
        "",
        "### 可安全引用（diagnostic only）",
        "",
        "1. **floor_rate=100%** → J-lens 零贡献；所有跨域分数均为 **probe-only (frozen/retrained)**，非 AGR 融合",
        "2. **零样本迁移失败**；Telecom sign-flip 0.201→0.799，说明 shopping 探针方向在 telecom **系统性反转**（符号约定问题，非「无信号」）",
        "3. **split leak_check=0** 仅保证 trajectory_id 不重叠；telecom test↔D_p quote 模板重叠 **87.9%**",
        "",
        "### 不可引用为 AGR 有效性 / 域内语义可分性",
        "",
        "1. Telecom in-domain AUROC=1.000 / F1=1.000 — 实为 L49 线性探针，且 train AUROC≈1.0（过拟合或模板伪特征）",
        "2. Airline in-domain AUROC=0.999 — 同上，probe-only",
        "3. 强正则 (C=0.01) 与 PCA-50 **未使** telecom test AUROC 回落（仍为 1.0）→ 完美分离非正则化 artifact，更可能来自**模板/表层线索**",
        "4. scenario-family blocked CV=1.0 **不等于** quote-template blocked CV；前者粒度更粗，**不能反驳** 87.9% 重叠",
        "5. 必须用 quote-template 零重叠切分重测 probe AUROC（见 `TAU3_QUOTE_TEMPLATE_AUDIT_REPORT.md`）",
        "",
        "### 论文表述建议",
        "",
        "- 跨域表行标签改为 `Probe-only (frozen)` / `Probe-only (retrained)`",
        "- RQ2 可报告：跨域 AGR 功能性分支失效 (floor=100%) + 探针方向不迁移/反转",
        "- RQ2 **不可**报告：「域内 AGR(slot) 重训达到近完美检测」",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="both", choices=["telecom", "airline", "both"])
    args = ap.parse_args(argv)

    domains = ["telecom", "airline"] if args.domain == "both" else [args.domain]
    audits: dict[str, dict[str, Any]] = {}
    for d in domains:
        cfg = get_tau3_domain(d)
        audit = run_red_flag_audit(cfg)
        out = cfg.report.parent / "agr_red_flag_audit.json"
        write_json(out, audit)
        audits[d] = audit
        print(json.dumps({d: {"out": str(out), "train_auroc": audit["indomain"]["train_D_p_auroc"], "test_auroc": audit["indomain"]["test_auroc"]}}, indent=2))

    write_audit_report_md(audits, TAU3_DIR / "TAU3_CROSS_DOMAIN_AUDIT_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
