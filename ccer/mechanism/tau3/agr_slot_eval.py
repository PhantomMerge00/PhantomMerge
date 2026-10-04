"""AGR(slot) zero-shot and in-domain evaluation for tau3 cross-domain."""
from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ccer.mechanism.activation_store import get_vector, load_activation_npz
from ccer.mechanism.agr.calibration import fit_threshold_f1
from ccer.mechanism.agr.signals import AgrProbeModel
from ccer.mechanism.bind_surprise import compute_bind_surprise, detection_metrics, try_auroc
from ccer.mechanism.claim_filter_audit import audit_gating, instances_to_dataframe
from ccer.mechanism.tau3.cohort import instance_npz_path, tau3_instance_rows
from ccer.mechanism.tau3.domain import Tau3DomainConfig, get_tau3_domain
from ccer.mechanism.tau3.frozen_probe import load_shopping_frozen_probe
from ccer.io_utils import load_jsonl
from ccer.paths import PRISM_SUMMARY

LINE_A_LAYER = 49
LINE_A_POSITION = "claim_onset"
SHOPPING_TAU_B_KEY = "tau_b"
PROBE_ONLY_FLOOR_THRESHOLD = 0.99


def _effective_method_label(floor_rate: float, mode: str) -> str:
    if floor_rate >= PROBE_ONLY_FLOOR_THRESHOLD:
        return "probe-only (frozen)" if mode == "zeroshot" else "probe-only (retrained)"
    return "AGR(slot)"


def _load_split_maps(cfg: Tau3DomainConfig) -> tuple[dict[str, str], dict[str, str]]:
    payload = json.loads(cfg.split_manifest.read_text(encoding="utf-8"))
    traj_split = {str(k): str(v) for k, v in (payload.get("trajectory_split") or {}).items()}
    inst_split = {str(k): str(v) for k, v in (payload.get("instance_split") or {}).items()}
    return traj_split, inst_split


def _shopping_tau_b() -> float:
    if PRISM_SUMMARY.is_file():
        return float(json.loads(PRISM_SUMMARY.read_text(encoding="utf-8"))[SHOPPING_TAU_B_KEY])
    return -12.714


def _load_readout_index(cfg: Tau3DomainConfig) -> dict[str, dict[str, Any]]:
    return {str(r["instance_audit_key"]): r for r in load_jsonl(cfg.slot_readout)}


def _hidden(iak: str, tid: str, cfg: Tau3DomainConfig) -> np.ndarray | None:
    path = instance_npz_path(iak, tid, cfg=cfg)
    if not path.is_file():
        return None
    loaded = load_activation_npz(path)
    return get_vector(loaded, position=LINE_A_POSITION, layer=LINE_A_LAYER)


def _train_domain_probe(cfg: Tau3DomainConfig, train_split: str = "D_p") -> AgrProbeModel:
    _, inst_split = _load_split_maps(cfg)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for row in tau3_instance_rows(require_activation=True, cfg=cfg):
        if inst_split.get(row.instance_audit_key) != train_split:
            continue
        h = _hidden(row.instance_audit_key, row.trajectory_id, cfg)
        if h is None:
            continue
        X_list.append(h)
        y_list.append(row.y)
    if not X_list:
        raise RuntimeError(f"No {cfg.domain} activations for split {train_split}")
    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    clf.fit(Xs, y)
    return AgrProbeModel(scaler=scaler, clf=clf, layer=LINE_A_LAYER, position=LINE_A_POSITION)


def build_agr_slot_rows(
    probe: AgrProbeModel,
    cfg: Tau3DomainConfig,
    *,
    readout_index: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    readout_index = readout_index or _load_readout_index(cfg)
    _, inst_split = _load_split_maps(cfg)
    rows: list[dict[str, Any]] = []
    for inst in tau3_instance_rows(require_activation=True, cfg=cfg):
        ro = readout_index.get(inst.instance_audit_key)
        if ro is None:
            continue
        h = _hidden(inst.instance_audit_key, inst.trajectory_id, cfg)
        if h is None:
            continue
        p_pm = probe.prob(h)
        log_slot_mass = float(ro.get("log_slot_mass") or float("-inf"))
        bs = compute_bind_surprise(p_pm, log_slot_mass)
        rows.append(
            {
                "instance_audit_key": inst.instance_audit_key,
                "trajectory_id": inst.trajectory_id,
                "split": inst_split.get(inst.instance_audit_key, inst.split),
                "y_pm": inst.y,
                "gold_verdict": inst.gold_verdict,
                "response_quote": inst.response_quote,
                "slot_norm": inst.slot_norm,
                "p_pm": p_pm,
                "log_slot_mass": bs["log_slot_mass"],
                "agr_slot_score": bs["bind_surprise"],
                "probe_logit": bs["probe_logit"],
            }
        )
    return rows


def _gating_metrics(rows: list[dict[str, Any]], tau_b: float) -> dict[str, Any]:
    instances = [
        {
            "instance_audit_key": r["instance_audit_key"],
            "trajectory_id": r["trajectory_id"],
            "split": r["split"],
            "y": r["y_pm"],
            "y_pm": r["y_pm"],
            "gold_verdict": r.get("gold_verdict", ""),
            "response_quote": r.get("response_quote", ""),
        }
        for r in rows
    ]
    probe_map = {str(r["instance_audit_key"]): float(r["agr_slot_score"]) for r in rows}
    df = instances_to_dataframe(instances, probe_map)
    audit = audit_gating(df, lambda rec, t=tau_b: float(rec.get("p_pm", 0.0)) > t)
    return audit.to_dict()


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    tau_b: float,
    mode: str,
    cfg: Tau3DomainConfig,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y_pm = sum(int(r["y_pm"]) for r in rows)
    det = detection_metrics(rows, score_key="agr_slot_score", threshold=tau_b)
    floor_rate = float(
        sum(1 for r in rows if float(r.get("log_slot_mass") or 0) <= -11.99) / max(len(rows), 1)
    )
    effective = "probe_only" if floor_rate >= PROBE_ONLY_FLOOR_THRESHOLD else "agr_slot"
    summary = {
        "schema": "tau3_agr_slot_eval_v1",
        "domain": cfg.domain,
        "mode": mode,
        "method": "AGR(slot)",
        "effective_method": effective,
        "effective_method_label": _effective_method_label(floor_rate, mode),
        "j_lens_contribution": 0 if effective == "probe_only" else 1,
        "n_claims": len(rows),
        "n_pm": y_pm,
        "n_clean": len(rows) - y_pm,
        "tau_b": tau_b,
        "auroc": try_auroc(rows, "agr_slot_score"),
        "probe_auroc": try_auroc(rows, "p_pm"),
        "detection": det,
        "gating": _gating_metrics(rows, tau_b),
        "floor_rate": floor_rate,
        "rq2_eligible": effective == "agr_slot",
    }
    if extra:
        summary.update(extra)
    return summary


def run_zeroshot_eval(cfg: Tau3DomainConfig | None = None) -> dict[str, Any]:
    cfg = cfg or get_tau3_domain("telecom")
    probe = load_shopping_frozen_probe()
    all_rows = build_agr_slot_rows(probe, cfg)
    tau_b = _shopping_tau_b()
    test_rows = [r for r in all_rows if r["split"] == "test"]
    leak = json.loads(cfg.split_manifest.read_text(encoding="utf-8"))["leak_check"]
    return _summarize_rows(
        test_rows,
        tau_b=tau_b,
        mode="zeroshot",
        cfg=cfg,
        extra={
            "probe_source": "shopping_train_split_frozen",
            "tau_b_source": "shopping_prism_frozen",
            "split_leak_check": leak,
            "n_all_claims_with_activation": len(all_rows),
        },
    )


def run_indomain_eval(cfg: Tau3DomainConfig | None = None) -> dict[str, Any]:
    cfg = cfg or get_tau3_domain("telecom")
    probe = _train_domain_probe(cfg, train_split="D_p")
    all_rows = build_agr_slot_rows(probe, cfg)
    fit_rows = [r for r in all_rows if r["split"] == "D_f"]
    test_rows = [r for r in all_rows if r["split"] == "test"]
    tau_b = fit_threshold_f1(fit_rows, "agr_slot_score")
    leak = json.loads(cfg.split_manifest.read_text(encoding="utf-8"))["leak_check"]
    return _summarize_rows(
        test_rows,
        tau_b=tau_b,
        mode="indomain",
        cfg=cfg,
        extra={
            "probe_source": f"{cfg.domain}_D_p_retrained",
            "tau_b_source": f"{cfg.domain}_D_f_f1",
            "split_leak_check": leak,
            "n_fit_D_f": len(fit_rows),
        },
    )


def run_eval(
    mode: Literal["zeroshot", "indomain", "both"],
    *,
    domain: str = "telecom",
) -> dict[str, Any]:
    cfg = get_tau3_domain(domain)
    out: dict[str, Any] = {"domain": cfg.domain}
    if mode in ("zeroshot", "both"):
        out["zeroshot"] = run_zeroshot_eval(cfg)
    if mode in ("indomain", "both"):
        out["indomain"] = run_indomain_eval(cfg)
    if mode == "both" and "zeroshot" in out and "indomain" in out:
        zs = out["zeroshot"]
        idm = out["indomain"]
        out["delta_indomain_minus_zeroshot"] = {
            "auroc": (idm.get("auroc") or 0) - (zs.get("auroc") or 0)
            if zs.get("auroc") is not None and idm.get("auroc") is not None
            else None,
            "f1": (idm.get("detection", {}).get("f1") or 0)
            - (zs.get("detection", {}).get("f1") or 0),
            "pm_rate_reduction_delta": (idm.get("gating", {}).get("pm_reduction") or 0)
            - (zs.get("gating", {}).get("pm_reduction") or 0),
            "cb_retention_delta": (idm.get("gating", {}).get("cb_retention_rate") or 0)
            - (zs.get("gating", {}).get("cb_retention_rate") or 0),
        }
    return out
