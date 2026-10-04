"""τ³ cross-domain mitigation (E2E axis): same rewrite method IDs as Shopping §0.4.

Probe-gated arms use the **best registered τ³ detector per domain**:
telecom/airline → in-domain AGR(slot) / BindSurprise (D_p probe, D_f τ);
retail → shopping frozen probe (``indomain_eligible=false``; probe AUROC > BindSurprise).

RARR / CoVe / Copy remain **native vendor** gates (not replaced by AGR).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ccer.mechanism.agr.calibration import fit_threshold_f1
from ccer.mechanism.claim_filter_audit import (
    audit_gating,
    bootstrap_traj_pm,
    instances_to_dataframe,
)
from ccer.mechanism.e2e_mitigation import apply_e2e_mitigation
from ccer.mechanism.e2e_mitigation_experiment import (
    E2E_METHOD_SPECS,
    _evaluate_block,
)
from ccer.mechanism.line_k_l_tiered import MethodId, apply_method_rewrite
from ccer.mechanism.prism_l_plus_experiment import (
    METHOD_SPECS as PROBE_GATED_METHOD_SPECS,
    _evaluate_method_block,
    _gate_checks,
)
from ccer.mechanism.line_k_claim_filter import _score_drop_factory
from ccer.mechanism.line_l_rewrite_quality import assess_rewrite_quality
from ccer.mechanism.anchor_resolve_heuristic import resolve_heuristic_anchor_pid
from ccer.mechanism.line_k_claim_filter import load_frozen_dev_thresholds
from ccer.mechanism.tau3.agr_slot_eval import (
    _shopping_tau_b,
    _train_domain_probe,
    build_agr_slot_rows,
)
from ccer.mechanism.tau3.cohort import tau3_instance_rows
from ccer.mechanism.tau3.domain import SUPPORTED_DOMAINS, get_tau3_domain
from ccer.mechanism.tau3.frozen_probe import load_shopping_frozen_probe
from ccer.io_utils import load_jsonl
import json

_LOG_SLOT_FLOOR = -11.99


def _load_tau3_trajectory_index(domain: str) -> dict[str, dict[str, Any]]:
    cfg = get_tau3_domain(domain)
    return {str(r["trajectory_id"]): r for r in load_jsonl(cfg.normalized)}


def _split_manifest_meta(cfg) -> dict[str, Any]:
    if not cfg.split_manifest.is_file():
        return {}
    return json.loads(cfg.split_manifest.read_text(encoding="utf-8"))


def _mitigation_detection_bundle(domain: str) -> dict[str, Any]:
    """Per-domain detector + τ (aligned with tau3 cross-domain reports)."""
    cfg = get_tau3_domain(domain)
    meta = _split_manifest_meta(cfg)
    indomain_ok = bool(meta.get("indomain_eligible", True))

    if not indomain_ok:
        probe = load_shopping_frozen_probe()
        all_rows = build_agr_slot_rows(probe, cfg)
        thresholds = load_frozen_dev_thresholds()
        tau_probe = float(thresholds["line_a_probe_aggressive"])
        tau_b = float(_shopping_tau_b())
        return {
            "cfg": cfg,
            "detection_mode": "zeroshot_shopping_frozen",
            "probe_source": "shopping_train_split_frozen",
            "indomain_eligible": False,
            "indomain_skip_reason": meta.get("indomain_skip_reason"),
            "all_rows": all_rows,
            "tau_b": tau_b,
            "tau_probe": tau_probe,
            "track_k_score_col": "p_pm",
            "track_k_tau": tau_probe,
            "track_k_label": "Track_K_E2E_frozen_probe_delete",
            "detection_note": (
                "Retail PM too sparse for D_p; gate on frozen L49 p_pm @ Shopping τ_agg "
                "(probe AUROC > BindSurprise on retail diagnostic)."
            ),
        }

    probe = _train_domain_probe(cfg, train_split="D_p")
    all_rows = build_agr_slot_rows(probe, cfg)
    fit_rows = [r for r in all_rows if r["split"] == "D_f"]
    if not fit_rows:
        raise RuntimeError(f"{domain}: no D_f rows for τ calibration")
    tau_b = fit_threshold_f1(fit_rows, "agr_slot_score")
    tau_probe = fit_threshold_f1(fit_rows, "p_pm")
    return {
        "cfg": cfg,
        "detection_mode": "indomain_agr_slot",
        "probe_source": f"{cfg.domain}_D_p_retrained",
        "indomain_eligible": True,
        "all_rows": all_rows,
        "tau_b": tau_b,
        "tau_probe": tau_probe,
        "n_fit_D_f": len(fit_rows),
        "track_k_score_col": "agr_slot_score",
        "track_k_tau": tau_b,
        "track_k_label": "Track_K_E2E_agr_slot_delete",
        "detection_note": "In-domain BindSurprise τ_B (D_f F1); PRISM on in-domain p_pm + τ_probe.",
    }


def _tau3_eval_dataframe(bundle: dict[str, Any]) -> pd.DataFrame:
    domain = bundle["cfg"].domain
    inst_map = {
        r.instance_audit_key: r
        for r in tau3_instance_rows(require_activation=False, cfg=bundle["cfg"])
    }
    instances: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    # PRISM / audit use domain probe probability; gating compares to tau_probe.
    probe_map = {str(r["instance_audit_key"]): float(r["p_pm"]) for r in bundle["all_rows"]}
    for r in bundle["all_rows"]:
        iak = str(r["instance_audit_key"])
        inst = inst_map.get(iak)
        log_mass = float(r.get("log_slot_mass") or float("-inf"))
        rec = {
            "instance_audit_key": iak,
            "trajectory_id": r["trajectory_id"],
            "split": r["split"],
            "y": int(r.get("y_pm", 0)),
            "y_pm": int(r.get("y_pm", 0)),
            "gold_verdict": r["gold_verdict"],
            "response_quote": r["response_quote"],
            "slot_norm": r.get("slot_norm") or (inst.slot_norm if inst else ""),
            "claim_value": inst.claim_value if inst else "",
            "agr_slot_score": float(r["agr_slot_score"]),
            "log_slot_mass": log_mass,
            "prism_slot_type_aligned": log_mass > _LOG_SLOT_FLOOR,
            "tau3_domain": domain,
        }
        instances.append(rec)
        enriched.append(rec)
    df = instances_to_dataframe(instances, probe_map)
    for col in (
        "slot_norm",
        "claim_value",
        "agr_slot_score",
        "log_slot_mass",
        "prism_slot_type_aligned",
        "tau3_domain",
    ):
        df[col] = [r.get(col) for r in enriched]
    df["slot"] = df["slot_norm"]
    return df


def _apply_e2e_arm_tau3(
    df: pd.DataFrame,
    traj_index: dict[str, dict[str, Any]],
    *,
    tau_probe: float,
    tau_b: float,
    arm: str,
    method: str,
    use_llm: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    from ccer.mechanism.e2e_mitigation import E2EMethodId

    method_id: E2EMethodId = method  # type: ignore[assignment]
    logs: list[dict[str, Any]] = []
    actions: list[str] = []
    branches: list[str] = []
    quotes_after: list[str | None] = []
    verified_flags: list[bool] = []

    for _, row in df.iterrows():
        rec = row.to_dict()
        rec["tau"] = tau_probe
        tid = str(rec["trajectory_id"])
        traj = traj_index.get(tid, {})
        anchor_pid = resolve_heuristic_anchor_pid(traj)

        if method_id == "track_k_e2e":
            score_col = str(rec.get("_track_k_score_col") or "agr_slot_score")
            gate_tau = float(rec.get("_track_k_tau") if rec.get("_track_k_tau") is not None else tau_b)
            score = float(rec.get(score_col, 0.0))
            quote = str(rec.get("response_quote") or "")
            if score <= gate_tau:
                outcome = {
                    "arm": arm,
                    "branch": "e2e_agr_slot_keep",
                    "action": "keep",
                    "quote_before": quote,
                    "quote_after": quote,
                    "method": "track_k_e2e",
                    "gate_source": rec.get("_track_k_gate_source") or "agr_slot_delete",
                    "tau_gate": gate_tau,
                }
            else:
                outcome = {
                    "arm": arm,
                    "branch": "e2e_agr_slot_delete",
                    "action": "delete",
                    "quote_before": quote,
                    "quote_after": None,
                    "method": "track_k_e2e",
                    "gate_source": rec.get("_track_k_gate_source") or "agr_slot_delete",
                    "tau_gate": gate_tau,
                }
        elif not anchor_pid:
            outcome = {
                "arm": arm,
                "branch": "e2e_no_anchor",
                "action": "delete",
                "quote_before": rec.get("response_quote"),
                "quote_after": None,
                "miss_reason": "heuristic_anchor_miss",
                "method": method_id,
            }
        else:
            outcome = apply_e2e_mitigation(
                rec,
                traj,
                method=method_id,
                arm=arm,
                tau=tau_probe,
                anchor_pid=anchor_pid,
                use_llm=use_llm,
            )

        q = assess_rewrite_quality({**outcome, **rec}, traj, anchor_pid=anchor_pid or "")
        enriched = {
            **rec,
            **outcome,
            "trajectory_id": tid,
            "anchor_pid": anchor_pid,
            "_verified_grounded": q.verified_grounded,
            "quality": q.to_dict(),
        }
        logs.append(enriched)
        actions.append(str(outcome.get("action") or "keep"))
        branches.append(str(outcome.get("branch") or "skip"))
        quotes_after.append(outcome.get("quote_after"))
        verified_flags.append(q.verified_grounded)

    out = df.copy()
    out["_line_l_action"] = actions
    out["_line_l_branch"] = branches
    out["quote_after"] = quotes_after
    out["_verified_grounded"] = verified_flags
    return out, logs


def _apply_method_arm_tau3(
    df: pd.DataFrame,
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    arm: str,
    method: MethodId,
    use_llm: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Probe-gated rewrite arms with τ³ heuristic anchor (matches E2E τ³ inference)."""
    logs: list[dict[str, Any]] = []
    actions: list[str] = []
    branches: list[str] = []
    quotes_after: list[str | None] = []
    verified_flags: list[bool] = []

    for _, row in df.iterrows():
        rec = row.to_dict()
        rec["tau"] = tau
        tid = str(rec["trajectory_id"])
        traj = traj_index.get(tid, {})
        anchor_pid = resolve_heuristic_anchor_pid(traj)

        if not anchor_pid:
            outcome = {
                "arm": arm,
                "branch": "heuristic_anchor_miss",
                "action": "delete",
                "quote_before": rec.get("response_quote"),
                "quote_after": None,
                "miss_reason": "heuristic_anchor_miss",
                "method": method,
            }
            q = assess_rewrite_quality({**outcome, **rec}, traj, anchor_pid="")
            enriched = {
                **rec,
                **outcome,
                "trajectory_id": tid,
                "anchor_pid": anchor_pid,
                "_verified_grounded": q.verified_grounded,
                "quality": q.to_dict(),
                "tau3_domain": rec.get("tau3_domain"),
            }
            logs.append(enriched)
            actions.append("delete")
            branches.append("heuristic_anchor_miss")
            quotes_after.append(None)
            verified_flags.append(q.verified_grounded)
            continue

        outcome = apply_method_rewrite(
            rec,
            traj,
            anchor_pid=anchor_pid,
            arm=arm,
            method=method,
            use_llm=use_llm,
        )
        q = assess_rewrite_quality({**outcome, **rec}, traj, anchor_pid=anchor_pid)
        enriched = {
            **rec,
            **outcome,
            "trajectory_id": tid,
            "anchor_pid": anchor_pid,
            "_verified_grounded": q.verified_grounded,
            "quality": q.to_dict(),
            "tau3_domain": rec.get("tau3_domain"),
        }
        logs.append(enriched)
        actions.append(str(outcome.get("action") or "keep"))
        branches.append(str(outcome.get("branch") or "skip_unflagged"))
        quotes_after.append(outcome.get("quote_after"))
        verified_flags.append(q.verified_grounded)

    out = df.copy()
    out["_line_l_action"] = actions
    out["_line_l_branch"] = branches
    out["quote_after"] = quotes_after
    out["_verified_grounded"] = verified_flags
    return out, logs


def _methods_done(summary: dict[str, Any] | None) -> set[str]:
    if not summary:
        return set()
    block = (summary.get("results_by_tau") or {}).get("indomain_agr_slot") or {}
    out: set[str] = set()
    for key, row in (block.get("methods") or {}).items():
        if row.get("status") == "error":
            continue  # retry on next run (not skipped)
        if row.get("audit_optimistic") or row.get("audit_conservative"):
            out.add(key)
    return out


def run_tau3_e2e_mitigation_experiment(
    domain: str,
    *,
    methods: list[str] | None = None,
    n_boot: int = 500,
    seed: int = 42,
    use_llm: bool = True,
    test_split: str = "test",
    skip_methods: set[str] | None = None,
    on_checkpoint: Any | None = None,
    prior_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    domain = str(domain).strip().lower()
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"Unsupported tau3 domain: {domain!r}")

    methods = methods or list(E2E_METHOD_SPECS.keys())
    skip_methods = skip_methods or set()
    bundle = _mitigation_detection_bundle(domain)
    tau_b = float(bundle["tau_b"])
    tau_probe = float(bundle["tau_probe"])
    track_k_col = str(bundle["track_k_score_col"])
    track_k_tau = float(bundle["track_k_tau"])
    traj_index = _load_tau3_trajectory_index(domain)
    df_all = _tau3_eval_dataframe(bundle)
    df_test = df_all[df_all["split"] == test_split].reset_index(drop=True)
    gate_src = (
        "frozen_probe_delete"
        if bundle["detection_mode"] == "zeroshot_shopping_frozen"
        else "agr_slot_delete"
    )
    df_test = df_test.assign(
        _track_k_score_col=track_k_col,
        _track_k_tau=track_k_tau,
        _track_k_gate_source=gate_src,
    )

    llm_methods = {"b1_rarr_e2e", "b2_cove_e2e", "prism_l_plus_a_e2e", "prism_l_plus_b_e2e"}

    all_logs: list[dict[str, Any]] = []
    tau_results: dict[str, Any] = {"tau": tau_b, "tau_probe": tau_probe, "methods": {}}
    if prior_summary:
        prev_block = (prior_summary.get("results_by_tau") or {}).get("indomain_agr_slot") or {}
        tau_results["methods"] = dict(prev_block.get("methods") or {})

    baseline = audit_gating(df_test, lambda _r: False).to_dict()
    n_traj = int(df_test["trajectory_id"].nunique()) if len(df_test) else 0

    def _flush(partial_status: str, methods_run: list[str]) -> None:
        if not on_checkpoint:
            return
        payload = _assemble_summary(
            domain=domain,
            bundle=bundle,
            tau_b=tau_b,
            tau_probe=tau_probe,
            tau_results=tau_results,
            baseline=baseline,
            n_traj=n_traj,
            n_claims=len(df_test),
            methods_run=methods_run,
            experiment_status=partial_status,
        )
        on_checkpoint(payload, all_logs)

    if "track_k_e2e" in methods and "track_k_e2e" not in skip_methods:
        drop_fn = _score_drop_factory(track_k_col, track_k_tau)
        tk = audit_gating(df_test, drop_fn)
        tau_results["methods"]["track_k_e2e"] = {
            "label": bundle.get("track_k_label", "Track_K_E2E"),
            "gate_description": str(bundle.get("detection_note") or bundle["detection_mode"]),
            "audit_optimistic": {
                **tk.to_dict(),
                **bootstrap_traj_pm(df_test, drop_fn, B=n_boot, seed=seed),
            },
            "audit_conservative": {
                **tk.to_dict(),
                **bootstrap_traj_pm(df_test, drop_fn, B=n_boot, seed=seed),
            },
        }
        _flush("in_progress", methods)

    for key in methods:
        if key == "track_k_e2e" or key in skip_methods:
            continue
        spec = E2E_METHOD_SPECS.get(key)
        if not spec:
            continue
        method_id = spec["method"]
        arm = str(spec["label"])
        try:
            df_arm, logs = _apply_e2e_arm_tau3(
                df_test,
                traj_index,
                tau_probe=tau_probe,
                tau_b=tau_b,
                arm=arm,
                method=method_id,
                use_llm=use_llm and key in llm_methods,
            )
            all_logs.extend(logs)
            block = _evaluate_block(
                df_arm,
                logs,
                traj_index,
                tau=tau_probe,
                method_key=key,
                n_boot=n_boot,
                seed=seed,
            )
            if key in ("prism_l_plus_a_e2e", "prism_l_plus_b_e2e"):
                block["gate_description"] = (
                    f"PRISM gate on in-domain probe p_pm (τ_probe={tau_probe:.4f} from D_f); "
                    + str(block.get("gate_description") or "")
                )
            tau_results["methods"][key] = block
        except Exception as exc:
            tau_results["methods"][key] = {
                "label": spec.get("label", key),
                "status": "error",
                "error": repr(exc),
            }
            all_logs.append(
                {
                    "method": key,
                    "status": "error",
                    "error": repr(exc),
                    "tau3_domain": domain,
                }
            )
        _flush("in_progress", methods)

    return _assemble_summary(
        domain=domain,
        bundle=bundle,
        tau_b=tau_b,
        tau_probe=tau_probe,
        tau_results=tau_results,
        baseline=baseline,
        n_traj=n_traj,
        n_claims=len(df_test),
        methods_run=methods,
        experiment_status="completed",
        _rewrite_logs=all_logs,
    )


def _assemble_summary(
    *,
    domain: str,
    bundle: dict[str, Any],
    tau_b: float,
    tau_probe: float,
    tau_results: dict[str, Any],
    baseline: dict[str, Any],
    n_traj: int,
    n_claims: int,
    methods_run: list[str],
    experiment_status: str,
    _rewrite_logs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cfg = get_tau3_domain(domain)
    ref_auroc = None
    if bundle.get("detection_mode") == "zeroshot_shopping_frozen" and cfg.zeroshot_summary.is_file():
        ref_auroc = json.loads(cfg.zeroshot_summary.read_text(encoding="utf-8")).get("probe_auroc")
    elif cfg.indomain_summary.is_file():
        ref_auroc = json.loads(cfg.indomain_summary.read_text(encoding="utf-8")).get("auroc")

    tau_b_src = (
        "shopping_prism_frozen (BindSurprise)"
        if bundle.get("detection_mode") == "zeroshot_shopping_frozen"
        else f"{domain}_D_f_f1 on agr_slot_score"
    )
    tau_probe_src = (
        "shopping line_k dev aggressive"
        if bundle.get("detection_mode") == "zeroshot_shopping_frozen"
        else f"{domain}_D_f_f1 on p_pm (PRISM-L+ gate)"
    )

    out: dict[str, Any] = {
        "schema_version": "ccer_tau3_e2e_mitigation_v2",
        "experiment_status": experiment_status,
        "tau3_domain": domain,
        "protocol": {
            "mode": "end_to_end",
            "aligned_with": "Shopping §0.4 rewrite methods; τ³-specific **detection** for probe-gated arms",
            "cohort": f"tau3 {domain} instance_split=test",
            "detection_mode": bundle.get("detection_mode"),
            "detection_probe_gated": bundle.get("detection_note") or bundle.get("detection_mode"),
            "probe_training": bundle["probe_source"],
            "track_k_gate": f"{bundle.get('track_k_score_col')} @ {bundle.get('track_k_tau')}",
            "tau_b_source": tau_b_src,
            "tau_probe_source": tau_probe_src,
            "tau_b": tau_b,
            "tau_probe": tau_probe,
            "detection_ref_auroc": ref_auroc,
            "indomain_eligible": bundle.get("indomain_eligible"),
            "vendor_native_gates": ["b1_rarr_e2e", "b2_cove_e2e", "b3_copy_e2e"],
            "inference": "heuristic anchor + parsed slot; gold eval-only",
            "llm_backend": "Qwen3-8B @8012 (Obs_τ + rewrite)",
            "methods_run": methods_run,
        },
        "test_baseline": {
            **baseline,
            "n_trajectories": n_traj,
            "n_claims": n_claims,
        },
        "results_by_tau": {"indomain_agr_slot": tau_results},
    }
    if _rewrite_logs is not None:
        out["_rewrite_logs"] = _rewrite_logs
    return out


def _methods_done_probe_gated(summary: dict[str, Any] | None) -> set[str]:
    if not summary:
        return set()
    block = (summary.get("results_by_tau") or {}).get("probe_gated") or {}
    out: set[str] = set()
    for key, row in (block.get("methods") or {}).items():
        if row.get("status") == "error":
            continue
        if row.get("audit_optimistic") or row.get("audit_conservative"):
            out.add(key)
    return out


def _assemble_probe_gated_summary(
    *,
    domain: str,
    bundle: dict[str, Any],
    tau_probe: float,
    results_by_tau: dict[str, Any],
    baseline: dict[str, Any],
    n_traj: int,
    n_claims: int,
    methods_run: list[str],
    experiment_status: str,
    quality_gates: dict[str, Any] | None = None,
    _rewrite_logs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cfg = get_tau3_domain(domain)
    ref_auroc = None
    if bundle.get("detection_mode") == "zeroshot_shopping_frozen" and cfg.zeroshot_summary.is_file():
        ref_auroc = json.loads(cfg.zeroshot_summary.read_text(encoding="utf-8")).get("probe_auroc")
    elif cfg.indomain_summary.is_file():
        ref_auroc = json.loads(cfg.indomain_summary.read_text(encoding="utf-8")).get("auroc")

    tau_probe_src = (
        "shopping line_k dev aggressive (frozen zeroshot)"
        if bundle.get("detection_mode") == "zeroshot_shopping_frozen"
        else f"{domain}_D_f_f1 on p_pm (PRISM-L+ gate)"
    )

    return {
        "schema_version": "ccer_tau3_probe_gated_mitigation_v1",
        "experiment_status": experiment_status,
        "tau3_domain": domain,
        "protocol": {
            "mode": "probe_gated",
            "aligned_with": "Shopping §4.2 prism_l_plus METHOD_SPECS",
            "cohort": f"tau3 {domain} instance_split=test",
            "detection_mode": bundle.get("detection_mode"),
            "detection_probe_gated": bundle.get("detection_note") or bundle.get("detection_mode"),
            "probe_training": bundle["probe_source"],
            "track_k_gate": f"{bundle.get('track_k_score_col')} @ {bundle.get('track_k_tau')}",
            "tau_probe_source": tau_probe_src,
            "tau_probe": tau_probe,
            "tau_b": float(bundle["tau_b"]),
            "detection_ref_auroc": ref_auroc,
            "indomain_eligible": bundle.get("indomain_eligible"),
            "inference": "heuristic anchor + probe/AGR gate; gold eval-only",
            "llm_backend": "Qwen3-8B @8012 (Obs_τ + rewrite)",
            "methods_run": methods_run,
        },
        "test_baseline": {
            **baseline,
            "n_trajectories": n_traj,
            "n_claims": n_claims,
        },
        "results_by_tau": results_by_tau,
        "quality_gates": quality_gates or {},
        **({"_rewrite_logs": _rewrite_logs} if _rewrite_logs is not None else {}),
    }


def run_tau3_probe_gated_mitigation_experiment(
    domain: str,
    *,
    methods: list[str] | None = None,
    n_boot: int = 500,
    seed: int = 42,
    use_llm: bool = True,
    test_split: str = "test",
    skip_methods: set[str] | None = None,
    on_checkpoint: Any | None = None,
    prior_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """τ³ probe-gated seven-arm mitigation (Shopping §4.2 method IDs)."""
    domain = str(domain).strip().lower()
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"Unsupported tau3 domain: {domain!r}")

    methods = methods or list(PROBE_GATED_METHOD_SPECS.keys())
    skip_methods = skip_methods or set()
    bundle = _mitigation_detection_bundle(domain)
    tau_probe = float(bundle["tau_probe"])
    track_k_col = str(bundle["track_k_score_col"])
    track_k_tau = float(bundle["track_k_tau"])
    traj_index = _load_tau3_trajectory_index(domain)
    df_all = _tau3_eval_dataframe(bundle)
    df_test = df_all[df_all["split"] == test_split].reset_index(drop=True)

    llm_methods = {
        "b1_rarr_official",
        "b2_cove_official",
        "prism_l_plus_a",
        "prism_l_plus_b",
        "prism_audit_only",
    }

    all_logs: list[dict[str, Any]] = []
    tau_block: dict[str, Any] = {"tau": tau_probe, "methods": {}}
    if prior_summary:
        prev = (prior_summary.get("results_by_tau") or {}).get("probe_gated") or {}
        tau_block["methods"] = dict(prev.get("methods") or {})

    baseline = audit_gating(df_test, lambda _r: False).to_dict()
    n_traj = int(df_test["trajectory_id"].nunique()) if len(df_test) else 0
    results_by_tau: dict[str, Any] = {"probe_gated": tau_block}

    def _flush(partial_status: str, methods_run: list[str]) -> None:
        if not on_checkpoint:
            return
        payload = _assemble_probe_gated_summary(
            domain=domain,
            bundle=bundle,
            tau_probe=tau_probe,
            results_by_tau=results_by_tau,
            baseline=baseline,
            n_traj=n_traj,
            n_claims=len(df_test),
            methods_run=methods_run,
            experiment_status=partial_status,
            quality_gates={"probe_gated": _gate_checks(all_logs, tau_probe)},
        )
        on_checkpoint(payload, all_logs)

    if "track_k" in methods and "track_k" not in skip_methods:
        drop_fn = _score_drop_factory(track_k_col, track_k_tau)
        tk = audit_gating(df_test, drop_fn)
        tau_block["methods"]["track_k"] = {
            "label": bundle.get("track_k_label", "Track_K_deletion"),
            "gate_description": str(bundle.get("detection_note") or bundle["detection_mode"]),
            "audit_optimistic": {
                **tk.to_dict(),
                **bootstrap_traj_pm(df_test, drop_fn, B=n_boot, seed=seed),
            },
            "audit_conservative": {
                **tk.to_dict(),
                **bootstrap_traj_pm(df_test, drop_fn, B=n_boot, seed=seed),
            },
        }
        _flush("in_progress", methods)

    for key in methods:
        if key == "track_k" or key in skip_methods:
            continue
        spec = PROBE_GATED_METHOD_SPECS.get(key)
        if not spec or spec.get("method") is None:
            continue
        method_id: MethodId = spec["method"]
        arm = str(spec.get("label") or key)
        try:
            df_arm, logs = _apply_method_arm_tau3(
                df_test,
                traj_index,
                tau=tau_probe,
                arm=arm,
                method=method_id,
                use_llm=use_llm and key in llm_methods,
            )
            all_logs.extend(logs)
            block = _evaluate_method_block(
                df_arm,
                logs,
                traj_index,
                tau=tau_probe,
                method_key=key,
                n_boot=n_boot,
                seed=seed,
            )
            block["label"] = spec["label"]
            tau_block["methods"][key] = block
        except Exception as exc:
            tau_block["methods"][key] = {
                "label": spec.get("label", key),
                "status": "error",
                "error": repr(exc),
            }
            all_logs.append(
                {
                    "method": key,
                    "status": "error",
                    "error": repr(exc),
                    "tau3_domain": domain,
                }
            )
        _flush("in_progress", methods)

    return _assemble_probe_gated_summary(
        domain=domain,
        bundle=bundle,
        tau_probe=tau_probe,
        results_by_tau=results_by_tau,
        baseline=baseline,
        n_traj=n_traj,
        n_claims=len(df_test),
        methods_run=methods,
        experiment_status="completed",
        quality_gates={"probe_gated": _gate_checks(all_logs, tau_probe)},
        _rewrite_logs=all_logs,
    )


def run_tau3_e2e_all_domains(
    *,
    domains: tuple[str, ...] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    domains = domains or tuple(sorted(SUPPORTED_DOMAINS))
    per_domain: dict[str, Any] = {}
    for d in domains:
        per_domain[d] = run_tau3_e2e_mitigation_experiment(d, **kwargs)
    return {
        "schema_version": "ccer_tau3_e2e_mitigation_pooled_v1",
        "domains": list(domains),
        "per_domain": {d: {k: v for k, v in per_domain[d].items() if k != "_rewrite_logs"} for d in domains},
    }
