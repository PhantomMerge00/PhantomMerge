"""PRISM: Probe-Readout Integrated Slot Mitigation — audit layer."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Literal

from ccer.mechanism.activation_store import get_vector, load_activation_npz
from ccer.mechanism.bind_surprise import (
    bind_surprise_rationale,
    classify_mechanism_quadrant,
    compute_bind_surprise,
    compute_slot_workspace_mass,
    detection_metrics,
    fit_log_slot_threshold,
    fit_tau_b,
    fit_tau_b_at_recall,
    pr_dominance_report,
    try_auroc,
)
from ccer.mechanism.jspace_readout import readout_topk
from ccer.mechanism.line_a_instance_activations import instance_npz_path
from ccer.mechanism.line_k_claim_filter import load_frozen_dev_thresholds, load_frozen_probe_scores
from ccer.io_utils import load_jsonl
from ccer.paths import INCREMENTAL_ADJUDICATION_JSONL
from ccer.replay.live_position import LINE_D_CLAIM_LAYER_DEFAULT
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

AuditState = Literal["confirmed_risk", "probe_only", "latent_slot", "clean"]
MechanismQuadrant = Literal["grounded_risk", "ungrounded_alarm", "latent_slot", "clean"]

_QUADRANT_TO_LEGACY: dict[str, AuditState] = {
    "grounded_risk": "confirmed_risk",
    "ungrounded_alarm": "probe_only",
    "latent_slot": "latent_slot",
    "clean": "clean",
}


def score_slot_alignment(
    topk: list[dict[str, Any]],
    slot_norm: str,
) -> dict[str, Any]:
    """Legacy binary slot alignment (derived from workspace mass)."""
    mass = compute_slot_workspace_mass(topk, slot_norm)
    hits = mass.get("slot_lexicon_hits") or []
    return {
        "slot_type_aligned": bool(hits),
        "slot_hint_tokens": hits,
        "slot_rank": mass.get("slot_rank"),
        "log_slot_mass": mass.get("log_slot_mass"),
        "slot_mass": mass.get("slot_mass"),
    }


def classify_audit_state(
    p_pm: float,
    tau: float,
    slot_type_aligned: bool,
) -> AuditState:
    """Legacy four-state labels (v1 hard intersection)."""
    probe_high = float(p_pm) > float(tau)
    if probe_high and slot_type_aligned:
        return "confirmed_risk"
    if probe_high and not slot_type_aligned:
        return "probe_only"
    if not probe_high and slot_type_aligned:
        return "latent_slot"
    return "clean"


def load_instance_hidden(
    instance_audit_key: str,
    trajectory_id: str,
    *,
    layer: int = LINE_D_CLAIM_LAYER_DEFAULT,
    position: str = "claim_onset",
) -> np.ndarray | None:
    path = instance_npz_path(instance_audit_key, trajectory_id)
    if not path.is_file():
        return None
    loaded = load_activation_npz(path)
    return get_vector(loaded, position=position, layer=layer)


def readout_claim_instance(
    h: np.ndarray,
    lens: JacobianLens,
    jlens_model: LensModel,
    tokenizer: Any,
    *,
    layer: int = LINE_D_CLAIM_LAYER_DEFAULT,
    k: int = 10,
) -> list[dict[str, Any]]:
    return readout_topk(
        lens,
        jlens_model,
        tokenizer,
        h,
        layer=layer,
        k=k,
        use_jacobian=True,
    )


def audit_single_claim(
    row: dict[str, Any],
    *,
    lens: JacobianLens,
    jlens_model: LensModel,
    tokenizer: Any,
    tau: float,
    layer: int = LINE_D_CLAIM_LAYER_DEFAULT,
    topk: int = 10,
    log_slot_threshold: float = -2.0,
) -> dict[str, Any]:
    iak = str(row["instance_audit_key"])
    tid = str(row["trajectory_id"])
    p_pm = float(row.get("p_pm") or 0.0)
    slot_norm = str(row.get("slot_norm") or "")
    h = load_instance_hidden(iak, tid, layer=layer)
    jlens_topk: list[dict[str, Any]] = []
    slot_align: dict[str, Any] = {
        "slot_type_aligned": False,
        "slot_hint_tokens": [],
        "slot_rank": None,
        "log_slot_mass": float("-inf"),
        "slot_mass": 0.0,
    }
    activation_ok = h is not None
    if h is not None:
        jlens_topk = readout_claim_instance(
            h, lens, jlens_model, tokenizer, layer=layer, k=topk
        )
        slot_align = score_slot_alignment(jlens_topk, slot_norm)

    bs = compute_bind_surprise(p_pm, float(slot_align.get("log_slot_mass") or float("-inf")))
    quadrant = classify_mechanism_quadrant(
        p_pm,
        tau,
        float(bs["log_slot_mass"]),
        log_slot_threshold,
    )
    audit_state = _QUADRANT_TO_LEGACY[quadrant]
    jlens_top5 = [str(t.get("token") or "") for t in jlens_topk[:5]]

    return {
        "instance_audit_key": iak,
        "trajectory_id": tid,
        "split": str(row.get("split") or ""),
        "p_pm": p_pm,
        "tau": float(tau),
        "y_pm": int(row.get("y") or 0),
        "gold_verdict": str(row.get("gold_verdict") or ""),
        "slot_norm": slot_norm,
        "claim_value": str(row.get("claim_value") or ""),
        "response_quote": str(row.get("response_quote") or ""),
        "slot_type_aligned": bool(slot_align["slot_type_aligned"]),
        "slot_hint_tokens": slot_align["slot_hint_tokens"],
        "slot_rank": slot_align["slot_rank"],
        "log_slot_mass": bs["log_slot_mass"],
        "slot_mass": slot_align.get("slot_mass"),
        "probe_logit": bs["probe_logit"],
        "bind_surprise": bs["bind_surprise"],
        "mechanism_quadrant": quadrant,
        "jlens_top5": jlens_top5,
        "jlens_topk": jlens_topk,
        "audit_state": audit_state,
        "activation_source": "line_a_instance",
        "activation_ok": activation_ok,
        "layer": layer,
        "position": "claim_onset",
        "method": "bind_surprise_v2",
    }


def _enrich_score_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adj = {str(r["instance_audit_key"]): r for r in load_jsonl(INCREMENTAL_ADJUDICATION_JSONL)}
    out: list[dict[str, Any]] = []
    for row in score_rows:
        iak = str(row["instance_audit_key"])
        adj_row = adj.get(iak) or {}
        out.append(
            {
                **row,
                "slot": str(adj_row.get("slot") or ""),
                "slot_norm": str(adj_row.get("slot_norm") or ""),
                "claim_value": str(adj_row.get("value") or ""),
                "committed_anchor_pid": str(adj_row.get("committed_anchor_pid") or ""),
                "anchor_evidence_quote": str(adj_row.get("anchor_evidence_quote") or ""),
                "donor_owners": adj_row.get("donor_owners") or [],
                "textual_selected_pid": str(adj_row.get("textual_selected_pid") or ""),
            }
        )
    return out


def run_prism_audit(
    *,
    lens: JacobianLens,
    jlens_model: LensModel,
    tokenizer: Any,
    tau: float | None = None,
    layer: int = LINE_D_CLAIM_LAYER_DEFAULT,
    topk: int = 50,
    split: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    score_rows = _enrich_score_rows(load_frozen_probe_scores())
    if tau is None:
        thresholds = load_frozen_dev_thresholds()
        tau = float(thresholds["line_a_probe_aggressive"])

    if split:
        score_rows = [r for r in score_rows if str(r.get("split") or "") == split]

    audit_rows: list[dict[str, Any]] = []
    for row in score_rows:
        rec = audit_single_claim(
            row,
            lens=lens,
            jlens_model=jlens_model,
            tokenizer=tokenizer,
            tau=tau,
            layer=layer,
            topk=topk,
        )
        audit_rows.append(rec)

    dev_rows = [r for r in audit_rows if r.get("split") == "dev"]
    log_slot_threshold = fit_log_slot_threshold(dev_rows)
    tau_fit = fit_tau_b(dev_rows)
    tau_b = float(tau_fit["tau_b"])
    probe_dev_recall = detection_metrics(dev_rows, score_key="p_pm", threshold=tau)["recall"]
    tau_recall_fit = fit_tau_b_at_recall(
        dev_rows,
        target_recall=float(probe_dev_recall or 0.92),
    )
    tau_b_recall = float(tau_recall_fit["tau_b"])

    for rec in audit_rows:
        rec["log_slot_threshold"] = log_slot_threshold
        rec["tau_b"] = tau_b
        rec["tau_b_recall_matched"] = tau_b_recall
        rec["bind_surprise_flagged"] = float(rec["bind_surprise"]) >= tau_b
        rec["bind_surprise_flagged_recall_matched"] = float(rec["bind_surprise"]) >= tau_b_recall
        rec["mechanism_quadrant"] = classify_mechanism_quadrant(
            float(rec["p_pm"]),
            tau,
            float(rec["log_slot_mass"]),
            log_slot_threshold,
        )
        rec["audit_state"] = _QUADRANT_TO_LEGACY[rec["mechanism_quadrant"]]
        rec["prism_audit_rationale"] = bind_surprise_rationale(
            rec["mechanism_quadrant"],
            slot_norm=rec["slot_norm"],
            bind_surprise=float(rec["bind_surprise"]),
            probe_logit=float(rec["probe_logit"]),
            log_slot_mass=float(rec["log_slot_mass"]),
            jlens_top5=rec["jlens_top5"],
        )

    summary = summarize_prism_audit(
        audit_rows,
        tau=tau,
        tau_b=tau_b,
        tau_fit=tau_fit,
        tau_recall_fit=tau_recall_fit,
        topk=topk,
    )
    return audit_rows, summary


def _discordance_2x2(rows: list[dict[str, Any]], tau: float) -> dict[str, Any]:
    cells: dict[str, int] = defaultdict(int)
    for r in rows:
        probe_high = float(r.get("p_pm") or 0) > float(tau)
        slot_al = bool(r.get("slot_type_aligned"))
        y = int(r.get("y_pm") or 0)
        key = f"probe_{int(probe_high)}__slot_{int(slot_al)}__y_{y}"
        cells[key] += 1
    return dict(cells)


def summarize_prism_audit(
    rows: list[dict[str, Any]],
    *,
    tau: float,
    tau_b: float | None = None,
    tau_fit: dict[str, Any] | None = None,
    tau_recall_fit: dict[str, Any] | None = None,
    topk: int = 50,
) -> dict[str, Any]:
    n = len(rows)
    state_counts = Counter(str(r.get("audit_state") or "") for r in rows)
    quad_counts = Counter(str(r.get("mechanism_quadrant") or "") for r in rows)
    y1 = [r for r in rows if int(r.get("y_pm") or 0) == 1]
    y0 = [r for r in rows if int(r.get("y_pm") or 0) == 0]
    flagged = [r for r in rows if float(r.get("p_pm") or 0) > float(tau)]

    def _rate(subset: list[dict[str, Any]], state: str) -> float | None:
        if not subset:
            return None
        return sum(1 for r in subset if r.get("audit_state") == state) / len(subset)

    by_verdict: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        gv = str(r.get("gold_verdict") or "unknown")
        by_verdict[gv][str(r.get("mechanism_quadrant") or r.get("audit_state") or "")] += 1

    activation_ok = sum(1 for r in rows if r.get("activation_ok"))
    slot_aligned_n = sum(1 for r in rows if r.get("slot_type_aligned"))

    probe_only_y0 = [r for r in y0 if r.get("audit_state") == "probe_only"]

    probe_metrics = detection_metrics(rows, score_key="p_pm", threshold=tau)
    bind_metrics: dict[str, Any] | None = None
    bind_recall_metrics: dict[str, Any] | None = None
    if tau_b is not None:
        bind_metrics = detection_metrics(rows, score_key="bind_surprise", threshold=tau_b)
    if tau_recall_fit is not None:
        bind_recall_metrics = detection_metrics(
            rows,
            score_key="bind_surprise",
            threshold=float(tau_recall_fit["tau_b"]),
        )

    test_rows = [r for r in rows if r.get("split") == "test"]
    pr_report = pr_dominance_report(test_rows) if test_rows else None

    return {
        "schema": "prism_bind_surprise_v2",
        "method": "BindSurprise: B(h,s)=probe_logit+log_slot_mass",
        "jlens_topk": topk,
        "n_claims": n,
        "tau_probe": tau,
        "tau_b": tau_b,
        "tau_b_fit": tau_fit,
        "tau_b_recall_matched_fit": tau_recall_fit,
        "detection_bind_surprise_recall_matched": bind_recall_metrics,
        "pr_dominance_test": pr_report,
        "layer": LINE_D_CLAIM_LAYER_DEFAULT,
        "position": "claim_onset",
        "activation_source": "line_a_instance",
        "n_activation_ok": activation_ok,
        "activation_coverage": (activation_ok / n) if n else None,
        "audit_state_counts": dict(state_counts),
        "mechanism_quadrant_counts": dict(quad_counts),
        "slot_align_rate": (slot_aligned_n / n) if n else None,
        "confirmed_risk_rate_among_y1": _rate(y1, "confirmed_risk"),
        "probe_only_rate_among_y1": _rate(y1, "probe_only"),
        "probe_only_rate_among_y0": _rate(y0, "probe_only"),
        "latent_slot_rate_among_y0": _rate(y0, "latent_slot"),
        "n_flagged_probe": len(flagged),
        "n_probe_only": state_counts.get("probe_only", 0),
        "n_probe_only_y0": len(probe_only_y0),
        "probe_fp_with_slot_mismatch": len(probe_only_y0),
        "detection_probe_only": probe_metrics,
        "detection_bind_surprise": bind_metrics,
        "auroc_probe_ppm": try_auroc(rows, "p_pm"),
        "auroc_bind_surprise": try_auroc(rows, "bind_surprise"),
        "by_gold_verdict_audit_state": {k: dict(v) for k, v in by_verdict.items()},
        "discordance_2x2": _discordance_2x2(rows, tau),
        "split_counts": dict(Counter(str(r.get("split") or "") for r in rows)),
    }


def build_audit_index(audit_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["instance_audit_key"]): r for r in audit_rows}


def merge_prism_audit_fields(
    log_row: dict[str, Any],
    audit_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    iak = str(log_row.get("instance_audit_key") or "")
    audit = audit_index.get(iak)
    if not audit:
        return log_row
    return {
        **log_row,
        "prism_audit_state": audit.get("audit_state"),
        "prism_mechanism_quadrant": audit.get("mechanism_quadrant"),
        "prism_slot_type_aligned": audit.get("slot_type_aligned"),
        "prism_jlens_top5": audit.get("jlens_top5"),
        "prism_bind_surprise": audit.get("bind_surprise"),
        "prism_probe_logit": audit.get("probe_logit"),
        "prism_log_slot_mass": audit.get("log_slot_mass"),
        "prism_tau_b": audit.get("tau_b"),
        "prism_audit_rationale": audit.get("prism_audit_rationale"),
    }


def select_case_studies(
    audit_rows: list[dict[str, Any]],
    *,
    n_each: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    grounded = [r for r in audit_rows if r.get("mechanism_quadrant") == "grounded_risk"]
    ungrounded = [r for r in audit_rows if r.get("mechanism_quadrant") == "ungrounded_alarm"]
    latent = [r for r in audit_rows if r.get("mechanism_quadrant") == "latent_slot"]
    return {
        "grounded_risk": grounded[:n_each],
        "ungrounded_alarm": ungrounded[:n_each],
        "latent_slot": latent[:n_each],
    }


def render_case_studies_md(cases: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["# PRISM BindSurprise Case Studies", ""]
    for label, rows in cases.items():
        lines.append(f"## {label}")
        lines.append("")
        for r in rows:
            lines.append(f"### {r.get('instance_audit_key')}")
            lines.append(f"- trajectory: `{r.get('trajectory_id')}` split={r.get('split')}")
            lines.append(f"- gold: {r.get('gold_verdict')} y_pm={r.get('y_pm')}")
            lines.append(
                f"- BindSurprise={r.get('bind_surprise'):.2f} "
                f"(ℓ_pm={r.get('probe_logit'):.2f}, log π_J={r.get('log_slot_mass'):.2f})"
            )
            lines.append(f"- quadrant={r.get('mechanism_quadrant')} tau_b={r.get('tau_b')}")
            lines.append(f"- slot_norm={r.get('slot_norm')} claim_value={r.get('claim_value')}")
            lines.append(f"- J-lens top-5: {r.get('jlens_top5')}")
            lines.append(f"- rationale: {r.get('prism_audit_rationale')}")
            lines.append("")
    return "\n".join(lines)


def render_diagnosis_report(
    summary: dict[str, Any],
    *,
    test_summary: dict[str, Any] | None = None,
) -> str:
    import json

    qc = summary.get("mechanism_quadrant_counts") or {}
    probe_det = summary.get("detection_probe_only") or {}
    bind_det = summary.get("detection_bind_surprise") or {}
    bind_rec = summary.get("detection_bind_surprise_recall_matched") or {}
    tau_fit = summary.get("tau_b_fit") or {}
    pr_dom = summary.get("pr_dominance_test") or {}

    lines = [
        "# PRISM BindSurprise Diagnosis Report",
        "",
        "## Unified score",
        "",
        "`B(h,s) = ℓ_pm(h) + log π_J(slot=s | h)`",
        "",
        "Slot-conditional binding likelihood: probe binding risk and Jacobian-slot",
        "workspace mass are **multiplicative factors** (additive in log-space), not an ensemble.",
        "",
        "## Setup",
        f"- Layer: L{summary.get('layer')} @ {summary.get('position')}",
        f"- Activation source: `{summary.get('activation_source')}`",
        f"- τ_probe: {summary.get('tau_probe')}",
        f"- τ_B (dev-frozen): {summary.get('tau_b')}",
        f"- Claims audited: {summary.get('n_claims')}",
        "",
        "## Mechanism quadrants (interpretation)",
        "",
        "| quadrant | count |",
        "|----------|-------|",
    ]
    for q in ("grounded_risk", "ungrounded_alarm", "latent_slot", "clean"):
        lines.append(f"| {q} | {qc.get(q, 0)} |")
    lines.extend(
        [
            "",
            "## Detection (full cohort)",
            "",
            "| method | threshold | precision | recall | F1 | AUROC |",
            "|--------|-----------|-----------|--------|-----|-------|",
            f"| Line A probe | {probe_det.get('threshold')} | "
            f"{probe_det.get('precision', 0):.3f} | {probe_det.get('recall', 0):.3f} | "
            f"{probe_det.get('f1', 0):.3f} | {summary.get('auroc_probe_ppm', '—')} |",
            f"| **BindSurprise** | {bind_det.get('threshold')} | "
            f"{bind_det.get('precision', 0):.3f} | {bind_det.get('recall', 0):.3f} | "
            f"{bind_det.get('f1', 0):.3f} | {summary.get('auroc_bind_surprise', '—')} |",
            "",
            f"τ_B dev fit: F1={tau_fit.get('dev_f1', 0):.3f} "
            f"P={tau_fit.get('dev_precision', 0):.3f} R={tau_fit.get('dev_recall', 0):.3f}",
            "",
            "## PR dominance (test, ranking-based)",
            "",
            f"- AUROC: probe {pr_dom.get('auroc_probe', '—')} → BindSurprise **{pr_dom.get('auroc_bind_surprise', '—')}**",
            f"- AP: probe {pr_dom.get('ap_probe', '—')} → BindSurprise **{pr_dom.get('ap_bind_surprise', '—')}**",
            f"- Pareto-dominates probe on recall grid: **{pr_dom.get('pareto_dominates_probe')}**",
            f"- Mean precision gain at matched recall: **{pr_dom.get('mean_precision_delta_at_recall', 0):+.3f}**",
            "",
            "### Matched-recall operating point (τ_B chosen on dev to match probe recall)",
            "",
            f"| method | P | R | F1 |",
            f"|--------|---|---|-----|",
            f"| probe @τ={probe_det.get('threshold')} | {probe_det.get('precision', 0):.3f} | "
            f"{probe_det.get('recall', 0):.3f} | {probe_det.get('f1', 0):.3f} |",
            f"| BindSurprise @τ_B^R | {bind_rec.get('precision', 0):.3f} | "
            f"{bind_rec.get('recall', 0):.3f} | {bind_rec.get('f1', 0):.3f} |",
            "",
            "## Discordance 2×2 (legacy probe × slot_aligned)",
            "",
            "```json",
            json.dumps(summary.get("discordance_2x2") or {}, indent=2),
            "```",
        ]
    )
    if test_summary:
        pd = test_summary.get("detection_probe_only") or {}
        bd = test_summary.get("detection_bind_surprise") or {}
        lines.extend(
            [
                "",
                "## Test split (τ_B frozen from dev)",
                "",
                f"| method | P | R | F1 |",
                f"|--------|---|---|-----|",
                f"| probe | {pd.get('precision', 0):.3f} | {pd.get('recall', 0):.3f} | {pd.get('f1', 0):.3f} |",
                f"| BindSurprise | {bd.get('precision', 0):.3f} | {bd.get('recall', 0):.3f} | {bd.get('f1', 0):.3f} |",
            ]
        )
    lines.extend(["", "See `audit_case_studies.md` for quadrant examples."])
    return "\n".join(lines)
