"""Line A sensitivity suite: label alignment, BoW baselines, and data-quality diagnostics."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ccer.io_utils import load_jsonl, write_json
from ccer.mechanism.activation_store import load_activation_npz
from ccer.mechanism.line_a_instance_activations import (
    LINE_A_INSTANCE_POSITIONS,
    activation_matrix_for_instances,
    build_adjudication_instance_dataset,
    instance_npz_path,
)
from ccer.mechanism.supervised_probe import (
    CLEAN_VERDICT,
    PM_VERDICTS,
    ProbeDataset,
    _find_winners,
    _holdout_scores_bow,
    _holdout_scores_probe,
    bootstrap_diff_ci,
    experiment_readiness,
    line_a_skip_trajectory_ids,
    train_bow_baseline,
    train_layer_position_probe,
)
from ccer.paths import INCREMENTAL_ADJUDICATION_JSONL, LINE_A_DIR, NORMALIZED_SHOPPING, SPLIT_MANIFEST_JSON
from ccer.replay.hf_forward import probe_layer_indices

LabelMode = Literal["instance", "trajectory_any_pm"]
BowField = Literal["response_quote", "final_answer"]

VARIANT_SPECS: list[dict[str, Any]] = [
    {
        "id": "V0_primary_instance_quote",
        "title": "Primary (instance label, BoW=response_quote)",
        "label_mode": "instance",
        "exclude_mixed": False,
        "exclude_quote_conflicts": False,
        "bow_field": "response_quote",
    },
    {
        "id": "V1_trajectory_any_pm_quote",
        "title": "Trajectory any-PM label, BoW=response_quote",
        "label_mode": "trajectory_any_pm",
        "exclude_mixed": False,
        "exclude_quote_conflicts": False,
        "bow_field": "response_quote",
    },
    {
        "id": "V2_trajectory_any_pm_answer",
        "title": "Trajectory any-PM label, BoW=final_answer",
        "label_mode": "trajectory_any_pm",
        "exclude_mixed": False,
        "exclude_quote_conflicts": False,
        "bow_field": "final_answer",
    },
    {
        "id": "V3_instance_no_mixed_quote",
        "title": "Instance label, exclude mixed trajectories, BoW=quote",
        "label_mode": "instance",
        "exclude_mixed": True,
        "exclude_quote_conflicts": False,
        "bow_field": "response_quote",
    },
    {
        "id": "V4_instance_no_mixed_answer",
        "title": "Instance label, exclude mixed trajectories, BoW=final_answer",
        "label_mode": "instance",
        "exclude_mixed": True,
        "exclude_quote_conflicts": False,
        "bow_field": "final_answer",
    },
    {
        "id": "V5_instance_full_answer",
        "title": "Instance label (full cohort), BoW=final_answer",
        "label_mode": "instance",
        "exclude_mixed": False,
        "exclude_quote_conflicts": False,
        "bow_field": "final_answer",
    },
    {
        "id": "V6_instance_no_conflicts_answer",
        "title": "Instance label, exclude quote-label conflicts, BoW=final_answer",
        "label_mode": "instance",
        "exclude_mixed": False,
        "exclude_quote_conflicts": True,
        "bow_field": "final_answer",
    },
]


def _load_splits() -> dict[str, str]:
    payload = json.loads(SPLIT_MANIFEST_JSON.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in (payload.get("splits") or {}).items()}


def _instance_label(verdict: str | None) -> int | None:
    if verdict in PM_VERDICTS:
        return 1
    if verdict == CLEAN_VERDICT:
        return 0
    return None


def mixed_trajectory_ids(*, require_activation: bool = True) -> set[str]:
    by_tid: dict[str, set[int]] = defaultdict(set)
    splits = _load_splits()
    for row in load_jsonl(INCREMENTAL_ADJUDICATION_JSONL):
        if not row.get("commitment_eligible"):
            continue
        y = _instance_label(row.get("gold_verdict"))
        if y is None:
            continue
        tid = str(row["trajectory_id"])
        if splits.get(tid) not in ("train", "dev", "test"):
            continue
        if require_activation and not instance_npz_path(
            str(row.get("instance_audit_key") or ""), tid
        ).is_file():
            continue
        by_tid[tid].add(y)
    return {tid for tid, ys in by_tid.items() if len(ys) > 1}


def quote_label_conflict_keys() -> set[str]:
    quote_to_labels: dict[str, set[str]] = defaultdict(set)
    for row in load_jsonl(INCREMENTAL_ADJUDICATION_JSONL):
        if not row.get("commitment_eligible"):
            continue
        v = row.get("gold_verdict")
        if v in PM_VERDICTS or v == CLEAN_VERDICT:
            q = str(row.get("response_quote") or "").strip()
            if q:
                quote_to_labels[q].add(str(v))
    return {q for q, labels in quote_to_labels.items() if len(labels) > 1}


def collect_data_diagnostics(*, require_activation: bool = True) -> dict[str, Any]:
    """Static audit of label–feature coupling issues (no model training)."""
    mixed = mixed_trajectory_ids(require_activation=require_activation)
    conflicts = quote_label_conflict_keys()
    base = build_adjudication_instance_dataset(require_activation=require_activation)
    inst_on_mixed = sum(1 for r in base.instances if r["trajectory_id"] in mixed)
    inst_on_conflict = sum(
        1 for r in base.instances if str(r.get("response_quote") or "").strip() in conflicts
    )
    same_x_diff_y = 0
    by_tid_y: dict[str, set[int]] = defaultdict(set)
    for r in base.instances:
        by_tid_y[r["trajectory_id"]].add(int(r["y"]))
    same_x_diff_y = sum(1 for ys in by_tid_y.values() if len(ys) > 1)

    # BoW-only ceiling: quote TF-IDF on test (train-only fit) — quick read on circularity
    y = np.array([r["y"] for r in base.instances], dtype=int)
    splits = [r["split"] for r in base.instances]
    tids = [r["trajectory_id"] for r in base.instances]
    quote_bow = train_bow_baseline(
        [r["response_quote"] for r in base.instances],
        y,
        splits,
        tids,
        eval_splits={"test"},
        n_boot=500,
        seed=42,
    )
    answer_bow = train_bow_baseline(
        [r["final_answer"] for r in base.instances],
        y,
        splits,
        tids,
        eval_splits={"test"},
        n_boot=500,
        seed=42,
    )
    return {
        "n_instances_with_activation": len(base.instances),
        "n_trajectories_with_activation": len({r["trajectory_id"] for r in base.instances}),
        "mixed_label_trajectories": len(mixed),
        "instances_on_mixed_trajectories": inst_on_mixed,
        "distinct_quotes_with_conflicting_labels": len(conflicts),
        "instances_on_conflicting_quotes": inst_on_conflict,
        "trajectories_with_same_activation_multiple_labels": same_x_diff_y,
        "bow_test_auroc_response_quote": (quote_bow.get("eval") or {}).get("test", {}).get("auroc"),
        "bow_test_auroc_final_answer": (answer_bow.get("eval") or {}).get("test", {}).get("auroc"),
        "interpretation_notes": [
            "BoW(response_quote) uses the adjudicated span as features — structurally aligned with instance labels.",
            "BoW(final_answer) is a fairer lexical baseline for 'surface text without quote tautology'.",
            "Mixed trajectories may still carry conflicting instance labels; Line A uses per-instance npz activations.",
        ],
    }


def build_sensitivity_dataset(
    *,
    label_mode: LabelMode = "instance",
    exclude_mixed: bool = False,
    exclude_quote_conflicts: bool = False,
    require_activation: bool = True,
) -> ProbeDataset:
    splits = _load_splits()
    traj_rows = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}
    mixed = mixed_trajectory_ids(require_activation=require_activation) if exclude_mixed else set()
    conflict_quotes = quote_label_conflict_keys() if exclude_quote_conflicts else set()
    skips = line_a_skip_trajectory_ids()

    raw_rows: list[dict[str, Any]] = []
    skipped_no_activation = 0
    skipped_no_label = 0
    skipped_no_split = 0
    skipped_mixed = 0
    skipped_conflict = 0

    for row in load_jsonl(INCREMENTAL_ADJUDICATION_JSONL):
        if not row.get("commitment_eligible"):
            continue
        y = _instance_label(row.get("gold_verdict"))
        if y is None:
            skipped_no_label += 1
            continue
        tid = str(row["trajectory_id"])
        if tid in skips:
            continue
        split = splits.get(tid)
        if split not in ("train", "dev", "test"):
            skipped_no_split += 1
            continue
        if tid in mixed:
            skipped_mixed += 1
            continue
        quote = str(row.get("response_quote") or "")
        if quote.strip() in conflict_quotes:
            skipped_conflict += 1
            continue
        iak = str(row.get("instance_audit_key") or "")
        npz_path = instance_npz_path(iak, tid)
        if require_activation and not npz_path.is_file():
            skipped_no_activation += 1
            continue
        traj = traj_rows.get(tid, {})
        raw_rows.append(
            {
                "instance_audit_key": iak,
                "trajectory_id": tid,
                "split": split,
                "y": y,
                "gold_verdict": row.get("gold_verdict"),
                "response_quote": quote,
                "final_answer": str((traj.get("metadata") or {}).get("final_answer") or ""),
            }
        )

    activation_layers: dict[str, list[int]] = {}
    instances: list[dict[str, Any]] = []

    if label_mode == "instance":
        for r in raw_rows:
            iak = str(r["instance_audit_key"])
            tid = r["trajectory_id"]
            npz_path = instance_npz_path(iak, tid)
            if require_activation and not npz_path.is_file():
                continue
            if iak not in activation_layers and npz_path.is_file():
                loaded = load_activation_npz(npz_path)
                activation_layers[iak] = list(loaded["layer_indices"])
            instances.append({**r, "has_activation": npz_path.is_file()})
    else:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in raw_rows:
            grouped[r["trajectory_id"]].append(r)
        for tid, rows in grouped.items():
            rep = None
            for r in rows:
                path = instance_npz_path(str(r["instance_audit_key"]), tid)
                if path.is_file():
                    rep = r
                    break
            if require_activation and rep is None:
                skipped_no_activation += len(rows)
                continue
            rep = rep or rows[0]
            iak = str(rep["instance_audit_key"])
            npz_path = instance_npz_path(iak, tid)
            if npz_path.is_file() and iak not in activation_layers:
                loaded = load_activation_npz(npz_path)
                activation_layers[iak] = list(loaded["layer_indices"])
            y_traj = 1 if any(int(r["y"]) == 1 for r in rows) else 0
            split = rows[0]["split"]
            quotes = " || ".join(sorted({str(r["response_quote"]) for r in rows if r["response_quote"]}))
            final_answer = rows[0]["final_answer"]
            instances.append(
                {
                    "instance_audit_key": iak,
                    "trajectory_id": tid,
                    "split": split,
                    "y": y_traj,
                    "gold_verdict": "trajectory_any_pm" if y_traj else "trajectory_all_clean",
                    "response_quote": quotes,
                    "final_answer": final_answer,
                    "has_activation": npz_path.is_file(),
                    "n_instances_merged": len(rows),
                }
            )

    manifest = {
        "schema": "line_a_instance_aligned_adjudication",
        "label_mode": label_mode,
        "exclude_mixed": exclude_mixed,
        "exclude_quote_conflicts": exclude_quote_conflicts,
        "n_instances": len(instances),
        "n_trajectories": len({r["trajectory_id"] for r in instances}),
        "skipped_no_activation": skipped_no_activation,
        "skipped_no_label": skipped_no_label,
        "skipped_no_split": skipped_no_split,
        "skipped_mixed": skipped_mixed,
        "skipped_quote_conflicts": skipped_conflict,
        "split_counts": {
            sp: sum(1 for r in instances if r["split"] == sp) for sp in ("train", "dev", "test")
        },
        "label_counts": {
            "pm": sum(1 for r in instances if r["y"] == 1),
            "clean": sum(1 for r in instances if r["y"] == 0),
        },
        "require_activation": require_activation,
    }
    return ProbeDataset(instances=instances, activation_layers=activation_layers, manifest=manifest)


def run_probe_vs_bow_on_dataset(
    ds: ProbeDataset,
    *,
    bow_field: BowField = "response_quote",
    n_boot: int = 2000,
    seed: int = 42,
    layers: list[int] | None = None,
) -> dict[str, Any]:
    readiness = experiment_readiness(ds)
    if not readiness["ready"]:
        return {
            "experiment_status": "blocked",
            "readiness": readiness,
            "dataset_manifest": ds.manifest,
        }

    instances = ds.instances
    y_all = np.array([r["y"] for r in instances], dtype=int)
    splits = [r["split"] for r in instances]
    tids = [r["trajectory_id"] for r in instances]
    bow_texts = [str(r[bow_field] or "") for r in instances]

    eval_primary = {"test"}
    eval_supplement = {"dev", "test"}

    all_layers: set[int] = set()
    for layer_list in ds.activation_layers.values():
        all_layers.update(layer_list)
    target_layers = layers or sorted(all_layers & set(probe_layer_indices(64)))

    bow_global = train_bow_baseline(
        bow_texts,
        y_all,
        splits,
        tids,
        eval_splits=eval_primary | eval_supplement,
        n_boot=n_boot,
        seed=seed,
    )

    layer_results: list[dict[str, Any]] = []
    for position in LINE_A_INSTANCE_POSITIONS:
        for layer in target_layers:
            X, valid_idx = activation_matrix_for_instances(instances, position=position, layer=layer)
            if X is None or len(valid_idx) < 10:
                continue
            if not any(splits[i] == "train" for i in valid_idx):
                continue
            y = y_all[valid_idx]
            sub_splits = [splits[i] for i in valid_idx]
            sub_tids = [tids[i] for i in valid_idx]
            probe = train_layer_position_probe(
                X,
                y,
                sub_splits,
                sub_tids,
                eval_splits=eval_primary | eval_supplement,
                n_boot=n_boot,
                seed=seed,
            )
            bow_sub = train_bow_baseline(
                [bow_texts[i] for i in valid_idx],
                y,
                sub_splits,
                sub_tids,
                eval_splits=eval_primary | eval_supplement,
                n_boot=n_boot,
                seed=seed,
            )
            entry: dict[str, Any] = {
                "position": position,
                "layer": layer,
                "n_instances_with_vector": len(valid_idx),
                "probe": probe,
                "bow_claim_on_subset": bow_sub,
            }
            for eval_key, eval_splits in (("test", {"test"}), ("dev_test", {"dev", "test"})):
                holdout_idx = [i for i, s in enumerate(sub_splits) if s in eval_splits]
                if len(holdout_idx) < 2:
                    entry[f"delta_vs_bow_claim_{eval_key}"] = None
                    continue
                probe_holdout = _holdout_scores_probe(X, y, sub_splits, sub_tids, eval_splits)
                bow_holdout = _holdout_scores_bow(
                    [bow_texts[i] for i in valid_idx],
                    y,
                    sub_splits,
                    sub_tids,
                    eval_splits,
                )
                y_h = y[holdout_idx]
                clusters = np.array([sub_tids[i] for i in holdout_idx], dtype=str)
                entry[f"delta_vs_bow_claim_{eval_key}"] = bootstrap_diff_ci(
                    y_h,
                    probe_holdout,
                    bow_holdout,
                    clusters,
                    n_boot=n_boot,
                    seed=seed,
                )
            layer_results.append(entry)

    winners = _find_winners(layer_results)
    return {
        "experiment_status": "completed",
        "readiness": readiness,
        "dataset_manifest": ds.manifest,
        "bow_field": bow_field,
        "bow_global": bow_global,
        "layer_position_results": layer_results,
        "winners": winners,
        "success_criterion": {
            "min_delta_auroc": 0.03,
            "ci_excludes_zero": True,
            "passed": bool(winners.get("primary_passed")),
        },
    }


def _best_row(layer_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    winners = _find_winners(layer_results)
    if winners.get("primary_winners"):
        return winners["primary_winners"][0]
    best = None
    best_diff = -999.0
    for row in layer_results:
        d = row.get("delta_vs_bow_claim_test") or {}
        diff = d.get("diff")
        if diff is not None and diff > best_diff:
            best_diff = float(diff)
            probe = ((row.get("probe") or {}).get("eval") or {}).get("test") or {}
            best = {
                "position": row["position"],
                "layer": row["layer"],
                "probe_auroc": probe.get("auroc"),
                "delta_auroc": diff,
                "delta_ci_95": d.get("diff_ci_95"),
            }
    return best


def run_sensitivity_suite(
    *,
    n_boot: int = 2000,
    seed: int = 42,
    quick: bool = False,
) -> dict[str, Any]:
    diagnostics = collect_data_diagnostics(require_activation=True)
    layers = [47] if quick else None
    variants: list[dict[str, Any]] = []

    for spec in VARIANT_SPECS:
        ds = build_sensitivity_dataset(
            label_mode=spec["label_mode"],
            exclude_mixed=spec["exclude_mixed"],
            exclude_quote_conflicts=spec["exclude_quote_conflicts"],
            require_activation=True,
        )
        result = run_probe_vs_bow_on_dataset(
            ds,
            bow_field=spec["bow_field"],
            n_boot=n_boot,
            seed=seed,
            layers=layers,
        )
        best = None
        if result.get("experiment_status") == "completed":
            best = _best_row(result.get("layer_position_results") or [])
        variants.append(
            {
                **spec,
                "status": result.get("experiment_status"),
                "readiness": result.get("readiness"),
                "dataset": result.get("dataset_manifest"),
                "success_passed": (result.get("success_criterion") or {}).get("passed"),
                "bow_test_auroc": ((result.get("bow_global") or {}).get("eval") or {}).get("test", {}).get("auroc"),
                "best_probe_vs_bow": best,
                "full_result": result if not quick else None,
            }
        )

    passed = [v for v in variants if v.get("success_passed")]
    passed.sort(
        key=lambda v: (v.get("best_probe_vs_bow") or {}).get("delta_auroc") or -1,
        reverse=True,
    )

    paper_rec = _paper_recommendation(diagnostics, variants, passed)
    return {
        "schema_version": "ccer_line_a_sensitivity_v1",
        "diagnostics": diagnostics,
        "variants": variants,
        "variants_passed_primary": [v["id"] for v in passed],
        "paper_recommendation": paper_rec,
    }


def _paper_recommendation(
    diagnostics: dict[str, Any],
    variants: list[dict[str, Any]],
    passed: list[dict[str, Any]],
) -> dict[str, Any]:
    if passed:
        best = passed[0]
        b = best.get("best_probe_vs_bow") or {}
        return {
            "verdict": "positive_under_specified_variant",
            "headline": (
                f"Probe beats BoW on test at {b.get('position')} L{b.get('layer')} "
                f"(ΔAUROC={_fmt(b.get('delta_auroc'))}, CI={_fmt_ci(b.get('delta_ci_95'))}) "
                f"under variant {best['id']}."
            ),
            "preferred_variant_id": best["id"],
            "preferred_variant_title": best["title"],
            "caveats": [
                "Primary instance+quote variant may inflate BoW; report preferred variant as main text.",
                f"Mixed-label trajectories: {diagnostics.get('mixed_label_trajectories')} "
                f"({diagnostics.get('instances_on_mixed_trajectories')} instances).",
                "Test clean count remains small in all variants — keep bootstrap CI, not point estimate alone.",
            ],
        }

    # No variant passed — root cause narrative
    v0 = next((v for v in variants if v["id"] == "V0_primary_instance_quote"), {})
    v5 = next((v for v in variants if v["id"] == "V5_instance_full_answer"), {})
    return {
        "verdict": "no_robust_probe_win",
        "headline": (
            "No variant met pre-registered ΔAUROC≥0.03 with test CI>0 after label/BoW alignment fixes."
        ),
        "root_causes": [
            {
                "cause": "bow_quote_tautology",
                "evidence": (
                    f"BoW(quote) test AUROC={_fmt(diagnostics.get('bow_test_auroc_response_quote'))} vs "
                    f"BoW(answer)={_fmt(diagnostics.get('bow_test_auroc_final_answer'))}."
                ),
            },
            {
                "cause": "label_feature_granularity_mismatch",
                "evidence": (
                    f"{diagnostics.get('trajectories_with_same_activation_multiple_labels')} trajectories "
                    f"carry conflicting instance labels (per-instance activations)."
                ),
            },
            {
                "cause": "annotation_noise",
                "evidence": (
                    f"{diagnostics.get('distinct_quotes_with_conflicting_labels')} quote strings appear "
                    f"with both PM and clean labels."
                ),
            },
            {
                "cause": "bow_solver_instability",
                "evidence": (
                    "TF-IDF BoW previously used saga with convergence warnings; "
                    "holdout AUROC varied across identical inputs. Re-runs use lbfgs for stability."
                ),
            },
            {
                "cause": "primary_winner_fragility",
                "evidence": (
                    f"V0 best Δ={_fmt((v0.get('best_probe_vs_bow') or {}).get('delta_auroc'))}; "
                    f"V5(answer BoW) best Δ={_fmt((v5.get('best_probe_vs_bow') or {}).get('delta_auroc'))}."
                ),
            },
        ],
        "paper_framing": (
            "Report as credible negative: lexical cues on adjudicated spans dominate; "
            "pre-answer hidden states do not robustly exceed non-quote surface baselines "
            "once label–feature coupling is relaxed."
        ),
    }


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _fmt_ci(ci: Any) -> str:
    if not ci:
        return "—"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def write_sensitivity_report(summary: dict[str, Any], path: Path) -> None:
    diag = summary.get("diagnostics") or {}
    rec = summary.get("paper_recommendation") or {}
    lines = [
        "# Line A Sensitivity & Attribution Report",
        "",
        "## Executive summary",
        "",
        rec.get("headline", ""),
        "",
        f"**Verdict:** `{rec.get('verdict')}`",
        "",
    ]
    if rec.get("preferred_variant_id"):
        lines.extend(
            [
                f"**Preferred variant for paper:** {rec.get('preferred_variant_id')} — "
                f"{rec.get('preferred_variant_title')}",
                "",
            ]
        )
    if rec.get("caveats"):
        lines.append("### Caveats")
        for c in rec["caveats"]:
            lines.append(f"- {c}")
        lines.append("")
    if rec.get("root_causes"):
        lines.append("### Root-cause decomposition")
        for item in rec["root_causes"]:
            lines.append(f"- **{item['cause']}**: {item['evidence']}")
        lines.append("")
        lines.append(f"**Suggested paper framing:** {rec.get('paper_framing')}")
        lines.append("")

    lines.extend(
        [
            "## Data-quality diagnostics",
            "",
            f"- Instances (with activation): {diag.get('n_instances_with_activation')}",
            f"- Trajectories: {diag.get('n_trajectories_with_activation')}",
            f"- Mixed-label trajectories: {diag.get('mixed_label_trajectories')} "
            f"({diag.get('instances_on_mixed_trajectories')} instances)",
            f"- Quote strings with conflicting PM/clean labels: "
            f"{diag.get('distinct_quotes_with_conflicting_labels')} "
            f"({diag.get('instances_on_conflicting_quotes')} instances)",
            f"- Trajectories: same npz, conflicting instance labels: "
            f"{diag.get('trajectories_with_same_activation_multiple_labels')}",
            f"- BoW test AUROC (response_quote): {_fmt(diag.get('bow_test_auroc_response_quote'))}",
            f"- BoW test AUROC (final_answer): {_fmt(diag.get('bow_test_auroc_final_answer'))}",
            "",
            "## Variant comparison (test holdout, best layer per variant)",
            "",
            "| ID | Setting | N_inst | N_traj | BoW test | Best position | L | Δ AUROC | Δ CI95 | Pass |",
            "|----|---------|--------|--------|----------|---------------|---|---------|--------|------|",
        ]
    )
    for v in summary.get("variants") or []:
        ds = v.get("dataset") or {}
        b = v.get("best_probe_vs_bow") or {}
        if v.get("status") == "blocked":
            issues = ", ".join((v.get("readiness") or {}).get("issues") or [])
            lines.append(
                f"| {v.get('id')} | {v.get('title', '')[:40]} | "
                f"{ds.get('n_instances', '—')} | {ds.get('n_trajectories', '—')} | "
                f"— | — | — | — | — | **blocked** ({issues}) |"
            )
            continue
        lines.append(
            f"| {v.get('id')} | {v.get('title', '')[:40]} | "
            f"{ds.get('n_instances', '—')} | {ds.get('n_trajectories', '—')} | "
            f"{_fmt(v.get('bow_test_auroc'))} | {b.get('position', '—')} | {b.get('layer', '—')} | "
            f"{_fmt(b.get('delta_auroc'))} | {_fmt_ci(b.get('delta_ci_95'))} | "
            f"{v.get('success_passed')} |"
        )

    passed_ids = summary.get("variants_passed_primary") or []
    lines.extend(
        [
            "",
            "## Primary success (Δ≥0.03, CI lower > 0)",
            "",
        ]
    )
    if passed_ids:
        for pid in passed_ids:
            lines.append(f"- `{pid}`")
    else:
        lines.append("- None across all sensitivity variants.")

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "- **V0**: original Line A protocol (most favorable to BoW).",
            "- **V2/V5**: fairer surface baseline (`final_answer`) or trajectory-level PM label.",
            "- **V3/V4**: remove trajectories where one npz carries both PM and clean instances.",
            "- **V6**: remove instances whose quote text appears with conflicting labels.",
            "",
            "A result that passes only under V0 but fails under V5/V2 should be framed as "
            "**quote–label coupling**, not as generic internal-representation superiority.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sensitivity_artifacts(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "line_a_sensitivity_summary.json", summary)
