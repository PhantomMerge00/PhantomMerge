"""ROUND10 core: strong evidence-mask + commitment patch (expert decisive experiment)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ccer.mechanism.pair_select import cem_valid_pair_ids
from ccer.mechanism.rank_sweep import measure_diff_rate
from ccer.mechanism.round6_verify import verify_dual_metrics_from_texts
from ccer.mechanism.round8_verify import PATCH_IMPLEMENTATION_NOTE, _row_behavioral_summary
from ccer.mechanism.stats import mcnemar_exact_p, wilson_ci

PAIR_AUDIT_JSONL = Path("${PHANTOM_MERGE_ROOT}/results/reports/manual_verification/round10_cem_cross_pair_audit.jsonl")


def _base_cross_pairs(
    track: str = "CEM",
    *,
    pool: str = "dev",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use native CEM pm×clean pairs — no multi_clean_expansion."""
    if pool == "mechanism_research":
        from ccer.mechanism.mechanism_pool import build_cem_mechanism_cross_pairs

        sel = build_cem_mechanism_cross_pairs(pool="mechanism_research")
        pairs = list(sel["pm_clean_cross_pairs"])
        meta = {
            "n_pairs": len(pairs),
            "pairing_source": "build_cem_mechanism_cross_pairs",
            "expansion_method": None,
            "pool": pool,
            "expansion_note": sel["pairing_meta"]["expansion_note"],
        }
        return pairs, meta
    sel = cem_valid_pair_ids(include_matched_clean=True)
    pairs = list(sel["pm_clean_cross_pairs"])
    meta = {
        "n_pairs": len(pairs),
        "pairing_source": "cem_valid_pair_ids",
        "expansion_method": None,
        "expansion_note": "Dev-only native pairs (rival_value_swap scorable filter).",
    }
    return pairs, meta


def _aggregate_from_pair_details(pair_details: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair-level product_id_diff is authoritative (fixes save_texts=False arms)."""
    n = len(pair_details)
    pid_k = sum(1 for p in pair_details if p.get("product_id_diff"))
    sel_k = sum(1 for p in pair_details if p.get("selected_product_block_diff"))
    full_k = sum(1 for p in pair_details if p.get("full_response_excerpt_diff"))
    ans_k = sum(1 for p in pair_details if p.get("answer_diff") or p.get("extract_answer_diff"))
    out_k = sum(1 for p in pair_details if p.get("output_diff"))
    return {
        "n_pairs": n,
        "n_product_id_diff": pid_k,
        "product_id_diff_rate": pid_k / n if n else 0.0,
        "product_id_ci95": wilson_ci(pid_k, n),
        "n_selected_product_block_diff": sel_k,
        "selected_product_block_diff_rate": sel_k / n if n else 0.0,
        "selected_product_block_ci95": wilson_ci(sel_k, n),
        "n_full_response_excerpt_diff": full_k,
        "full_response_excerpt_diff_rate": full_k / n if n else 0.0,
        "full_response_excerpt_ci95": wilson_ci(full_k, n),
        "n_extract_answer_diff": ans_k,
        "answer_diff_rate": ans_k / n if n else 0.0,
        "n_output_diff": out_k,
        "output_diff_rate": out_k / n if n else 0.0,
    }


def _sync_dual_metrics(row: dict[str, Any]) -> dict[str, Any]:
    dual = verify_dual_metrics_from_texts(row.get("pair_details") or [])
    verified_pairs = {d["pm_trajectory_id"]: d for d in dual.get("pair_details") or []}
    synced_details: list[dict[str, Any]] = []
    for pr in row.get("pair_details") or []:
        key = pr.get("pm_trajectory_id")
        merged = dict(pr)
        if key in verified_pairs:
            for field in (
                "answer_diff",
                "extract_answer_diff",
                "selected_product_block_diff",
                "full_response_excerpt_diff",
                "product_id_diff",
                "compared_section_only_diff",
            ):
                if field in verified_pairs[key]:
                    merged[field] = verified_pairs[key][field]
        synced_details.append(merged)
    row["pair_details"] = synced_details
    dual_n = int(dual.get("n_pairs") or 0)
    orig_n = len(synced_details)
    if dual_n == orig_n and dual_n > 0:
        for metric_key, row_fields in (
            ("product_id_diff", ("n_product_id_diff", "product_id_diff_rate", "product_id_ci95")),
            ("selected_product_block_diff", ("n_selected_product_block_diff", "selected_product_block_diff_rate", "selected_product_block_ci95")),
            ("full_response_excerpt_diff", ("n_full_response_excerpt_diff", "full_response_excerpt_diff_rate", "full_response_excerpt_ci95")),
        ):
            block = dual.get(metric_key) or {}
            if isinstance(block, dict) and block.get("k") is not None:
                row[row_fields[0]] = block["k"]
                row[row_fields[1]] = block["rate"]
                row[row_fields[2]] = block["ci95"]
    else:
        row.update(_aggregate_from_pair_details(synced_details))
    return {"measure_row": row, "dual_metrics": dual}


def _load_anchor_strata() -> dict[str, str]:
    if not PAIR_AUDIT_JSONL.is_file():
        return {}
    out: dict[str, str] = {}
    for line in PAIR_AUDIT_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["pm_trajectory_id"])] = str(row.get("anchor_title_quality") or "unknown")
    return out


def _stratified_product_id_summary(pair_details: list[dict[str, Any]], strata: dict[str, str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"shell_pid_only": [], "real_text": [], "unknown": []}
    for pr in pair_details:
        tid = str(pr.get("pm_trajectory_id") or "")
        key = strata.get(tid, "unknown")
        groups.setdefault(key, []).append(pr)
    out: dict[str, Any] = {}
    for key, rows in groups.items():
        if not rows:
            continue
        k = sum(1 for r in rows if r.get("product_id_diff"))
        n = len(rows)
        out[key] = {
            "n_pairs": n,
            "product_id_k": k,
            "product_id_rate": k / n if n else 0.0,
            "product_id_ci95": wilson_ci(k, n),
        }
    return out


def _paired_mcnemar_prep(masked_details: list[dict[str, Any]], unmasked_details: list[dict[str, Any]]) -> dict[str, Any]:
    by_pm = {str(d["pm_trajectory_id"]): d for d in unmasked_details}
    both_flip = mask_only = unmask_only = neither = 0
    pairs: list[dict[str, Any]] = []
    for md in masked_details:
        pm = str(md["pm_trajectory_id"])
        ud = by_pm.get(pm)
        if not ud:
            continue
        m = bool(md.get("product_id_diff"))
        u = bool(ud.get("product_id_diff"))
        if m and u:
            both_flip += 1
        elif m and not u:
            mask_only += 1
        elif not m and u:
            unmask_only += 1
        else:
            neither += 1
        pairs.append({"pm_trajectory_id": pm, "masked_diff": m, "unmasked_diff": u})
    p_exact = mcnemar_exact_p(mask_only, unmask_only)
    return {
        "n_paired": len(pairs),
        "both_flip": both_flip,
        "mask_only_flip": mask_only,
        "unmask_only_flip": unmask_only,
        "neither_flip": neither,
        "discordant_pairs": mask_only + unmask_only,
        "mcnemar_exact_p_two_sided": p_exact,
        "pair_table": pairs,
        "note": "Use mask_only vs unmask_only for McNemar (expert paired design).",
    }


def _shell_only_pairs(pairs: list[dict[str, Any]], strata: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cp in pairs:
        pm = str(cp.get("pm_trajectory_id") or cp.get("cap_trajectory_id") or "")
        if strata.get(pm) == "shell_pid_only":
            out.append(cp)
    return out


def run_evidence_mask_commitment(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer: int = 32,
    position: str = "commitment",
    rank: int = 16,
    probe_max_tokens: int = 384,
    alpha: float = 1.0,
    shard_index: int = 0,
    num_shards: int = 1,
    pool: str = "dev",
) -> dict[str, Any]:
    """Strong evidence mask + commitment interchange vs same-run unmasked baseline."""
    pairs, pair_meta = _base_cross_pairs(track, pool=pool)
    ni_cache: dict[str, str] = {}

    masked_row = measure_diff_rate(
        model=model,
        tokenizer=tokenizer,
        track=track,
        layer=layer,
        position=position,
        rank=rank,
        mode="interchange",
        probe_max_tokens=probe_max_tokens,
        evidence_mask=True,
        alpha=alpha,
        cross_pairs_override=pairs,
        ni_cache=ni_cache,
        shard_index=shard_index,
        num_shards=num_shards,
        save_texts=True,
    )
    unmasked_row = measure_diff_rate(
        model=model,
        tokenizer=tokenizer,
        track=track,
        layer=layer,
        position=position,
        rank=rank,
        mode="interchange",
        probe_max_tokens=probe_max_tokens,
        evidence_mask=False,
        alpha=alpha,
        cross_pairs_override=pairs,
        ni_cache=ni_cache,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    masked_sync = _sync_dual_metrics(masked_row)
    unmasked_sync = _sync_dual_metrics(unmasked_row)
    strata = _load_anchor_strata()

    masked_details = masked_sync["measure_row"].get("pair_details") or []
    unmasked_details = unmasked_sync["measure_row"].get("pair_details") or []

    return {
        "schema_version": "ccer_round10_strong_evidence_mask_v1",
        "experiment_id": "strong_evidence_mask_plus_commitment_patch",
        "primary_metric": "product_id_diff",
        "mask_mode": "strong",
        "layer": layer,
        "position": position,
        "rank": rank,
        "intervention_alpha": alpha,
        "alpha_confirmed_full_strength": float(alpha) == 1.0,
        "pair_meta": pair_meta,
        "patch_implementation": PATCH_IMPLEMENTATION_NOTE,
        "anchor_stratification": {
            "shell_pid_only": _stratified_product_id_summary(masked_details, strata).get("shell_pid_only"),
            "real_text": _stratified_product_id_summary(masked_details, strata).get("real_text"),
            "note": "Report product_id_diff separately; do not merge shell vs real anchors.",
        },
        "paired_mcnemar_prep": _paired_mcnemar_prep(masked_details, unmasked_details),
        "with_strong_evidence_mask": {
            **_row_behavioral_summary(masked_sync["measure_row"]),
            "dual_metrics": masked_sync["dual_metrics"],
            "measure_row": masked_sync["measure_row"],
        },
        "without_evidence_mask_same_run_baseline": {
            **_row_behavioral_summary(unmasked_sync["measure_row"]),
            "dual_metrics": unmasked_sync["dual_metrics"],
            "measure_row": unmasked_sync["measure_row"],
        },
        "shard_index": shard_index,
        "num_shards": num_shards,
    }


def run_shell_wrong_owner_donor_control(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer: int = 32,
    position: str = "commitment",
    rank: int = 16,
    probe_max_tokens: int = 384,
    alpha: float = 1.0,
    pool: str = "dev",
) -> dict[str, Any]:
    """ROUND11: shell anchors only — strong mask + wrong_owner_donor (recipient vec as wrong donor)."""
    pairs, pair_meta = _base_cross_pairs(track, pool=pool)
    strata = _load_anchor_strata()
    shell_pairs = _shell_only_pairs(pairs, strata)
    ni_cache: dict[str, str] = {}

    correct_row = measure_diff_rate(
        model=model,
        tokenizer=tokenizer,
        track=track,
        layer=layer,
        position=position,
        rank=rank,
        mode="interchange",
        probe_max_tokens=probe_max_tokens,
        evidence_mask=True,
        alpha=alpha,
        cross_pairs_override=shell_pairs,
        ni_cache=ni_cache,
        control_id="target_interchange",
    )
    wrong_row = measure_diff_rate(
        model=model,
        tokenizer=tokenizer,
        track=track,
        layer=layer,
        position=position,
        rank=rank,
        mode="interchange",
        probe_max_tokens=probe_max_tokens,
        evidence_mask=True,
        alpha=alpha,
        cross_pairs_override=shell_pairs,
        ni_cache=ni_cache,
        control_id="wrong_owner_donor",
    )
    correct_sync = _sync_dual_metrics(correct_row)
    wrong_sync = _sync_dual_metrics(wrong_row)
    correct_details = correct_sync["measure_row"].get("pair_details") or []
    wrong_details = wrong_sync["measure_row"].get("pair_details") or []

    by_pm_wrong = {str(d["pm_trajectory_id"]): d for d in wrong_details}
    correct_only = wrong_only = neither = both = 0
    pair_cmp: list[dict[str, Any]] = []
    for cd in correct_details:
        pm = str(cd["pm_trajectory_id"])
        wd = by_pm_wrong.get(pm) or {}
        c_flip = bool(cd.get("product_id_diff"))
        w_flip = bool(wd.get("product_id_diff"))
        if c_flip and w_flip:
            both += 1
        elif c_flip and not w_flip:
            correct_only += 1
        elif not c_flip and w_flip:
            wrong_only += 1
        else:
            neither += 1
        pair_cmp.append(
            {
                "pm_trajectory_id": pm,
                "correct_donor_diff": c_flip,
                "wrong_owner_donor_diff": w_flip,
            }
        )

    correct_sum = _row_behavioral_summary(correct_sync["measure_row"])
    wrong_sum = _row_behavioral_summary(wrong_sync["measure_row"])

    return {
        "schema_version": "ccer_round11_shell_wrong_owner_donor_v1",
        "experiment_id": "shell_strong_mask_correct_vs_wrong_owner_donor",
        "primary_metric": "product_id_diff",
        "mask_mode": "strong",
        "layer": layer,
        "position": position,
        "rank": rank,
        "intervention_alpha": alpha,
        "anchor_filter": "shell_pid_only",
        "wrong_donor_definition": "recipient_pm_vec_at_commitment (wrong owner; not clean donor)",
        "pair_meta": {
            **pair_meta,
            "n_shell_pairs": len(shell_pairs),
            "n_full_cohort": len(pairs),
        },
        "with_strong_mask_correct_donor": {
            **correct_sum,
            "dual_metrics": correct_sync["dual_metrics"],
            "measure_row": correct_sync["measure_row"],
        },
        "with_strong_mask_wrong_owner_donor": {
            **wrong_sum,
            "dual_metrics": wrong_sync["dual_metrics"],
            "measure_row": wrong_sync["measure_row"],
        },
        "paired_correct_vs_wrong_donor": {
            "n_paired": len(pair_cmp),
            "both_flip": both,
            "correct_donor_only_flip": correct_only,
            "wrong_owner_only_flip": wrong_only,
            "neither_flip": neither,
            "mcnemar_exact_p_two_sided": mcnemar_exact_p(correct_only, wrong_only),
            "pair_table": pair_cmp,
            "interpretation_gate": (
                "Expert gate: support masking hypothesis only if correct_donor >> wrong_owner "
                "(~3/15 vs ~0/15); if wrong_owner also ~20%, shell instability not causal donor."
            ),
        },
    }
