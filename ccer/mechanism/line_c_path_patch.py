"""Line C: Attention vs MLP path patching — full n=22 + wrong_owner_donor control."""
from __future__ import annotations

from typing import Any

from ccer.mechanism.mechanism_pool import MECHANISM_POOL_MANIFEST, build_mechanism_research_pool_manifest
from ccer.mechanism.pair_select import cem_valid_pair_ids
from ccer.mechanism.rank_sweep import measure_diff_rate, merge_measure_rows
from ccer.mechanism.round12_line_b import _eligible_cross_pairs
from ccer.mechanism.round8_verify import PATCH_IMPLEMENTATION_NOTE, _row_behavioral_summary
from ccer.mechanism.round9_verify import _sync_dual_metrics
from ccer.mechanism.stats import mcnemar_exact_p


def _base_cross_pairs(
    track: str = "CEM",
    *,
    pool: str = "mechanism_research",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cross pairs for path patching. Default: Line B mechanism_research pool (activation-eligible only)."""
    if pool == "mechanism_research":
        pairs = _eligible_cross_pairs("mechanism_research")
        manifest = build_mechanism_research_pool_manifest()
        meta = {
            "n_pairs": len(pairs),
            "n_cross_pairs_total": int(manifest.get("n_cross_pairs") or 0),
            "pairing_source": "build_cem_mechanism_cross_pairs",
            "pool": "mechanism_research",
            "pool_manifest": str(MECHANISM_POOL_MANIFEST),
            "activation_coverage_artifact": "results/p3/round12_line_b/activation_coverage.json",
            "expansion_method": "adjudication_confirmed_train_test_dev",
            "expansion_note": (
                "Reuses Line B expanded pool + P3-0 activations; "
                "filters to L32/commitment activation-eligible pairs only."
            ),
        }
        return pairs, meta

    sel = cem_valid_pair_ids(include_matched_clean=True)
    pairs = list(sel["pm_clean_cross_pairs"])
    meta = {
        "n_pairs": len(pairs),
        "pairing_source": "cem_valid_pair_ids",
        "pool": "native_dev",
        "expansion_method": None,
        "expansion_note": "Native n=22 dev cross-pairs; fallback when mechanism_research unavailable.",
    }
    return pairs, meta


def _pool_for_measure(pool: str) -> str:
    """Map Line C pool arg to measure_diff_rate / fit_owner_pca pool name."""
    if pool == "mechanism_research":
        return "mechanism_research"
    return "dev"


def _paired_correct_vs_wrong(
    correct_details: list[dict[str, Any]],
    wrong_details: list[dict[str, Any]],
) -> dict[str, Any]:
    by_pm_wrong = {str(d["pm_trajectory_id"]): d for d in wrong_details}
    correct_only = wrong_only = neither = both = 0
    pair_table: list[dict[str, Any]] = []
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
        pair_table.append(
            {
                "pm_trajectory_id": pm,
                "correct_donor_diff": c_flip,
                "wrong_owner_donor_diff": w_flip,
            }
        )
    return {
        "n_paired": len(pair_table),
        "both_flip": both,
        "correct_donor_only_flip": correct_only,
        "wrong_owner_only_flip": wrong_only,
        "neither_flip": neither,
        "mcnemar_exact_p_two_sided": mcnemar_exact_p(correct_only, wrong_only),
        "pair_table": pair_table,
    }


def _paired_site_vs_site(
    left_details: list[dict[str, Any]],
    right_details: list[dict[str, Any]],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    by_right = {str(d["pm_trajectory_id"]): d for d in right_details}
    left_only = right_only = neither = both = 0
    pair_table: list[dict[str, Any]] = []
    for ld in left_details:
        pm = str(ld["pm_trajectory_id"])
        rd = by_right.get(pm) or {}
        l_flip = bool(ld.get("product_id_diff"))
        r_flip = bool(rd.get("product_id_diff"))
        if l_flip and r_flip:
            both += 1
        elif l_flip and not r_flip:
            left_only += 1
        elif not l_flip and r_flip:
            right_only += 1
        else:
            neither += 1
        pair_table.append(
            {
                "pm_trajectory_id": pm,
                f"{left_label}_diff": l_flip,
                f"{right_label}_diff": r_flip,
            }
        )
    return {
        "left_label": left_label,
        "right_label": right_label,
        "n_paired": len(pair_table),
        "both_flip": both,
        f"{left_label}_only_flip": left_only,
        f"{right_label}_only_flip": right_only,
        "neither_flip": neither,
        "mcnemar_exact_p_two_sided": mcnemar_exact_p(left_only, right_only),
        "pair_table": pair_table,
    }


def _site_arm(
    *,
    model: Any,
    tokenizer: Any,
    track: str,
    layer: int,
    position: str,
    rank: int,
    probe_max_tokens: int,
    inject_site: str,
    cross_pairs: list[dict[str, Any]],
    ni_cache: dict[str, str],
    shard_index: int,
    num_shards: int,
    alpha: float,
    pool: str,
) -> dict[str, Any]:
    pca_pool = _pool_for_measure(pool)
    correct_row = measure_diff_rate(
        model=model,
        tokenizer=tokenizer,
        track=track,
        layer=layer,
        position=position,
        rank=rank,
        mode="interchange",
        probe_max_tokens=probe_max_tokens,
        inject_site=inject_site,
        cross_pairs_override=cross_pairs,
        ni_cache=ni_cache,
        shard_index=shard_index,
        num_shards=num_shards,
        control_id="target_interchange",
        alpha=alpha,
        pool=pca_pool,
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
        inject_site=inject_site,
        cross_pairs_override=cross_pairs,
        ni_cache=ni_cache,
        shard_index=shard_index,
        num_shards=num_shards,
        control_id="wrong_owner_donor",
        alpha=alpha,
        pool=pca_pool,
    )
    correct_sync = _sync_dual_metrics(correct_row)
    wrong_sync = _sync_dual_metrics(wrong_row)
    correct_details = correct_sync["measure_row"].get("pair_details") or []
    wrong_details = wrong_sync["measure_row"].get("pair_details") or []
    return {
        "inject_site": inject_site,
        "correct_donor": {
            **_row_behavioral_summary(correct_sync["measure_row"]),
            "dual_metrics": correct_sync["dual_metrics"],
            "measure_row": correct_sync["measure_row"],
        },
        "wrong_owner_donor": {
            **_row_behavioral_summary(wrong_sync["measure_row"]),
            "dual_metrics": wrong_sync["dual_metrics"],
            "measure_row": wrong_sync["measure_row"],
        },
        "paired_correct_vs_wrong_donor": _paired_correct_vs_wrong(correct_details, wrong_details),
    }


def run_path_patch_expanded(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer: int = 32,
    position: str = "commitment",
    rank: int = 16,
    probe_max_tokens: int = 384,
    shard_index: int = 0,
    num_shards: int = 1,
    alpha: float = 1.0,
    inject_sites: tuple[str, ...] | None = None,
    pool: str = "mechanism_research",
) -> dict[str, Any]:
    """Full-pool attention vs MLP path patching with wrong_owner_donor control."""
    sites = tuple(inject_sites or ("attention", "mlp"))
    cross_pairs, pair_meta = _base_cross_pairs(track, pool=pool)
    ni_cache: dict[str, str] = {}

    by_site: dict[str, Any] = {}
    for site in sites:
        if site not in ("attention", "mlp"):
            raise ValueError(f"invalid inject_site: {site}")
        by_site[site] = _site_arm(
            model=model,
            tokenizer=tokenizer,
            track=track,
            layer=layer,
            position=position,
            rank=rank,
            probe_max_tokens=probe_max_tokens,
            inject_site=site,
            cross_pairs=cross_pairs,
            ni_cache=ni_cache,
            shard_index=shard_index,
            num_shards=num_shards,
            alpha=alpha,
            pool=pool,
        )

    att_correct_details = (
        (by_site.get("attention") or {}).get("correct_donor", {}).get("measure_row") or {}
    ).get("pair_details") or []
    mlp_correct_details = (
        (by_site.get("mlp") or {}).get("correct_donor", {}).get("measure_row") or {}
    ).get("pair_details") or []

    out: dict[str, Any] = {
        "schema_version": "ccer_line_c_path_patch_v1",
        "experiment_id": "attention_vs_mlp_path_patch_full_pool",
        "primary_metric": "product_id_diff",
        "secondary_metrics": ["selected_product_block_diff", "full_response_excerpt_diff"],
        "layer": layer,
        "position": position,
        "rank": rank,
        "intervention_alpha": alpha,
        "probe_max_tokens": probe_max_tokens,
        "wrong_donor_definition": "recipient_pm_vec_at_commitment (wrong owner; not clean donor)",
        "pool": pool,
        "pca_subspace_pool": _pool_for_measure(pool),
        "pca_subspace_note": "PCA owner subspace fit uses same pool as Line B (not dev-only when pool=mechanism_research).",
        "pair_meta": pair_meta,
        "patch_implementation": PATCH_IMPLEMENTATION_NOTE,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "inject_sites_run": list(sites),
        "by_inject_site": by_site,
    }
    if att_correct_details and mlp_correct_details:
        out["attention_vs_mlp_correct_donor"] = _paired_site_vs_site(
            att_correct_details,
            mlp_correct_details,
            left_label="attention",
            right_label="mlp",
        )
    return out


def merge_path_patch_site_partials(attention_partial: dict[str, Any], mlp_partial: dict[str, Any]) -> dict[str, Any]:
    """Merge attention-only + mlp-only shard outputs into one full Line C artifact."""
    base = dict(attention_partial)
    att_site = (attention_partial.get("by_inject_site") or {}).get("attention") or {}
    mlp_site = (mlp_partial.get("by_inject_site") or {}).get("mlp") or {}
    base["by_inject_site"] = {"attention": att_site, "mlp": mlp_site}
    base["inject_sites_run"] = ["attention", "mlp"]
    att_details = (att_site.get("correct_donor") or {}).get("measure_row", {}).get("pair_details") or []
    mlp_details = (mlp_site.get("correct_donor") or {}).get("measure_row", {}).get("pair_details") or []
    if att_details and mlp_details:
        base["attention_vs_mlp_correct_donor"] = _paired_site_vs_site(
            att_details,
            mlp_details,
            left_label="attention",
            right_label="mlp",
        )
    base.pop("shard_index", None)
    base["num_shards_merged"] = max(
        int(attention_partial.get("num_shards") or 1),
        int(mlp_partial.get("num_shards") or 1),
    )
    return base


def _merge_site_arm(arms: list[dict[str, Any]], key: str) -> dict[str, Any]:
    parts = [a.get(key) or {} for a in arms]
    measure_rows = [p.get("measure_row") or {} for p in parts if p.get("measure_row")]
    merged_row = merge_measure_rows(measure_rows) if measure_rows else {}
    synced = _sync_dual_metrics(merged_row) if merged_row.get("pair_details") else {"measure_row": merged_row, "dual_metrics": {}}
    return {
        **_row_behavioral_summary(synced["measure_row"]),
        "dual_metrics": synced.get("dual_metrics"),
        "measure_row": synced["measure_row"],
    }


def merge_path_patch_shard_results(shard_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge shard-local Line C path-patch outputs into one aggregate."""
    if not shard_results:
        return {"error": "no_shard_results"}
    if len(shard_results) == 1:
        out = dict(shard_results[0])
        out.pop("shard_index", None)
        out["num_shards_merged"] = 1
        return out

    base = dict(shard_results[0])
    base.pop("shard_index", None)
    merged_sites: dict[str, Any] = {}
    for site in ("attention", "mlp"):
        site_shards = [s.get("by_inject_site", {}).get(site) or {} for s in shard_results]
        correct = _merge_site_arm(site_shards, "correct_donor")
        wrong = _merge_site_arm(site_shards, "wrong_owner_donor")
        correct_details = (correct.get("measure_row") or {}).get("pair_details") or []
        wrong_details = (wrong.get("measure_row") or {}).get("pair_details") or []
        merged_sites[site] = {
            "inject_site": site,
            "correct_donor": correct,
            "wrong_owner_donor": wrong,
            "paired_correct_vs_wrong_donor": _paired_correct_vs_wrong(correct_details, wrong_details),
        }

    att_details = (merged_sites["attention"]["correct_donor"].get("measure_row") or {}).get("pair_details") or []
    mlp_details = (merged_sites["mlp"]["correct_donor"].get("measure_row") or {}).get("pair_details") or []
    base["by_inject_site"] = merged_sites
    base["attention_vs_mlp_correct_donor"] = _paired_site_vs_site(
        att_details,
        mlp_details,
        left_label="attention",
        right_label="mlp",
    )
    base["num_shards_merged"] = len(shard_results)
    return base
