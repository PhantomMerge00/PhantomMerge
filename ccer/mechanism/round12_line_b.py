"""Line B: DAS full-pool training + rank sweep vs PCA (product_id_diff primary)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ccer.io_utils import load_json
from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.paths import SPLIT_MANIFEST_JSON
from ccer.mechanism.das_train import compare_pca_vs_das, fit_das_subspace
from ccer.mechanism.line_b_protocol import (
    DEFAULT_MIN_EVAL_PAIR_QUALITY,
    DEFAULT_MIN_PAIR_QUALITY,
    LINE_B_PROTOCOL_VERSION,
    filter_cross_pairs_by_quality,
    line_a_roi_defaults,
    line_b_select_roi_layer,
)
from ccer.mechanism.mechanism_pool import (
    MECHANISM_RESEARCH_SPLITS,
    build_cem_mechanism_cross_pairs,
    build_mechanism_research_pool_manifest,
)
from ccer.mechanism.owner_pca import collect_diff_vectors, fit_owner_pca
from ccer.mechanism.rank_sweep import measure_diff_rate
from ccer.mechanism.round8_verify import _row_behavioral_summary
from ccer.mechanism.stats import wilson_ci

LINE_B_RANKS = [16, 32, 64]
LINE_B_CONTROLS = ("target_interchange", "wrong_owner_donor")
LINE_B_SUPPLEMENTAL_CONTROLS = ("random_same_rank",)  # + full_vector mode (ceiling)


def _pm_ids_for_splits(splits: tuple[str, ...] | None) -> set[str] | None:
    if splits is None:
        return None
    split_map = load_json(SPLIT_MANIFEST_JSON).get("splits") or {}
    return {tid for tid, sp in split_map.items() if sp in splits}


def _u_to_list(u: Any) -> list[list[float]] | None:
    if u is None:
        return None
    arr = np.asarray(u, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr.tolist()


def _u_from_list(data: list[list[float]] | None) -> np.ndarray | None:
    if not data:
        return None
    return np.asarray(data, dtype=np.float32)
SWEEP_CHECKPOINT_SCHEMA = "ccer_line_b_das_sweep_checkpoint_v1"


def _condition_key(subspace_source: str, control_id: str) -> str:
    return f"{subspace_source}_{control_id}"


def _fit_summary_pca(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        k: fit.get(k)
        for k in ("n_vectors", "pc_variance_explained", "error", "rank")
        if k in fit
    }


def _fit_summary_das(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        k: fit.get(k)
        for k in (
            "n_triplets",
            "das_method",
            "best_interchange_mse",
            "final_interchange_mse",
            "error",
        )
        if k in fit
    }


def load_sweep_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SWEEP_CHECKPOINT_SCHEMA:
        return None
    return data


def save_sweep_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _new_checkpoint_state(
    *,
    track: str,
    layer: int,
    position: str,
    pool_name: str,
    ranks: list[int],
) -> dict[str, Any]:
    return {
        "schema_version": SWEEP_CHECKPOINT_SCHEMA,
        "track": track.upper(),
        "layer": layer,
        "position": position,
        "pool": pool_name,
        "ranks_requested": ranks,
        "completed_rank_rows": [],
        "rank_partial": {},
    }


def _u_or_empty(fit: dict[str, Any], key: str) -> np.ndarray:
    u = fit.get(key)
    return u if u is not None else np.array([])


def activation_coverage_for_pool(
    *,
    pool: str = "mechanism_research",
    layer: int | None = None,
    position: str | None = None,
) -> dict[str, Any]:
    roi = line_a_roi_defaults()
    layer = int(layer if layer is not None else roi["layer"])
    position = str(position if position is not None else roi["position"])
    """Audit activation coverage for expanded mechanism pool cross pairs."""
    pool_name = "mechanism_research" if pool == "mechanism_research" else "dev"
    cross = build_cem_mechanism_cross_pairs(pool=pool_name)["pm_clean_cross_pairs"]
    rows: list[dict[str, Any]] = []
    pm_ok = clean_ok = both_ok = 0
    for cp in cross:
        pm_tid = str(cp["pm_trajectory_id"])
        clean_tid = str(cp["clean_trajectory_id"])
        pm_path = activation_path(pm_tid, "original")
        clean_path = activation_path(clean_tid, "original")
        pm_vec = clean_vec = None
        if pm_path.is_file():
            pm_vec = get_vector(load_activation_npz(pm_path), position=position, layer=layer)
        if clean_path.is_file():
            clean_vec = get_vector(load_activation_npz(clean_path), position=position, layer=layer)
        row = {
            "pm_trajectory_id": pm_tid,
            "clean_trajectory_id": clean_tid,
            "pm_activation_ok": pm_vec is not None,
            "clean_activation_ok": clean_vec is not None,
            "eval_eligible": pm_vec is not None and clean_vec is not None,
        }
        rows.append(row)
        if pm_vec is not None:
            pm_ok += 1
        if clean_vec is not None:
            clean_ok += 1
        if row["eval_eligible"]:
            both_ok += 1
    return {
        "schema_version": "ccer_line_b_activation_coverage_v1",
        "pool": pool_name,
        "layer": layer,
        "position": position,
        "n_cross_pairs": len(cross),
        "n_pm_activation_ok": pm_ok,
        "n_clean_activation_ok": clean_ok,
        "n_eval_eligible": both_ok,
        "coverage_rate": both_ok / len(cross) if cross else 0.0,
        "missing_activation_note": (
            "Expanded pool requires P3-0 activation extraction for train/test trajectories "
            "before eval n increases beyond dev-only coverage."
        ),
        "pair_rows": rows,
    }


def _eligible_cross_pairs(
    pool: str,
    *,
    layer: int | None = None,
    position: str | None = None,
    eval_splits: tuple[str, ...] | None = None,
    min_pair_quality: float = 0.0,
) -> list[dict[str, Any]]:
    roi = line_a_roi_defaults()
    layer = int(layer if layer is not None else roi["layer"])
    position = str(position if position is not None else roi["position"])
    pool_name = "mechanism_research" if pool == "mechanism_research" else "dev"
    cross = build_cem_mechanism_cross_pairs(pool=pool_name)["pm_clean_cross_pairs"]
    eval_allow = _pm_ids_for_splits(eval_splits)
    eligible: list[dict[str, Any]] = []
    for cp in cross:
        pm_tid = str(cp["pm_trajectory_id"])
        if eval_allow is not None and pm_tid not in eval_allow:
            continue
        clean_tid = str(cp["clean_trajectory_id"])
        pm_path = activation_path(pm_tid, "original")
        clean_path = activation_path(clean_tid, "original")
        if not pm_path.is_file() or not clean_path.is_file():
            continue
        pm_npz = load_activation_npz(pm_path)
        clean_npz = load_activation_npz(clean_path)
        if get_vector(pm_npz, position=position, layer=layer) is None:
            continue
        if get_vector(clean_npz, position=position, layer=layer) is None:
            continue
        eligible.append(cp)
    if min_pair_quality > 0 and eligible:
        eligible, _pq = filter_cross_pairs_by_quality(eligible, min_quality=min_pair_quality)
    return eligible


def run_das_vs_pca_rank_sweep(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer: int | None = None,
    position: str | None = None,
    ranks: list[int] | None = None,
    pool: str = "mechanism_research",
    probe_max_tokens: int = 96,
    das_steps: int = 500,
    das_lr: float = 0.02,
    pair_limit: int | None = None,
    checkpoint_path: Path | None = None,
    subspace_train_splits: tuple[str, ...] | None = None,
    eval_splits: tuple[str, ...] | None = None,
    min_train_pair_quality: float = DEFAULT_MIN_PAIR_QUALITY,
    min_eval_pair_quality: float = DEFAULT_MIN_EVAL_PAIR_QUALITY,
    include_full_vector_ceiling: bool = True,
    auto_layer_scan: bool = True,
    n_layers: int = 64,
) -> dict[str, Any]:
    """
    DAS vs PCA at each rank, with target_interchange + wrong_owner_donor controls.
    Primary metric: product_id_diff (never evaluate_iia_row).
    """
    roi = line_a_roi_defaults()
    layer = int(layer if layer is not None else roi["layer"])
    position = str(position if position is not None else roi["position"])
    ranks = ranks or LINE_B_RANKS
    pool_name = "mechanism_research" if pool == "mechanism_research" else "dev"
    layer_scan_meta: dict[str, Any] | None = None
    if auto_layer_scan:
        layer, layer_scan_meta = line_b_select_roi_layer(
            n_layers=n_layers,
            position=position,
            track=track,
            pool=pool_name,
        )
    train_allow = _pm_ids_for_splits(subspace_train_splits)
    all_eligible = _eligible_cross_pairs(
        pool,
        layer=layer,
        position=position,
        eval_splits=eval_splits,
        min_pair_quality=0.0,
    )
    cross_pairs, eval_quality_meta = filter_cross_pairs_by_quality(
        all_eligible, min_quality=min_eval_pair_quality,
    )
    train_pairs, pair_quality_meta = filter_cross_pairs_by_quality(
        all_eligible, min_quality=min_train_pair_quality,
    )
    if train_pairs:
        train_pm_ids = {str(cp["pm_trajectory_id"]) for cp in train_pairs}
        if train_allow is not None:
            train_allow = train_allow & train_pm_ids
        elif pool_name == "mechanism_research":
            train_allow = train_pm_ids
    if pair_limit is not None:
        cross_pairs = cross_pairs[:pair_limit]

    pool_manifest = build_mechanism_research_pool_manifest()
    coverage = activation_coverage_for_pool(pool=pool_name, layer=layer, position=position)

    ni_cache: dict[str, str] = {}
    t0 = time.time()

    ckpt: dict[str, Any] | None = None
    if checkpoint_path is not None:
        loaded = load_sweep_checkpoint(checkpoint_path)
        if loaded and (
            loaded.get("pool") == pool_name
            and loaded.get("layer") == layer
            and loaded.get("position") == position
            and loaded.get("protocol_version") == LINE_B_PROTOCOL_VERSION
            and loaded.get("subspace_train_splits") == (list(subspace_train_splits) if subspace_train_splits else None)
            and loaded.get("eval_splits") == (list(eval_splits) if eval_splits else None)
        ):
            ckpt = loaded
            print(
                f"[line_b] resume checkpoint: completed_ranks="
                f"{[r.get('rank') for r in ckpt.get('completed_rank_rows') or []]} "
                f"partial={list((ckpt.get('rank_partial') or {}).keys())}",
                flush=True,
            )
        elif loaded:
            print("[line_b] checkpoint pool/layer/position mismatch — starting fresh", flush=True)
        if ckpt is None:
            ckpt = _new_checkpoint_state(
                track=track, layer=layer, position=position, pool_name=pool_name, ranks=ranks
            )
            ckpt["subspace_train_splits"] = list(subspace_train_splits) if subspace_train_splits else None
            ckpt["eval_splits"] = list(eval_splits) if eval_splits else None
            ckpt["protocol_version"] = LINE_B_PROTOCOL_VERSION
            ckpt["pair_quality_meta"] = pair_quality_meta

    completed_by_rank = {
        int(r["rank"]): r for r in (ckpt.get("completed_rank_rows") or []) if ckpt
    }
    rows: list[dict[str, Any]] = []

    def _flush_checkpoint() -> None:
        if checkpoint_path is not None and ckpt is not None:
            save_sweep_checkpoint(checkpoint_path, ckpt)

    if ckpt is not None and checkpoint_path is not None:
        _flush_checkpoint()
        print(
            f"[line_b] checkpoint initialized -> {checkpoint_path} "
            f"ranks={ranks} layer=L{layer} position={position} pairs={len(cross_pairs)}",
            flush=True,
        )

    for rank in ranks:
        if rank in completed_by_rank:
            rows.append(completed_by_rank[rank])
            continue

        rank_key = str(rank)
        partial: dict[str, Any] | None = None
        if ckpt is not None:
            partial = (ckpt.get("rank_partial") or {}).get(rank_key)
            if not partial:
                partial = {"conditions": {}}
                ckpt.setdefault("rank_partial", {})[rank_key] = partial

        rank_meta = (partial or {}).get("rank_meta")
        if rank_meta and (
            rank_meta.get("phase") == "fit_done" or rank_meta.get("U_das") is not None
        ):
            print(
                f"[line_b] rank={rank} resume: fit done, eval conditions "
                f"(pairs={len(cross_pairs)})...",
                flush=True,
            )
        elif rank_meta and rank_meta.get("phase") == "pca_done" and not rank_meta.get("U_das"):
            print(f"[line_b] rank={rank} resume: PCA done, running DAS ({das_steps} steps)...", flush=True)
            bundle = collect_diff_vectors(
                position=position,
                layer=layer,
                track=track,
                pool=pool_name,
                pm_trajectory_allowlist=train_allow,
            )
            diff_vecs = bundle["within_vectors"] + bundle["cross_vectors"]
            das_fit = fit_das_subspace(
                diff_vecs,
                rank=rank,
                layer=layer,
                position=position,
                track=track,
                pool=pool_name,
                steps=das_steps,
                lr=das_lr,
                pm_trajectory_allowlist=train_allow,
                log_every=25,
            )
            das_compare = compare_pca_vs_das(
                u_pca=_u_from_list(rank_meta.get("U_owner")),
                u_das=_u_or_empty(das_fit, "U_das"),
            )
            rank_meta = {
                **rank_meta,
                "n_training_triplets_das": int(das_fit.get("n_triplets") or 0),
                "das_method": das_fit.get("das_method"),
                "das_interchange_mse": das_fit.get("best_interchange_mse")
                or das_fit.get("final_interchange_mse"),
                "pca_vs_das_subspace_alignment": das_compare,
                "das_fit": _fit_summary_das(das_fit),
                "U_das": _u_to_list(das_fit.get("U_das")),
                "phase": "fit_done",
            }
            if ckpt is not None and partial is not None:
                partial["rank_meta"] = rank_meta
                _flush_checkpoint()
        elif not rank_meta or rank_meta.get("phase") != "fit_done":
            print(f"[line_b] rank={rank} (1/3) fitting PCA...", flush=True)
            pca_fit = fit_owner_pca(
                layer=layer,
                position=position,
                rank=rank,
                track=track,
                pool=pool_name,
                pm_trajectory_allowlist=train_allow,
            )
            bundle = collect_diff_vectors(
                position=position,
                layer=layer,
                track=track,
                pool=pool_name,
                pm_trajectory_allowlist=train_allow,
            )
            diff_vecs = bundle["within_vectors"] + bundle["cross_vectors"]
            pca_partial = {
                "phase": "pca_done",
                "n_training_vectors_pca": int(pca_fit.get("n_vectors") or 0),
                "pca_fit": _fit_summary_pca(pca_fit),
                "U_owner": _u_to_list(pca_fit.get("U_owner")),
            }
            if ckpt is not None and partial is not None:
                partial["rank_meta"] = pca_partial
                _flush_checkpoint()
            print(
                f"[line_b] rank={rank} (2/3) DAS training steps={das_steps} "
                f"vectors={len(diff_vecs)}...",
                flush=True,
            )
            das_fit = fit_das_subspace(
                diff_vecs,
                rank=rank,
                layer=layer,
                position=position,
                track=track,
                pool=pool_name,
                steps=das_steps,
                lr=das_lr,
                pm_trajectory_allowlist=train_allow,
                log_every=25,
            )
            das_compare = compare_pca_vs_das(
                u_pca=_u_or_empty(pca_fit, "U_owner"),
                u_das=_u_or_empty(das_fit, "U_das"),
            )
            rank_meta = {
                **pca_partial,
                "n_training_triplets_das": int(das_fit.get("n_triplets") or 0),
                "das_method": das_fit.get("das_method"),
                "das_interchange_mse": das_fit.get("best_interchange_mse")
                or das_fit.get("final_interchange_mse"),
                "pca_vs_das_subspace_alignment": das_compare,
                "das_fit": _fit_summary_das(das_fit),
                "U_das": _u_to_list(das_fit.get("U_das")),
                "phase": "fit_done",
            }
            if ckpt is not None and partial is not None:
                partial["rank_meta"] = rank_meta
                _flush_checkpoint()
            print(f"[line_b] rank={rank} (3/3) eval conditions (pairs={len(cross_pairs)})...", flush=True)
        pca_fit = (rank_meta or {}).get("pca_fit") or {}
        das_fit = (rank_meta or {}).get("das_fit") or {}
        u_pca_cached = _u_from_list(rank_meta.get("U_owner"))
        u_das_cached = _u_from_list(rank_meta.get("U_das"))

        rank_row: dict[str, Any] = {
            "rank": rank,
            "n_training_vectors_pca": rank_meta.get("n_training_vectors_pca"),
            "n_training_triplets_das": rank_meta.get("n_training_triplets_das"),
            "das_method": rank_meta.get("das_method"),
            "das_interchange_mse": rank_meta.get("das_interchange_mse"),
            "pca_vs_das_subspace_alignment": rank_meta.get("pca_vs_das_subspace_alignment"),
            "conditions": {},
        }

        for control_id in LINE_B_CONTROLS:
            for subspace_source in ("pca", "das"):
                key = _condition_key(subspace_source, control_id)
                cond_partial = (partial.get("conditions") or {}).get(key) if partial is not None else None
                if cond_partial and cond_partial.get("complete"):
                    rank_row["conditions"][key] = cond_partial.get("summary") or {}
                    continue
                if subspace_source == "das" and das_fit.get("error"):
                    err = {"error": das_fit.get("error")}
                    rank_row["conditions"][key] = err
                    if ckpt is not None and partial is not None:
                        partial.setdefault("conditions", {})[key] = {
                            "complete": True,
                            "summary": err,
                            "pair_details": [],
                        }
                        _flush_checkpoint()
                    continue

                resume_details = (
                    list(cond_partial.get("pair_details") or []) if cond_partial else None
                )
                if resume_details:
                    print(
                        f"[line_b] resume rank={rank} {key}: {len(resume_details)}/{len(cross_pairs)} pairs",
                        flush=True,
                    )
                else:
                    print(
                        f"[line_b] rank={rank} eval {key} ({len(cross_pairs)} pairs)...",
                        flush=True,
                    )

                def _on_pair_done(
                    details: list[dict[str, Any]],
                    *,
                    _key: str = key,
                ) -> None:
                    if ckpt is None or partial is None:
                        return
                    partial.setdefault("conditions", {})[_key] = {
                        "complete": False,
                        "pair_details": details,
                    }
                    _flush_checkpoint()
                    print(
                        f"[line_b] rank={rank} {_key} pair={len(details)}/{len(cross_pairs)} "
                        f"checkpoint",
                        flush=True,
                    )

                u_ov = u_pca_cached if subspace_source == "pca" else u_das_cached
                mr = measure_diff_rate(
                    model=model,
                    tokenizer=tokenizer,
                    track=track,
                    layer=layer,
                    position=position,
                    rank=rank,
                    mode="interchange",
                    probe_max_tokens=probe_max_tokens,
                    subspace_source=subspace_source,
                    control_id=control_id,
                    cross_pairs_override=cross_pairs,
                    ni_cache=ni_cache,
                    pool=pool_name,
                    resume_pair_details=resume_details,
                    on_pair_complete=_on_pair_done,
                    u_override=u_ov,
                    das_steps=das_steps,
                    use_cache=False,
                )
                summary = _row_behavioral_summary(mr)
                rank_row["conditions"][key] = summary
                if ckpt is not None and partial is not None:
                    partial.setdefault("conditions", {})[key] = {
                        "complete": True,
                        "summary": summary,
                        "pair_details": mr.get("pair_details") or [],
                    }
                    _flush_checkpoint()

        if include_full_vector_ceiling:
            ceil_key = "full_vector_target_interchange"
            ceil_partial = (partial.get("conditions") or {}).get(ceil_key) if partial else None
            if ceil_partial and ceil_partial.get("complete"):
                rank_row["conditions"][ceil_key] = ceil_partial.get("summary") or {}
            else:
                ceil_mr = measure_diff_rate(
                    model=model,
                    tokenizer=tokenizer,
                    track=track,
                    layer=layer,
                    position=position,
                    rank=rank,
                    mode="full_vector",
                    probe_max_tokens=probe_max_tokens,
                    control_id="target_interchange",
                    cross_pairs_override=cross_pairs,
                    ni_cache=ni_cache,
                    pool=pool_name,
                    use_cache=False,
                )
                ceil_summary = _row_behavioral_summary(ceil_mr)
                rank_row["conditions"][ceil_key] = ceil_summary
                if ckpt is not None and partial is not None:
                    partial.setdefault("conditions", {})[ceil_key] = {
                        "complete": True,
                        "summary": ceil_summary,
                        "pair_details": ceil_mr.get("pair_details") or [],
                    }
                    _flush_checkpoint()

        rows.append(rank_row)
        if ckpt is not None:
            ckpt.setdefault("completed_rank_rows", [])
            ckpt["completed_rank_rows"] = [
                r for r in ckpt["completed_rank_rows"] if int(r.get("rank") or 0) != rank
            ]
            ckpt["completed_rank_rows"].append(rank_row)
            ckpt.get("rank_partial", {}).pop(rank_key, None)
            _flush_checkpoint()

    return {
        "schema_version": "ccer_line_b_das_vs_pca_v2",
        "primary_metric": "product_id_diff",
        "evaluator": "measure_diff_rate (NOT evaluate_iia_row)",
        "protocol_version": LINE_B_PROTOCOL_VERSION,
        "roi_source": roi.get("source"),
        "layer_scan": layer_scan_meta,
        "min_eval_pair_quality": min_eval_pair_quality,
        "pair_quality_meta": pair_quality_meta,
        "eval_quality_meta": eval_quality_meta,
        "min_train_pair_quality": min_train_pair_quality,
        "track": track.upper(),
        "layer": layer,
        "position": position,
        "pool": pool_name,
        "splits_included": list(MECHANISM_RESEARCH_SPLITS) if pool_name == "mechanism_research" else ["dev"],
        "n_cross_pairs_total": int(pool_manifest.get("n_cross_pairs") or 0),
        "n_cross_pairs_eval_eligible": len(cross_pairs),
        "activation_coverage": coverage,
        "pool_manifest_summary": {
            "n_cem_confirmed": pool_manifest.get("n_cem_confirmed"),
            "n_cem_by_split": pool_manifest.get("n_cem_by_split"),
            "methodology_note": pool_manifest.get("methodology_note"),
        },
        "ranks": ranks,
        "controls": list(LINE_B_CONTROLS),
        "subspace_train_splits": list(subspace_train_splits) if subspace_train_splits else None,
        "eval_splits": list(eval_splits) if eval_splits else None,
        "methodology_note": (
            "Subspace trained on train+dev PM when subspace_train_splits set; "
            "else full mechanism_research pool. Eval pairs filtered by eval_splits when set."
        ),
        "rank_sweep_rows": rows,
        "elapsed_s": round(time.time() - t0, 1),
        "comparison_table": build_comparison_table(rows),
    }


def run_supplemental_controls_sweep(
    *,
    model: Any,
    tokenizer: Any,
    track: str = "CEM",
    layer: int | None = None,
    position: str | None = None,
    ranks: list[int] | None = None,
    pool: str = "mechanism_research",
    probe_max_tokens: int = 96,
    das_steps: int = 500,
) -> dict[str, Any]:
    """Expert-recommended controls: full_vector ceiling + random_same_rank (PCA subspace)."""
    roi = line_a_roi_defaults()
    layer = int(layer if layer is not None else roi["layer"])
    position = str(position if position is not None else roi["position"])
    ranks = ranks or LINE_B_RANKS
    pool_name = "mechanism_research" if pool == "mechanism_research" else "dev"
    cross_pairs = _eligible_cross_pairs(pool, layer=layer, position=position)
    ni_cache: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    for rank in ranks:
        pca_fit = fit_owner_pca(
            layer=layer, position=position, rank=rank, track=track, pool=pool_name
        )
        u_pca = pca_fit.get("U_owner")
        rank_row: dict[str, Any] = {"rank": rank, "conditions": {}}

        ceil_mr = measure_diff_rate(
            model=model,
            tokenizer=tokenizer,
            track=track,
            layer=layer,
            position=position,
            rank=rank,
            mode="full_vector",
            probe_max_tokens=probe_max_tokens,
            cross_pairs_override=cross_pairs,
            ni_cache=ni_cache,
            pool=pool_name,
            control_id="target_interchange",
            das_steps=das_steps,
            use_cache=False,
        )
        rank_row["conditions"]["full_vector_target_interchange"] = _row_behavioral_summary(ceil_mr)

        rand_mr = measure_diff_rate(
            model=model,
            tokenizer=tokenizer,
            track=track,
            layer=layer,
            position=position,
            rank=rank,
            mode="interchange",
            probe_max_tokens=probe_max_tokens,
            subspace_source="pca",
            control_id="random_same_rank",
            cross_pairs_override=cross_pairs,
            ni_cache=ni_cache,
            pool=pool_name,
            u_override=u_pca,
            das_steps=das_steps,
            use_cache=False,
        )
        rank_row["conditions"]["pca_random_same_rank"] = _row_behavioral_summary(rand_mr)
        rows.append(rank_row)

    return {
        "schema_version": "ccer_line_b_supplemental_controls_v1",
        "primary_metric": "product_id_diff",
        "layer": layer,
        "position": position,
        "pool": pool_name,
        "n_cross_pairs_eval_eligible": len(cross_pairs),
        "ranks": ranks,
        "supplemental_controls": ["full_vector_target_interchange", "pca_random_same_rank"],
        "rank_sweep_rows": rows,
    }


def run_das_training_diagnostics(
    *,
    layer: int | None = None,
    position: str | None = None,
    track: str = "CEM",
    ranks: list[int] | None = None,
    pool: str = "mechanism_research",
    das_steps: int = 500,
    das_lr: float = 0.02,
) -> dict[str, Any]:
    """CPU-only: train DAS vs PCA subspaces and report interchange MSE (no GPU eval)."""
    roi = line_a_roi_defaults()
    layer = int(layer if layer is not None else roi["layer"])
    position = str(position if position is not None else roi["position"])
    ranks = ranks or LINE_B_RANKS
    pool_name = "mechanism_research" if pool == "mechanism_research" else "dev"
    pool_manifest = build_mechanism_research_pool_manifest()
    coverage = activation_coverage_for_pool(pool=pool_name, layer=layer, position=position)
    rows: list[dict[str, Any]] = []

    for rank in ranks:
        bundle = collect_diff_vectors(position=position, layer=layer, track=track, pool=pool_name)
        diff_vecs = bundle["within_vectors"] + bundle["cross_vectors"]
        pca_fit = fit_owner_pca(
            layer=layer, position=position, rank=rank, track=track, pool=pool_name
        )
        das_fit = fit_das_subspace(
            diff_vecs,
            rank=rank,
            layer=layer,
            position=position,
            track=track,
            pool=pool_name,
            steps=das_steps,
            lr=das_lr,
        )
        cmp = compare_pca_vs_das(
            u_pca=_u_or_empty(pca_fit, "U_owner"),
            u_das=_u_or_empty(das_fit, "U_das"),
        )
        rows.append(
            {
                "rank": rank,
                "n_diff_vectors": len(diff_vecs),
                "n_triplets_das": das_fit.get("n_triplets"),
                "das_method": das_fit.get("das_method"),
                "das_interchange_mse": das_fit.get("best_interchange_mse")
                or das_fit.get("final_interchange_mse"),
                "pca_variance_explained": pca_fit.get("pc_variance_explained"),
                "pca_vs_das_alignment": cmp,
                "das_error": das_fit.get("error"),
                "pca_error": pca_fit.get("error"),
            }
        )

    return {
        "schema_version": "ccer_line_b_das_training_diagnostics_v1",
        "layer": layer,
        "position": position,
        "pool": pool_name,
        "pool_manifest_summary": {
            "n_cem_confirmed": pool_manifest.get("n_cem_confirmed"),
            "n_cross_pairs": pool_manifest.get("n_cross_pairs"),
        },
        "activation_coverage": coverage,
        "training_rows": rows,
        "note": "Subspace training only; product_id_diff eval requires GPU das-sweep stage.",
    }


def build_comparison_table(rank_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten rank sweep into DAS vs PCA comparison rows with Wilson CIs."""
    table: list[dict[str, Any]] = []
    for rr in rank_rows:
        rank = rr.get("rank")
        ceil = (rr.get("conditions") or {}).get("full_vector_target_interchange") or {}
        if ceil and not ceil.get("error"):
            table.append(
                {
                    "rank": rank,
                    "subspace": "full_vector",
                    "control_id": "target_interchange",
                    "product_id_k": int(ceil.get("product_id_k") or 0),
                    "n_pairs": int(ceil.get("n_pairs") or 0),
                    "product_id_rate": ceil.get("product_id_rate"),
                    "product_id_ci95": ceil.get("product_id_ci95"),
                }
            )
        for subspace in ("pca", "das"):
            for control_id in LINE_B_CONTROLS:
                cond = (rr.get("conditions") or {}).get(f"{subspace}_{control_id}") or {}
                if cond.get("error"):
                    table.append(
                        {
                            "rank": rank,
                            "subspace": subspace,
                            "control_id": control_id,
                            "error": cond.get("error"),
                        }
                    )
                    continue
                k = int(cond.get("product_id_k") or 0)
                n = int(cond.get("n_pairs") or 0)
                table.append(
                    {
                        "rank": rank,
                        "subspace": subspace,
                        "control_id": control_id,
                        "product_id_k": k,
                        "n_pairs": n,
                        "product_id_rate": cond.get("product_id_rate"),
                        "product_id_ci95": cond.get("product_id_ci95"),
                        "selected_product_block_k": cond.get("selected_block_k"),
                        "selected_product_block_rate": cond.get("selected_block_rate"),
                        "selected_product_block_ci95": cond.get("selected_block_ci95"),
                    }
                )
    return table
