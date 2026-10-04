"""Contrastive Activation Addition (CAA) steering vectors for Line E mitigation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ccer.io_utils import load_json
from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.mechanism_pool import PoolName, build_cem_mechanism_cross_pairs
from ccer.mechanism.pair_select import TrackName, load_trajectory_index
from ccer.paths import COHORT_MANIFEST_JSON, SPLIT_MANIFEST_JSON

SteeringControlId = Literal[
    "caa_true",
    "caa_random",
    "caa_orthogonal",
    "paired_caa",
    "paired_full",
    "donor_restore",
    "no_intervention",
]
MitigationDirection = Literal["subtract", "add"]


@dataclass(frozen=True)
class CAAVectorResult:
    vector: np.ndarray
    layer: int
    position: str
    n_pm: int
    n_clean: int
    pm_norm: float
    clean_norm: float
    vector_norm: float
    pm_trajectory_ids: tuple[str, ...]
    clean_trajectory_ids: tuple[str, ...]


def _trajectory_track(tid: str, cohort: dict[str, Any]) -> TrackName | None:
    cem = set(cohort.get("cohorts", {}).get("CEM", []))
    cap = set(cohort.get("cohorts", {}).get("CAP", []))
    if tid in cem:
        return "CEM"
    if tid in cap:
        return "CAP"
    return None


def pm_pool_with_activations(*, track: TrackName | None = None) -> list[str]:
    """PM trajectories (CEM/CAP cohort) with cached P3-0 original activations."""
    cohort = load_json(COHORT_MANIFEST_JSON)
    rows = load_trajectory_index()
    pool: list[str] = []
    for track_name in ("CEM", "CAP"):
        if track is not None and track != track_name:
            continue
        for tid in cohort.get("cohorts", {}).get(track_name, []):
            if tid not in rows:
                continue
            if rows[tid].get("trajectory_outcome") != "has_pm_core":
                continue
            if activation_path(tid, "original").is_file():
                pool.append(tid)
    return sorted(pool)


def clean_pool_with_activations() -> list[str]:
    cohort = load_json(COHORT_MANIFEST_JSON)
    rows = load_trajectory_index()
    out: list[str] = []
    for tid in cohort.get("cohorts", {}).get("matched_clean", []):
        if tid in rows and activation_path(tid, "original").is_file():
            out.append(tid)
    return sorted(out)


def _has_activation(tid: str, *, layer: int, position: str) -> bool:
    from ccer.mechanism.activation_store import fast_npz_roi_ok

    ok, _ = fast_npz_roi_ok(tid, position=position, layer=layer, require_line_d_v3=False)
    return ok


def mechanism_research_pm_with_activations(*, layer: int, position: str) -> list[str]:
    """CEM-primary PM trajectories from Line B mechanism_research pool (reuse P3-0 cache)."""
    cross = build_cem_mechanism_cross_pairs(pool="mechanism_research")["pm_clean_cross_pairs"]
    pm_ids = sorted({str(cp["pm_trajectory_id"]) for cp in cross})
    return [tid for tid in pm_ids if _has_activation(tid, layer=layer, position=position)]


def paired_clean_trajectory_id(pm_trajectory_id: str, *, pool: PoolName = "mechanism_research") -> str | None:
    cross = build_cem_mechanism_cross_pairs(pool=pool)["pm_clean_cross_pairs"]
    for cp in cross:
        if str(cp["pm_trajectory_id"]) == pm_trajectory_id:
            return str(cp["clean_trajectory_id"])
    return None


def paired_activation_vectors(
    pm_trajectory_id: str,
    *,
    layer: int,
    position: str,
    pool: PoolName = "mechanism_research",
) -> dict[str, Any] | dict[str, str]:
    """Per-trajectory PM/clean activations and paired CAA delta (PM - clean)."""
    clean_tid = paired_clean_trajectory_id(pm_trajectory_id, pool=pool)
    if not clean_tid:
        return {"error": "no_paired_clean", "pm_trajectory_id": pm_trajectory_id}
    pm_vecs, pm_used = collect_layer_activations([pm_trajectory_id], layer=layer, position=position)
    clean_vecs, clean_used = collect_layer_activations([clean_tid], layer=layer, position=position)
    if not pm_vecs or not clean_vecs:
        return {
            "error": "missing_paired_activation",
            "pm_trajectory_id": pm_trajectory_id,
            "clean_trajectory_id": clean_tid,
        }
    pm_vec = pm_vecs[0]
    clean_vec = clean_vecs[0]
    delta = (pm_vec - clean_vec).astype(np.float32)
    return {
        "pm_trajectory_id": pm_trajectory_id,
        "clean_trajectory_id": clean_tid,
        "pm_vec": pm_vec,
        "clean_vec": clean_vec,
        "paired_caa_vector": delta,
        "paired_norm": float(np.linalg.norm(delta)),
        "layer": layer,
        "position": position,
    }


def mechanism_research_clean_for_build(
    build_pm_ids: list[str],
    *,
    layer: int,
    position: str,
) -> list[str]:
    """Clean donors paired to build-set PM (Line B cross-pair manifest; no re-pairing)."""
    build_set = set(build_pm_ids)
    cross = build_cem_mechanism_cross_pairs(pool="mechanism_research")["pm_clean_cross_pairs"]
    clean_ids: set[str] = set()
    for cp in cross:
        pm_tid = str(cp["pm_trajectory_id"])
        if pm_tid not in build_set:
            continue
        clean_tid = str(cp["clean_trajectory_id"])
        if _has_activation(clean_tid, layer=layer, position=position):
            clean_ids.add(clean_tid)
    return sorted(clean_ids)


def assign_split_manifest_holdout(
    pm_trajectory_ids: list[str],
    *,
    build_splits: tuple[str, ...] = ("train", "dev"),
    eval_splits: tuple[str, ...] = ("test",),
) -> dict[str, Any]:
    """Train/dev build vs test eval using split_manifest (requires Line B P3-0 activations)."""
    split_map = load_json(SPLIT_MANIFEST_JSON).get("splits") or {}
    build_pm = sorted(tid for tid in pm_trajectory_ids if split_map.get(tid) in build_splits)
    eval_pm = sorted(tid for tid in pm_trajectory_ids if split_map.get(tid) in eval_splits)
    return {
        "holdout_method": "split_manifest",
        "build_splits": list(build_splits),
        "eval_splits": list(eval_splits),
        "pool": "mechanism_research",
        "n_pm_total": len(pm_trajectory_ids),
        "build_pm_trajectory_ids": build_pm,
        "eval_pm_trajectory_ids": eval_pm,
        "n_build_pm": len(build_pm),
        "n_eval_pm": len(eval_pm),
        "build_split_counts": {sp: sum(1 for tid in build_pm if split_map.get(tid) == sp) for sp in build_splits},
        "eval_split_counts": {sp: sum(1 for tid in eval_pm if split_map.get(tid) == sp) for sp in eval_splits},
        "line_b_activation_reuse": (
            "Reuses P3-0 activations extracted by Line B (--stage extract-activations); "
            "see results/p3/round12_line_b/activation_coverage.json"
        ),
        "line_a_roi_note": (
            "Line A v3 peak L49/claim_onset uses instance-aligned probes on dense layers; "
            "P3-0 sparse layers are [0,16,32,48,63]. Primary steering ROI L32/commitment "
            "aligns with Line B frozen ROI; L48/claim_onset available as sensitivity only."
        ),
    }


def _group_holdout_key(tid: str, rows: dict[str, dict[str, Any]]) -> str:
    traj = rows.get(tid) or {}
    anchor = str((traj.get("commitment") or {}).get("action_anchor") or "")
    query = str((traj.get("metadata") or {}).get("query") or "")[:80]
    base = f"{anchor}|{query}" if anchor or query else tid
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def assign_build_eval_holdout(
    pm_trajectory_ids: list[str],
    *,
    holdout_frac: float = 0.35,
    seed: int = 42,
) -> dict[str, Any]:
    """Legacy group-level holdout within dev-only cohort (fallback when pool=dev)."""
    cohort = load_json(COHORT_MANIFEST_JSON)
    rows = load_trajectory_index()
    groups: dict[str, list[str]] = {}
    for tid in pm_trajectory_ids:
        groups.setdefault(_group_holdout_key(tid, rows), []).append(tid)

    group_ids = sorted(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids)
    n_holdout = max(1, int(round(len(group_ids) * holdout_frac)))
    holdout_groups = set(group_ids[:n_holdout])
    build_pm: list[str] = []
    eval_pm: list[str] = []
    for gid, tids in groups.items():
        if gid in holdout_groups:
            eval_pm.extend(tids)
        else:
            build_pm.extend(tids)

    by_track_build: dict[str, int] = {"CEM": 0, "CAP": 0}
    by_track_eval: dict[str, int] = {"CEM": 0, "CAP": 0}
    for tid in build_pm:
        tr = _trajectory_track(tid, cohort)
        if tr:
            by_track_build[tr] += 1
    for tid in eval_pm:
        tr = _trajectory_track(tid, cohort)
        if tr:
            by_track_eval[tr] += 1

    split_map = load_json(SPLIT_MANIFEST_JSON).get("splits") or {}
    return {
        "holdout_method": "group_level_within_dev_activation_pool",
        "holdout_frac": holdout_frac,
        "seed": seed,
        "n_pm_total": len(pm_trajectory_ids),
        "n_groups": len(group_ids),
        "n_holdout_groups": len(holdout_groups),
        "build_pm_trajectory_ids": sorted(build_pm),
        "eval_pm_trajectory_ids": sorted(eval_pm),
        "n_build_pm": len(build_pm),
        "n_eval_pm": len(eval_pm),
        "build_by_track": by_track_build,
        "eval_by_track": by_track_eval,
        "cohort_selection_rules": (cohort.get("selection_rules") or {}),
        "split_manifest_note": (
            "cohort trajectories are dev_only; activations are not available on "
            "split_manifest train/test PM trajectories without P3-0 re-extraction"
        ),
        "eval_split_counts": {
            sp: sum(1 for tid in eval_pm if split_map.get(tid) == sp) for sp in ("train", "dev", "test")
        },
    }


def collect_layer_activations(
    trajectory_ids: list[str],
    *,
    layer: int,
    position: str,
    condition_id: str = "original",
) -> tuple[list[np.ndarray], list[str]]:
    vectors: list[np.ndarray] = []
    used: list[str] = []
    for tid in trajectory_ids:
        path = activation_path(tid, condition_id)
        if not path.is_file():
            continue
        loaded = load_activation_npz(path)
        vec = get_vector(loaded, position=position, layer=layer)
        if vec is None:
            continue
        vectors.append(np.asarray(vec, dtype=np.float32).reshape(-1))
        used.append(tid)
    return vectors, used


def build_caa_vector(
    pm_trajectory_ids: list[str],
    clean_trajectory_ids: list[str],
    *,
    layer: int,
    position: str,
) -> CAAVectorResult | dict[str, Any]:
    pm_vecs, pm_used = collect_layer_activations(pm_trajectory_ids, layer=layer, position=position)
    clean_vecs, clean_used = collect_layer_activations(clean_trajectory_ids, layer=layer, position=position)
    if len(pm_vecs) < 2:
        return {"error": "insufficient_pm_activations", "n_pm": len(pm_vecs), "layer": layer, "position": position}
    if len(clean_vecs) < 2:
        return {"error": "insufficient_clean_activations", "n_clean": len(clean_vecs), "layer": layer, "position": position}
    pm_stack = np.stack(pm_vecs, axis=0)
    clean_stack = np.stack(clean_vecs, axis=0)
    pm_mean = pm_stack.mean(axis=0)
    clean_mean = clean_stack.mean(axis=0)
    vec = (pm_mean - clean_mean).astype(np.float32)
    return CAAVectorResult(
        vector=vec,
        layer=layer,
        position=position,
        n_pm=len(pm_used),
        n_clean=len(clean_used),
        pm_norm=float(np.linalg.norm(pm_mean)),
        clean_norm=float(np.linalg.norm(clean_mean)),
        vector_norm=float(np.linalg.norm(vec)),
        pm_trajectory_ids=tuple(pm_used),
        clean_trajectory_ids=tuple(clean_used),
    )


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-8:
        return vec.astype(np.float32)
    return (vec / n).astype(np.float32)


def build_paired_steering_controls(
    paired_caa_vector: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Per-trajectory norm-matched random/orthogonal controls (Line E v2)."""
    v = np.asarray(paired_caa_vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        raise ValueError("paired CAA vector has zero norm")
    rng = np.random.default_rng(seed)
    rand_dir = _unit_vector(rng.standard_normal(v.shape[0]).astype(np.float32)) * norm
    u = _unit_vector(v)
    rand_raw = rng.standard_normal(v.shape[0]).astype(np.float32)
    orth = rand_raw - u * float(np.dot(u, rand_raw))
    orth_dir = _unit_vector(orth) * norm
    return {"caa_random": rand_dir, "caa_orthogonal": orth_dir}


def build_steering_controls(
    caa_vector: np.ndarray,
    *,
    seed: int = 0,
) -> dict[SteeringControlId, np.ndarray]:
    v = np.asarray(caa_vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        raise ValueError("CAA vector has zero norm; cannot build norm-matched controls")

    rng = np.random.default_rng(seed)
    rand_dir = rng.standard_normal(v.shape[0]).astype(np.float32)
    rand_dir = _unit_vector(rand_dir) * norm

    # Orthogonal component with matched norm (Gram-Schmidt on random direction).
    u = _unit_vector(v)
    rand_raw = rng.standard_normal(v.shape[0]).astype(np.float32)
    orth = rand_raw - u * float(np.dot(u, rand_raw))
    orth_dir = _unit_vector(orth) * norm

    return {
        "caa_true": v,
        "caa_random": rand_dir,
        "caa_orthogonal": orth_dir,
    }


def save_caa_bundle(
    path: Any,
    *,
    caa: CAAVectorResult,
    controls: dict[str, np.ndarray],
    holdout: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    arrays: dict[str, Any] = {
        "caa_vector": caa.vector,
        "layer": np.array(caa.layer, dtype=np.int32),
        "position": np.array(caa.position),
        "n_pm": np.array(caa.n_pm, dtype=np.int32),
        "n_clean": np.array(caa.n_clean, dtype=np.int32),
        "vector_norm": np.array(caa.vector_norm, dtype=np.float32),
        "pm_trajectory_ids": np.array(list(caa.pm_trajectory_ids)),
        "clean_trajectory_ids": np.array(list(caa.clean_trajectory_ids)),
        "build_pm_trajectory_ids": np.array(holdout.get("build_pm_trajectory_ids") or []),
        "eval_pm_trajectory_ids": np.array(holdout.get("eval_pm_trajectory_ids") or []),
    }
    for key, vec in controls.items():
        arrays[f"control__{key}"] = np.asarray(vec, dtype=np.float32)
    if metadata:
        arrays["meta_json"] = np.array(str(metadata))
    np.savez_compressed(path, **arrays)


def load_caa_bundle(path: Any) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    controls: dict[str, np.ndarray] = {}
    for key in data.files:
        if key.startswith("control__"):
            controls[key.replace("control__", "", 1)] = data[key].astype(np.float32)
    return {
        "caa_vector": data["caa_vector"].astype(np.float32),
        "layer": int(data["layer"]),
        "position": str(data["position"]),
        "n_pm": int(data["n_pm"]),
        "n_clean": int(data["n_clean"]),
        "vector_norm": float(data["vector_norm"]),
        "pm_trajectory_ids": [str(x) for x in data["pm_trajectory_ids"].tolist()],
        "clean_trajectory_ids": [str(x) for x in data["clean_trajectory_ids"].tolist()],
        "build_pm_trajectory_ids": [str(x) for x in data["build_pm_trajectory_ids"].tolist()],
        "eval_pm_trajectory_ids": [str(x) for x in data["eval_pm_trajectory_ids"].tolist()],
        "controls": controls,
    }
