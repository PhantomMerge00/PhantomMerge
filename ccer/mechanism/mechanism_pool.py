"""Mechanism research pool: CEM-primary trajectories across splits (Line B)."""
from __future__ import annotations

from typing import Any, Literal

from ccer.adjudication.loader import load_adjudication_index
from ccer.io_utils import load_json
from ccer.mechanism.pair_select import _normalize_slot, _query_slot_from_trajectory, load_trajectory_index
from ccer.paths import COHORT_MANIFEST_JSON, SPLIT_MANIFEST_JSON

PoolName = Literal["dev", "mechanism_research"]
MECHANISM_RESEARCH_SPLITS = ("dev", "train", "test")
MECHANISM_POOL_MANIFEST = (
    __import__("ccer.paths", fromlist=["P3_DIR"]).P3_DIR / "mechanism_research_pool.json"
)


def adjudication_mechanism_pools_multi(*, splits: tuple[str, ...] = MECHANISM_RESEARCH_SPLITS) -> dict[str, list[str]]:
    """Confirmed CEM/AH trajectory IDs unioned across the given splits."""
    index = load_adjudication_index()
    split_map = load_json(SPLIT_MANIFEST_JSON).get("splits", {})
    allowed = set(splits)
    cem: list[str] = []
    ah: list[str] = []
    for tid in index.by_trajectory:
        if split_map.get(tid) not in allowed:
            continue
        if index.cem_trajectory_status.get(tid) == "confirmed":
            cem.append(tid)
        if index.ah_trajectory_status.get(tid) == "confirmed":
            ah.append(tid)
    return {"CEM": sorted(cem), "AH": sorted(ah)}


def build_mechanism_research_pool_manifest() -> dict[str, Any]:
    """Freeze mechanism-research pool metadata (input pool expansion, not cherry-picking)."""
    index = load_adjudication_index()
    split_map = load_json(SPLIT_MANIFEST_JSON).get("splits", {})
    pools = adjudication_mechanism_pools_multi()
    cem_ids = pools["CEM"]
    by_split: dict[str, int] = {s: 0 for s in MECHANISM_RESEARCH_SPLITS}
    for tid in cem_ids:
        sp = split_map.get(tid)
        if sp in by_split:
            by_split[sp] += 1
    cross = build_cem_mechanism_cross_pairs(pool="mechanism_research")
    return {
        "schema_version": "ccer_mechanism_research_pool_v1",
        "pool_name": "mechanism_research",
        "splits_included": list(MECHANISM_RESEARCH_SPLITS),
        "methodology_note": (
            "Expand input pool from train+test+dev confirmed CEM-primary trajectories "
            "(adjudication v2.1). Not outcome cherry-picking; round 3 justified."
        ),
        "n_cem_confirmed": len(cem_ids),
        "n_cem_by_split": by_split,
        "n_ah_confirmed": len(pools["AH"]),
        "n_cross_pairs": len(cross["pm_clean_cross_pairs"]),
        "n_cross_pairs_dev_only": len(
            build_cem_mechanism_cross_pairs(pool="dev")["pm_clean_cross_pairs"]
        ),
        "cem_trajectory_ids": cem_ids,
        "cross_pair_ids": cross["pm_clean_cross_pairs"],
        "pairing_meta": cross["pairing_meta"],
        "label_authority": "cem_ah_strict_v2.1",
    }


def _cem_pool_ids(pool: PoolName) -> list[str]:
    if pool == "dev":
        from ccer.counterfactual.bundles import adjudication_mechanism_pools

        return list(adjudication_mechanism_pools(split="dev")["CEM"])
    return list(adjudication_mechanism_pools_multi()["CEM"])


def build_cem_mechanism_cross_pairs(
    *,
    pool: PoolName = "mechanism_research",
    include_matched_clean: bool = True,
) -> dict[str, Any]:
    """PM×clean cross pairs for mechanism research (no multi_clean_expansion)."""
    rows = load_trajectory_index()
    cem_ids = _cem_pool_ids(pool)
    cohort = load_json(COHORT_MANIFEST_JSON)
    clean_ids = list(cohort.get("cohorts", {}).get("matched_clean", [])) if include_matched_clean else []

    pm_pairs: list[dict[str, Any]] = []
    for tid in cem_ids:
        traj = rows.get(tid)
        if not traj:
            continue
        cem_claim = next(
            (c for c in (traj.get("claims") or []) if c.get("legacy_label") == "cross_object_merge"),
            None,
        )
        if not cem_claim:
            continue
        pm_pairs.append(
            {
                "trajectory_id": tid,
                "cohort": "CEM",
                "slot_norm": str(cem_claim.get("slot_norm") or ""),
                "claim_value": str(cem_claim.get("value") or ""),
            }
        )

    clean_rows: list[dict[str, Any]] = []
    for tid in clean_ids:
        traj = rows.get(tid)
        if not traj:
            continue
        clean_rows.append(
            {
                "trajectory_id": tid,
                "cohort": "matched_clean",
                "slot_norm": _query_slot_from_trajectory(traj),
            }
        )

    slot_to_clean: dict[str, list[str]] = {}
    for cr in clean_rows:
        slot_to_clean.setdefault(_normalize_slot(cr["slot_norm"]), []).append(cr["trajectory_id"])

    cross_pairs: list[dict[str, Any]] = []
    used_clean: set[str] = set()
    unmatched: list[dict[str, Any]] = []
    for pair in pm_pairs:
        slot = _normalize_slot(pair["slot_norm"])
        candidates = [c for c in slot_to_clean.get(slot, []) if c not in used_clean]
        if not candidates:
            unmatched.append(pair)
            continue
        clean_tid = candidates[0]
        used_clean.add(clean_tid)
        cross_pairs.append(
            {
                "pm_trajectory_id": pair["trajectory_id"],
                "clean_trajectory_id": clean_tid,
                "slot_norm": pair["slot_norm"],
                "pairing_method": "slot_norm_match",
                "track": "CEM",
                "pool": pool,
            }
        )

    clean_pool = [cr["trajectory_id"] for cr in clean_rows if cr["trajectory_id"] not in used_clean]
    if not clean_pool:
        clean_pool = [cr["trajectory_id"] for cr in clean_rows]
    for i, pair in enumerate(unmatched):
        if not clean_pool:
            break
        clean_tid = clean_pool[i % len(clean_pool)]
        cross_pairs.append(
            {
                "pm_trajectory_id": pair["trajectory_id"],
                "clean_trajectory_id": clean_tid,
                "slot_norm": pair["slot_norm"],
                "pairing_method": "cohort_index_fallback",
                "track": "CEM",
                "pool": pool,
            }
        )

    return {
        "pool": pool,
        "cem_pairs": pm_pairs,
        "matched_clean_original": clean_rows,
        "pm_clean_cross_pairs": cross_pairs,
        "n_cem_in_pool": len(cem_ids),
        "n_cem_with_claim": len(pm_pairs),
        "n_cross_pairs": len(cross_pairs),
        "pairing_meta": {
            "pairing_source": "build_cem_mechanism_cross_pairs",
            "expansion_method": None,
            "splits_included": list(MECHANISM_RESEARCH_SPLITS) if pool == "mechanism_research" else ["dev"],
            "expansion_note": (
                f"Mechanism research pool: {len(cem_ids)} confirmed CEM-primary across "
                f"{'dev+train+test' if pool == 'mechanism_research' else 'dev'}; "
                f"{len(cross_pairs)} native PM×clean pairs (no multi_clean_expansion)."
            ),
        },
    }
