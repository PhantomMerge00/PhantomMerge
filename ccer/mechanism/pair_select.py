"""Select CAP / CEM / matched_clean pairing for P3 mechanism analysis."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from ccer.counterfactual.bundles import build_cap_bundle, build_cem_bundle
from ccer.io_utils import load_json, load_jsonl
from ccer.paths import COHORT_MANIFEST_JSON, NORMALIZED_SHOPPING, P1_DIR

TrackName = Literal["CAP", "CEM"]


def load_trajectory_index() -> dict[str, dict[str, Any]]:
    return {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}


def invalid_cap_ids() -> set[str]:
    out: set[str] = set()
    for row in load_jsonl(P1_DIR / "invalid_counterfactuals.jsonl"):
        if str(row.get("condition_id") or "") != "query_value_swap":
            continue
        tid = str(row.get("trajectory_id") or row.get("root_id") or "")
        if tid:
            out.add(tid)
    return out


def cap_valid_pair_ids(*, include_matched_clean: bool = True) -> dict[str, Any]:
    cohort = load_json(COHORT_MANIFEST_JSON)
    invalid = invalid_cap_ids()
    cap_ids = [tid for tid in cohort.get("cohorts", {}).get("CAP", []) if tid not in invalid]
    rows = load_trajectory_index()
    pairs: list[dict[str, Any]] = []
    for tid in cap_ids:
        traj = rows.get(tid)
        if not traj:
            continue
        cap_claim = next(
            (c for c in (traj.get("claims") or []) if c.get("legacy_label") == "constraint_projection"),
            None,
        )
        if not cap_claim:
            continue
        value = str(cap_claim.get("value") or "")
        cf_value = f"CF_{value}"
        pairs.append(
            {
                "trajectory_id": tid,
                "cohort": "CAP",
                "slot_norm": str(cap_claim.get("slot_norm") or ""),
                "claim_value": value,
                "cf_value": cf_value,
                "conditions": ["original", "query_value_swap", "position_permutation"],
            }
        )

    clean_ids = cohort.get("cohorts", {}).get("matched_clean", []) if include_matched_clean else []
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
                "conditions": ["original"],
            }
        )

    slot_to_clean: dict[str, list[str]] = {}
    for cr in clean_rows:
        slot_to_clean.setdefault(cr["slot_norm"], []).append(cr["trajectory_id"])

    cross_pairs: list[dict[str, Any]] = []
    used_clean: set[str] = set()
    unmatched_caps: list[dict[str, Any]] = []
    for pair in pairs:
        slot = _normalize_slot(pair["slot_norm"])
        candidates = [c for c in slot_to_clean.get(slot, []) if c not in used_clean]
        if not candidates:
            unmatched_caps.append(pair)
            continue
        clean_tid = candidates[0]
        used_clean.add(clean_tid)
        cross_pairs.append(
            {
                "cap_trajectory_id": pair["trajectory_id"],
                "clean_trajectory_id": clean_tid,
                "slot_norm": pair["slot_norm"],
                "pairing_method": "slot_norm_match",
            }
        )

    clean_pool = [cr["trajectory_id"] for cr in clean_rows if cr["trajectory_id"] not in used_clean]
    if not clean_pool:
        clean_pool = [cr["trajectory_id"] for cr in clean_rows]
    for i, pair in enumerate(unmatched_caps):
        clean_tid = clean_pool[i % len(clean_pool)]
        cross_pairs.append(
            {
                "cap_trajectory_id": pair["trajectory_id"],
                "clean_trajectory_id": clean_tid,
                "slot_norm": pair["slot_norm"],
                "pairing_method": "cohort_index_fallback",
            }
        )

    return {
        "cap_pairs": pairs,
        "matched_clean_original": clean_rows,
        "pm_clean_cross_pairs": cross_pairs,
        "n_cap_valid": len(pairs),
        "n_cross_pairs": len(cross_pairs),
    }


def cem_valid_pair_ids(*, include_matched_clean: bool = True) -> dict[str, Any]:
    """CEM track: rival_value_swap pairs. Often n<32 — must disclose in reports."""
    cohort = load_json(COHORT_MANIFEST_JSON)
    cem_ids = list(cohort.get("cohorts", {}).get("CEM", []))
    rows = load_trajectory_index()
    scorable_rival = set()
    for row in load_jsonl(P1_DIR / "source_effects_rows.jsonl"):
        if row.get("cohort") != "CEM" or row.get("condition_id") != "rival_value_swap":
            continue
        if row.get("scorable"):
            scorable_rival.add(str(row.get("trajectory_id")))

    pairs: list[dict[str, Any]] = []
    for tid in cem_ids:
        if tid not in scorable_rival:
            continue
        traj = rows.get(tid)
        if not traj:
            continue
        cem_claim = next(
            (c for c in (traj.get("claims") or []) if c.get("legacy_label") == "cross_object_merge"),
            None,
        )
        if not cem_claim:
            continue
        value = str(cem_claim.get("value") or "")
        pairs.append(
            {
                "trajectory_id": tid,
                "cohort": "CEM",
                "slot_norm": str(cem_claim.get("slot_norm") or ""),
                "claim_value": value,
                "cf_value": f"CF_{value}",
                "conditions": ["original", "rival_value_swap", "position_permutation"],
            }
        )

    base = cap_valid_pair_ids(include_matched_clean=include_matched_clean)
    cross_pairs = [
        {
            "pm_trajectory_id": p["trajectory_id"],
            "clean_trajectory_id": cp["clean_trajectory_id"],
            "slot_norm": p.get("slot_norm"),
            "pairing_method": cp.get("pairing_method"),
            "track": "CEM",
        }
        for p, cp in zip(pairs, base["pm_clean_cross_pairs"])
    ]
    if len(cross_pairs) < len(pairs):
        clean_rows = base.get("matched_clean_original") or []
        pool = [c["trajectory_id"] for c in clean_rows]
        for i, p in enumerate(pairs[len(cross_pairs) :]):
            if not pool:
                break
            cross_pairs.append(
                {
                    "pm_trajectory_id": p["trajectory_id"],
                    "clean_trajectory_id": pool[i % len(pool)],
                    "slot_norm": p.get("slot_norm"),
                    "pairing_method": "cohort_index_fallback",
                    "track": "CEM",
                }
            )

    return {
        "cem_pairs": pairs,
        "matched_clean_original": base.get("matched_clean_original", []),
        "pm_clean_cross_pairs": cross_pairs,
        "n_cem_valid": len(pairs),
        "n_cross_pairs": len(cross_pairs),
        "below_32_threshold": len(pairs) < 32,
        "below_32_reason": "P1 rival_value_swap scorable n=10; cohort pool 43/64",
    }


def select_track_pairs(track: TrackName, *, include_matched_clean: bool = True) -> dict[str, Any]:
    if track == "CEM":
        out = cem_valid_pair_ids(include_matched_clean=include_matched_clean)
        out["track"] = "CEM"
        out["pairs"] = out["cem_pairs"]
        return out
    out = cap_valid_pair_ids(include_matched_clean=include_matched_clean)
    out["track"] = "CAP"
    out["pairs"] = out["cap_pairs"]
    out["below_32_threshold"] = False
    return out


def commitment_mismatch_pairs() -> dict[str, Any]:
    """Commitment-State Restoration cohort (expert_recorrect §3.2)."""
    cohort = load_json(COHORT_MANIFEST_JSON)
    cem_ids = set(cohort.get("cohorts", {}).get("CEM", []))
    rows = load_trajectory_index()
    mismatch: list[dict[str, Any]] = []
    consistent: list[dict[str, Any]] = []
    for tid in cem_ids:
        traj = rows.get(tid)
        if not traj:
            continue
        rel = (traj.get("commitment") or {}).get("commitment_relation")
        slot = _query_slot_from_trajectory(traj)
        rec = {"trajectory_id": tid, "slot_norm": slot, "commitment_relation": rel}
        if rel == "mismatch":
            mismatch.append(rec)
        elif rel == "consistent":
            consistent.append(rec)
    consistent_by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in consistent:
        consistent_by_slot[str(row.get("slot_norm") or "unknown")].append(row)

    pairs: list[dict[str, Any]] = []
    unmatched: list[str] = []
    slot_cursor: dict[str, int] = defaultdict(int)
    for m in mismatch:
        slot = str(m.get("slot_norm") or "unknown")
        pool = consistent_by_slot.get(slot) or []
        if not pool:
            unmatched.append(m["trajectory_id"])
            continue
        idx = slot_cursor[slot] % len(pool)
        slot_cursor[slot] += 1
        donor = pool[idx]
        pairs.append(
            {
                "recipient_trajectory_id": m["trajectory_id"],
                "donor_trajectory_id": donor["trajectory_id"],
                "slot_norm": slot,
                "donor_slot_norm": str(donor.get("slot_norm") or slot),
                "intervention": "commitment_state_restoration",
            }
        )
    return {
        "n_mismatch": len(mismatch),
        "n_consistent": len(consistent),
        "n_pairs_unmatched_slot": len(unmatched),
        "unmatched_recipient_ids": unmatched,
        "restoration_pairs": pairs,
    }


def _normalize_slot(slot: str) -> str:
    return str(slot or "unknown").strip().lower()


def _query_slot_from_trajectory(traj: dict[str, Any]) -> str:
    for ev in traj.get("evidence") or []:
        if ev.get("scope") == "query_constraint" and ev.get("slot_norm"):
            return _normalize_slot(str(ev["slot_norm"]))
    for claim in traj.get("claims") or []:
        if claim.get("legacy_label") == "constraint_projection" and claim.get("slot_norm"):
            return _normalize_slot(str(claim["slot_norm"]))
    user = str((traj.get("messages_final_call") or [{}])[-1].get("content") or "")
    import re

    m = re.search(r"Hard user requirement:\s*([^.]+)", user, re.I)
    if m:
        return _normalize_slot(m.group(1).split()[0])
    return "unknown"


def _primary_slot(traj: dict[str, Any]) -> str:
    return _query_slot_from_trajectory(traj)


def messages_for_condition(
    trajectory: dict[str, Any],
    condition_id: str,
    *,
    track: TrackName = "CAP",
) -> list[dict[str, str]] | None:
    if condition_id == "original":
        return list(trajectory.get("messages_final_call") or [])
    bundle_fn = build_cap_bundle if track == "CAP" else build_cem_bundle
    for bundle_name, op_fn, kwargs in bundle_fn(trajectory):
        if bundle_name == condition_id and op_fn is not None:
            result = op_fn(trajectory["messages_final_call"], **kwargs)
            if result.invalid_reason or result.semantic_validation == "invalid":
                return None
            return list(result.messages)
    return None
