"""Line E v4 ultimate protocol: claim_onset mitigation without silent fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.line_e_donor import (
    build_anchor_aligned_bundle,
    build_paired_cross_bundle,
    build_slot_consistent_bundle,
    list_missing_bilateral_pairs,
    trajectory_commitment_stratum,
    ultimate_eligible_trajectory,
)
from ccer.mechanism.steering_vector import (
    build_paired_steering_controls,
    mechanism_research_pm_with_activations,
    paired_clean_trajectory_id,
)
from ccer.replay.live_position import (
    LINE_D_CLAIM_LAYER_DEFAULT,
    intervention_steering_apply,
    line_d_layer_for_position,
    pair_has_symmetric_claim_anchor,
)

# v4 ultimate supersedes v3; keep alias for downstream imports.
LINE_E_V3_PROTOCOL_VERSION = "line_e_v4_ultimate_mitigation"
LINE_E_ULTIMATE_PROTOCOL_VERSION = LINE_E_V3_PROTOCOL_VERSION

LINE_E_V3_PRIMARY_POSITION = "claim_onset"
LINE_E_V3_SECONDARY_POSITION = "pre_value"
LINE_E_V3_LEGACY_POSITION = "commitment"
LINE_E_ULTIMATE_POSITIONS: tuple[str, ...] = ("claim_onset", "pre_value")

LINE_E_V3_STEERING_DIRECTION = "subtract"
LINE_E_V3_VECTOR_SEMANTICS = "pm_minus_donor_subtract_toward_donor"

# Ordered by mechanistic specificity (R2): slot/anchor donors before cross-pair CAA.
LINE_E_ULTIMATE_CONTROLS: tuple[str, ...] = (
    "slot_consistent_restore",
    "anchor_aligned_restore",
    "donor_restore",
    "paired_caa",
    "caa_random",
    "caa_orthogonal",
)
LINE_E_V3_CONTROLS = LINE_E_ULTIMATE_CONTROLS

LINE_E_V3_ALPHA_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
LINE_E_PROBE_ALPHAS = (0.5, 1.0, 1.5)
LINE_E_MECHANISM_OBS_ALPHAS = (1.0,)


@dataclass(frozen=True)
class LineESteeringConfig:
    protocol_version: str
    layer: int
    position: str
    steering_apply: str
    steering_direction: str
    control_ids: tuple[str, ...]
    vector_semantics: str
    use_live_position_resolver: bool
    baseline_control: str
    strict_bilateral_gate: bool
    allow_legacy_fallback: bool

    @property
    def schema_tag(self) -> str:
        return "v4_ultimate"


def line_e_layer_for_position(position: str, *, claim_layer: int = LINE_D_CLAIM_LAYER_DEFAULT) -> int:
    return line_d_layer_for_position(position, claim_layer=claim_layer)


def line_e_steering_apply_for_position(position: str) -> str:
    return intervention_steering_apply(position)


def line_e_v3_config(
    *,
    position: str = LINE_E_V3_PRIMARY_POSITION,
    claim_layer: int = LINE_D_CLAIM_LAYER_DEFAULT,
    strict_bilateral_gate: bool = True,
    allow_legacy_fallback: bool = False,
) -> LineESteeringConfig:
    return LineESteeringConfig(
        protocol_version=LINE_E_ULTIMATE_PROTOCOL_VERSION,
        layer=line_e_layer_for_position(position, claim_layer=claim_layer),
        position=position,
        steering_apply=line_e_steering_apply_for_position(position),
        steering_direction=LINE_E_V3_STEERING_DIRECTION,
        control_ids=LINE_E_ULTIMATE_CONTROLS,
        vector_semantics=LINE_E_V3_VECTOR_SEMANTICS,
        use_live_position_resolver=position in LINE_E_ULTIMATE_POSITIONS,
        baseline_control="paired_caa",
        strict_bilateral_gate=strict_bilateral_gate,
        allow_legacy_fallback=allow_legacy_fallback,
    )


def line_e_ultimate_config(**kwargs: Any) -> LineESteeringConfig:
    return line_e_v3_config(**kwargs)


def trajectory_has_position_activation(
    trajectory_id: str,
    *,
    layer: int,
    position: str,
) -> bool:
    path = activation_path(trajectory_id, "original")
    if not path.is_file():
        return False
    loaded = load_activation_npz(path)
    if get_vector(loaded, position=position, layer=layer) is None:
        return False
    stored_idx = (loaded.get("positions") or {}).get(position)
    return stored_idx is not None and int(stored_idx) >= 0


def line_e_v3_eligible_trajectory(traj: dict[str, Any], *, layer: int, position: str) -> bool:
    ok, _ = ultimate_eligible_trajectory(traj, layer=layer, position=position)
    return ok


def filter_v3_eval_trajectories(
    trajectory_ids: list[str],
    rows_index: dict[str, dict[str, Any]],
    *,
    layer: int,
    position: str,
) -> list[str]:
    out: list[str] = []
    for tid in trajectory_ids:
        traj = rows_index.get(tid)
        if not traj:
            continue
        traj = dict(traj)
        traj.setdefault("trajectory_id", tid)
        if line_e_v3_eligible_trajectory(traj, layer=layer, position=position):
            out.append(tid)
    return sorted(out)


def audit_v3_pool_coverage(
    pm_trajectory_ids: list[str],
    *,
    layer: int,
    position: str,
) -> dict[str, Any]:
    missing = list_missing_bilateral_pairs(pm_trajectory_ids, layer=layer, position=position)
    eligible = [tid for tid in pm_trajectory_ids if tid not in {m["pm_trajectory_id"] for m in missing}]
    return {
        "n_pm_total": len(pm_trajectory_ids),
        "n_eligible": len(eligible),
        "n_excluded": len(missing),
        "missing_bilateral_pairs": missing,
        "eligible_pm_trajectory_ids": sorted(eligible),
    }


def mechanism_research_pm_v3_pool(*, position: str = LINE_E_V3_PRIMARY_POSITION) -> list[str]:
    layer = line_e_layer_for_position(position)
    raw = mechanism_research_pm_with_activations(layer=layer, position=position)
    audit = audit_v3_pool_coverage(raw, layer=layer, position=position)
    return audit["eligible_pm_trajectory_ids"]


def build_trajectory_steering_bundle(
    pm_trajectory_id: str,
    *,
    layer: int,
    position: str,
    seed: int = 0,
    allow_legacy_fallback: bool = False,
) -> dict[str, Any]:
    """Per-trajectory vectors for Line E v4 ultimate — no silent commitment fallback."""
    paired_cross = build_paired_cross_bundle(pm_trajectory_id, layer=layer, position=position)
    if paired_cross.get("error"):
        if allow_legacy_fallback and position in LINE_E_ULTIMATE_POSITIONS:
            from ccer.mechanism.steering_vector import paired_activation_vectors

            fallback_layer = line_e_layer_for_position(LINE_E_V3_LEGACY_POSITION)
            paired_cross = paired_activation_vectors(
                pm_trajectory_id,
                layer=fallback_layer,
                position=LINE_E_V3_LEGACY_POSITION,
            )
            if paired_cross.get("error"):
                return paired_cross
            paired_cross["vector_transfer"] = (
                f"{LINE_E_V3_LEGACY_POSITION}_L{fallback_layer}_to_{position}_L{layer}"
            )
            paired_cross["donor_kind"] = "legacy_fallback"
        else:
            return paired_cross

    pm_vec = paired_cross["pm_vec"]
    clean_vec = paired_cross["clean_vec"]
    caa_vec = paired_cross["paired_caa_vector"]
    controls = build_paired_steering_controls(caa_vec, seed=seed + hash(pm_trajectory_id) % 10000)

    slot_bundle = build_slot_consistent_bundle(pm_trajectory_id, layer=layer, position=position)
    anchor_bundle = build_anchor_aligned_bundle(pm_trajectory_id, layer=layer, position=position)

    from ccer.mechanism.pair_select import load_trajectory_index

    traj = load_trajectory_index().get(pm_trajectory_id) or {}
    stratum = trajectory_commitment_stratum(traj)

    return {
        **paired_cross,
        "pm_trajectory_id": pm_trajectory_id,
        "clean_trajectory_id": paired_clean_trajectory_id(pm_trajectory_id),
        "mitigation_vector": caa_vec,
        "mitigation_direction": LINE_E_V3_STEERING_DIRECTION,
        "controls": controls,
        "donor_restore": {"pm_vec": pm_vec, "clean_vec": clean_vec},
        "slot_consistent_restore": (
            {
                "pm_vec": slot_bundle["pm_vec"],
                "donor_vec": slot_bundle["donor_vec"],
                "donor_trajectory_id": slot_bundle["donor_trajectory_id"],
            }
            if not slot_bundle.get("error")
            else None
        ),
        "anchor_aligned_restore": (
            {
                "pm_vec": anchor_bundle["pm_vec"],
                "donor_vec": anchor_bundle["donor_vec"],
                "donor_trajectory_id": anchor_bundle["donor_trajectory_id"],
            }
            if not anchor_bundle.get("error")
            else None
        ),
        "commitment_stratum": stratum,
        "vector_transfer": paired_cross.get("vector_transfer"),
        "slot_consistent_error": slot_bundle.get("error"),
        "anchor_aligned_error": anchor_bundle.get("error"),
        "strict_bilateral_gate": not allow_legacy_fallback,
    }
