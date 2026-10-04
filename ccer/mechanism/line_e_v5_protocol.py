"""Line E v5 extreme in-generation mitigation protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ccer.mechanism.line_e_protocol import (
    LINE_E_V3_PRIMARY_POSITION,
    line_e_layer_for_position,
    line_e_steering_apply_for_position,
)

LINE_E_V5_PROTOCOL_VERSION = "line_e_v5_extreme_in_generation"

# B-style interchange arms (fixed α=1.0; strongest in-generation signal from Line B v3).
LINE_E_V5_INTERCHANGE_ARMS: tuple[tuple[str, str, str, tuple[float, ...]], ...] = (
    ("pca_target_interchange", "target_interchange", "interchange", (1.0,)),
    ("full_vector_target", "target_interchange", "full_vector", (1.0,)),
    ("wrong_owner_donor", "wrong_owner_donor", "interchange", (1.0,)),
)

# Probe-aligned hidden steering + owner-guided decode logit bias.
LINE_E_V5_STEER_ARMS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("probe_steer", (0.5, 1.0, 1.5)),
    ("anchor_logit_bias", (0.5, 1.0, 1.5)),
)

LINE_E_V5_ALPHAS = [0.0, 0.5, 1.0, 1.5]
LINE_E_V5_PCA_RANK = 16
LINE_E_V5_PROBE_POOL = "mechanism_research"
LINE_E_V5_ANCHOR_BIAS_PER_TOKEN = 3.0


@dataclass(frozen=True)
class LineEV5Config:
    protocol_version: str
    layer: int
    position: str
    steering_apply: str
    pca_rank: int
    interchange_arms: tuple[tuple[str, str, str, tuple[float, ...]], ...]
    steer_arms: tuple[tuple[str, tuple[float, ...]], ...]
    anchor_bias_per_token: float

    @property
    def schema_tag(self) -> str:
        return "v5_extreme"


def line_e_v5_config(
    *,
    position: str = LINE_E_V3_PRIMARY_POSITION,
    claim_layer: int | None = None,
    pca_rank: int = LINE_E_V5_PCA_RANK,
) -> LineEV5Config:
    layer = line_e_layer_for_position(position, claim_layer=claim_layer or 49)
    return LineEV5Config(
        protocol_version=LINE_E_V5_PROTOCOL_VERSION,
        layer=layer,
        position=position,
        steering_apply=line_e_steering_apply_for_position(position),
        pca_rank=pca_rank,
        interchange_arms=LINE_E_V5_INTERCHANGE_ARMS,
        steer_arms=LINE_E_V5_STEER_ARMS,
        anchor_bias_per_token=LINE_E_V5_ANCHOR_BIAS_PER_TOKEN,
    )


def v5_control_ids(cfg: LineEV5Config | None = None) -> tuple[str, ...]:
    cfg = cfg or line_e_v5_config()
    ids: list[str] = [arm[0] for arm in cfg.interchange_arms]
    ids.extend(arm[0] for arm in cfg.steer_arms)
    return tuple(ids)
