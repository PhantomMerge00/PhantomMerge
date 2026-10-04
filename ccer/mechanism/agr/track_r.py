"""AGR Track R: routing oracle consistency (Proposition-content)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

PmType = Literal["CEM", "CAP", "ASF", "clean"]


@dataclass
class SyntheticClaim:
    claim_id: str
    pm_type: PmType
    is_pm: bool
    anchor_has_value: bool
    extraction_hit: bool
    v_anchor: str | None
    claim_value: str
    is_clean_bullet: bool


def oracle_rho(claim: SyntheticClaim) -> float:
    """rho(c) = 1[src(c) != a_tau] approximated by gold PM label."""
    return 1.0 if claim.is_pm else 0.0


def route_claim(
    claim: SyntheticClaim,
    *,
    rho: float,
    theta: float = 0.5,
) -> Literal["keep", "delete", "rewrite"]:
    """Section 3 extraction+routing: flagged claims attempt extractive rewrite else delete."""
    if rho < theta:
        return "keep"
    if claim.extraction_hit and claim.v_anchor:
        return "rewrite"
    return "delete"


def outcome_is_pm_after_routing(claim: SyntheticClaim, action: str) -> bool:
    if action == "keep":
        return claim.is_pm
    if action == "delete":
        return False
    # rewrite: claim replaced with anchor literal — oracle extraction is correct
    return False


def run_oracle_routing_batch(claims: list[SyntheticClaim]) -> dict[str, Any]:
    n_pm_in = sum(1 for c in claims if c.is_pm)
    n_cb_in = sum(1 for c in claims if c.is_clean_bullet)
    n_pm_out = 0
    n_cb_kept = 0
    for c in claims:
        rho = oracle_rho(c)
        action = route_claim(c, rho=rho)
        if outcome_is_pm_after_routing(c, action):
            n_pm_out += 1
        if c.is_clean_bullet and action == "keep":
            n_cb_kept += 1
    pm_rate = n_pm_out / max(len(claims), 1)
    cb_retention = n_cb_kept / max(n_cb_in, 1)
    return {
        "n_claims": len(claims),
        "n_pm_in": n_pm_in,
        "n_cb_in": n_cb_in,
        "pm_rate": pm_rate,
        "cb_retention": cb_retention,
    }


def default_synthetic_suite() -> list[SyntheticClaim]:
    """CEM/CAP/ASF + clean; extraction hit/miss variants."""
    out: list[SyntheticClaim] = []
    idx = 0
    for pm_type, is_pm in [("CEM", True), ("CAP", True), ("ASF", True), ("clean", False)]:
        for anchor_has, hit in [(True, True), (True, False), (False, False)]:
            idx += 1
            v = "anchor_val" if hit else None
            cv = "wrong" if is_pm else "anchor_val"
            out.append(
                SyntheticClaim(
                    claim_id=f"syn_{idx}",
                    pm_type=pm_type,
                    is_pm=is_pm,
                    anchor_has_value=anchor_has,
                    extraction_hit=hit and anchor_has,
                    v_anchor=v,
                    claim_value=cv,
                    is_clean_bullet=not is_pm,
                )
            )
    return out
