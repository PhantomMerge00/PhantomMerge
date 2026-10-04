"""Line L+ Phase 4: PM-type conditional rewrite routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ccer.mechanism.line_l_anchor_extract import detect_donor_pollution, lookup_obs_tau_slot

OnMiss = Literal["delete", "copy", "rarr", "cove", "skip"]


@dataclass(frozen=True)
class StrategySpec:
    pm_type: str
    retrieve_pids: tuple[str, ...]
    allow_llm: bool
    allow_copy: bool
    on_miss: OnMiss
    skip_rewrite: bool
    reason: str | None = None
    allow_snippet_fallback: bool = False


def _donor_pids(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for d in row.get("donor_owners") or []:
        if isinstance(d, dict):
            pid = str(d.get("pid") or "").strip()
        else:
            pid = str(d).strip()
        if pid:
            out.append(pid)
    return out


def route_rewrite_strategy(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    anchor_pid: str,
) -> StrategySpec:
    verdict = str(row.get("gold_verdict") or "")
    slot_norm = str(row.get("slot_norm") or "").strip()

    if not slot_norm:
        return StrategySpec(
            pm_type="no_slot",
            retrieve_pids=(anchor_pid,),
            allow_llm=False,
            allow_copy=False,
            on_miss="skip",
            skip_rewrite=True,
            reason="no_slot_norm",
        )

    if verdict == "trajectory_clean":
        return StrategySpec(
            pm_type="clean_expansion",
            retrieve_pids=(anchor_pid,),
            allow_llm=False,
            allow_copy=False,
            on_miss="skip",
            skip_rewrite=True,
            reason="clean_expansion_no_rewrite",
        )

    if verdict == "cross_object_merge":
        donors = _donor_pids(row)
        pids = tuple([anchor_pid] + [d for d in donors if d != anchor_pid])
        return StrategySpec(
            pm_type="CEM",
            retrieve_pids=pids,
            allow_llm=False,
            allow_copy=True,
            on_miss="delete",
            skip_rewrite=False,
        )

    if verdict == "constraint_projection":
        return StrategySpec(
            pm_type="CAP",
            retrieve_pids=(anchor_pid,),
            allow_llm=False,
            allow_copy=True,
            on_miss="delete",
            skip_rewrite=False,
            reason="anchor_only_shell_delete_on_miss",
        )

    if verdict == "anchored_hallucination":
        return StrategySpec(
            pm_type="AH",
            retrieve_pids=(anchor_pid,),
            allow_llm=True,
            allow_copy=True,
            on_miss="rarr",
            skip_rewrite=False,
        )

    return StrategySpec(
        pm_type="other",
        retrieve_pids=(anchor_pid,),
        allow_llm=False,
        allow_copy=True,
        on_miss="delete",
        skip_rewrite=False,
    )


def route_rewrite_strategy_runtime(
    row: dict[str, Any],
    traj: dict[str, Any],
    acceptance_pid: str,
) -> StrategySpec:
    """Structure-only PM routing (no gold_verdict) for deployment rewrite stack."""
    slot_norm = str(row.get("slot_norm") or "").strip()
    anchor_pid = acceptance_pid

    if not slot_norm:
        return StrategySpec(
            pm_type="no_slot",
            retrieve_pids=(anchor_pid,),
            allow_llm=False,
            allow_copy=False,
            on_miss="skip",
            skip_rewrite=True,
            reason="no_slot_norm",
        )

    if detect_donor_pollution(row, traj, acceptance_pid, slot_norm):
        donors = _donor_pids(row)
        pids = tuple([anchor_pid] + [d for d in donors if d != anchor_pid])
        return StrategySpec(
            pm_type="CEM",
            retrieve_pids=pids,
            allow_llm=False,
            allow_copy=True,
            on_miss="delete",
            skip_rewrite=False,
        )

    obs = lookup_obs_tau_slot(traj, anchor_pid=acceptance_pid, slot_norm=slot_norm)
    if obs.branch == "extraction_miss" and obs.miss_reason in (
        "obs_tau_slot_missing",
        "empty_obs_tau",
    ):
        return StrategySpec(
            pm_type="CAP",
            retrieve_pids=(anchor_pid,),
            allow_llm=False,
            allow_copy=False,
            on_miss="delete",
            skip_rewrite=False,
            reason="anchor_only_shell_delete_on_miss",
        )

    claim = str(row.get("claim_value") or "").strip()
    if claim and len(claim) > 60:
        return StrategySpec(
            pm_type="AH",
            retrieve_pids=(anchor_pid,),
            allow_llm=True,
            allow_copy=True,
            on_miss="rarr",
            skip_rewrite=False,
        )

    return StrategySpec(
        pm_type="other",
        retrieve_pids=(anchor_pid,),
        allow_llm=False,
        allow_copy=True,
        on_miss="delete",
        skip_rewrite=False,
    )
