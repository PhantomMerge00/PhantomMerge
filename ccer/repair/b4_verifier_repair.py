"""B4 bounded verifier repair using evidence plan."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ccer.repair.baselines import b4_verifier_repair_prompt, build_evidence_plan_from_trajectory
from ccer.replay.vllm_replay import vllm_generate

ROOT = Path("${PHANTOM_MERGE_ROOT}")

def _simple_unsupported_issues(trajectory: dict[str, Any], answer: str) -> list[str]:
    plan = build_evidence_plan_from_trajectory(trajectory)
    issues: list[str] = []
    for c in trajectory.get("claims") or []:
        slot = c.get("slot_norm") or ""
        val = str(c.get("value") or "")
        if val and val.lower() in answer.lower():
            st = (plan.get("slots") or {}).get(slot, {}).get("status")
            if st != "supported" and c.get("legacy_label") in ("constraint_projection", "cross_object_merge"):
                issues.append(f"{slot}={val} lacks anchor support in E_seen")
    return issues[:5]


def run_b4(
    trajectory: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    max_retries: int = 2,
) -> dict[str, Any]:
    retries = 0
    current_messages = messages
    history: list[dict[str, Any]] = []
    while retries <= max_retries:
        text, meta = vllm_generate(current_messages)
        issues = _simple_unsupported_issues(trajectory, text)
        history.append({"retry": retries, "issues": issues, "latency": meta})
        if not issues or retries >= max_retries:
            return {"answer": text, "retries": retries, "history": history}
        current_messages = b4_verifier_repair_prompt(messages, issues=issues)
        retries += 1
    return {"answer": text, "retries": retries, "history": history}
