"""B0–B4 / M0 repair baselines (§7)."""
from __future__ import annotations

import copy
import json
import re
from typing import Any

ANCHOR_PROMPT = (
    "\n\n[CCER-B1] Only assert attributes of the Selected product that are directly "
    "supported by tool observations visible in this dialogue. If unsupported, state "
    "that the attribute is not confirmed by current evidence."
)

M0_FORMAT_INSTRUCTION = (
    "\n\n[CCER-M0] Respond using structured sections: Evidence Plan (slot table), "
    "Selected product ID, About (supported facts only), Compared. Do not add unsupported claims."
)


def b0_original(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return copy.deepcopy(messages)


def b1_anchor_prompt(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out = copy.deepcopy(messages)
    for m in out:
        if m.get("role") == "system":
            m["content"] = str(m.get("content") or "") + ANCHOR_PROMPT
    return out


def b2_evidence_plan(messages: list[dict[str, str]], evidence_plan: dict[str, Any]) -> list[dict[str, str]]:
    out = copy.deepcopy(messages)
    plan_text = "### Anchor Evidence Plan (E_seen only)\n" + json.dumps(evidence_plan, ensure_ascii=False, indent=2)
    for m in out:
        if m.get("role") == "user":
            m["content"] = plan_text + "\n\n" + str(m.get("content") or "")
    return out


def b3_anchor_only(messages: list[dict[str, str]], *, anchor_pid: str | None) -> list[dict[str, str]]:
    out = copy.deepcopy(messages)
    user = ""
    for m in out:
        if m.get("role") == "user":
            user = str(m.get("content") or "")
    if anchor_pid:
        blocks = re.split(r"(?=<obs>)", user)
        kept = [blocks[0]] if blocks else [user]
        for b in blocks[1:]:
            if anchor_pid in b[:500]:
                kept.append(b)
        new_user = "".join(kept)
        for m in out:
            if m.get("role") == "user":
                m["content"] = (
                    f"[B3 anchor-only disclosure: comparison evidence for non-{anchor_pid} removed]\n"
                    + new_user
                )
    return out


def b4_verifier_repair_prompt(messages: list[dict[str, str]], *, issues: list[str]) -> list[dict[str, str]]:
    out = copy.deepcopy(messages)
    fix = "\n".join(f"- {i}" for i in issues)
    for m in out:
        if m.get("role") == "system":
            m["content"] = str(m.get("content") or "") + (
                f"\n\n[CCER-B4 verifier] Fix these unsupported claims (max 1 retry):\n{fix}"
            )
    return out


def m0_format_only(messages: list[dict[str, str]], evidence_plan: dict[str, Any]) -> list[dict[str, str]]:
    out = b2_evidence_plan(messages, evidence_plan)
    for m in out:
        if m.get("role") == "system":
            m["content"] = str(m.get("content") or "") + M0_FORMAT_INSTRUCTION
    return out


def build_evidence_plan_from_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Anchor Evidence Plan from E_seen only — no hidden catalog."""
    anchor = (trajectory.get("commitment") or {}).get("action_anchor")
    slots: dict[str, Any] = {}
    for ev in trajectory.get("evidence") or []:
        if ev.get("entity_role") == "anchor" or (anchor and anchor in (ev.get("entity_ids") or [])):
            slot = ev.get("slot_norm") or "unknown"
            slots[slot] = {"status": "supported", "value": ev.get("value_norm"), "source": ev.get("source_id")}
    for c in trajectory.get("claims") or []:
        slot = c.get("slot_norm") or "unknown"
        if slot not in slots:
            slots[slot] = {"status": "unknown", "value": None, "note": "no anchor evidence in E_seen"}
    return {"anchor_pid": anchor, "slots": slots}
