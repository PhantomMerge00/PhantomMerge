"""Load cem_ah_strict_v2.1 adjudication and resolve mechanism swap targets."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ccer.counterfactual.base import find_pid_blocks, find_product_record_snippets
from ccer.paths import ROOT

ADJUDICATION_JSONL = (
    ROOT / "results/reports/manual_verification/cem_ah_final_adjudication.jsonl"
)


def _user_content(traj: dict[str, Any]) -> str:
    for m in traj.get("messages_final_call") or []:
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def _value_in_text(text: str, value: str) -> bool:
    if not value or not text:
        return False
    if value in text:
        return True
    vl = value.strip().lower()
    return bool(vl and vl in text.lower())


def _value_in_block(user: str, pid: str, value: str) -> bool:
    if not value or not pid:
        return False
    for _, _, snippet in find_pid_blocks(user, pid):
        if value in snippet:
            return True
        vl = value.strip().lower()
        if vl and vl in snippet.lower():
            return True
    return False


def trajectory_status(instance_statuses: list[str], cohort: str) -> str:
    tag = f"confirmed_{cohort}"
    if tag in instance_statuses:
        return "confirmed"
    if "uncertain" in instance_statuses:
        return "uncertain"
    return "rejected"


def _clean_slot_value(value: str, response_quote: str = "") -> str:
    """Normalize adjudication value to prompt/answer surface form."""
    if response_quote and ":" in response_quote:
        return response_quote.split(":", 1)[1].strip()
    core = re.split(r"\s*\(", value, maxsplit=1)[0].strip()
    return core or value.strip()


def _ah_claim_surface_value(row: dict[str, Any]) -> str:
    """Prefer canonical slot value when response_quote is a long model paraphrase."""
    value = str(row.get("value") or "").strip()
    quote = str(row.get("response_quote") or "")
    if quote and ":" in quote:
        after = quote.split(":", 1)[1].strip()
        if value and len(after) > max(len(value) * 3, 40):
            return value
        return after
    return value or quote


def _strip_leading_symbols(text: str) -> str:
    return re.sub(r"^[\s✅✔✨•\-]+", "", text).strip()


_SLOT_HINTS: dict[str, tuple[str, ...]] = {
    "material": ("material", "fabric", "canvas", "cotton", "plastic", "glass", "rubber", "leather", "metal", "wood"),
    "color": ("color", "black", "white", "red", "gold", "silver", "blue", "green", "pink", "beige"),
}


def _infer_ah_incompatible_from_evidence(row: dict[str, Any], anchor_block: str) -> str | None:
    """Fallback when claim value is absent from anchor block (model paraphrase AH)."""
    slot = str(row.get("slot_norm") or row.get("slot") or "").lower()
    ev = str(row.get("anchor_evidence_quote") or "")
    parts: list[str] = []
    if ev.startswith("{"):
        try:
            obj = json.loads(ev)
            title = str(obj.get("title") or "")
            parts.extend(title.split(";"))
        except json.JSONDecodeError:
            pass
    best: str | None = None
    best_score = -10**9
    for part in parts:
        core = _strip_leading_symbols(part)
        if len(core) < 4 or not _value_in_text(anchor_block, core):
            continue
        score = len(core)
        if slot in _SLOT_HINTS:
            if any(h in core.lower() for h in _SLOT_HINTS[slot]):
                score += 120
        if any(x in core.lower() for x in ("size chart", " cm", "measurement", "mistake of customers")):
            score -= 200
        if score > best_score:
            best_score = score
            best = core
    return best


def _pick_anchor_swap_value(row: dict[str, Any], anchor_block: str) -> str | None:
    """Pick a value present in anchor block to swap (AH incompatible surface form)."""
    for cand in row.get("anchor_incompatible_value") or []:
        cv = str(cand).strip()
        if cv and _value_in_text(anchor_block, cv):
            return cv
    claim = _ah_claim_surface_value(row)
    if claim and _value_in_text(anchor_block, claim):
        return claim
    inferred = _infer_ah_incompatible_from_evidence(row, anchor_block)
    if inferred:
        return inferred
    slot = str(row.get("slot_norm") or row.get("slot") or "")
    if slot:
        m = re.search(rf"{re.escape(slot)}\s*:\s*([^;,\n]+)", anchor_block, re.I)
        if m:
            return m.group(1).strip()
    return None


def resolve_donor_pid(row: dict[str, Any], *, user: str | None = None) -> str | None:
    """Pick causal donor PID for a confirmed_CEM instance."""
    donors = [str(d.get("pid") or "") for d in (row.get("donor_owners") or []) if d.get("pid")]
    donors = [d for d in donors if d]
    textual = str(row.get("textual_selected_pid") or "")
    value = str(row.get("value") or "")
    owner = str(row.get("owner_uniqueness") or "")

    if owner in {"resolved_by_textual_anchor", "resolved_by_compound_copy"} and textual:
        if textual in donors or not donors:
            return textual or None
    if textual and textual in donors:
        return textual
    if user and value:
        for pid in donors:
            if _value_in_block(user, pid, value):
                return pid
    if owner == "unique" and len(donors) == 1:
        return donors[0]
    if donors:
        return donors[0]
    return textual or None


@dataclass
class CemSwapTarget:
    trajectory_id: str
    claim_value: str
    slot_norm: str
    donor_pid: str
    committed_anchor_pid: str
    textual_selected_pid: str
    instance_audit_key: str
    owner_uniqueness: str
    response_quote: str


@dataclass
class AhSwapTarget:
    trajectory_id: str
    claim_value: str
    slot_norm: str
    anchor_pid: str
    instance_audit_key: str
    response_quote: str


@dataclass
class AdjudicationIndex:
    rows: list[dict[str, Any]] = field(default_factory=list)
    by_trajectory: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    cem_trajectory_status: dict[str, str] = field(default_factory=dict)
    ah_trajectory_status: dict[str, str] = field(default_factory=dict)

    def confirmed_cem_instances(self, trajectory_id: str) -> list[dict[str, Any]]:
        return [
            r
            for r in self.by_trajectory.get(trajectory_id, [])
            if r.get("adjudication_status") == "confirmed_CEM"
        ]

    def confirmed_ah_instances(self, trajectory_id: str) -> list[dict[str, Any]]:
        return [
            r
            for r in self.by_trajectory.get(trajectory_id, [])
            if r.get("adjudication_status") == "confirmed_AH"
        ]

    def dev_confirmed_cem_trajectories(self) -> list[str]:
        out: list[str] = []
        for tid, status in self.cem_trajectory_status.items():
            if status != "confirmed":
                continue
            rows = self.by_trajectory.get(tid, [])
            if rows and str(rows[0].get("split") or "") == "dev":
                out.append(tid)
        return sorted(out)

    def dev_confirmed_ah_trajectories(self) -> list[str]:
        out: list[str] = []
        for tid, status in self.ah_trajectory_status.items():
            if status != "confirmed":
                continue
            rows = self.by_trajectory.get(tid, [])
            if rows and str(rows[0].get("split") or "") == "dev":
                out.append(tid)
        return sorted(out)

    def pick_primary_cem_instance(self, trajectory_id: str) -> dict[str, Any] | None:
        confirmed = self.confirmed_cem_instances(trajectory_id)
        if not confirmed:
            return None
        # Prefer commitment mismatch + textual resolution
        def score(r: dict[str, Any]) -> tuple[int, int]:
            s = 0
            if r.get("commitment_relation") == "mismatch":
                s += 2
            if r.get("owner_uniqueness") in {
                "resolved_by_textual_anchor",
                "resolved_by_compound_copy",
                "unique",
            }:
                s += 1
            return (-s, int(r.get("quote_start") or 0))

        return sorted(confirmed, key=score)[0]

    def pick_primary_ah_instance(self, trajectory_id: str) -> dict[str, Any] | None:
        confirmed = self.confirmed_ah_instances(trajectory_id)
        if not confirmed:
            return None
        return confirmed[0]


@lru_cache(maxsize=1)
def load_adjudication_index(path: Path | None = None) -> AdjudicationIndex:
    path = path or ADJUDICATION_JSONL
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_trajectory: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_trajectory.setdefault(str(row["trajectory_id"]), []).append(row)

    cem_status: dict[str, str] = {}
    ah_status: dict[str, str] = {}
    for tid, inst_rows in by_trajectory.items():
        statuses = [str(r.get("adjudication_status") or "") for r in inst_rows]
        if any(s == "confirmed_CEM" for s in statuses):
            cem_status[tid] = trajectory_status(statuses, "CEM")
        if any(s == "confirmed_AH" for s in statuses):
            ah_status[tid] = trajectory_status(statuses, "AH")

    return AdjudicationIndex(
        rows=rows,
        by_trajectory=by_trajectory,
        cem_trajectory_status=cem_status,
        ah_trajectory_status=ah_status,
    )


def resolve_cem_swap_target(
    trajectory: dict[str, Any],
    index: AdjudicationIndex | None = None,
) -> CemSwapTarget | None:
    index = index or load_adjudication_index()
    tid = str(trajectory.get("trajectory_id") or "")
    row = index.pick_primary_cem_instance(tid)
    if not row:
        return None
    user = _user_content(trajectory)
    donor = resolve_donor_pid(row, user=user)
    if not donor:
        return None
    value = _clean_slot_value(str(row.get("value") or ""), str(row.get("response_quote") or ""))
    if not value:
        return None
    return CemSwapTarget(
        trajectory_id=tid,
        claim_value=value,
        slot_norm=str(row.get("slot_norm") or row.get("slot") or ""),
        donor_pid=donor,
        committed_anchor_pid=str(row.get("committed_anchor_pid") or ""),
        textual_selected_pid=str(row.get("textual_selected_pid") or ""),
        instance_audit_key=str(row.get("instance_audit_key") or ""),
        owner_uniqueness=str(row.get("owner_uniqueness") or ""),
        response_quote=str(row.get("response_quote") or ""),
    )


def resolve_ah_swap_target(
    trajectory: dict[str, Any],
    index: AdjudicationIndex | None = None,
) -> AhSwapTarget | None:
    index = index or load_adjudication_index()
    tid = str(trajectory.get("trajectory_id") or "")
    row = index.pick_primary_ah_instance(tid)
    if not row:
        return None
    user = _user_content(trajectory)
    anchor = str(row.get("committed_anchor_pid") or row.get("textual_selected_pid") or "")
    anchor_block = (find_product_record_snippets(user, anchor) or [""])[0] if anchor else ""
    swap_value = _pick_anchor_swap_value(row, anchor_block)
    if not swap_value or not anchor:
        return None
    return AhSwapTarget(
        trajectory_id=tid,
        claim_value=swap_value,
        slot_norm=str(row.get("slot_norm") or row.get("slot") or ""),
        anchor_pid=anchor,
        instance_audit_key=str(row.get("instance_audit_key") or ""),
        response_quote=str(row.get("response_quote") or ""),
    )
