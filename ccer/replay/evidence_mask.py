"""Strong evidence masking for commitment-patch bypass test (expert ROUND9/10)."""
from __future__ import annotations

import copy
import json
import re
from typing import Any

from ccer.adjudication.loader import resolve_cem_swap_target
from ccer.counterfactual.base import Edit, apply_text_edits, find_product_record_snippets
from ccer.counterfactual.operators import HARD_REQ_RE

STRONG_MASK_TOKEN = "[EVIDENCE_MASKED]"


def _user_content(messages: list[dict[str, str]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def _placeholder_record(pid: str, mask_token: str) -> str:
    return f'{{"product_id":"{pid}","evidence":"{mask_token}"}}'


def _dedupe_edits(edits: list[Edit]) -> list[Edit]:
    ordered = sorted(edits, key=lambda e: len(e.old_text), reverse=True)
    kept: list[Edit] = []
    for ed in ordered:
        if any(ed.old_text in k.old_text and ed.old_text != k.old_text for k in kept):
            continue
        kept.append(ed)
    return kept


def _entity_pids(target: Any, trajectory: dict[str, Any]) -> tuple[str, str]:
    donor_pid = str(getattr(target, "donor_pid", "") or "")
    anchor_pid = str(
        getattr(target, "committed_anchor_pid", "")
        or getattr(target, "textual_selected_pid", "")
        or ""
    )
    commitment = trajectory.get("commitment") or {}
    if not anchor_pid:
        anchor_pid = str(commitment.get("action_anchor") or "")
    return donor_pid, anchor_pid


def _rival_pids_from_evidence(trajectory: dict[str, Any]) -> list[str]:
    from ccer.mechanism.anchor_resolve_heuristic import _evidence_records

    out: list[str] = []
    for ev in _evidence_records(trajectory):
        if str(ev.get("entity_role") or "") != "rival":
            continue
        for eid in ev.get("entity_ids") or []:
            pid = str(eid or "")
            if pid and pid not in out:
                out.append(pid)
    return out


def is_shell_anchor_title(title: str, anchor_pid: str) -> bool:
    """Anchor title is only its own PID (no real product description)."""
    t = str(title or "").strip()
    if not t or not anchor_pid:
        return False
    if t == anchor_pid:
        return True
    if re.fullmatch(r"\d{6,12}", t):
        return True
    return False


def classify_anchor_title_quality(trajectory: dict[str, Any], anchor_pid: str) -> dict[str, Any]:
    user = ""
    for m in trajectory.get("messages_final_call") or []:
        if m.get("role") == "user":
            user = str(m.get("content") or "")
    snippets = find_product_record_snippets(user, anchor_pid) if anchor_pid else []
    title = ""
    if snippets:
        try:
            obj = json.loads(snippets[0])
            title = str(obj.get("title") or "")
        except json.JSONDecodeError:
            m = re.search(r'"title"\s*:\s*"([^"]*)"', snippets[0])
            title = m.group(1) if m else snippets[0][:80]
    shell = is_shell_anchor_title(title, anchor_pid)
    return {
        "anchor_pid": anchor_pid,
        "anchor_title_preview": title[:120],
        "anchor_title_quality": "shell_pid_only" if shell else "real_text",
        "has_pins_not_viewed": "pins_not_viewed" in user,
    }


def _mask_bare_pids_outside_placeholders(user: str, pids: list[str], mask_token: str) -> tuple[str, list[dict[str, Any]]]:
    """Replace any bare PID occurrence outside masked placeholder records."""
    edits_meta: list[dict[str, Any]] = []
    out = user
    for pid in pids:
        if not pid:
            continue
        placeholder = _placeholder_record(pid, mask_token)
        parts = out.split(placeholder)
        rebuilt: list[str] = []
        for i, part in enumerate(parts):
            if i > 0:
                rebuilt.append(placeholder)
            if pid in part:
                n = part.count(pid)
                part = part.replace(pid, mask_token)
                edits_meta.append({"op": "mask_bare_pid", "pid": pid, "count": n})
            rebuilt.append(part)
        out = "".join(rebuilt)
    return out, edits_meta


def _mask_claim_in_user_turns(user: str, claim_value: str, mask_token: str) -> tuple[str, bool]:
    """Mask claim/slot value in <user>...</user> dialogue turns (not only hard requirement)."""
    if not claim_value or len(claim_value.strip()) < 3:
        return user, False

    def _repl_user_block(m: re.Match[str]) -> str:
        block = m.group(0)
        pat = re.compile(re.escape(claim_value), re.I)
        new_block, n = pat.subn(mask_token, block, count=0)
        return new_block if n else block

    new_user, n_blocks = 0, 0
    out = re.sub(r"<user>[\s\S]*?</user>", _repl_user_block, user)
    if out != user:
        n_blocks = 1
    return out, n_blocks > 0


def audit_prompt_leaks(
    edited_user: str,
    *,
    donor_pid: str,
    anchor_pid: str,
    claim_value: str,
    rival_pids: list[str],
    mask_token: str,
) -> dict[str, Any]:
    """Explicit post-mask leak audit (document every check, not empty-by-default)."""
    checks: list[dict[str, Any]] = []

    def _add(name: str, passed: bool, detail: str, locations: list[str] | None = None) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail, "locations": locations or []})

    for pid in (donor_pid, anchor_pid):
        if not pid:
            continue
        snips = find_product_record_snippets(edited_user, pid)
        if not snips:
            _add(f"record_present:{pid}", False, "masked record missing")
            continue
        bad = [s for s in snips if mask_token not in s or ('"title"' in s and '"evidence"' not in s)]
        _add(
            f"record_masked:{pid}",
            len(bad) == 0,
            "ok" if not bad else f"{len(bad)} snippet(s) still expose title/value",
            bad[:2],
        )

    if claim_value and len(claim_value.strip()) >= 4:
        idx = edited_user.find(claim_value)
        _add(
            "claim_value_absent_globally",
            idx < 0,
            "not found" if idx < 0 else f"found at char {idx}",
            [edited_user[max(0, idx - 40) : idx + len(claim_value) + 40]] if idx >= 0 else [],
        )

    for rpid in rival_pids:
        if not rpid or rpid in (donor_pid, anchor_pid):
            continue
        # Outside placeholder records: bare PID still in prompt?
        stripped = edited_user
        for snip in find_product_record_snippets(edited_user, rpid):
            stripped = stripped.replace(snip, "")
        bare_hits = [m.start() for m in re.finditer(re.escape(rpid), stripped)]
        _add(
            f"rival_pid_bare_outside_record:{rpid}",
            len(bare_hits) == 0,
            "none" if not bare_hits else f"{len(bare_hits)} bare occurrence(s)",
            [stripped[max(0, i - 30) : i + len(rpid) + 30] for i in bare_hits[:3]],
        )

    m = HARD_REQ_RE.search(edited_user)
    if m:
        body = m.group(2)
        _add(
            "query_clause_masked",
            mask_token in body or body.strip() in ("", "."),
            f"query body={body[:60]}",
        )

    failures = [c for c in checks if not c["passed"]]
    return {
        "checks_run": len(checks),
        "checks": checks,
        "failures": failures,
        "leak_free": len(failures) == 0,
    }


def mask_cem_evidence_spans(
    messages: list[dict[str, str]],
    trajectory: dict[str, Any],
    *,
    mask_token: str = STRONG_MASK_TOKEN,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    target = resolve_cem_swap_target(trajectory)
    user = _user_content(messages)
    if not user:
        return copy.deepcopy(messages), {"masked": False, "reason": "empty_user", "mask_mode": "strong"}

    donor_pid, anchor_pid = _entity_pids(target, trajectory) if target else ("", "")
    claim_value = str(getattr(target, "claim_value", "") or "") if target else ""
    slot_norm = str(getattr(target, "slot_norm", "") or "") if target else ""
    rival_pids = _rival_pids_from_evidence(trajectory)

    edits: list[Edit] = []
    masked_pids: list[str] = []

    for pid in (donor_pid, anchor_pid):
        if not pid:
            continue
        seen_snips: set[str] = set()
        for snippet in find_product_record_snippets(user, pid):
            if snippet in seen_snips:
                continue
            seen_snips.add(snippet)
            edits.append(
                Edit(
                    "user",
                    snippet,
                    _placeholder_record(pid, mask_token),
                    f"strong_mask_product_record:{pid}",
                )
            )
        if seen_snips:
            masked_pids.append(pid)

    m = HARD_REQ_RE.search(user)
    if m:
        edits.append(
            Edit("user", m.group(0), f"Hard user requirement: {mask_token}.", "strong_mask_query_clause")
        )

    for ev in trajectory.get("evidence") or []:
        role = str(ev.get("entity_role") or "")
        if role not in ("rival", "anchor"):
            continue
        span = ev.get("char_span")
        if not span or len(span) != 2:
            continue
        start, end = int(span[0]), int(span[1])
        if start < 0 or end > len(user) or start >= end:
            continue
        old = user[start:end]
        if not old.strip() or mask_token in old:
            continue
        edits.append(
            Edit("user", old, mask_token, f"strong_mask_evidence_span:{role}", char_span=(start, end))
        )

    edits = _dedupe_edits(edits)
    if not edits:
        return copy.deepcopy(messages), {
            "masked": False,
            "reason": "no_editable_spans",
            "mask_mode": "strong",
            "donor_pid": donor_pid,
            "anchor_pid": anchor_pid,
        }

    edited, edit_manifest = apply_text_edits(messages, edits)
    edited_user = _user_content(edited)

    # Post-pass: mask claim in <user> turns and bare PIDs in tool summaries / product_ids lists.
    edited_user, user_turn_masked = _mask_claim_in_user_turns(edited_user, claim_value, mask_token)
    if claim_value and (len(claim_value.strip()) >= 4 or " " in claim_value):
        pat = re.compile(re.escape(claim_value), re.I)
        edited_user, n_global = pat.subn(mask_token, edited_user)
        global_claim_masked = n_global > 0
    else:
        global_claim_masked = False
    edited_user, bare_pid_edits = _mask_bare_pids_outside_placeholders(
        edited_user, [donor_pid, anchor_pid] + rival_pids, mask_token
    )
    if user_turn_masked or bare_pid_edits:
        for m in edited:
            if m.get("role") == "user":
                m["content"] = edited_user

    leak_audit = audit_prompt_leaks(
        edited_user,
        donor_pid=donor_pid,
        anchor_pid=anchor_pid,
        claim_value=claim_value,
        rival_pids=rival_pids,
        mask_token=mask_token,
    )

    return edited, {
        "masked": True,
        "mask_mode": "strong",
        "n_edits": len(edit_manifest),
        "donor_pid": donor_pid,
        "anchor_pid": anchor_pid,
        "claim_value": claim_value,
        "slot_norm": slot_norm,
        "masked_pids": masked_pids,
        "rival_pids_in_evidence": rival_pids,
        "mask_token": mask_token,
        "edit_manifest": edit_manifest,
        "post_mask_pass": {
            "user_turn_claim_masked": user_turn_masked,
            "global_claim_masked": global_claim_masked,
            "bare_pid_replacements": bare_pid_edits,
        },
        "leak_audit": leak_audit,
        "leak_free": leak_audit["leak_free"],
    }
