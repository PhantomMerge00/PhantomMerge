"""Eight evidence-layer counterfactual operators (§6)."""
from __future__ import annotations

import copy
import re
from typing import Any

from ccer.counterfactual.base import (
    Edit,
    OperatorResult,
    apply_text_edits,
    filter_held_fixed_present,
    find_pid_blocks,
    find_product_record_snippets,
    observation_block_tags,
    validate_held_fixed,
    validate_user_unchanged,
)

HARD_REQ_RE = re.compile(r"(Hard user requirement:\s*)([^.]+)(\.)", re.I)
OPERATOR_VERSION = "2"


def _fail(condition_id: str, reason: str) -> OperatorResult:
    return OperatorResult(
        condition_id=condition_id,
        operator_version=OPERATOR_VERSION,
        messages=[],
        edit_manifest=[],
        held_fixed=[],
        expected_high_level_change={},
        semantic_validation="invalid",
        invalid_reason=reason,
    )


def _json_unescape(text: str) -> str:
    return text.replace("\\/", "/").replace('\\"', '"')


def _value_replace_variants(value: str) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()

    def add(v: str) -> None:
        v = str(v or "").strip()
        if v and v not in seen:
            seen.add(v)
            variants.append(v)

    add(value)
    add(_json_unescape(value))
    if "/" in value:
        add(value.replace("/", "\\/"))
    m = re.match(r"^([\d.]+)\s*([A-Za-z]+)?", value.strip())
    if m:
        num, unit = m.group(1), (m.group(2) or "").strip()
        add(num)
        if unit:
            add(f"{num} {unit}")
            add(f"{num}{unit}")
            add(f"{num} {unit.lower()}")
            add(f"{num} {unit.upper()}")
    return variants


def _replace_once_in_snippet(snippet: str, old: str, new: str) -> str | None:
    for variant in _value_replace_variants(old):
        if variant in snippet:
            return snippet.replace(variant, new, 1)
        esc = variant.replace("/", "\\/")
        if esc in snippet:
            return snippet.replace(esc, new.replace("/", "\\/"), 1)
    old_stripped = old.strip()
    for sep in (",", ";", "/"):
        if sep not in snippet:
            continue
        parts = snippet.split(sep)
        for i, part in enumerate(parts):
            if old_stripped and (old_stripped == part.strip() or old_stripped in part):
                new_part = part.replace(old, new, 1) if old in part else part.replace(part.strip(), new, 1)
                parts[i] = new_part
                return sep.join(parts)
    # Case-insensitive token/substring match (e.g. black/Black, Type-C/Type-c).
    if old_stripped and len(old_stripped) <= 80:
        m = re.search(re.escape(old_stripped), snippet, re.I)
        if m:
            return snippet[: m.start()] + new + snippet[m.end() :]
    # Price field fallback: "91.5 PHP" -> replace numeric in "price":91.5
    m = re.match(r"^([\d.]+)", old.strip())
    if m:
        num = m.group(1)
        pm = re.search(rf'"price"\s*:\s*{re.escape(num)}\b', snippet)
        if pm:
            return snippet[: pm.start()] + f'"price":CF_{num}' + snippet[pm.end() :]
    return None


def _all_product_record_snippets(user: str) -> list[str]:
    pids = sorted(set(re.findall(r'"product_id"\s*:\s*"?(\d{6,12})"?', user)))
    seen: set[str] = set()
    out: list[str] = []
    for pid in pids:
        for snip in find_product_record_snippets(user, pid):
            if snip not in seen:
                seen.add(snip)
                out.append(snip)
    return out


def _rival_held_fixed(user: str, rival_pid: str, edited_snippet: str) -> list[str]:
    held: list[str] = []
    for snip in _all_product_record_snippets(user):
        if snip == edited_snippet:
            continue
        if rival_pid and f'"product_id":"{rival_pid}"' in snip:
            continue
        if rival_pid and f'"product_id":{rival_pid}' in snip:
            continue
        held.append(snip)
    return held


def rival_value_swap(
    messages: list[dict[str, str]],
    *,
    rival_pid: str,
    slot_value_old: str,
    slot_value_new: str,
    anchor_pid: str | None = None,
) -> OperatorResult:
    cid = "rival_value_swap"
    user = _user(messages)
    snippets = find_product_record_snippets(user, rival_pid)
    if not snippets:
        return _fail(cid, f"rival_block_not_found:{rival_pid}")

    edit_old: str | None = None
    edit_new: str | None = None
    for snippet in snippets:
        candidate_new = _replace_once_in_snippet(snippet, slot_value_old, slot_value_new)
        if candidate_new and candidate_new != snippet:
            edit_old = snippet
            edit_new = candidate_new
            break

    if not edit_old or not edit_new:
        return _fail(cid, f"rival_value_not_in_block:{slot_value_old}")

    edits = [Edit("user", edit_old, edit_new, f"rival:{rival_pid}:{slot_value_old}->{slot_value_new}")]
    edited, manifest = apply_text_edits(messages, edits)
    held = _rival_held_fixed(user, rival_pid, edit_old)
    if anchor_pid:
        for snip in find_product_record_snippets(user, anchor_pid):
            if snip not in held and snip != edit_old:
                held.append(snip)
    held_present = filter_held_fixed_present(messages, held)
    ok, fails = validate_held_fixed(messages, edited, held_present)
    return OperatorResult(
        condition_id=cid,
        operator_version=OPERATOR_VERSION,
        messages=edited,
        edit_manifest=manifest,
        held_fixed=held_present,
        expected_high_level_change={
            "rival_value": slot_value_new,
            "rival_value_old": slot_value_old,
            "rival_pid": rival_pid,
        },
        semantic_validation="ok" if manifest and ok else "invalid",
        invalid_reason=";".join(fails) if fails else None,
    )


def query_value_swap(
    messages: list[dict[str, str]],
    *,
    old_value: str,
    new_value: str,
) -> OperatorResult:
    cid = "query_value_swap"
    user = _user(messages)
    m = HARD_REQ_RE.search(user)
    if not m or old_value.lower() not in m.group(2).lower():
        return _fail(cid, "hard_requirement_value_not_found")
    req_body = m.group(2)
    pat = re.compile(re.escape(old_value), re.I)
    new_body, n = pat.subn(new_value, req_body, count=1)
    if n != 1:
        return _fail(cid, "hard_requirement_value_replace_failed")
    old_clause = m.group(0)
    new_clause = m.group(1) + new_body + m.group(3)
    if old_clause == new_clause:
        return _fail(cid, "query_edit_noop")
    edits = [Edit("user", old_clause, new_clause, f"query:{old_value}->{new_value}")]
    edited, manifest = apply_text_edits(messages, edits)
    obs_tags = filter_held_fixed_present(messages, observation_block_tags(user))
    ok, fails = validate_held_fixed(messages, edited, obs_tags)
    return OperatorResult(
        condition_id=cid,
        operator_version=OPERATOR_VERSION,
        messages=edited,
        edit_manifest=manifest,
        held_fixed=obs_tags,
        expected_high_level_change={"query_constraint_value": new_value},
        semantic_validation="ok" if manifest and ok else "invalid",
        invalid_reason=";".join(fails) if fails else None,
    )


def owner_reassignment(
    messages: list[dict[str, str]],
    *,
    value: str,
    from_pid: str,
    to_pid: str,
) -> OperatorResult:
    """Move same value attribution from rival block to anchor block (relation change, not rename)."""
    cid = "owner_reassignment"
    user = _user(messages)
    blocks_from = find_pid_blocks(user, from_pid)
    blocks_to = find_pid_blocks(user, to_pid)
    if not blocks_from or value not in blocks_from[0][2]:
        return _fail(cid, "value_not_in_from_pid_block")
    if not blocks_to:
        return _fail(cid, "missing_to_pid_block")
    # Remove value near from_pid, append to to_pid block
    snippet_from = blocks_from[0][2]
    new_from = snippet_from.replace(value, "[removed]", 1)
    snippet_to = blocks_to[0][2]
    new_to = snippet_to + f"; {value}"
    edits = [
        Edit("user", snippet_from, new_from, f"remove_value_from:{from_pid}"),
        Edit("user", snippet_to, new_to, f"add_value_to:{to_pid}"),
    ]
    edited, manifest = apply_text_edits(messages, edits)
    return OperatorResult(
        condition_id=cid,
        operator_version=OPERATOR_VERSION,
        messages=edited,
        edit_manifest=manifest,
        held_fixed=[from_pid, to_pid],
        expected_high_level_change={"owner_of_value": to_pid},
        semantic_validation="ok" if len(manifest) == 2 else "invalid",
    )


def anchor_value_swap(
    messages: list[dict[str, str]],
    *,
    anchor_pid: str,
    old_value: str,
    new_value: str,
) -> OperatorResult:
    cid = "anchor_value_swap"
    user = _user(messages)
    snippets = find_product_record_snippets(user, anchor_pid) or [
        snip for _, _, snip in find_pid_blocks(user, anchor_pid)
    ]
    if not snippets:
        return _fail(cid, "anchor_block_not_found")
    snippet = snippets[0]
    new_snippet = _replace_once_in_snippet(snippet, old_value, new_value)
    if not new_snippet or new_snippet == snippet:
        return _fail(cid, "old_value_not_in_anchor_block")
    edits = [Edit("user", snippet, new_snippet, f"anchor:{old_value}->{new_value}")]
    edited, manifest = apply_text_edits(messages, edits)
    return OperatorResult(
        condition_id=cid,
        operator_version=OPERATOR_VERSION,
        messages=edited,
        edit_manifest=manifest,
        held_fixed=[],
        expected_high_level_change={"anchor_supported_value": new_value},
        semantic_validation="ok" if manifest else "invalid",
    )


def commitment_swap(
    messages: list[dict[str, str]],
    *,
    pid_a: str,
    pid_b: str,
) -> OperatorResult:
    cid = "commitment_swap"
    user = _user(messages)
    if pid_a not in user or pid_b not in user:
        return _fail(cid, "both_pids_must_appear_in_context")
    # Swap recommend context markers only in dialogue history tool results (not global rename)
    edits = [
        Edit("user", f'"product_id": "{pid_a}"', f'"product_id": "{pid_b}"', "commitment_tool_ref"),
        Edit("user", f"product_id={pid_a}", f"product_id={pid_b}", "commitment_summary_ref"),
    ]
    edited, manifest = apply_text_edits(messages, edits)
    manifest = [m for m in manifest if m["old"] in user]
    return OperatorResult(
        condition_id=cid,
        operator_version=OPERATOR_VERSION,
        messages=edited,
        edit_manifest=manifest,
        held_fixed=[],
        expected_high_level_change={"committed_pid": pid_b},
        semantic_validation="ok" if manifest else "invalid",
    )


def pid_rename(messages: list[dict[str, str]], *, old_pid: str, new_pid: str) -> OperatorResult:
    cid = "pid_rename"
    user = _user(messages)
    if old_pid not in user:
        return _fail(cid, "pid_not_in_context")
    edits = [Edit("user", old_pid, new_pid, f"global_pid_rename:{old_pid}->{new_pid}")]
    edited, manifest = apply_text_edits(messages, edits)
    count = user.count(old_pid)
    new_count = _user(edited).count(new_pid)
    valid = count > 0 and old_pid not in _user(edited)
    return OperatorResult(
        condition_id=cid,
        operator_version=OPERATOR_VERSION,
        messages=edited,
        edit_manifest=manifest,
        held_fixed=[],
        expected_high_level_change={"nuisance_pid_rename": True, "n_replaced": count, "new_pid_count": new_count},
        semantic_validation="ok" if valid else "invalid",
    )


def position_permutation(messages: list[dict[str, str]]) -> OperatorResult:
    cid = "position_permutation"
    user = _user(messages)
    parts = user.split("\n\n")
    if len(parts) < 3:
        return _fail(cid, "insufficient_blocks_to_permute")
    # Swap first two obs-adjacent blocks after header
    if len(parts) >= 4:
        parts[1], parts[2] = parts[2], parts[1]
    new_user = "\n\n".join(parts)
    edited = copy.deepcopy(messages)
    for m in edited:
        if m.get("role") == "user":
            m["content"] = new_user
    return OperatorResult(
        condition_id=cid,
        operator_version=OPERATOR_VERSION,
        messages=edited,
        edit_manifest=[{"path": "user.content", "op": "swap_blocks", "semantic_object": "nuisance_order"}],
        held_fixed=[],
        expected_high_level_change={"order_only": True},
        semantic_validation="ok",
    )


def source_null(
    messages: list[dict[str, str]],
    *,
    source_pid: str,
    neutral_text: str = "[source removed for counterfactual]",
) -> OperatorResult:
    cid = "source_null"
    user = _user(messages)
    blocks = find_pid_blocks(user, source_pid)
    if not blocks:
        return _fail(cid, "source_block_not_found")
    edits = []
    for _, _, snippet in blocks:
        edits.append(Edit("user", snippet, neutral_text, f"null_source:{source_pid}"))
    edited, manifest = apply_text_edits(messages, edits[:1])  # one block at a time
    return OperatorResult(
        condition_id=cid,
        operator_version=OPERATOR_VERSION,
        messages=edited,
        edit_manifest=manifest,
        held_fixed=[],
        expected_high_level_change={"source_removed": source_pid},
        semantic_validation="ok" if manifest else "invalid",
    )


def instruction_pressure_neutral(messages: list[dict[str, str]]) -> OperatorResult:
    cid = "instruction_pressure_neutral"
    edited = copy.deepcopy(messages)
    for m in edited:
        if m.get("role") == "system":
            content = str(m.get("content") or "")
            content = content.replace(
                "You **must** recommend even when no candidate fully satisfies every hard field",
                "Recommend a suitable product when possible",
            )
            content = content.replace("must appear as **positive About facts on Selected**", "may appear in About when supported")
            m["content"] = content
    ok, fails = validate_user_unchanged(messages, edited)
    return OperatorResult(
        condition_id=cid,
        operator_version=OPERATOR_VERSION,
        messages=edited,
        edit_manifest=[{"path": "system.content", "op": "neutralize_pressure"}],
        held_fixed=[],
        expected_high_level_change={"instruction_pressure": "reduced"},
        semantic_validation="ok" if ok else "invalid",
        invalid_reason=";".join(fails) if fails else None,
    )


def _user(messages: list[dict[str, str]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


OPERATORS = {
    "rival_value_swap": rival_value_swap,
    "query_value_swap": query_value_swap,
    "owner_reassignment": owner_reassignment,
    "anchor_value_swap": anchor_value_swap,
    "commitment_swap": commitment_swap,
    "pid_rename": pid_rename,
    "position_permutation": position_permutation,
    "source_null": source_null,
    "instruction_pressure_neutral": instruction_pressure_neutral,
}
