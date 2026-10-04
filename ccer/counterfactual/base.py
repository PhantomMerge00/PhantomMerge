"""Diff validation and edit manifest utilities."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ccer.io_utils import sha256_json


@dataclass
class Edit:
    path: str
    old_text: str
    new_text: str
    semantic_object: str
    char_span: tuple[int, int] | None = None


@dataclass
class OperatorResult:
    condition_id: str
    operator_version: str
    messages: list[dict[str, str]]
    edit_manifest: list[dict[str, Any]]
    held_fixed: list[str]
    expected_high_level_change: dict[str, Any]
    semantic_validation: str = "pending"
    token_alignment_validation: str = "pending"
    invalid_reason: str | None = None

    def to_counterfactual_record(
        self,
        *,
        root_id: str,
        pair_id: str,
        estimand: str,
        split: str,
        base_hash: str,
    ) -> dict[str, Any]:
        edited_hash = sha256_json(self.messages)
        return {
            "root_id": root_id,
            "pair_id": pair_id,
            "condition_id": self.condition_id,
            "operator_version": self.operator_version,
            "estimand": estimand,
            "base_input_hash": base_hash,
            "edited_input_hash": edited_hash,
            "edit_manifest": self.edit_manifest,
            "held_fixed": self.held_fixed,
            "expected_high_level_change": self.expected_high_level_change,
            "semantic_validation": self.semantic_validation,
            "token_alignment_validation": self.token_alignment_validation,
            "split": split,
            "invalid_reason": self.invalid_reason,
        }


def apply_text_edits(
    messages: list[dict[str, str]],
    edits: list[Edit],
    *,
    target_role: str = "user",
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    out = copy.deepcopy(messages)
    manifest: list[dict[str, Any]] = []
    for i, m in enumerate(out):
        if m.get("role") != target_role:
            continue
        content = str(m.get("content") or "")
        for ed in edits:
            if ed.old_text not in content:
                continue
            content = content.replace(ed.old_text, ed.new_text, 1)
            manifest.append(
                {
                    "path": f"messages[{i}].content",
                    "old": ed.old_text,
                    "new": ed.new_text,
                    "semantic_object": ed.semantic_object,
                    "char_span": list(ed.char_span) if ed.char_span else None,
                }
            )
        m["content"] = content
    return out, manifest


def _user_content(messages: list[dict[str, str]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def observation_block_tags(user_content: str, *, min_len: int = 40) -> list[str]:
    """Stable product/tool observation blocks for held-fixed validation."""
    tags: list[str] = []
    for block in user_content.split("\n\n"):
        b = block.strip()
        if len(b) < min_len:
            continue
        if b.lower().startswith("hard user requirement"):
            continue
        if re.search(r"product[_ ]?id|[\"']?\d{8,12}[\"']?", b, re.I):
            tags.append(b)
    return tags


def validate_user_unchanged(
    base_messages: list[dict[str, str]],
    edited_messages: list[dict[str, str]],
) -> tuple[bool, list[str]]:
    if _user_content(base_messages) != _user_content(edited_messages):
        return False, ["user_content_changed"]
    return True, []


def filter_held_fixed_present(base_messages: list[dict[str, str]], held_fixed: list[str]) -> list[str]:
    """Keep only held-fixed tags that literally appear in base user content."""
    base_user = _user_content(base_messages)
    return [tag for tag in held_fixed if tag and tag in base_user]


def validate_held_fixed(
    base_messages: list[dict[str, str]],
    edited_messages: list[dict[str, str]],
    held_fixed: list[str],
) -> tuple[bool, list[str]]:
    """Verify declared held-fixed substrings unchanged after edit."""
    failures: list[str] = []
    base_user = _user_content(base_messages)
    edit_user = _user_content(edited_messages)
    for tag in held_fixed:
        if tag not in base_user:
            failures.append(f"missing_in_base:{tag[:40]}")
            continue
        if base_user.count(tag) != edit_user.count(tag):
            failures.append(f"count_changed:{tag[:40]}")
        elif tag in base_user and tag not in edit_user:
            failures.append(f"removed:{tag[:40]}")
    return len(failures) == 0, failures


def find_pid_blocks(user_content: str, pid: str) -> list[tuple[int, int, str]]:
    """Return (start, end, snippet) for regions mentioning pid."""
    blocks: list[tuple[int, int, str]] = []
    start = 0
    while True:
        idx = user_content.find(pid, start)
        if idx < 0:
            break
        end = min(len(user_content), idx + 400)
        blocks.append((idx, end, user_content[idx:end]))
        start = idx + len(pid)
    return blocks


def find_product_record_snippets(user_content: str, pid: str) -> list[str]:
    """Return tight product-record snippets for a PID (preferred over wide pid blocks)."""
    if not pid:
        return []
    patterns = [
        re.compile(rf'\{{[^{{}}]*"product_id"\s*:\s*"{re.escape(pid)}"[^{{}}]*\}}'),
        re.compile(rf'\{{[^{{}}]*"product_id"\s*:\s*{re.escape(pid)}[^{{}}]*\}}'),
    ]
    seen: set[str] = set()
    out: list[str] = []
    for pat in patterns:
        for m in pat.finditer(user_content or ""):
            snip = m.group(0)
            if snip not in seen:
                seen.add(snip)
                out.append(snip)
    if out:
        return out
    return [snip for _, _, snip in find_pid_blocks(user_content, pid)]
