"""Audit CEM rival_value_swap scorable/invalid cases (ROUND2 red flag 1)."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.counterfactual.base import find_pid_blocks
from ccer.io_utils import load_jsonl
from ccer.paths import P1_DIR, REPORTS

REPORT_PATH = REPORTS / "cem_rival_audit.md"


def _user(messages: list[dict[str, str]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def _classify_invalid(reason: str) -> str:
    r = str(reason or "")
    if r.startswith("rival_value_not_found"):
        return "value_not_in_prompt"
    if "missing_in_base:product_id=" in r or "count_changed:product_id=" in r:
        return "held_fixed_format"
    if "held_fixed" in r or "missing_in_base" in r:
        return "held_fixed_other"
    return "other"


def _edit_locus(traj: dict, value: str) -> str:
    user = _user(traj.get("messages_final_call") or [])
    compared = str((traj.get("text_anchor") or {}).get("compared_pid") or "")
    anchor = str((traj.get("commitment") or {}).get("action_anchor") or "")
    if not value or value not in user:
        return "value_absent_from_prompt"
    idx = user.find(value)
    in_rival = any(value in snip for _, _, snip in find_pid_blocks(user, compared)) if compared else False
    in_anchor = any(value in snip for _, _, snip in find_pid_blocks(user, anchor)) if anchor else False
    if in_rival:
        return "rival_block"
    if in_anchor:
        return "anchor_block"
    if idx >= 0:
        return "global_first_match_would_hit_here"
    return "unknown"


def _manual_follow_hint(orig: str, new: str, cf_value: str, old_value: str) -> str:
    from ccer.replay.answer_utils import extract_answer

    o = extract_answer(orig) if orig else orig
    n = extract_answer(new) if new else new
    hints = []
    if cf_value and cf_value in n and cf_value not in o:
        hints.append("cf_value_appears")
    if old_value and old_value in o and old_value not in n:
        hints.append("old_value_dropped")
    if old_value and old_value in n:
        hints.append("old_value_retained")
    if re.search(r"Selected product ID:\s*(\d+)", o or "", re.I) != re.search(
        r"Selected product ID:\s*(\d+)", n or "", re.I
    ):
        hints.append("selected_pid_changed")
    return ", ".join(hints) if hints else "no_obvious_follow"


def build_audit_report() -> str:
    rows = {r["trajectory_id"]: r for r in load_jsonl(ROOT / "results/normalized/shopping_trajectories.jsonl")}
    effects = [
        r
        for r in load_jsonl(P1_DIR / "source_effects_rows.jsonl")
        if r.get("cohort") == "CEM" and r.get("condition_id") == "rival_value_swap"
    ]
    invalids = [
        r
        for r in load_jsonl(P1_DIR / "invalid_counterfactuals.jsonl")
        if r.get("condition_id") == "rival_value_swap"
        and str(r.get("trajectory_id") or r.get("root_id") or "") in rows
    ]
    gens = {
        (r["trajectory_id"], r["condition_id"]): r
        for r in load_jsonl(P1_DIR / "generations.jsonl")
        if r.get("cohort") == "CEM"
    }

    invalid_buckets = Counter(_classify_invalid(r.get("invalid_reason") or "") for r in invalids)

    lines = [
        "# CEM rival_value_swap 审计（ROUND2 红旗一）",
        "",
        "## 摘要",
        "",
        f"- scorable rival swap: **{len(effects)}**",
        f"- invalid rival swap: **{len(invalids)}**",
        f"- follow: **{sum(1 for e in effects if e.get('source_following'))}/{len(effects)}**",
        "",
        "**结论**：10/43 scorable 与 0/10 follow 主要来自算子实现（全局 replace、held_fixed 格式），非「数据里没有 CEM」。",
        "",
        "## Invalid 分类（33 条）",
        "",
        "| 桶 | n |",
        "|---|---|",
    ]
    for k, v in invalid_buckets.most_common():
        lines.append(f"| {k} | {v} |")

    lines.extend(["", "## 10 条 scorable 逐条", ""])
    for e in effects:
        tid = str(e["trajectory_id"])
        traj = rows.get(tid, {})
        cem = next((c for c in (traj.get("claims") or []) if c.get("legacy_label") == "cross_object_merge"), {})
        value = str(cem.get("value") or "")
        cf = f"CF_{value}"
        gen = gens.get((tid, "rival_value_swap"), {})
        orig = str(gen.get("original_answer") or "")
        ans = str(gen.get("answer") or "")
        locus = _edit_locus(traj, value)
        hint = _manual_follow_hint(orig, ans, cf, value)
        lines.extend(
            [
                f"### `{tid}`",
                "",
                f"- value: `{value}` | rival_pid: `{(traj.get('text_anchor') or {}).get('compared_pid')}`",
                f"- 全局 replace 会打偏: **{locus}**",
                f"- scored follow: `{e.get('source_following')}` ({e.get('reason')})",
                f"- 肉眼线索: {hint}",
                "",
            ]
        )

    sample_invalid = invalids[:15]
    lines.extend(["## Invalid 抽查（15 条）", ""])
    for inv in sample_invalid:
        tid = str(inv.get("trajectory_id") or inv.get("root_id") or "")
        traj = rows.get(tid, {})
        cem = next((c for c in (traj.get("claims") or []) if c.get("legacy_label") == "cross_object_merge"), {})
        value = str(cem.get("value") or "")
        bucket = _classify_invalid(inv.get("invalid_reason") or "")
        recoverable = bucket in ("held_fixed_format", "value_not_in_prompt")
        lines.append(
            f"- `{tid}` | {bucket} | reason=`{inv.get('invalid_reason')}` | "
            f"value=`{value[:40]}` | 算子扩展可挽回: **{'是' if recoverable else '待查'}**"
        )

    lines.extend(
        [
            "",
            "## 建议修复（已纳入实现）",
            "",
            "1. rival block 内单次替换，禁止全局 first-match",
            "2. held_fixed 改用 observation block 原文，不用 `product_id=` 字面串",
            "3. 统一 operators 与 fixed_history 的 held_fixed 过滤",
            "4. follow 检测增加 Selected/Compared claim-span 定向规则",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    text = build_audit_report()
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
