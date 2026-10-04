"""Export CEM cohort + AH dev manual verification packs (MD + JSONL)."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.counterfactual.base import find_pid_blocks
from ccer.counterfactual.bundles import resolve_cem_rival_value
from ccer.io_utils import load_json, load_jsonl, write_jsonl
from ccer.paths import NORMALIZED_SHOPPING, P1_DIR, REPORTS, SPLIT_MANIFEST_JSON
from ccer.replay.answer_utils import extract_answer, normalize_final_synthesis_text

OUT_DIR = REPORTS / "manual_verification"
PID_RE = re.compile(r'"product_id"\s*:\s*"(\d{6,14})"', re.I)
COMMON_VALUE_RE = re.compile(
    r"^(s|m|l|xl|xxl|xs|xxs|black|white|red|blue|green|yellow|brown|gray|grey|"
    r"silver|gold|pink|orange|purple|type-c|usb|simple|\d{1,2})$",
    re.I,
)


def _user_content(traj: dict[str, Any]) -> str:
    for m in traj.get("messages_final_call") or []:
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def _answer_text(traj: dict[str, Any]) -> str:
    raw = str((traj.get("metadata") or {}).get("final_answer") or "")
    return extract_answer(normalize_final_synthesis_text(raw))


def _all_pids(user: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for pid in PID_RE.findall(user):
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _entity_snippet(user: str, pid: str, *, window: int = 320) -> str:
    blocks = find_pid_blocks(user, pid)
    if not blocks:
        return ""
    return blocks[0][2][:window]


def _value_match_variants(value: str) -> list[str]:
    v = value.strip()
    variants = {v, v.lower()}
    m = re.match(r"^([\d.]+)\s*(\w+)?$", v)
    if m:
        num, unit = m.group(1), (m.group(2) or "").strip()
        if unit:
            variants.add(f"{num} {unit}")
            variants.add(f"{num}{unit}")
        variants.add(num)
    return [x for x in variants if x]


def _value_in_text(text: str, value: str) -> bool:
    if not value or not text:
        return False
    tl, vl = text.lower(), value.lower()
    if vl in tl:
        return True
    for variant in _value_match_variants(value):
        if variant.lower() in tl:
            return True
    return False


def _value_locations(user: str, value: str) -> dict[str, list[str]]:
    if not value:
        return {}
    locs: dict[str, list[str]] = {}
    for pid in _all_pids(user):
        snippet = _entity_snippet(user, pid, window=500)
        if not snippet:
            continue
        if _value_in_text(snippet, value):
            locs.setdefault(pid, []).append(snippet)
    return locs


def _primary_cem(traj: dict[str, Any]) -> dict[str, Any] | None:
    claims = [c for c in (traj.get("claims") or []) if c.get("legacy_label") == "cross_object_merge"]
    if not claims:
        return None
    resolved = resolve_cem_rival_value(traj)
    if resolved:
        value, rival_pid = resolved
        for c in claims:
            if str(c.get("value") or "").strip() == value:
                return c
    return claims[0]


def _primary_ah(traj: dict[str, Any]) -> dict[str, Any] | None:
    for c in traj.get("claims") or []:
        if c.get("legacy_label") == "anchored_hallucination":
            return c
    return None


def _anchor_evidence_for_slot(traj: dict[str, Any], slot_norm: str, anchor_pid: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    user = _user_content(traj)
    slot_key = (slot_norm or "").strip().lower()
    for ev in traj.get("evidence") or []:
        ents = [str(x) for x in (ev.get("entity_ids") or ev.get("source_id") or [])]
        if anchor_pid not in ents:
            continue
        ev_slot = str(ev.get("slot_norm") or ev.get("slot_raw") or "").strip().lower()
        if slot_key and ev_slot and slot_key not in ev_slot and ev_slot not in slot_key:
            continue
        out.append(
            {
                "evidence_id": ev.get("evidence_id"),
                "slot_norm": ev.get("slot_norm"),
                "value_norm": ev.get("value_norm"),
                "value_raw": ev.get("value_raw"),
                "evidence_kind": ev.get("evidence_kind"),
                "char_span": ev.get("char_span"),
            }
        )
    if not out and anchor_pid:
        snippet = _entity_snippet(user, anchor_pid, window=500)
        if snippet:
            out.append(
                {
                    "evidence_id": None,
                    "slot_norm": slot_norm,
                    "value_norm": None,
                    "value_raw": snippet,
                    "evidence_kind": "prompt_snippet_fallback",
                    "char_span": None,
                }
            )
    return out


def _p1_rival_status(tid: str) -> dict[str, Any]:
    effects = {
        str(r["trajectory_id"]): r
        for r in load_jsonl(P1_DIR / "source_effects_rows.jsonl")
        if r.get("cohort") == "CEM" and r.get("condition_id") == "rival_value_swap"
    }
    invalids = {
        str(r.get("trajectory_id") or r.get("root_id") or ""): r
        for r in load_jsonl(P1_DIR / "invalid_counterfactuals.jsonl")
        if r.get("condition_id") == "rival_value_swap"
    }
    if tid in effects:
        e = effects[tid]
        return {
            "operator": "rival_value_swap",
            "scorable": True,
            "source_following": e.get("source_following"),
            "reason": e.get("reason"),
        }
    if tid in invalids:
        inv = invalids[tid]
        return {
            "operator": "rival_value_swap",
            "scorable": False,
            "invalid_reason": inv.get("invalid_reason"),
        }
    return {"operator": "rival_value_swap", "scorable": None, "note": "not_in_p1_effects"}


def _heuristic_cem_label(
    *,
    value: str,
    rival_pid: str,
    anchor_pid: str,
    value_locs: dict[str, list[str]],
    claim: dict[str, Any],
) -> tuple[str, str]:
    reasons: list[str] = []
    rival_hits = rival_pid in value_locs
    anchor_hits = anchor_pid in value_locs
    other_pids = [p for p in value_locs if p not in {rival_pid, anchor_pid}]

    if not rival_hits:
        reasons.append("claim值未出现在标注rival实体的prompt片段中")
    if anchor_hits:
        reasons.append("claim值也出现在anchor实体片段中（非纯rival来源）")
    if len(other_pids) >= 1:
        reasons.append(f"另有{len(other_pids)}个实体prompt片段也含该值")
    if COMMON_VALUE_RE.match(value.strip()):
        reasons.append("取值空间极小（颜色/尺码/单字符等）")

    if rival_hits and not anchor_hits and not other_pids:
        return "true_CEM", "；".join(reasons) if reasons else "rival唯一含值且anchor不含"
    if not rival_hits and not value_locs:
        return "ambiguous", "prompt中找不到该值的明确实体归属；" + "；".join(reasons)
    if other_pids or anchor_hits or COMMON_VALUE_RE.match(value.strip()):
        if len(other_pids) >= 2 or (COMMON_VALUE_RE.match(value.strip()) and other_pids):
            return "coincidence", "；".join(reasons)
        return "ambiguous", "；".join(reasons)
    return "ambiguous", "；".join(reasons) or "需人工判断rival追溯唯一性"


def _heuristic_ah_label(
    *,
    value: str,
    anchor_pid: str,
    slot_norm: str,
    anchor_evs: list[dict[str, Any]],
    claim: dict[str, Any],
) -> tuple[str, str]:
    reasons: list[str] = []
    contradictions = []
    for ev in anchor_evs:
        ev_val = str(ev.get("value_norm") or ev.get("value_raw") or "").strip()
        if ev_val and value.lower() not in ev_val.lower() and ev_val.lower() not in value.lower():
            contradictions.append(ev_val)
    if contradictions:
        return "true_AH", f"anchor证据显示'{contradictions[0][:60]}'与claim值'{value}'矛盾"
    if not anchor_evs:
        return "ambiguous", "未找到anchor侧同slot结构化证据，仅有claim标注"
    if COMMON_VALUE_RE.match(value.strip()):
        return "ambiguous", "取值常见且anchor证据未形成清晰矛盾"
    return "ambiguous", "anchor证据与claim关系不明确，需人工读片段"


def _build_cem_pack(tid: str, traj: dict[str, Any]) -> dict[str, Any]:
    user = _user_content(traj)
    text_anchor = traj.get("text_anchor") or {}
    commitment = traj.get("commitment") or {}
    anchor_pid = str(commitment.get("action_anchor") or text_anchor.get("selected_pid") or "")
    rival_pid = str(text_anchor.get("compared_pid") or "")
    claim = _primary_cem(traj)
    all_cem_claims = [c for c in (traj.get("claims") or []) if c.get("legacy_label") == "cross_object_merge"]

    value = str((claim or {}).get("value") or "")
    slot_norm = str((claim or {}).get("slot_norm") or "")
    value_locs = _value_locations(user, value)
    heuristic_label, heuristic_reason = _heuristic_cem_label(
        value=value,
        rival_pid=rival_pid,
        anchor_pid=anchor_pid,
        value_locs=value_locs,
        claim=claim or {},
    )

    selected_pid = str(text_anchor.get("selected_pid") or "")
    flags: list[str] = []
    if anchor_pid and rival_pid and anchor_pid == rival_pid:
        flags.append("anchor_pid_eq_rival_pid")
    if selected_pid and value_locs.get(selected_pid) and not value_locs.get(rival_pid):
        flags.append("value_on_selected_not_on_labeled_rival")
    if selected_pid and rival_pid and selected_pid != rival_pid and value_locs.get(selected_pid):
        flags.append("value_also_on_selected_product")

    return {
        "pack_type": "CEM",
        "trajectory_id": tid,
        "split": traj.get("split"),
        "annotation_target": {
            "claim_id": (claim or {}).get("claim_id"),
            "slot_norm": slot_norm,
            "claim_value": value,
            "response_span": (claim or {}).get("response_span"),
            "selected_pid": selected_pid,
            "anchor_pid": anchor_pid,
            "rival_pid": rival_pid,
            "referent_ids": (claim or {}).get("referent_ids"),
            "candidate_source_types": (claim or {}).get("candidate_source_types"),
            "quality_flags": flags,
        },
        "all_cem_claims": [
            {
                "claim_id": c.get("claim_id"),
                "slot_norm": c.get("slot_norm"),
                "value": c.get("value"),
                "response_span": c.get("response_span"),
            }
            for c in all_cem_claims
        ],
        "model_answer_excerpt": _answer_text(traj)[:2500],
        "evidence_for_annotator": {
            "rival_pid": rival_pid,
            "rival_snippets": [_entity_snippet(user, rival_pid, window=500)] if rival_pid else [],
            "rival_value_snippets": value_locs.get(rival_pid, []),
            "anchor_pid": anchor_pid,
            "anchor_snippets": [_entity_snippet(user, anchor_pid, window=500)] if anchor_pid else [],
            "anchor_has_value": anchor_pid in value_locs,
            "other_entities_with_same_value": [
                {"pid": pid, "snippets": snippets[:2]}
                for pid, snippets in value_locs.items()
                if pid not in {rival_pid, anchor_pid}
            ],
            "query_requirements": [
                ln.strip()
                for ln in user.splitlines()
                if ln.strip().lower().startswith("hard user requirement")
            ],
        },
        "operator_status": _p1_rival_status(tid),
        "heuristic_pre_label": heuristic_label,
        "heuristic_reason": heuristic_reason,
        "human_annotation": {
            "label": None,
            "notes": None,
            "annotator": None,
            "annotated_at": None,
        },
        "trajectory": traj,
    }


def _build_ah_pack(tid: str, traj: dict[str, Any]) -> dict[str, Any]:
    user = _user_content(traj)
    text_anchor = traj.get("text_anchor") or {}
    commitment = traj.get("commitment") or {}
    anchor_pid = str(commitment.get("action_anchor") or text_anchor.get("selected_pid") or "")
    claim = _primary_ah(traj)
    value = str((claim or {}).get("value") or "")
    slot_norm = str((claim or {}).get("slot_norm") or "")
    anchor_evs = _anchor_evidence_for_slot(traj, slot_norm, anchor_pid)
    heuristic_label, heuristic_reason = _heuristic_ah_label(
        value=value,
        anchor_pid=anchor_pid,
        slot_norm=slot_norm,
        anchor_evs=anchor_evs,
        claim=claim or {},
    )

    return {
        "pack_type": "AH",
        "trajectory_id": tid,
        "split": traj.get("split"),
        "annotation_target": {
            "claim_id": (claim or {}).get("claim_id"),
            "slot_norm": slot_norm,
            "claim_value": value,
            "response_span": (claim or {}).get("response_span"),
            "anchor_pid": anchor_pid,
            "compared_pid": str(text_anchor.get("compared_pid") or ""),
            "candidate_source_types": (claim or {}).get("candidate_source_types"),
            "commitment_relation": (claim or {}).get("normalized_axes", {}).get("commitment_relation"),
        },
        "model_answer_excerpt": _answer_text(traj)[:2500],
        "evidence_for_annotator": {
            "anchor_pid": anchor_pid,
            "anchor_snippets": [_entity_snippet(user, anchor_pid, window=500)] if anchor_pid else [],
            "anchor_structured_evidence": anchor_evs,
            "claim_in_answer": value in _answer_text(traj) if value else False,
            "query_requirements": [
                ln.strip()
                for ln in user.splitlines()
                if ln.strip().lower().startswith("hard user requirement")
            ],
        },
        "heuristic_pre_label": heuristic_label,
        "heuristic_reason": heuristic_reason,
        "human_annotation": {
            "label": None,
            "notes": None,
            "annotator": None,
            "annotated_at": None,
        },
        "trajectory": traj,
    }


def _md_header(title: str, n: int, label_options: str, extra: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- 生成时间: {ts}",
            f"- 条目数: **{n}**",
            f"- 配套 JSONL: 同目录下同名 `.jsonl`（含完整 `trajectory` 字段）",
            "",
            "## 标注说明",
            "",
            "请逐条阅读 **模型答案**、**claim** 与 **证据片段**，判断标注是否成立。",
            "",
            "### 标签定义",
            "",
            label_options,
            "",
            "### 填写方式",
            "",
            "1. 在 JSONL 对应行的 `human_annotation` 填入 `label` / `notes` / `annotator` / `annotated_at`。",
            "2. 或在本文档每条末尾的 `【人工标注】` 处手写（事后同步到 JSONL）。",
            "3. `heuristic_pre_label` 仅为机器启发，**不能替代人工判断**。",
            "",
            extra,
            "---",
            "",
        ]
    )


def _md_cem_case(i: int, pack: dict[str, Any]) -> str:
    t = pack["annotation_target"]
    ev = pack["evidence_for_annotator"]
    op = pack["operator_status"]
    lines = [
        f"## {i}. `{pack['trajectory_id']}`",
        "",
        "| 字段 | 值 |",
        "|---|---|",
        f"| slot | `{t.get('slot_norm')}` |",
        f"| **claim value** | `{t.get('claim_value')}` |",
        f"| selected pid（答案选中） | `{t.get('selected_pid')}` |",
        f"| anchor pid | `{t.get('anchor_pid')}` |",
        f"| **rival pid（标注 owner）** | `{t.get('rival_pid')}` |",
        f"| 质量标记 | `{', '.join(t.get('quality_flags') or []) or '—'}` |",
        f"| 启发式预标 | `{pack['heuristic_pre_label']}` — {pack['heuristic_reason']} |",
        f"| P1 rival_value_swap | scorable=`{op.get('scorable')}`"
        + (f", follow=`{op.get('source_following')}`" if op.get("scorable") else f", reason=`{op.get('invalid_reason') or op.get('note')}`"),
        "",
        f"**答案中的 claim span**: {t.get('response_span')}",
        "",
        "### 模型答案（摘录）",
        "",
        "```",
        pack["model_answer_excerpt"][:1200],
        "```",
        "",
        "### Rival 证据片段（标注来源实体）",
        "",
    ]
    for j, snip in enumerate(ev.get("rival_value_snippets") or ev.get("rival_snippets") or ["*(未找到含值片段)*"]):
        lines.extend([f"**rival #{j+1}**", "", "```", snip[:800], "```", ""])
    lines.extend(["### Anchor 证据片段（选中实体）", ""])
    for j, snip in enumerate(ev.get("anchor_snippets") or ["*(无)*"]):
        lines.extend([f"**anchor #{j+1}**", "", "```", snip[:800], "```", ""])
    others = ev.get("other_entities_with_same_value") or []
    lines.extend([f"### 其他也含该值的实体（共 {len(others)} 个）", ""])
    if others:
        for o in others[:5]:
            lines.extend([f"- pid `{o['pid']}`:", "", "```", (o["snippets"][0] if o["snippets"] else "")[:400], "```", ""])
    else:
        lines.append("- （无）")
    if pack.get("all_cem_claims") and len(pack["all_cem_claims"]) > 1:
        lines.extend(["", "### 同轨迹其他 CEM claim", ""])
        for c in pack["all_cem_claims"]:
            if c.get("claim_id") != t.get("claim_id"):
                lines.append(f"- `{c.get('slot_norm')}` = `{c.get('value')}`")
    lines.extend(
        [
            "",
            "**【人工标注】** label: ______ （`true_CEM` / `ambiguous` / `coincidence`） | notes: ______",
            "",
            "---",
            "",
        ]
    )
    return "\n".join(lines)


def _md_ah_case(i: int, pack: dict[str, Any]) -> str:
    t = pack["annotation_target"]
    ev = pack["evidence_for_annotator"]
    lines = [
        f"## {i}. `{pack['trajectory_id']}`",
        "",
        "| 字段 | 值 |",
        "|---|---|",
        f"| slot | `{t.get('slot_norm')}` |",
        f"| **claim value（答案中）** | `{t.get('claim_value')}` |",
        f"| anchor pid | `{t.get('anchor_pid')}` |",
        f"| compared pid | `{t.get('compared_pid')}` |",
        f"| 启发式预标 | `{pack['heuristic_pre_label']}` — {pack['heuristic_reason']}` |",
        "",
        f"**答案中的 claim span**: {t.get('response_span')}",
        "",
        "### 模型答案（摘录）",
        "",
        "```",
        pack["model_answer_excerpt"][:1200],
        "```",
        "",
        "### Anchor 证据（应支持「anchor有证据但与claim矛盾」）",
        "",
    ]
    for j, ev_row in enumerate(ev.get("anchor_structured_evidence") or []):
        lines.extend(
            [
                f"**evidence #{j+1}** slot=`{ev_row.get('slot_norm')}` value=`{ev_row.get('value_norm')}`",
                "",
                "```",
                str(ev_row.get("value_raw") or "")[:800],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "### Anchor prompt 片段",
            "",
            "```",
            (ev.get("anchor_snippets") or [""])[0][:800],
            "```",
            "",
            "**【人工标注】** label: ______ （`true_AH` / `ambiguous` / `coincidence`） | notes: ______",
            "",
            "---",
            "",
        ]
    )
    return "\n".join(lines)


def _ah_dev_ids(rows: list[dict], split: dict[str, str]) -> list[str]:
    out: list[str] = []
    for r in rows:
        tid = r["trajectory_id"]
        if split.get(tid) != "dev":
            continue
        if not (r.get("eligible") or {}).get("counterfactual"):
            continue
        primary = None
        for c in r.get("claims") or []:
            if c.get("legacy_label") == "anchored_hallucination":
                primary = "AH"
                break
            if c.get("legacy_label") in ("cross_object_merge", "constraint_projection"):
                primary = "other"
                break
        if primary == "AH":
            out.append(tid)
    return sorted(out)


def export_manual_verification() -> dict[str, Any]:
    cohort = load_json(ROOT / "results/cohort_manifest.json")["cohorts"]["CEM"]
    split = load_json(SPLIT_MANIFEST_JSON)["splits"]
    rows = load_jsonl(NORMALIZED_SHOPPING)
    by_id = {r["trajectory_id"]: r for r in rows}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cem_packs = [_build_cem_pack(tid, by_id[tid]) for tid in cohort if tid in by_id]
    ah_ids = _ah_dev_ids(rows, split)
    ah_packs = [_build_ah_pack(tid, by_id[tid]) for tid in ah_ids if tid in by_id]

    cem_jsonl = OUT_DIR / "cem_manual_verification.jsonl"
    ah_jsonl = OUT_DIR / "ah_manual_verification.jsonl"
    write_jsonl(cem_jsonl, cem_packs)
    write_jsonl(ah_jsonl, ah_packs)

    cem_label_help = (
        "- **`true_CEM`**: claim 值可**唯一、明确**追溯到 rival 实体的证据片段（非巧合撞值）。\n"
        "- **`ambiguous`**: 多个实体都符合、或 rival 来源不清晰、或证据跨块难定位。\n"
        "- **`coincidence`**: 值太常见 / 多实体同值，更像随机撞值而非从 rival 抄取。"
    )
    ah_label_help = (
        "- **`true_AH`**: anchor 侧有明确证据，且与 claim 值形成**清晰矛盾**（非歧义）。\n"
        "- **`ambiguous`**: anchor 证据不足、矛盾不清晰，或 slot 对齐有歧义。\n"
        "- **`coincidence`**: 看似矛盾但可能来自同义表达、截断、或常见值误判。"
    )

    cem_md_parts = [
        _md_header("CEM Cohort 人工核验表（43 条）", len(cem_packs), cem_label_help),
    ]
    cem_heur = Counter(p["heuristic_pre_label"] for p in cem_packs)
    cem_md_parts.extend(
        [
            "## 启发式预标分布（仅供参考）",
            "",
            "| 预标 | n |",
            "|---|---:|",
        ]
    )
    for k, v in cem_heur.most_common():
        cem_md_parts.append(f"| {k} | {v} |")
    cem_md_parts.extend(["", "---", ""])
    for i, pack in enumerate(cem_packs, 1):
        cem_md_parts.append(_md_cem_case(i, pack))

    ah_md_parts = [
        _md_header(
            "AH Dev 人工核验表",
            len(ah_packs),
            ah_label_help,
            extra="> AH 全库共 74 条 primary-AH 轨迹；本表仅含 **dev split** 中 eligible 的条目。",
        ),
    ]
    ah_heur = Counter(p["heuristic_pre_label"] for p in ah_packs)
    ah_md_parts.extend(
        [
            "## 启发式预标分布（仅供参考）",
            "",
            "| 预标 | n |",
            "|---|---:|",
        ]
    )
    for k, v in ah_heur.most_common():
        ah_md_parts.append(f"| {k} | {v} |")
    ah_md_parts.extend(["", "---", ""])
    for i, pack in enumerate(ah_packs, 1):
        ah_md_parts.append(_md_ah_case(i, pack))

    cem_md = OUT_DIR / "cem_manual_verification.md"
    ah_md = OUT_DIR / "ah_manual_verification.md"
    cem_md.write_text("\n".join(cem_md_parts), encoding="utf-8")
    ah_md.write_text("\n".join(ah_md_parts), encoding="utf-8")

    summary = {
        "cem_n": len(cem_packs),
        "ah_n": len(ah_packs),
        "cem_heuristic": dict(cem_heur),
        "ah_heuristic": dict(ah_heur),
        "outputs": {
            "cem_md": str(cem_md),
            "cem_jsonl": str(cem_jsonl),
            "ah_md": str(ah_md),
            "ah_jsonl": str(ah_jsonl),
        },
    }
    (OUT_DIR / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = export_manual_verification()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
