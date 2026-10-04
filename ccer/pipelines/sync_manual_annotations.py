"""Sync manual annotation drafts into verification JSONL + MD, and emit audit summaries."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
MANUAL_DIR = ROOT / "results/reports/manual_verification"
DRAFT_DIR = MANUAL_DIR / "_draft_annotations"
AH_SPECIFICITY_JSON = ROOT / "results/reports/ah_specificity_check.json"

ANNOTATION_LINE_RE = re.compile(
    r"(\*\*【人工标注】\*\* label: )`[^`]*`( \| notes: ).*",
    re.MULTILINE,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_draft(pack: str) -> dict[str, dict[str, Any]]:
    path = DRAFT_DIR / f"{pack.lower()}_annotations.jsonl"
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["trajectory_id"]] = row
    return out


def _backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)


def merge_jsonl(pack: str, annotator: str, annotated_at: str | None = None) -> list[dict[str, Any]]:
    jsonl_path = MANUAL_DIR / f"{pack.lower()}_manual_verification.jsonl"
    draft = _load_draft(pack)
    ts = annotated_at or _now_iso()
    _backup(jsonl_path)

    updated: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tid = row["trajectory_id"]
        ann = draft.get(tid)
        if ann is None:
            raise KeyError(f"Missing draft annotation for {tid} in {pack}")
        row["human_annotation"] = {
            "label": ann["label"],
            "notes": ann["notes"],
            "annotator": annotator,
            "annotated_at": ts,
        }
        updated.append(row)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in updated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return updated


def sync_md(pack: str, rows: list[dict[str, Any]]) -> None:
    md_path = MANUAL_DIR / f"{pack.lower()}_manual_verification.md"
    text = md_path.read_text(encoding="utf-8")

    for row in rows:
        tid = row["trajectory_id"]
        ha = row["human_annotation"]
        label = ha["label"]
        notes = ha["notes"].replace("|", "\\|")
        pattern = re.compile(
            rf"(## \d+\. `{tid}`.*?)(\*\*【人工标注】\*\* label: )`[^`]*`( \| notes: ).*?(\n)",
            re.DOTALL,
        )

        def _repl(match: re.Match[str]) -> str:
            return f"{match.group(1)}{match.group(2)}`{label}`{match.group(3)}{notes}{match.group(4)}"

        new_text, n = pattern.subn(_repl, text, count=1)
        if n != 1:
            raise ValueError(f"Could not update MD annotation block for {tid}")
        text = new_text

    md_path.write_text(text, encoding="utf-8")


def _heuristic_agreement(rows: list[dict[str, Any]], draft: dict[str, dict[str, Any]]) -> tuple[int, int]:
    agree = 0
    total = 0
    for row in rows:
        tid = row["trajectory_id"]
        ann = draft.get(tid)
        if not ann:
            continue
        total += 1
        if ann.get("agrees_with_heuristic") is True:
            agree += 1
        elif ann.get("agrees_with_heuristic") is False:
            pass
        elif row.get("heuristic_pre_label") == ann["label"]:
            agree += 1
    return agree, total


def write_cem_summary(rows: list[dict[str, Any]], draft: dict[str, dict[str, Any]], annotator: str) -> None:
    labels = Counter(r["human_annotation"]["label"] for r in rows)
    heur = Counter(r.get("heuristic_pre_label") for r in rows)
    agree, total = _heuristic_agreement(rows, draft)

    ctab: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        flags = row["annotation_target"].get("quality_flags") or []
        flag = flags[0] if flags else "(none)"
        ctab[flag][row["human_annotation"]["label"]] += 1

    true_ids = [r["trajectory_id"] for r in rows if r["human_annotation"]["label"] == "true_CEM"]

    p1_all = {"n": 0, "scorable": 0, "follow": 0}
    p1_true = {"n": 0, "scorable": 0, "follow": 0}
    for row in rows:
        op = row.get("operator_status") or {}
        if op.get("operator") != "rival_value_swap":
            continue
        p1_all["n"] += 1
        if op.get("scorable"):
            p1_all["scorable"] += 1
            if op.get("source_following"):
                p1_all["follow"] += 1
        if row["human_annotation"]["label"] == "true_CEM":
            p1_true["n"] += 1
            if op.get("scorable"):
                p1_true["scorable"] += 1
                if op.get("source_following"):
                    p1_true["follow"] += 1

    patterns = Counter()
    for row in rows:
        if row["human_annotation"]["label"] != "coincidence":
            continue
        flags = set(row["annotation_target"].get("quality_flags") or [])
        if "value_also_on_selected_product" in flags:
            patterns["值在 anchor/选中侧可见"] += 1
        if "anchor_pid_eq_rival_pid" in flags:
            patterns["anchor_pid=rival_pid 标注混乱"] += 1
        if "value_on_selected_not_on_labeled_rival" in flags:
            patterns["值在选中侧不在标注 rival"] += 1
        patterns["多实体同值撞值"] += 1

    lines = [
        "# CEM 标注审计摘要",
        "",
        f"- 审计时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- 标注员: **{annotator}**",
        f"- 条目数: **{len(rows)}**",
        "",
        "## 标签分布",
        "",
        "| 标签 | n |",
        "|---|---|",
    ]
    for label in ("true_CEM", "ambiguous", "coincidence"):
        lines.append(f"| `{label}` | {labels.get(label, 0)} |")
    lines.extend(
        [
            "",
            "## 与启发式预标一致率",
            "",
            f"- 启发式分布: {dict(heur)}",
            f"- 人工一致: **{agree}/{total}** ({100 * agree / total:.1f}%)" if total else "- 人工一致: n/a",
            "",
            "## 质量标记 × 标签交叉表",
            "",
            "| quality_flag | true_CEM | ambiguous | coincidence |",
            "|---|---|---|---|",
        ]
    )
    for flag in sorted(ctab):
        lines.append(
            f"| `{flag}` | {ctab[flag].get('true_CEM', 0)} | {ctab[flag].get('ambiguous', 0)} | {ctab[flag].get('coincidence', 0)} |"
        )
    lines.extend(["", "## Clean CEM 子集 (true_CEM)", "", f"- n = **{len(true_ids)}**", ""])
    for tid in true_ids:
        lines.append(f"- `{tid}`")
    lines.extend(
        [
            "",
            "## P1 rival_value_swap 对照",
            "",
            "| 子集 | n | scorable | follow | follow率 |",
            "|---|---|---|---|---|",
        ]
    )
    for name, p1 in [("全体", p1_all), ("true_CEM clean", p1_true)]:
        rate = f"{100 * p1['follow'] / p1['scorable']:.1f}%" if p1["scorable"] else "—"
        lines.append(f"| {name} | {p1['n']} | {p1['scorable']} | {p1['follow']} | {rate} |")
    lines.extend(["", "## Top 误判模式 (coincidence 条目)", ""])
    for pat, n in patterns.most_common(5):
        lines.append(f"- **{pat}**: {n} 条")
    lines.append("")
    (MANUAL_DIR / "cem_annotation_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_ah_summary(rows: list[dict[str, Any]], draft: dict[str, dict[str, Any]], annotator: str) -> None:
    labels = Counter(r["human_annotation"]["label"] for r in rows)
    heur = Counter(r.get("heuristic_pre_label") for r in rows)
    agree, total = _heuristic_agreement(rows, draft)
    true_ids = [r["trajectory_id"] for r in rows if r["human_annotation"]["label"] == "true_AH"]

    spec = json.loads(AH_SPECIFICITY_JSON.read_text(encoding="utf-8"))
    spec_by_id = {r["trajectory_id"]: r for r in spec.get("results", [])}
    scorable = sum(1 for r in spec.get("results", []) if r.get("scorable"))
    follow = sum(1 for r in spec.get("results", []) if r.get("source_following"))

    patterns = Counter()
    for row in rows:
        tid = row["trajectory_id"]
        label = row["human_annotation"]["label"]
        heur_label = row.get("heuristic_pre_label")
        ann = draft.get(tid, {})
        if heur_label == "true_AH" and label == "ambiguous":
            patterns["启发式 true_AH 但 query/CAP 边界降为 ambiguous"] += 1
        if ann.get("primitives", {}).get("slot_absent"):
            patterns["slot 缺失而非矛盾"] += 1
        if label == "coincidence":
            patterns["子串/规格误读或 anchor 实际支持"] += 1
        if heur_label == "true_AH" and label == "coincidence":
            patterns["启发式 true_AH 但 anchor 实际支持/子串误读"] += 1

    lines = [
        "# AH 标注审计摘要",
        "",
        f"- 审计时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- 标注员: **{annotator}**",
        f"- 条目数: **{len(rows)}**",
        "",
        "## 标签分布",
        "",
        "| 标签 | n |",
        "|---|---|",
    ]
    for label in ("true_AH", "ambiguous", "coincidence"):
        lines.append(f"| `{label}` | {labels.get(label, 0)} |")
    lines.extend(["", f"## true_AH 子集: n = **{len(true_ids)}**", ""])
    for tid in true_ids:
        lines.append(f"- `{tid}`")
    if not true_ids:
        lines.append("- （无；dev 集在 EXPERT_FREEZE 下均为 query 硬约束复述或 slot 缺失，无 clean AH）")
    lines.extend(
        [
            "",
            "## 与启发式预标一致率",
            "",
            f"- 启发式分布: {dict(heur)}",
            f"- 人工一致: **{agree}/{total}** ({100 * agree / total:.1f}%)" if total else "- 人工一致: n/a",
            "",
            "## ah_specificity_check 对照",
            "",
            f"- P1 scorable: **{scorable}** / follow: **{follow}**",
            "",
            "| trajectory_id | 人工标签 | scorable | follow | invalid_reason |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        tid = row["trajectory_id"]
        s = spec_by_id.get(tid, {})
        inv = s.get("invalid_reason") or s.get("reason") or "—"
        follow_val = s.get("source_following")
        follow_str = "—" if follow_val is None else str(follow_val)
        lines.append(
            f"| `{tid}` | `{row['human_annotation']['label']}` | {s.get('scorable', False)} | {follow_str} | {inv} |"
        )
    lines.extend(["", "## 不应标为 AH 的案例模式", ""])
    for pat, n in patterns.most_common():
        lines.append(f"- **{pat}**: {n} 条")
    lines.append("")
    (MANUAL_DIR / "ah_annotation_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", default="john")
    parser.add_argument("--annotated-at", default=None)
    parser.add_argument("--pack", choices=("CEM", "AH", "both"), default="both")
    parser.add_argument("--skip-jsonl", action="store_true")
    parser.add_argument("--skip-md", action="store_true")
    parser.add_argument("--summaries-only", action="store_true")
    args = parser.parse_args()

    packs = ["CEM", "AH"] if args.pack == "both" else [args.pack]
    for pack in packs:
        draft = _load_draft(pack)
        if args.summaries_only:
            jsonl_path = MANUAL_DIR / f"{pack.lower()}_manual_verification.jsonl"
            rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            if not args.skip_jsonl:
                rows = merge_jsonl(pack, args.annotator, args.annotated_at)
            else:
                jsonl_path = MANUAL_DIR / f"{pack.lower()}_manual_verification.jsonl"
                rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not args.skip_md:
                sync_md(pack, rows)

        if pack == "CEM":
            write_cem_summary(rows, draft, args.annotator)
        else:
            write_ah_summary(rows, draft, args.annotator)

    print(json.dumps({"packs": packs, "annotator": args.annotator, "status": "ok"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
