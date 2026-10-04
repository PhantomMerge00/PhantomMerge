"""Generate PRISM-L+ forensic case report from prior Line L+ and PRISM audits."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import load_jsonl, write_json
from ccer.mechanism.prism_audit import build_audit_index
from ccer.mechanism.supervised_probe import PM_VERDICTS
from ccer.paths import (
    LINE_L_PLUS_REWRITE_LOG,
    PRISM_AUDIT_JSONL,
    PRISM_L_PLUS_DIR,
    PRISM_L_PLUS_FORENSIC,
    PRISM_L_PLUS_FORENSIC_JSON,
)

_PID_VALUE_RE = re.compile(r"^\d{8,12}$")


def _load_logs(path: Path) -> dict[tuple, dict]:
    if not path.is_file():
        return {}
    seen: dict[tuple, dict] = {}
    for row in load_jsonl(path):
        key = (
            str(row.get("arm") or ""),
            float(row.get("tau", 0)),
            str(row.get("instance_audit_key") or row.get("claim_id") or ""),
        )
        seen[key] = row
    return seen


def _by_arm(logs: dict[tuple, dict], arm: str, tau: float = 0.05) -> dict[str, dict]:
    return {
        str(k[2]): v
        for k, v in logs.items()
        if k[0] == arm and abs(k[1] - tau) < 1e-9
    }


def _is_pm(row: dict) -> bool:
    if int(row.get("y_pm", row.get("y", 0)) or 0) == 1:
        return True
    return str(row.get("gold_verdict") or "") in PM_VERDICTS


def run_forensic_diff(*, tau: float = 0.05) -> dict:
    logs = _load_logs(LINE_L_PLUS_REWRITE_LOG)
    b1 = _by_arm(logs, "B1_RARR_adapted", tau)
    b2 = _by_arm(logs, "B2_CoVe_adapted", tau)
    ours = _by_arm(logs, "Ours_Line_L_plus_v3", tau)
    b3 = _by_arm(logs, "B3_Copy_constrained", tau)

    audit_index = build_audit_index(list(load_jsonl(PRISM_AUDIT_JSONL))) if PRISM_AUDIT_JSONL.is_file() else {}

    b1_ours_diff: list[dict] = []
    for iak, r1 in b1.items():
        ro = ours.get(iak)
        if not ro:
            continue
        if r1.get("action") == "rewrite" and ro.get("action") == "delete":
            aud = audit_index.get(iak, {})
            b1_ours_diff.append(
                {
                    "instance_audit_key": iak,
                    "gold_verdict": r1.get("gold_verdict"),
                    "slot_norm": r1.get("slot_norm"),
                    "prism_audit_state": aud.get("audit_state"),
                    "y_pm": r1.get("y_pm"),
                    "b1_branch": r1.get("branch"),
                    "ours_branch": ro.get("branch"),
                    "ours_miss": ro.get("miss_reason"),
                    "quote_before": r1.get("quote_before"),
                    "b1_after": r1.get("quote_after"),
                    "v_anchor": r1.get("v_anchor"),
                }
            )

    pid_bad: list[dict] = []
    for iak, r1 in b1.items():
        if r1.get("branch") != "rarr_verify_pass":
            continue
        v = str(r1.get("v_anchor") or "")
        if _PID_VALUE_RE.match(v.strip()):
            pid_bad.append(
                {
                    "instance_audit_key": iak,
                    "slot_norm": r1.get("slot_norm"),
                    "quote_before": r1.get("quote_before"),
                    "quote_after": r1.get("quote_after"),
                    "v_anchor": v,
                }
            )

    b1_b2_same = 0
    b1_b2_total = 0
    for iak, r1 in b1.items():
        r2 = b2.get(iak)
        if not r2:
            continue
        b1_b2_total += 1
        if r1.get("action") == r2.get("action") and r1.get("quote_after") == r2.get("quote_after"):
            b1_b2_same += 1

    by_verdict = Counter(r.get("gold_verdict") for r in b1_ours_diff)
    by_prism = Counter(r.get("prism_audit_state") for r in b1_ours_diff)
    confirmed_risk = [
        r for r in b1_ours_diff if r.get("prism_audit_state") == "confirmed_risk"
    ]

    return {
        "tau": tau,
        "n_b1_rewrite_ours_delete": len(b1_ours_diff),
        "by_gold_verdict": dict(by_verdict),
        "by_prism_audit_state": dict(by_prism),
        "n_confirmed_risk_in_diff": len(confirmed_risk),
        "n_pid_as_value_bad": len(pid_bad),
        "b1_b2_identical_rate": (b1_b2_same / b1_b2_total) if b1_b2_total else None,
        "b1_b2_identical_count": b1_b2_same,
        "b1_b2_total": b1_b2_total,
        "samples_b1_ours_diff": b1_ours_diff[:20],
        "samples_pid_bad": pid_bad[:15],
        "n_b3_rewrite": sum(1 for r in b3.values() if r.get("action") == "rewrite"),
        "n_ours_rewrite": sum(1 for r in ours.values() if r.get("action") == "rewrite"),
        "n_b1_rewrite": sum(1 for r in b1.values() if r.get("action") == "rewrite"),
    }


def _render_md(summary: dict) -> str:
    lines = [
        "# PRISM-L+ Forensic Cases",
        "",
        f"τ = {summary.get('tau')}",
        "",
        "## Summary",
        "",
        f"- B1 rewrite / Ours delete: **{summary.get('n_b1_rewrite_ours_delete')}**",
        f"- PID-as-value bad rewrites (B1): **{summary.get('n_pid_as_value_bad')}**",
        f"- B1 vs B2 identical outputs: **{summary.get('b1_b2_identical_count')}/{summary.get('b1_b2_total')}** "
        f"({summary.get('b1_b2_identical_rate', 0):.1%})",
        f"- Rewrite counts: B1={summary.get('n_b1_rewrite')}, B3={summary.get('n_b3_rewrite')}, "
        f"Ours={summary.get('n_ours_rewrite')}",
        "",
        "## B1✓ / Ours✗ by gold_verdict",
        "",
        "```json",
        json.dumps(summary.get("by_gold_verdict") or {}, indent=2),
        "```",
        "",
        "## B1✓ / Ours✗ by PRISM audit_state",
        "",
        "```json",
        json.dumps(summary.get("by_prism_audit_state") or {}, indent=2),
        "```",
        "",
        f"Confirmed-risk in diff subset: **{summary.get('n_confirmed_risk_in_diff')}**",
        "",
        "## Sample B1 rewrite / Ours delete",
        "",
    ]
    for i, r in enumerate(summary.get("samples_b1_ours_diff") or [], 1):
        lines.append(f"### Case {i}: `{r.get('instance_audit_key', '')[:50]}`")
        lines.append(f"- verdict: {r.get('gold_verdict')} | prism: {r.get('prism_audit_state')}")
        lines.append(f"- before: {str(r.get('quote_before') or '')[:80]}")
        lines.append(f"- B1 after: {str(r.get('b1_after') or '')[:80]}")
        lines.append(f"- Ours: {r.get('ours_branch')} ({r.get('ours_miss')})")
        lines.append("")

    lines.append("## Sample PID-as-value bad rewrites (B1)")
    lines.append("")
    for i, r in enumerate(summary.get("samples_pid_bad") or [], 1):
        lines.append(
            f"{i}. `{r.get('slot_norm')}`: {str(r.get('quote_before') or '')[:40]} → "
            f"{str(r.get('quote_after') or '')[:50]}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    PRISM_L_PLUS_DIR.mkdir(parents=True, exist_ok=True)
    summary = run_forensic_diff(tau=0.05)
    write_json(PRISM_L_PLUS_FORENSIC_JSON, summary)
    PRISM_L_PLUS_FORENSIC.write_text(_render_md(summary), encoding="utf-8")
    print(f"Wrote {PRISM_L_PLUS_FORENSIC}")
    print(f"Wrote {PRISM_L_PLUS_FORENSIC_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
