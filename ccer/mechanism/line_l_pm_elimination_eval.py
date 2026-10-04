"""PM elimination metrics (acceptance gate) vs attribution metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Literal

from ccer.mechanism.line_l_attribution_eval import anchor_evidence_attribution_rate
from ccer.mechanism.line_l_rewrite_acceptance import (
    RewriteVerifyPolicy,
    acceptance_verify_bundle,
    get_rewrite_verify_policy,
    verify_rewrite_row_acceptance,
)
from ccer.mechanism.line_l_rewrite_quality import build_evidence_corpus, extract_value_from_quote
from ccer.mechanism.line_l_wrong_anchor import (
    resolve_acceptance_pid_committed,
    resolve_acceptance_pid_textual,
)

_PM_BUCKET = {
    "cross_object_merge": "CEM",
    "constraint_projection": "CAP",
    "anchored_hallucination": "AH",
}


def _rewrite_rows(logs: list[dict[str, Any]], tau: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in logs:
        if float(row.get("p_pm", 0)) <= float(tau):
            continue
        if str(row.get("action") or "") != "rewrite":
            continue
        out.append(row)
    return out


def pm_elimination_for_rewrite(
    row: dict[str, Any],
    traj: dict[str, Any],
    *,
    acceptance_mode: Literal["textual", "committed"] = "textual",
    policy: RewriteVerifyPolicy | None = None,
) -> dict[str, Any]:
    if acceptance_mode == "committed":
        pid = resolve_acceptance_pid_committed(row, traj)
    else:
        pid = resolve_acceptance_pid_textual(row, traj)
    bundle = verify_rewrite_row_acceptance(
        row, traj, acceptance_pid=pid, policy=policy
    )
    return {**bundle, "acceptance_mode": acceptance_mode, "acceptance_pid": pid}


def aggregate_pm_elimination_metrics(
    logs: list[dict[str, Any]],
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    policy: RewriteVerifyPolicy | None = None,
) -> dict[str, Any]:
    pol = policy or get_rewrite_verify_policy()
    rewrites = _rewrite_rows(logs, tau)
    if not rewrites:
        return {
            "n_rewrite": 0,
            "policy": pol,
            "pm_elimination_rate_textual": None,
            "pm_elimination_rate_committed": None,
            "still_pm_proxy_textual": None,
            "still_pm_proxy_committed": None,
            "n_pid_mismatch": 0,
            "by_gold_verdict": {},
        }

    ok_text = ok_com = 0
    mismatch = 0
    by_verdict: dict[str, dict[str, int]] = {}

    for row in rewrites:
        tid = str(row.get("trajectory_id") or "")
        traj = traj_index.get(tid, {})
        pt = resolve_acceptance_pid_textual(row, traj)
        pc = resolve_acceptance_pid_committed(row, traj)
        if pt and pc and pt != pc:
            mismatch += 1
        bt = pm_elimination_for_rewrite(
            row, traj, acceptance_mode="textual", policy=pol
        )
        bc = pm_elimination_for_rewrite(
            row, traj, acceptance_mode="committed", policy=pol
        )
        ok_text += int(bool(bt.get("ok")))
        ok_com += int(bool(bc.get("ok")))
        gv = str(row.get("gold_verdict") or "unknown")
        bucket = _PM_BUCKET.get(gv, "other")
        slot = by_verdict.setdefault(bucket, {"n": 0, "ok_textual": 0, "ok_committed": 0})
        slot["n"] += 1
        slot["ok_textual"] += int(bool(bt.get("ok")))
        slot["ok_committed"] += int(bool(bc.get("ok")))

    n = len(rewrites)
    by_rates = {
        k: {
            **v,
            "pm_elimination_rate_textual": v["ok_textual"] / v["n"] if v["n"] else None,
            "pm_elimination_rate_committed": v["ok_committed"] / v["n"] if v["n"] else None,
        }
        for k, v in by_verdict.items()
    }
    return {
        "n_rewrite": n,
        "policy": pol,
        "pm_elimination_rate_textual": ok_text / n,
        "pm_elimination_rate_committed": ok_com / n,
        "still_pm_proxy_textual": 1.0 - ok_text / n,
        "still_pm_proxy_committed": 1.0 - ok_com / n,
        "n_pid_mismatch": mismatch,
        "by_gold_verdict": by_rates,
    }


def aggregate_rewrite_quality_v2(
    logs: list[dict[str, Any]],
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    policy: RewriteVerifyPolicy | None = None,
) -> dict[str, Any]:
    pol = policy or get_rewrite_verify_policy()
    flagged = [r for r in logs if float(r.get("p_pm", 0)) > float(tau)]
    rewrites = [r for r in flagged if str(r.get("action") or "") == "rewrite"]
    n_delete = sum(1 for r in flagged if str(r.get("action") or "") == "delete")
    n_rules = n_nli = n_both = 0
    for row in rewrites:
        tid = str(row.get("trajectory_id") or "")
        traj = traj_index.get(tid, {})
        pid = resolve_acceptance_pid_textual(row, traj)
        v = str(row.get("v_anchor") or extract_value_from_quote(str(row.get("quote_after") or "")) or "")
        corpus = build_evidence_corpus(traj, pid)
        src = str(row.get("candidate_source") or row.get("source_field") or "")
        if "obs_tau" in src:
            src = "obs_tau_window"
        b = acceptance_verify_bundle(
            v,
            str(row.get("slot_norm") or ""),
            corpus,
            row=row,
            traj=traj,
            acceptance_pid=pid,
            candidate_source=src,
            policy=pol,
        )
        n_rules += int(bool(b.get("rules_ok")))
        n_nli += int(bool(b.get("nli_ok")))
        n_both += int(bool(b.get("rules_ok")) and bool(b.get("nli_ok")))
    n = len(rewrites)
    return {
        "policy": pol,
        "n_flagged": len(flagged),
        "n_rewrite": n,
        "n_delete": n_delete,
        "n_pass_rules": n_rules,
        "n_pass_nli": n_nli,
        "n_pass_both": n_both,
    }


def render_fac_rewrite_metrics_markdown(
    blocks: list[dict[str, Any]],
    *,
    policy_ablation: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# FAC rewrite metrics: attribution vs PM elimination",
        "",
        "| Method | n_rewrite | Attribution (committed) | PM elim. (textual) | PM elim. (committed) | still_pm proxy (textual) |",
        "|--------|-----------|-------------------------|--------------------|----------------------|--------------------------|",
    ]
    for b in blocks:
        attr = b.get("attribution_rate_committed")
        pm = b.get("pm_elimination_metrics") or {}
        lines.append(
            f"| {b.get('label', b.get('method_key', ''))} "
            f"| {pm.get('n_rewrite', b.get('n_rewrite', ''))} "
            f"| {attr if attr is not None else '—'} "
            f"| {pm.get('pm_elimination_rate_textual', '—')} "
            f"| {pm.get('pm_elimination_rate_committed', '—')} "
            f"| {pm.get('still_pm_proxy_textual', '—')} |"
        )
    lines.extend(
        [
            "",
            "## 策略选型（57 条盲审队列 replay）",
            "",
        ]
    )
    if policy_ablation:
        for pol, stats in policy_ablation.items():
            lines.append(
                f"- **{pol}**: pass={stats.get('n_pass')}/{stats.get('n_total')} "
                f"({stats.get('pass_rate')})"
            )
    else:
        lines.append("- （待 ablation 脚本填充）")
    lines.append("")
    lines.append(
        "> 旧版 anchor_evidence_attribution 57/57 **不等于** PM 已消除；以 `pm_elimination_rate_*` 为准。"
    )
    return "\n".join(lines)


def replay_audit_queue_verify_policies(
    queue_path: Path,
    traj_index: dict[str, dict[str, Any]],
    policies: tuple[RewriteVerifyPolicy, ...] = (
        "rules_only",
        "nli_only",
        "rules_and_nli",
        "rules_or_nli",
    ),
) -> dict[str, Any]:
    """Offline ablation on human audit queue rows (uses v_anchor replay)."""
    rows: list[dict[str, Any]] = []
    if queue_path.is_file():
        for line in queue_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    results: dict[str, Any] = {}
    for pol in policies:
        n_pass = 0
        for row in rows:
            tid = str(row.get("trajectory_id") or "")
            traj = traj_index.get(tid, {})
            pid = resolve_acceptance_pid_textual(
                {
                    **row,
                    "committed_anchor_pid": row.get("committed_anchor_pid"),
                    "textual_selected_pid": row.get("textual_selected_pid"),
                },
                traj,
            ) or str(row.get("committed_anchor_pid") or "")
            replay_row = {
                "slot_norm": row.get("slot_norm"),
                "v_anchor": row.get("v_anchor"),
                "quote_after": row.get("quote_after"),
                "candidate_source": "obs_tau_window" if "ts:[" in str(row.get("v_anchor") or "") else "",
            }
            b = verify_rewrite_row_acceptance(replay_row, traj, acceptance_pid=pid, policy=pol)
            n_pass += int(bool(b.get("ok")))
        n = len(rows)
        results[pol] = {
            "n_total": n,
            "n_pass": n_pass,
            "pass_rate": round(n_pass / n, 4) if n else None,
        }
    return results


def build_method_eval_bundle(
    logs: list[dict[str, Any]],
    traj_index: dict[str, dict[str, Any]],
    *,
    tau: float,
    method_key: str,
    label: str,
) -> dict[str, Any]:
    pm = aggregate_pm_elimination_metrics(logs, traj_index, tau=tau)
    qv2 = aggregate_rewrite_quality_v2(logs, traj_index, tau=tau)
    attr = anchor_evidence_attribution_rate(
        logs,
        traj_index,
        tau=tau,
        anchor_pid_fn=lambda r, t: resolve_acceptance_pid_committed(r, t),
    )
    return {
        "method_key": method_key,
        "label": label,
        "pm_elimination_metrics": pm,
        "rewrite_quality_v2": qv2,
        "attribution_rate_committed": attr.get("anchor_evidence_attribution_rate"),
        "n_rewrite": pm.get("n_rewrite"),
    }
