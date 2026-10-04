"""P2 B0–B4/M0 comparison on dev cohort."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.eval.repair_eval import aggregate_repair_comparison, evaluate_repair_output
from ccer.io_utils import load_json, load_jsonl, write_json, write_jsonl
from ccer.paths import COHORT_MANIFEST_JSON, NORMALIZED_SHOPPING, P2_DIR, REPORTS
from ccer.repair.b4_verifier_repair import run_b4
from ccer.repair.baselines import (
    b0_original,
    b1_anchor_prompt,
    b2_evidence_plan,
    b3_anchor_only,
    build_evidence_plan_from_trajectory,
    m0_format_only,
)
from ccer.replay.fixed_history import extract_answer
from ccer.replay.vllm_replay import check_vllm_available, vllm_generate


def run_p2_dev(*, dry_run: bool = False, limit: int = 5) -> dict[str, Any]:
    rows = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}
    cohort = load_json(COHORT_MANIFEST_JSON)
    dev_tids = (
        cohort.get("cohorts", {}).get("CEM", [])[:limit]
        + cohort.get("cohorts", {}).get("CAP", [])[:limit]
        + cohort.get("cohorts", {}).get("matched_clean", [])[:limit]
    )
    dev_tids = list(dict.fromkeys(dev_tids))[: limit * 3]
    print(f"[P2] n_trajectories={len(dev_tids)} limit={limit}", flush=True)

    if not dry_run and not check_vllm_available():
        return {"status": "blocked", "reason": "vllm_unavailable"}

    eval_rows: list[dict] = []
    P2_DIR.mkdir(parents=True, exist_ok=True)

    for ti, tid in enumerate(dev_tids, 1):
        traj = rows.get(tid)
        if not traj or not traj.get("eligible", {}).get("native_replay"):
            continue
        print(f"[P2] {ti}/{len(dev_tids)} {tid}", flush=True)
        base = traj["messages_final_call"]
        plan = build_evidence_plan_from_trajectory(traj)
        anchor = (traj.get("commitment") or {}).get("action_anchor")

        methods = {
            "B0": b0_original(base),
            "B1": b1_anchor_prompt(base),
            "B2": b2_evidence_plan(base, plan),
            "B3": b3_anchor_only(base, anchor_pid=anchor),
            "M0": m0_format_only(base, plan),
        }

        for mid, msgs in methods.items():
            if dry_run:
                eval_rows.append({"method_id": mid, "trajectory_id": tid, "dry_run": True})
                continue
            print(f"[P2] {ti}/{len(dev_tids)} {tid} {mid}", flush=True)
            text, meta = vllm_generate(msgs)
            ans = extract_answer(text)
            eval_rows.append(
                evaluate_repair_output(traj, method_id=mid, generated_answer=ans, latency=meta)
            )

        if not dry_run:
            b4_out = run_b4(traj, base, max_retries=2)
            eval_rows.append(
                evaluate_repair_output(
                    traj,
                    method_id="B4",
                    generated_answer=extract_answer(b4_out["answer"]),
                    latency=b4_out["history"][-1]["latency"] if b4_out.get("history") else None,
                )
            )

    summary = aggregate_repair_comparison([r for r in eval_rows if not r.get("dry_run")])
    if not summary and eval_rows:
        summary = {"dry_run": True, "n_planned_evals": len(eval_rows)}
    write_jsonl(P2_DIR / "repair_eval_rows.jsonl", eval_rows)
    write_json(P2_DIR / "repair_comparison.json", summary)
    REPORTS.mkdir(parents=True, exist_ok=True)
    md = ["# P2 repair trade-off", ""]
    for mid, s in summary.items():
        md.append(f"## {mid}")
        md.append(f"- n={s['n']} mean_new_pm={s['mean_new_pm']:.3f} retention={s['mean_supported_retention']}")
    (REPORTS / "p2_tradeoff.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    print(json.dumps(run_p2_dev(dry_run=args.dry_run, limit=args.limit), indent=2, default=str))
