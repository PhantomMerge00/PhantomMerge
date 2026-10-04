"""Task-correct re-judge on Line-A v3 test cohort after mitigation (RQ3)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "data/benchmarks/shoppingbench/scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from qa_rollout_integrity import last_tool_pids  # noqa: E402

from ccer.io_utils import load_jsonl, write_json
from ccer.mechanism.e2e_mitigation_experiment import E2E_METHOD_SPECS
from ccer.mechanism.line_k_claim_filter import load_frozen_probe_scores
from ccer.mechanism.prism_l_plus_experiment import METHOD_SPECS as PROBE_SPECS
from ccer.mechanism.pair_select import load_trajectory_index

DEFAULT_ROLLOUT = (
    ROOT
    / "data/rollouts/shopping/shopping_qwen3-32b_rollout/rollout.jsonl"
)
DEFAULT_GOLD = ROOT / "data/benchmarks/shoppingbench/data/synthesize_product_seal_mix.jsonl"
E2E_LOG = ROOT / "results/e2e_mitigation/rewrite_log.jsonl"
PROBE_LOG = ROOT / "results/prism_l_plus/rewrite_log.jsonl"
OUT_JSON = ROOT / "results/rq3/shopping_mitigation_task_rejudge.json"
OUT_MD = ROOT / "results/rq3/SHOPPING_MITIGATION_TASK_REJUDGE.md"
TAU = 0.05

E2E_ARM_TO_KEY = {str(v["label"]): k for k, v in E2E_METHOD_SPECS.items()}
PROBE_ARM_TO_KEY = {str(v["label"]): k for k, v in PROBE_SPECS.items()}


def seal_id_from_steps(steps: list) -> str | None:
    for s in steps:
        if not isinstance(s, dict):
            continue
        ei = s.get("extra_info") or {}
        tid = ei.get("seal_task_id") or ei.get("trajectory_id")
        if tid:
            return str(tid)
    return None


def load_gold(path: Path) -> dict[str, str]:
    gold_by: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        tid = str(g.get("seal_task_id") or "").strip()
        if tid:
            gid = str((g.get("reward") or {}).get("product_id") or "")
            gold_by[tid] = gid
    return gold_by


def load_rollouts(path: Path) -> dict[str, list[dict]]:
    by_tid: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        steps = json.loads(line)
        if not isinstance(steps, list):
            continue
        tid = seal_id_from_steps(steps)
        if tid:
            by_tid[tid] = steps
    return by_tid


def build_traj_to_rollout_id(traj_index: dict[str, dict]) -> dict[str, str]:
    """Map Line-A trajectory_id -> rollout seal_task_id key."""
    out: dict[str, str] = {}
    for tid, traj in traj_index.items():
        raw = traj.get("raw_ref") or {}
        roll_tid = str(raw.get("trajectory_id") or tid)
        out[tid] = roll_tid
    return out


def apply_edits_to_steps(steps: list[dict], edits: list[dict]) -> list[dict]:
    steps = copy.deepcopy(steps)
    for ed in edits:
        qb = str(ed.get("quote_before") or ed.get("response_quote") or "")
        if not qb:
            continue
        action = str(ed.get("action") or "keep")
        qa = ed.get("quote_after")
        replaced = False
        for step in steps:
            if not isinstance(step, dict):
                continue
            for msg in step.get("prompt") or []:
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content") or ""
                if qb in content:
                    if action == "delete" or qa is None:
                        msg["content"] = content.replace(qb, "").strip()
                    else:
                        msg["content"] = content.replace(qb, str(qa), 1)
                    replaced = True
            ei = step.get("extra_info") or {}
            for msg in ei.get("completion") or []:
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content") or ""
                if qb in content:
                    if action == "delete" or qa is None:
                        msg["content"] = content.replace(qb, "").strip()
                    else:
                        msg["content"] = content.replace(qb, str(qa), 1)
                    replaced = True
        if not replaced:
            for step in steps:
                ei = step.get("extra_info") or {}
                comp = ei.get("agent_completion") or ei.get("final_response")
                if isinstance(comp, str) and qb in comp:
                    if action == "delete" or qa is None:
                        ei["agent_completion"] = comp.replace(qb, "").strip()
                    else:
                        ei["agent_completion"] = comp.replace(qb, str(qa), 1)
    return steps


def score_steps(steps: list[dict], gold_pid: str) -> dict[str, Any]:
    rec = last_tool_pids(steps, "recommend_product")
    if not rec:
        return {"status": "no_recommend", "top1_hit": False}
    if len(rec) > 2:
        return {"status": "non_compliant_gt2", "top1_hit": False, "rec": rec[:3]}
    return {"status": "ok", "top1_hit": rec[0] == gold_pid, "rec": rec[:2]}


def method_key_from_row(row: dict, *, axis: str) -> str | None:
    arm = str(row.get("arm") or "")
    if axis == "e2e":
        if arm in E2E_ARM_TO_KEY:
            return E2E_ARM_TO_KEY[arm]
        m = str(row.get("method") or "")
        if m in E2E_METHOD_SPECS:
            return m
        if m == "track_k_e2e":
            return "track_k_e2e"
    else:
        if arm in PROBE_ARM_TO_KEY:
            return PROBE_ARM_TO_KEY[arm]
        m = str(row.get("method") or "")
        if m in PROBE_SPECS:
            return m
    return None


def group_logs(
    path: Path, axis: str, test_trajs: set[str], *, tau: float
) -> dict[str, dict[str, list[dict]]]:
    """method_key -> trajectory_id -> edits."""
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    if not path.is_file():
        return out
    for row in load_jsonl(path):
        if str(row.get("split") or "") != "test":
            continue
        if abs(float(row.get("tau", tau)) - tau) > 1e-6:
            continue
        tid = str(row.get("trajectory_id") or "")
        if tid not in test_trajs:
            continue
        mk = method_key_from_row(row, axis=axis)
        if not mk:
            continue
        out[mk][tid].append(row)
    return out


def eval_method(
    method_key: str,
    edits_by_traj: dict[str, list[dict]],
    test_trajs: set[str],
    traj_to_rollout: dict[str, str],
    rollouts: dict[str, list[dict]],
    gold: dict[str, str],
) -> dict[str, Any]:
    top1 = 0
    n_tc = 0
    pm_trajs = set()
    pm_top1 = 0
    miss_rollout = 0
    miss_gold = 0
    for tid in test_trajs:
        roll_key = traj_to_rollout.get(tid, tid)
        steps = rollouts.get(roll_key)
        if not steps:
            miss_rollout += 1
            continue
        gid = gold.get(roll_key)
        if not gid:
            miss_gold += 1
            continue
        edited = apply_edits_to_steps(steps, edits_by_traj.get(tid, []))
        sc = score_steps(edited, gid)
        if sc["status"] != "ok":
            continue
        n_tc += 1
        if sc["top1_hit"]:
            top1 += 1
    return {
        "method_key": method_key,
        "tc_n": n_tc,
        "tc_top1": top1,
        "tc_rate": (top1 / n_tc) if n_tc else None,
        "miss_rollout": miss_rollout,
        "miss_gold": miss_gold,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout", type=Path, default=DEFAULT_ROLLOUT)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args(argv)
    tau = float(args.tau)

    scores = load_frozen_probe_scores()
    test_trajs = {str(r["trajectory_id"]) for r in scores if r.get("split") == "test"}
    traj_index = load_trajectory_index()
    traj_to_rollout = build_traj_to_rollout_id(traj_index)
    rollouts = load_rollouts(args.rollout)
    gold = load_gold(args.gold)

    baseline_top1 = 0
    baseline_n = 0
    for tid in test_trajs:
        roll_key = traj_to_rollout.get(tid, tid)
        steps = rollouts.get(roll_key)
        gid = gold.get(roll_key)
        if not steps or not gid:
            continue
        sc = score_steps(steps, gid)
        if sc["status"] != "ok":
            continue
        baseline_n += 1
        if sc["top1_hit"]:
            baseline_top1 += 1
    baseline_rate = (baseline_top1 / baseline_n) if baseline_n else None

    e2e_groups = group_logs(E2E_LOG, "e2e", test_trajs, tau=tau)
    probe_groups = group_logs(PROBE_LOG, "probe", test_trajs, tau=tau)

    results: list[dict[str, Any]] = []
    for mk in E2E_METHOD_SPECS:
        if mk == "track_k_e2e":
            continue
        block = eval_method(mk, e2e_groups.get(mk, {}), test_trajs, traj_to_rollout, rollouts, gold)
        block["axis"] = "e2e"
        block["delta_tc"] = (
            (block["tc_rate"] - baseline_rate) if block["tc_rate"] is not None and baseline_rate is not None else None
        )
        results.append(block)
    if "track_k_e2e" in E2E_METHOD_SPECS:
        block = eval_method(
            "track_k_e2e", e2e_groups.get("track_k_e2e", {}), test_trajs, traj_to_rollout, rollouts, gold
        )
        block["axis"] = "e2e"
        block["delta_tc"] = (
            (block["tc_rate"] - baseline_rate) if block["tc_rate"] is not None and baseline_rate is not None else None
        )
        results.append(block)

    for mk in PROBE_SPECS:
        block = eval_method(mk, probe_groups.get(mk, {}), test_trajs, traj_to_rollout, rollouts, gold)
        block["axis"] = "probe_gated"
        block["delta_tc"] = (
            (block["tc_rate"] - baseline_rate) if block["tc_rate"] is not None and baseline_rate is not None else None
        )
        results.append(block)

    payload = {
        "schema_version": "ccer_shopping_mitigation_task_rejudge_v1",
        "cohort": "line_a_v3_test",
        "n_trajectories": len(test_trajs),
        "tau": tau,
        "baseline": {
            "tc_n": baseline_n,
            "tc_top1": baseline_top1,
            "tc_rate": baseline_rate,
        },
        "methods": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out_json, payload)

    lines = [
        "# Shopping mitigation — task-correct re-judge (Line-A test)",
        "",
        f"**Cohort**: {len(test_trajs)} trajectories | **τ** = {tau}",
        f"**Baseline TC (no filter)**: {baseline_top1}/{baseline_n} "
        f"({baseline_rate:.3f} rate)" if baseline_rate is not None else "",
        "",
        "| Axis | Method | TC top1 | TC N | Rate | Δ vs baseline |",
        "|------|--------|--------:|-----:|-----:|--------------:|",
    ]
    for row in results:
        rate = row.get("tc_rate")
        delta = row.get("delta_tc")
        rate_s = f"{rate:.3f}" if rate is not None else "—"
        delta_s = f"{delta:+.3f}" if delta is not None else "—"
        lines.append(
            f"| {row.get('axis')} | {row.get('method_key')} | {row.get('tc_top1')} | "
            f"{row.get('tc_n')} | {rate_s} | {delta_s} |"
        )
    lines.append("")
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
