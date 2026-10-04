"""Run LLM fallback only on extraction_miss flagged claims (resumable cache)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_jsonl
from ccer.mechanism.line_k_claim_filter import load_frozen_dev_thresholds, load_frozen_probe_scores
from ccer.mechanism.line_k_l_tiered import apply_tiered_mitigation
from ccer.mechanism.line_l_generative_rewrite import load_llm_cache
from ccer.mechanism.line_l_claim_filter import _enrich_score_rows
from ccer.mechanism.line_l_wrong_anchor import resolve_committed_anchor_pid
from ccer.mechanism.pair_select import load_trajectory_index
from ccer.paths import LINE_L_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-name", default="aggressive", choices=("aggressive", "balanced"))
    ap.add_argument("--limit", type=int, default=0, help="Max LLM calls (0=all misses)")
    args = ap.parse_args()

    thresholds = load_frozen_dev_thresholds()
    tau = float(
        thresholds["line_a_probe_aggressive"]
        if args.tau_name == "aggressive"
        else thresholds["line_a_probe_balanced"]
    )
    traj_index = load_trajectory_index()
    load_llm_cache()

    rows = [r for r in _enrich_score_rows(load_frozen_probe_scores()) if r["split"] == "test"]
    misses = []
    for row in rows:
        if float(row["p_pm"]) <= tau:
            continue
        traj = traj_index.get(row["trajectory_id"], {})
        anchor = resolve_committed_anchor_pid(dict(row), traj)
        ext = apply_tiered_mitigation(
            {**row, "tau": tau},
            traj,
            anchor_pid=anchor,
            policy="delete_or_rewrite_v1",
            rewrite_fallback="none",
        )
        if ext.get("branch") in ("extraction_miss", "skip_no_slot"):
            misses.append((row, traj, anchor))

    if args.limit > 0:
        misses = misses[: args.limit]

    out_path = LINE_L_DIR / f"llm_fallback_only_{args.tau_name}.jsonl"
    results = []
    for i, (row, traj, anchor) in enumerate(misses, 1):
        out = apply_tiered_mitigation(
            {**row, "tau": tau},
            traj,
            anchor_pid=anchor,
            rewrite_fallback="llm",
        )
        results.append(out)
        if i % 10 == 0:
            print(f"[llm_fallback] {i}/{len(misses)}", flush=True)

    write_jsonl(out_path, results)
    n_llm = sum(1 for r in results if r.get("evidence_kind") == "fallback_llm")
    print(json.dumps({"n_miss": len(misses), "n_llm": n_llm, "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
