#!/usr/bin/env python3
"""P2 vendor PM detection baselines (clone + ADAPTATION)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arms",
        default="selfcheck_nli,selfcheck_nli_multi,factool_kbqa,factscore",
        help="selfcheck_nli,selfcheck_nli_multi,factool_kbqa,factscore",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selfcheck-k", type=int, default=5)
    ap.add_argument("--factscore-limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--factool-limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/rq2/p2_vendor")
    args = ap.parse_args()
    arms = {a.strip() for a in args.arms.split(",") if a.strip()}
    results: dict = {}
    if "selfcheck_nli" in arms:
        from ccer.vendor_baselines.selfcheck_nli_obs_tau import run_selfcheck_nli_obs_tau

        results["selfcheck_nli"] = run_selfcheck_nli_obs_tau(
            device=args.device, out_dir=args.out_dir / "selfcheck_nli", multi_sample_k=1
        )
    if "selfcheck_nli_multi" in arms:
        from ccer.vendor_baselines.selfcheck_nli_obs_tau import run_selfcheck_nli_obs_tau

        results["selfcheck_nli_multi"] = run_selfcheck_nli_obs_tau(
            device=args.device,
            out_dir=args.out_dir / "selfcheck_nli_multi",
            multi_sample_k=max(2, args.selfcheck_k),
        )
    if "factool_kbqa" in arms:
        from ccer.vendor_baselines.factool_kbqa_obs_tau import run_factool_kbqa_obs_tau
        from ccer.vendor_baselines.p2_checkpoint import load_scored_keys

        factool_dir = args.out_dir / "factool_kbqa"
        flim = args.factool_limit if args.factool_limit > 0 else None
        want = flim or 1244
        done = len(load_scored_keys(factool_dir / "scores.jsonl"))
        if done >= want and flim is None:
            print(f"[p2] factool_kbqa skip: scores.jsonl already has {done} rows", flush=True)
            summary = factool_dir / "summary.json"
            if summary.is_file():
                results["factool_kbqa"] = json.loads(summary.read_text(encoding="utf-8"))
            else:
                results["factool_kbqa"] = run_factool_kbqa_obs_tau(out_dir=factool_dir, limit=flim)
        else:
            results["factool_kbqa"] = run_factool_kbqa_obs_tau(out_dir=factool_dir, limit=flim)
    if "factscore" in arms:
        from ccer.vendor_baselines.factscore_obs_tau import run_factscore_obs_tau

        lim = args.factscore_limit if args.factscore_limit > 0 else None
        results["factscore"] = run_factscore_obs_tau(out_dir=args.out_dir / "factscore", limit=lim)
    summary = args.out_dir / "p2_vendor_detection_summary.json"
    summary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
