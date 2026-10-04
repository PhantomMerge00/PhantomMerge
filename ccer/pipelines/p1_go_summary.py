"""P1 Tier-1 go/no-go summary from incremental source_effects_rows."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.eval.source_following import aggregate_following_stats
from ccer.io_utils import load_json, load_jsonl, write_json
from ccer.paths import P1_DIR, REPORTS

DEFAULT_MIN_EFFECTS = 10
DEFAULT_MIN_OVERALL_RATE = 0.10
DEFAULT_MIN_COHORT_N = 5
DEFAULT_MIN_COHORT_RATE = 0.15
DEFAULT_MAX_HARNESS_INVALID_FRAC = 0.20


def build_p1_summary(
    *,
    min_effects: int = DEFAULT_MIN_EFFECTS,
    min_overall_rate: float = DEFAULT_MIN_OVERALL_RATE,
    min_cohort_n: int = DEFAULT_MIN_COHORT_N,
    min_cohort_rate: float = DEFAULT_MIN_COHORT_RATE,
    max_harness_invalid_frac: float = DEFAULT_MAX_HARNESS_INVALID_FRAC,
) -> dict[str, Any]:
    effects = load_jsonl(P1_DIR / "source_effects_rows.jsonl")
    stats = aggregate_following_stats(effects)
    by_cohort = stats["by_cohort"]
    n_scorable = int(stats["n_scorable"] or 0)
    n_harness_invalid = int(stats["n_harness_invalid"] or 0)
    n_effects_total = int(stats["n_effects_total"] or 0)
    overall_rate = stats["source_following_rate"]
    n_follow = int(stats["n_follow"] or 0)

    harness_invalid_frac = (
        n_harness_invalid / n_effects_total if n_effects_total else 0.0
    )

    if n_effects_total == 0:
        decision = "BLOCKED_HARNESS"
        reasons = ["no P1 effects recorded"]
    elif harness_invalid_frac > max_harness_invalid_frac or n_scorable < min_effects:
        decision = "BLOCKED_HARNESS"
        reasons = [
            f"harness/scorable gate failed: n_scorable={n_scorable} (need>={min_effects}), "
            f"n_harness_invalid={n_harness_invalid}/{n_effects_total} "
            f"(frac={harness_invalid_frac:.3f}, max={max_harness_invalid_frac})",
        ]
    else:
        cohort_go = any(
            (c.get("n") or 0) >= min_cohort_n and (c.get("rate") or 0) >= min_cohort_rate
            for c in by_cohort.values()
        )
        overall_go = n_scorable >= min_effects and (overall_rate or 0) >= min_overall_rate
        if overall_go or cohort_go:
            decision = "GO"
            reasons = []
            if overall_go:
                reasons.append(
                    f"overall source_following {overall_rate:.3f} >= {min_overall_rate} (n_scorable={n_scorable})"
                )
            if cohort_go:
                hit = [
                    f"{name}:{c['rate']:.3f}(n={c['n']})"
                    for name, c in by_cohort.items()
                    if (c.get("n") or 0) >= min_cohort_n and (c.get("rate") or 0) >= min_cohort_rate
                ]
                reasons.append(f"cohort threshold met: {', '.join(hit)}")
        else:
            decision = "NO_GO"
            reasons = [
                f"valid harness but insufficient Tier-1 source-following "
                f"(n_scorable={n_scorable}, overall_rate={overall_rate}, by_cohort={by_cohort})"
            ]

    prior = load_json(P1_DIR / "source_effects.json") if (P1_DIR / "source_effects.json").is_file() else {}
    next_step = {
        "GO": "run_p2_dev_limit_16",
        "NO_GO": "stop_tier2_reassess_operators_or_positioning",
        "BLOCKED_HARNESS": "fix_replay_harness_before_interpreting_tier1",
    }[decision]

    return {
        "schema_version": "ccer_p1_go_v2",
        "decision": decision,
        "reasons": reasons,
        "thresholds": {
            "min_effects": min_effects,
            "min_overall_rate": min_overall_rate,
            "min_cohort_n": min_cohort_n,
            "min_cohort_rate": min_cohort_rate,
            "max_harness_invalid_frac": max_harness_invalid_frac,
        },
        "n_effects_total": n_effects_total,
        "n_scorable": n_scorable,
        "n_harness_invalid": n_harness_invalid,
        "n_follow": n_follow,
        "source_following_rate": overall_rate,
        "by_cohort": by_cohort,
        "by_condition": stats.get("by_condition") or {},
        "vllm_available": prior.get("vllm_available"),
        "offline_mode": prior.get("offline_mode"),
        "next_step": next_step,
    }


def write_p1_go_report(report: dict[str, Any] | None = None) -> dict[str, Any]:
    report = report or build_p1_summary()
    write_json(
        P1_DIR / "source_effects.json",
        {
            "n_generations": len(load_jsonl(P1_DIR / "generations.jsonl")),
            "n_invalid": len(load_jsonl(P1_DIR / "invalid_counterfactuals.jsonl")),
            "n_counterfactuals": len(load_jsonl(P1_DIR / "counterfactuals.jsonl")),
            "n_effects": report["n_effects_total"],
            "n_scorable": report["n_scorable"],
            "n_harness_invalid": report["n_harness_invalid"],
            "source_following_rate": report["source_following_rate"],
            "by_cohort": report["by_cohort"],
            "by_condition": report.get("by_condition") or {},
            "go_decision": report["decision"],
            "vllm_available": report.get("vllm_available"),
            "offline_mode": report.get("offline_mode"),
        },
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS / "p1_go_decision.json", report)
    md = [
        "# P1 Tier-1 go/no-go",
        "",
        f"- **decision**: {report['decision']}",
        f"- **source_following_rate** (scorable only): {report['source_following_rate']}",
        f"- **n_scorable**: {report['n_scorable']}",
        f"- **n_harness_invalid**: {report['n_harness_invalid']}",
        "",
        "## Reasons",
        *[f"- {r}" for r in report["reasons"]],
        "",
        f"- **next_step**: {report['next_step']}",
    ]
    (REPORTS / "p1_go_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--query", choices=("decision", "json"), default="json")
    ap.add_argument("--min-effects", type=int, default=DEFAULT_MIN_EFFECTS)
    ap.add_argument("--min-overall-rate", type=float, default=DEFAULT_MIN_OVERALL_RATE)
    ap.add_argument("--min-cohort-n", type=int, default=DEFAULT_MIN_COHORT_N)
    ap.add_argument("--min-cohort-rate", type=float, default=DEFAULT_MIN_COHORT_RATE)
    args = ap.parse_args()
    report = write_p1_go_report(
        build_p1_summary(
            min_effects=args.min_effects,
            min_overall_rate=args.min_overall_rate,
            min_cohort_n=args.min_cohort_n,
            min_cohort_rate=args.min_cohort_rate,
        )
    )
    if args.query == "decision":
        print(report["decision"])
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
