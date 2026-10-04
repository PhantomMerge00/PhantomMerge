"""CLI: reproduce and verify frozen-result metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reproduce.l1.reproduce_l1 import run_l1
from reproduce.l1.verify_l1 import run_verify

ROOT = Path(__file__).resolve().parents[2]


def _cmd_reproduce(args: argparse.Namespace) -> int:
    metrics = run_l1(Path(args.bundle), Path(args.out))
    print(json.dumps({"ok": True, "n_readouts": len(metrics.get("readouts", {}))}))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    ref = Path(args.reference) if args.reference else ROOT / "docs" / "reference_values.json"
    report = run_verify(Path(args.bundle), Path(args.results), ref)
    print(json.dumps({"passed": report["passed"], "n": len(report["comparisons"])}))
    return 0 if report["passed"] else 1


def _cmd_demo(_args: argparse.Namespace) -> int:
    print("Not implemented; see docs/ARTIFACT.md §2", file=sys.stderr)
    return 2


def _cmd_run(_args: argparse.Namespace) -> int:
    print("Not implemented; see reproduce/REPLICATION.md", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phantom-merge")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rep = sub.add_parser("reproduce")
    p_rep.add_argument("--bundle", default="results")
    p_rep.add_argument("--out", default="outputs/metrics")
    p_rep.set_defaults(func=_cmd_reproduce)

    p_ver = sub.add_parser("verify")
    p_ver.add_argument("--bundle", default="results")
    p_ver.add_argument("--results", default="outputs/metrics")
    p_ver.add_argument("--reference", default=None)
    p_ver.set_defaults(func=_cmd_verify)

    p_demo = sub.add_parser("demo")
    p_demo.add_argument("--config", required=True)
    p_demo.add_argument("--out", required=True)
    p_demo.set_defaults(func=_cmd_demo)

    p_run = sub.add_parser("run")
    p_run.add_argument("--config", required=True)
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
