"""Phase 0a: normalize tau3 gold export trajectories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.normalize.airline_qwen_1k import run_normalize_airline_qwen_1k
from ccer.normalize.retail_qwen_1k import run_normalize_retail_qwen_1k
from ccer.normalize.telecom import run_normalize_telecom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="telecom", choices=["telecom", "airline", "retail"])
    args = ap.parse_args()
    if args.domain == "airline":
        stats = run_normalize_airline_qwen_1k()
    elif args.domain == "retail":
        stats = run_normalize_retail_qwen_1k()
    else:
        stats = run_normalize_telecom()
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
