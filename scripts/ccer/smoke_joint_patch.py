#!/usr/bin/env python3
"""Preflight smoke for Task I joint patch (1 pair, 4 positions)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
sys.path.insert(0, str(ROOT))

from ccer.pipelines.run_p3_line_i import _run_smoke  # noqa: E402
from ccer.replay.hf_forward import load_hf_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test Task I joint patch")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--probe-max-tokens", type=int, default=96)
    args = ap.parse_args()
    print("[smoke_joint_patch] loading model...", flush=True)
    model, tokenizer, _ = load_hf_model(device_map=args.device_map)
    _run_smoke(
        model,
        tokenizer,
        rank=args.rank,
        probe_max_tokens=args.probe_max_tokens,
    )


if __name__ == "__main__":
    main()
