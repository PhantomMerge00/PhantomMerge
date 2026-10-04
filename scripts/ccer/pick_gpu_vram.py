#!/usr/bin/env python3
"""Pick or verify GPU(s) with enough free VRAM for Qwen3-32B HF (~64GB).

Single-GPU if one card has enough free; otherwise greedily combine cards until
total free >= threshold (for HF device_map=auto sharding).

Usage (auto-pick):
  eval "$(python3 pick_gpu_vram.py --min-gb 65)"

Usage (verify user-set CUDA_VISIBLE_DEVICES):
  CUDA_VISIBLE_DEVICES=1,4 eval "$(python3 pick_gpu_vram.py --min-gb 65)"

Prints 'export CUDA_VISIBLE_DEVICES=...' to stdout; status to stderr.
Exits 1 if no suitable GPU set.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _query_gpus() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"ERROR: nvidia-smi failed: {exc}", file=sys.stderr)
        sys.exit(2)

    gpus: list[dict] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx, name, free_mib, total_mib = parts[0], parts[1], parts[2], parts[3]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "free_mib": int(float(free_mib)),
                "total_mib": int(float(total_mib)),
            }
        )
    return gpus


def _pick_gpus(gpus: list[dict], min_mib: int) -> list[dict] | None:
    eligible = [g for g in gpus if g["free_mib"] >= min_mib]
    if eligible:
        return [max(eligible, key=lambda g: g["free_mib"])]

    # Greedy multi-GPU: prefer cards with most free VRAM; skip near-empty 49GB Ada
    # unless needed (total capacity still counts toward sum).
    min_per_gpu_mib = 8 * 1024  # each shard needs >=8GB usable
    candidates = sorted(gpus, key=lambda x: -x["free_mib"])
    picked: list[dict] = []
    total = 0
    for g in candidates:
        if g["free_mib"] < min_per_gpu_mib:
            continue
        picked.append(g)
        total += g["free_mib"]
        if total >= min_mib:
            return picked
    return None


def _format_pick(picked: list[dict]) -> str:
    return ",".join(str(g["index"]) for g in picked)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick/verify GPU(s) for 32B HF load")
    ap.add_argument("--min-gb", type=float, default=65.0, help="Minimum total free VRAM (GB)")
    args = ap.parse_args()
    min_mib = int(args.min_gb * 1024)

    gpus = _query_gpus()
    if not gpus:
        print("ERROR: no GPUs reported by nvidia-smi", file=sys.stderr)
        sys.exit(1)

    print(f"GPU scan (need >={args.min_gb:.0f}GB free total, multi-GPU OK):", file=sys.stderr)
    for g in gpus:
        free_gb = g["free_mib"] / 1024
        total_gb = g["total_mib"] / 1024
        print(
            f"  nvidia-smi GPU {g['index']}: {g['name']}  "
            f"free={free_gb:.1f}GB / {total_gb:.1f}GB",
            file=sys.stderr,
        )

    forced = os.environ.get("CUDA_VISIBLE_DEVICES")
    if forced is not None and str(forced).strip() != "":
        indices = [int(x.strip()) for x in str(forced).split(",") if x.strip()]
        selected = [g for g in gpus if g["index"] in indices]
        missing = set(indices) - {g["index"] for g in selected}
        if missing:
            print(f"ERROR: CUDA_VISIBLE_DEVICES={forced} unknown indices {sorted(missing)}", file=sys.stderr)
            sys.exit(1)
        total_free = sum(g["free_mib"] for g in selected)
        if total_free < min_mib:
            print(
                f"ERROR: GPUs {forced} have only {total_free/1024:.1f}GB free combined "
                f"(need {args.min_gb:.0f}GB). Pick another set or wait.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"Using GPUs {forced} ({total_free/1024:.1f}GB free combined, sharded load)",
            file=sys.stderr,
        )
        print(f"export CUDA_VISIBLE_DEVICES={forced}")
        return

    picked = _pick_gpus(gpus, min_mib)
    if not picked:
        print(
            f"ERROR: no GPU set with >={args.min_gb:.0f}GB free (single or combined). "
            "Set CUDA_VISIBLE_DEVICES after freeing cards, or wait.",
            file=sys.stderr,
        )
        sys.exit(1)

    total_free = sum(g["free_mib"] for g in picked)
    label = _format_pick(picked)
    if len(picked) == 1:
        g = picked[0]
        print(
            f"Auto-picked nvidia-smi GPU {g['index']} ({g['name']}), "
            f"{g['free_mib']/1024:.1f}GB free",
            file=sys.stderr,
        )
    else:
        parts = ", ".join(f"GPU{g['index']}({g['free_mib']/1024:.1f}GB)" for g in picked)
        print(
            f"Auto-picked multi-GPU {label}: {parts} = {total_free/1024:.1f}GB free (sharded)",
            file=sys.stderr,
        )
    print(f"export CUDA_VISIBLE_DEVICES={label}")


if __name__ == "__main__":
    main()
