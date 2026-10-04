"""Extract instance-aligned activations for incremental adjudication (Line A PM/clean instances)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import load_jsonl, write_json
from ccer.mechanism.line_a_instance_activations import (
    extract_line_a_instance_trajectory,
    group_instances_by_trajectory,
    line_a_adjudication_instance_rows,
)
from ccer.mechanism.supervised_probe import line_a_skip_trajectory_ids
from ccer.paths import LINE_A_DIR, NORMALIZED_SHOPPING
from ccer.replay.hf_forward import load_hf_model, probe_layer_indices


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--shared-gpu", action="store_true", default=True)
    ap.add_argument("--gpu-reserve-gib", type=int, default=18)
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    args = ap.parse_args(argv)

    rows = line_a_adjudication_instance_rows(require_activation=False)
    grouped = group_instances_by_trajectory(rows)
    traj_rows = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}
    skips = line_a_skip_trajectory_ids()
    tids = sorted(t for t in grouped if t not in skips)
    if args.shard_id is not None and args.num_shards is not None and args.num_shards > 1:
        tids = [tid for i, tid in enumerate(tids) if i % args.num_shards == args.shard_id]
    if args.limit:
        tids = tids[: args.limit]

    model, tokenizer, n_layers = load_hf_model(
        device_map=args.device_map,
        shared_gpu=args.shared_gpu,
        gpu_reserve_gib=args.gpu_reserve_gib,
    )
    layer_indices = probe_layer_indices(n_layers)
    print(f"[line_a] adjudication extract: {len(tids)} trajectories", flush=True)

    stats = {"trajectories": 0, "instances_written": 0, "errors": []}
    for i, tid in enumerate(tids, 1):
        insts = grouped[tid]
        traj = traj_rows.get(tid)
        if not traj:
            continue
        print(f"[{i}/{len(tids)}] {tid}", flush=True)
        try:
            out = extract_line_a_instance_trajectory(
                model=model,
                tokenizer=tokenizer,
                trajectory=traj,
                instances=insts,
                layer_indices=layer_indices,
            )
            stats["trajectories"] += 1
            stats["instances_written"] += out["n_written"]
            stats["errors"].extend(out.get("errors") or [])
        except Exception as exc:  # noqa: BLE001
            stats["errors"].append(f"{tid}:{exc}")

    out_path = LINE_A_DIR / "adjudication_extract_manifest.json"
    if args.shard_id is not None:
        out_path = LINE_A_DIR / f"adjudication_extract_manifest_s{args.shard_id}.json"
    write_json(out_path, stats)
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
