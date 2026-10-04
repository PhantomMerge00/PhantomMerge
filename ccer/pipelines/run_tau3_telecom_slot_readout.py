"""Phase 0d: J-lens slot readout per tau3 claim."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jlens
from jlens.lens import JacobianLens

from ccer.io_utils import write_jsonl
from ccer.mechanism.activation_store import get_vector, load_activation_npz
from ccer.mechanism.bind_surprise import compute_slot_workspace_mass
from ccer.mechanism.prism_audit import readout_claim_instance
from ccer.mechanism.tau3.cohort import instance_npz_path, tau3_instance_rows
from ccer.mechanism.tau3.domain import get_tau3_domain
from ccer.paths import JSPACE_LENS_L49
from ccer.replay.hf_forward import load_hf_model
from ccer.replay.live_position import LINE_D_CLAIM_LAYER_DEFAULT

LINE_A_LAYER = 49


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="telecom", choices=["telecom", "airline", "retail"])
    parser.add_argument("--lens-path", default=str(JSPACE_LENS_L49))
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    cfg = get_tau3_domain(args.domain)

    rows = tau3_instance_rows(require_activation=True, cfg=cfg)
    if args.limit:
        rows = rows[: args.limit]

    lens = JacobianLens.load(args.lens_path)
    model, tokenizer, _ = load_hf_model(device_map=args.device_map)
    jlens_model = jlens.from_hf(model, tokenizer)

    out_rows: list[dict[str, Any]] = []
    missing = 0
    for row in rows:
        path = instance_npz_path(row.instance_audit_key, row.trajectory_id, cfg=cfg)
        if not path.is_file():
            missing += 1
            continue
        loaded = load_activation_npz(path)
        h = get_vector(loaded, position="claim_onset", layer=LINE_A_LAYER)
        if h is None:
            missing += 1
            continue
        topk = readout_claim_instance(
            h.astype("float32"),
            lens,
            jlens_model,
            tokenizer,
            layer=LINE_D_CLAIM_LAYER_DEFAULT,
            k=args.topk,
        )
        slot_mass = compute_slot_workspace_mass(topk, row.slot_norm)
        out_rows.append(
            {
                "instance_audit_key": row.instance_audit_key,
                "trajectory_id": row.trajectory_id,
                "split": row.split,
                "y_pm": row.y,
                "gold_verdict": row.gold_verdict,
                "slot_norm": row.slot_norm,
                "claim_value": row.claim_value,
                "response_quote": row.response_quote,
                "log_slot_mass": slot_mass.get("log_slot_mass"),
                "slot_mass": slot_mass.get("slot_mass"),
                "slot_lexicon_hits": slot_mass.get("slot_lexicon_hits"),
                "slot_rank": slot_mass.get("slot_rank"),
                "jlens_top5": [str(t.get("token") or "") for t in topk[:5]],
            }
        )

    cfg.slot_readout.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(cfg.slot_readout, out_rows)
    stats = {
        "domain": cfg.domain,
        "n_readout": len(out_rows),
        "n_missing_activation": missing,
        "out_path": str(cfg.slot_readout),
    }
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
