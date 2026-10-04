"""Build anchor-value J-lens mass cache for AGR Round 3."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_JLENS = ROOT / "third_party" / "jacobian-lens"
if _JLENS.is_dir() and str(_JLENS) not in sys.path:
    sys.path.insert(0, str(_JLENS))

import jlens
from jlens.lens import JacobianLens

from ccer.mechanism.agr.anchor_mass_cache import build_anchor_mass_cache_rows, write_anchor_mass_cache
from ccer.paths import AGR_ANCHOR_VALUE_MASS_CACHE
from ccer.replay.hf_forward import load_hf_model

LENS_PATH = ROOT / "artifacts" / "ccer" / "jspace" / "J_l_qwen32b_L49.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens-path", default=str(LENS_PATH))
    ap.add_argument("--device-map", default="cpu")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    lens = JacobianLens.load(args.lens_path)
    model, tokenizer, _ = load_hf_model(device_map=args.device_map)
    jlens_model = jlens.from_hf(model, tokenizer)

    rows = build_anchor_mass_cache_rows(
        lens=lens,
        jlens_model=jlens_model,
        tokenizer=tokenizer,
        topk=args.topk,
    )
    write_anchor_mass_cache(rows)
    prov = {}
    for r in rows:
        p = str(r.get("anchor_provenance") or "missing")
        prov[p] = prov.get(p, 0) + 1
    print(f"Wrote {len(rows)} rows to {AGR_ANCHOR_VALUE_MASS_CACHE}")
    print(f"Provenance counts: {prov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
