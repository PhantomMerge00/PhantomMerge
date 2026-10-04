"""CLI: ModernBERT-large + LR on Line-A v3 (quote + BCP variants)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.mechanism.modernbert_pm_detector import run_modernbert_detection  # noqa: E402

DEFAULT_OUT = ROOT / "results/rq2"


def main() -> None:
    ap = argparse.ArgumentParser(description="ModernBERT PM detection baseline (CCER)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--skip-encode", action="store_true", help="Use cached .npy only")
    ap.add_argument(
        "--variants",
        default="quote,bcp_fair,bcp_oracle",
        help="Comma-separated: quote,bcp_fair,bcp_oracle",
    )
    args = ap.parse_args()
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())  # type: ignore
    payload = run_modernbert_detection(
        out_dir=args.out_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        skip_encode=args.skip_encode,
        variants=variants,  # type: ignore[arg-type]
    )
    print(json.dumps(payload.get("variants", {}), indent=2))


if __name__ == "__main__":
    main()
