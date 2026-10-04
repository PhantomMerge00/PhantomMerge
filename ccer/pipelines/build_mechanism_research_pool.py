"""Build and freeze mechanism research pool manifest (Line B step 1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import write_json
from ccer.mechanism.mechanism_pool import (
    MECHANISM_POOL_MANIFEST,
    build_mechanism_research_pool_manifest,
)
from ccer.mechanism.round12_line_b import activation_coverage_for_pool


def main() -> None:
    manifest = build_mechanism_research_pool_manifest()
    coverage = activation_coverage_for_pool(pool="mechanism_research")
    manifest["activation_coverage_L32_commitment"] = {
        "n_eval_eligible": coverage.get("n_eval_eligible"),
        "n_cross_pairs": coverage.get("n_cross_pairs"),
        "coverage_rate": coverage.get("coverage_rate"),
    }
    write_json(MECHANISM_POOL_MANIFEST, manifest)
    print(json.dumps(
        {
            "n_cem_confirmed": manifest["n_cem_confirmed"],
            "n_cem_by_split": manifest["n_cem_by_split"],
            "n_cross_pairs": manifest["n_cross_pairs"],
            "n_eval_eligible": coverage.get("n_eval_eligible"),
        },
        indent=2,
    ))
    print(f"wrote {MECHANISM_POOL_MANIFEST}")


if __name__ == "__main__":
    main()
