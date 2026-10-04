"""CPU audit: recompute dual metrics on saved ni/ti texts with corrected extraction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.mechanism.round6_verify import verify_dual_metrics_from_texts
from ccer.replay.answer_utils import extract_answer, extract_selected_product_id


def _legacy_first_block_extract(text: str) -> str:
    import re

    m = re.search(r"<response>(.+?)</response>", text, re.DOTALL | re.I)
    return m.group(1).strip() if m else text.strip()


def audit_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("measure_row", {}).get("pair_details") or data.get("pair_details") or []
    fixed = verify_dual_metrics_from_texts(rows)
    legacy_answer_diff = 0
    for row in rows:
        ni, ti = str(row.get("ni_text") or ""), str(row.get("ti_text") or "")
        if _legacy_first_block_extract(ni) != _legacy_first_block_extract(ti):
            legacy_answer_diff += 1
    return {
        "source": str(path),
        "n_pairs": len(rows),
        "legacy_first_block_extract_answer_diff_k": legacy_answer_diff,
        "fixed_metrics": fixed,
        "sample_pair_0": {
            "legacy_excerpt_head": _legacy_first_block_extract(str(rows[0].get("ni_text") or ""))[:80],
            "fixed_excerpt_head": extract_answer(str(rows[0].get("ni_text") or ""))[:80],
            "fixed_product_id_ni": extract_selected_product_id(str(rows[0].get("ni_text") or "")),
        }
        if rows
        else {},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=str(ROOT / "results/p3/round8/dual_metric_n18.json"),
    )
    ap.add_argument(
        "--output",
        default=str(ROOT / "results/p3/round8/dual_metric_n18_reaudit.json"),
    )
    args = ap.parse_args()
    result = audit_json(Path(args.input))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
