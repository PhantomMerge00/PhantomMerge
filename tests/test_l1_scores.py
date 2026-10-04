import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results/rq2/table2_prf_audit/per_claim_scores_test274.jsonl"


def test_scores_cohort_size():
    assert SCORES.is_file()
    rows = [json.loads(l) for l in SCORES.read_text().splitlines() if l.strip()]
    assert len(rows) == 274
    assert sum(int(r["y_pm"]) for r in rows) == 156
