"""Rebuild cohort_manifest.json from cem_ah_strict_v2.1 adjudication (v2)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.adjudication.loader import ADJUDICATION_JSONL, load_adjudication_index
from ccer.counterfactual.bundles import adjudication_mechanism_pools
from ccer.io_utils import load_json, write_json
from ccer.paths import COHORT_MANIFEST_JSON, SPLIT_MANIFEST_JSON

BACKUP_PATH = COHORT_MANIFEST_JSON.with_suffix(".v1.json")


def build_manifest(*, split: str = "dev") -> dict:
    old = load_json(COHORT_MANIFEST_JSON) if COHORT_MANIFEST_JSON.is_file() else {}
    index = load_adjudication_index()
    pools = adjudication_mechanism_pools(split=split)

    cap = list(old.get("cohorts", {}).get("CAP", []))
    matched_clean = list(old.get("cohorts", {}).get("matched_clean", []))

    cem_all = sorted(
        tid for tid, st in index.cem_trajectory_status.items() if st == "confirmed"
    )
    ah_all = sorted(tid for tid, st in index.ah_trajectory_status.items() if st == "confirmed")

    manifest = {
        "schema_version": "ccer_cohort_v2",
        "label_authority": str(ADJUDICATION_JSONL.relative_to(ROOT)),
        "rubric_version": "cem_ah_strict_v2.1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "primary_endpoints_frozen": old.get(
            "primary_endpoints_frozen",
            [
                "paired_source_contrast_CEM",
                "paired_source_contrast_CAP",
                "pm_reduction_with_known_slot_coverage",
            ],
        ),
        "dev_targets": {
            "CEM": 64,
            "CAP": 64,
            "matched_clean": 64,
            "AH": 18,
        },
        "actual_n": {
            "CEM": len(pools["CEM"]),
            "CAP": len(cap),
            "matched_clean": len(matched_clean),
            "AH": len(pools["AH"]),
        },
        "available_n": {
            "CEM": len(cem_all),
            "CAP": old.get("available_n", {}).get("CAP", len(cap)),
            "matched_clean": len(matched_clean),
            "AH": len(ah_all),
        },
        "cohorts": {
            "CEM": pools["CEM"],
            "CAP": cap,
            "matched_clean": matched_clean,
            "AH": pools["AH"],
        },
        "legacy_cohort_v1": {
            "CEM_n": len(old.get("cohorts", {}).get("CEM", [])),
            "note": "v1 used compared_pid; v2 uses adjudication donor_owners/textual_selected_pid",
        },
        "selection_rules": {
            "split": f"{split}_only",
            "CEM": "adjudication_status=confirmed_CEM (trajectory rollup)",
            "AH": "adjudication_status=confirmed_AH (trajectory rollup)",
            "CAP": "unchanged from ccer_cohort_v1",
            "matched_clean": "unchanged from ccer_cohort_v1",
        },
    }
    return manifest


def main() -> None:
    if COHORT_MANIFEST_JSON.is_file() and not BACKUP_PATH.is_file():
        import shutil

        shutil.copy2(COHORT_MANIFEST_JSON, BACKUP_PATH)
        print(f"backed up v1 -> {BACKUP_PATH}")

    manifest = build_manifest()
    write_json(COHORT_MANIFEST_JSON, manifest)
    print(json.dumps(manifest["actual_n"], indent=2))
    print(f"wrote {COHORT_MANIFEST_JSON}")


if __name__ == "__main__":
    main()
