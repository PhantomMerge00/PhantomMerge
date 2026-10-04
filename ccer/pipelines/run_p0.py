"""P0 orchestration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.audit.data_audit import run_data_audit
from ccer.audit.model_manifest import build_model_manifest
from ccer.audit.split_manifest import run_split_manifest
from ccer.normalize.fhir import run_normalize_fhir
from ccer.normalize.shopping import run_normalize_shopping
from ccer.replay.latency_pilot import run_replay_pilot


def main() -> int:
    print("P0.1 normalize shopping...")
    shop_stats = run_normalize_shopping()
    print(json.dumps(shop_stats, indent=2))

    print("P0.2 normalize fhir...")
    fhir_stats = run_normalize_fhir()
    print(json.dumps(fhir_stats, indent=2))

    print("P0.3 model_manifest...")
    build_model_manifest()

    print("P0.4 data_audit...")
    run_data_audit()

    print("P0.5 split_manifest...")
    run_split_manifest()

    print("P0.6 replay pilot...")
    audit = run_replay_pilot(dry_run=False)
    print(json.dumps({"go": audit.get("go_no_go"), "pilot_n": audit.get("pilot_n")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
