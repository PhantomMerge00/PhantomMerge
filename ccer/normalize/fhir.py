"""FHIR lightweight normalization — eligible flags default false for P1/P2."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ccer.io_utils import load_jsonl, sha256_json, write_jsonl
from ccer.paths import ANNOTATION_VERSION, FHIR_ROLLOUT_PATHS, NORMALIZED_FHIR
from ccer.schema.validate import validate_trajectory


def normalize_fhir_row(row: dict[str, Any], *, source_file: str, line_no: int) -> dict[str, Any]:
    tid = str(row.get("question_id") or row.get("trajectory_id") or f"fhir_line_{line_no}")
    trace = row.get("trace") or []
    messages_final_call = []
    for m in trace:
        if isinstance(m, dict) and m.get("role"):
            messages_final_call.append({"role": m.get("role"), "content": m.get("content", "")})
    exclusion = [
        "fhir_best_effort_reconstruction_only",
        "hf_tools_not_reattached",
        "counterfactual_evidence_edit_not_ready",
    ]
    if row.get("agent_answer") is None:
        exclusion.append("missing_agent_answer")
    if str(row.get("error") or ""):
        exclusion.append(f"rollout_error:{row.get('error')}")

    out = {
        "root_id": tid,
        "trajectory_id": tid,
        "raw_ref": {"rollout_file": source_file, "line_no": line_no},
        "split": "unassigned",
        "group_id": tid,
        "annotation_version": ANNOTATION_VERSION,
        "domain": "fhir",
        "schema_version": "ccer_v1",
        "is_calibration": False,
        "trajectory_outcome": "unknown",
        "pm_core_count": 0,
        "messages_raw": {"trace": trace},
        "messages_final_call": messages_final_call,
        "model_manifest_id": "qwen3-32b_fhir_v1",
        "input_hash": sha256_json(messages_final_call) if messages_final_call else None,
        "actions": [],
        "commitment": {"action_anchor": None, "commitment_relation": "unobservable", "ambiguity": []},
        "text_anchor": {"selected_pid": None, "compared_pid": None, "declared_pids": []},
        "evidence": [],
        "claims": [],
        "eligible": {
            "semantic_eval": False,
            "native_replay": False,
            "counterfactual": False,
            "patching": False,
        },
        "exclusion_reason": exclusion,
        "metadata": {
            "replay_fidelity_level": "best_effort_reconstruction",
            "patient_fhir_id": row.get("patient_fhir_id"),
        },
    }
    validate_trajectory(out)
    return out


def run_normalize_fhir(out_path: Path = NORMALIZED_FHIR) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in FHIR_ROLLOUT_PATHS:
        if not path.is_file():
            continue
        for i, row in enumerate(load_jsonl(path), 1):
            rows.append(normalize_fhir_row(row, source_file=str(path), line_no=i))
    write_jsonl(out_path, rows)
    return {"n_normalized": len(rows), "eligible_native_replay": 0}


if __name__ == "__main__":
    print(json.dumps(run_normalize_fhir(), indent=2))
