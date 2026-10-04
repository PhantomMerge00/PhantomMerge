"""CCER schema validation (§3 data contract)."""
from __future__ import annotations

from typing import Any

EVIDENCE_KINDS = frozenset(
    {
        "tool_observation",
        "filter_result",
        "user_factual_assertion",
        "user_requirement",
        "assistant_derived_summary",
    }
)

CLAIM_MODES = frozenset(
    {
        "assertion",
        "negation",
        "comparison",
        "request",
        "qualified_unknown",
        "ambiguous",
    }
)

ELIGIBLE_KEYS = frozenset(
    {"semantic_eval", "native_replay", "counterfactual", "patching"}
)

REQUIRED_TRAJECTORY_KEYS = frozenset(
    {
        "root_id",
        "trajectory_id",
        "raw_ref",
        "split",
        "group_id",
        "annotation_version",
        "domain",
        "messages_raw",
        "messages_final_call",
        "model_manifest_id",
        "input_hash",
        "actions",
        "commitment",
        "text_anchor",
        "evidence",
        "claims",
        "eligible",
        "exclusion_reason",
    }
)


class SchemaValidationError(Exception):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def _require(obj: dict[str, Any], key: str, path: str) -> Any:
    if key not in obj:
        raise SchemaValidationError(path, f"missing required key {key!r}")
    return obj[key]


def validate_evidence(ev: dict[str, Any], path: str = "evidence[]") -> None:
    for key in (
        "evidence_id",
        "source_id",
        "slot_norm",
        "value_norm",
        "scope",
        "polarity",
        "evidence_kind",
    ):
        _require(ev, key, path)
    kind = ev["evidence_kind"]
    if kind not in EVIDENCE_KINDS:
        raise SchemaValidationError(path, f"invalid evidence_kind {kind!r}")


def validate_claim(cl: dict[str, Any], path: str = "claims[]") -> None:
    for key in (
        "claim_id",
        "slot_norm",
        "value",
        "mode",
        "legacy_label",
        "human_status",
    ):
        _require(cl, key, path)
    if cl["mode"] not in CLAIM_MODES:
        raise SchemaValidationError(path, f"invalid mode {cl['mode']!r}")


def validate_eligible(eligible: dict[str, Any], path: str = "eligible") -> None:
    for key in ELIGIBLE_KEYS:
        if key not in eligible:
            raise SchemaValidationError(path, f"missing eligible.{key}")
        if not isinstance(eligible[key], bool):
            raise SchemaValidationError(path, f"eligible.{key} must be bool")


def validate_trajectory(row: dict[str, Any], *, strict: bool = True) -> list[str]:
    """Validate trajectory record. Returns list of warnings; raises on hard errors if strict."""
    warnings: list[str] = []
    missing = REQUIRED_TRAJECTORY_KEYS - set(row.keys())
    if missing and strict:
        raise SchemaValidationError("trajectory", f"missing keys {sorted(missing)}")
    if missing:
        warnings.extend(f"missing key {k}" for k in sorted(missing))

    if "eligible" in row:
        validate_eligible(row["eligible"])
    if "evidence" in row:
        for i, ev in enumerate(row["evidence"]):
            if isinstance(ev, dict):
                validate_evidence(ev, f"evidence[{i}]")
    if "claims" in row:
        for i, cl in enumerate(row["claims"]):
            if isinstance(cl, dict):
                validate_claim(cl, f"claims[{i}]")
    if "exclusion_reason" in row and not isinstance(row["exclusion_reason"], list):
        raise SchemaValidationError("exclusion_reason", "must be a list")
    return warnings


def validate_counterfactual(cf: dict[str, Any]) -> None:
    for key in (
        "root_id",
        "pair_id",
        "condition_id",
        "operator_version",
        "estimand",
        "edit_manifest",
        "held_fixed",
        "split",
    ):
        _require(cf, key, "counterfactual")
    if cf["estimand"] not in {
        "fixed_history_final_synthesis",
        "downstream_replay",
        "teacher_forced_score",
    }:
        raise SchemaValidationError("counterfactual", f"invalid estimand {cf['estimand']!r}")
