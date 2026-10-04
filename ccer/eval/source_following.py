"""Tier-1 source-following scoring on extracted final answers (not raw tool_call junk)."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from ccer.replay.answer_utils import extract_answer, validate_replay_output


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _contains_value(text: str, value: str) -> bool:
    if not value:
        return False
    if value in text:
        return True
    return _norm(value) in _norm(text)


def _manifest_old_snippet(edit_manifest: list[dict[str, Any]] | None) -> str:
    if not edit_manifest:
        return ""
    return str(edit_manifest[0].get("old") or "")


def _claim_excerpt(text: str) -> str:
    from ccer.replay.answer_utils import extract_answer

    excerpt = extract_answer(text) if "<response>" in text.lower() else text
    for marker in ("Selected", "Compared", "About Selected", "About Compared"):
        pos = excerpt.find(marker)
        if pos >= 0:
            return excerpt[pos : pos + 800]
    return excerpt


def evaluate_source_following(
    *,
    condition_id: str,
    original_answer: str,
    new_answer: str,
    expected: dict[str, Any],
    edit_manifest: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score one CF pair. source_following=None when not scorable (harness/axis)."""
    new_excerpt, harness_err = validate_replay_output(new_answer)
    if harness_err:
        return {
            "source_following": None,
            "scorable": False,
            "harness_valid": False,
            "reason": harness_err,
        }

    orig_excerpt = extract_answer(original_answer) if "<response>" in original_answer.lower() else str(original_answer or "")

    if condition_id in ("instruction_pressure_neutral",):
        return {
            "source_following": None,
            "scorable": False,
            "harness_valid": True,
            "reason": "non_source_axis_control",
        }

    if condition_id == "position_permutation":
        sim = SequenceMatcher(None, _norm(orig_excerpt), _norm(new_excerpt)).ratio()
        follow = sim >= 0.75
        return {
            "source_following": follow,
            "scorable": True,
            "harness_valid": True,
            "reason": f"placebo_invariance_similarity={sim:.3f}",
            "metric": "invariance",
        }

    if condition_id == "rival_value_swap":
        new_val = str(expected.get("rival_value") or "")
        old_val = str(expected.get("rival_value_old") or "")
        if not old_val:
            old_snip = _manifest_old_snippet(edit_manifest)
            if "->" in old_snip:
                old_val = old_snip.split("->", 1)[0].split(":")[-1].strip()
        orig_claim = _claim_excerpt(orig_excerpt)
        new_claim = _claim_excerpt(new_excerpt)
        follow = _contains_value(new_claim, new_val) and not _contains_value(orig_claim, new_val)
        if not follow and old_val:
            if _contains_value(orig_claim, old_val) and not _contains_value(new_claim, old_val):
                follow = True
            if _contains_value(new_claim, new_val) and not _contains_value(new_excerpt, new_val):
                follow = True
        return {
            "source_following": follow,
            "scorable": True,
            "harness_valid": True,
            "reason": "rival_value_adoption_or_old_dropped_in_claim_span",
            "metric": "directional",
        }

    if condition_id == "anchor_value_swap":
        new_val = str(expected.get("anchor_supported_value") or "")
        orig_claim = _claim_excerpt(orig_excerpt)
        new_claim = _claim_excerpt(new_excerpt)
        follow = _contains_value(new_claim, new_val) and not _contains_value(orig_claim, new_val)
        return {
            "source_following": follow,
            "scorable": True,
            "harness_valid": True,
            "reason": "anchor_supported_value_in_claim_span",
            "metric": "directional",
        }

    if condition_id == "query_value_swap":
        new_val = str(expected.get("query_constraint_value") or "")
        follow = _contains_value(new_excerpt, new_val) and not _contains_value(orig_excerpt, new_val)
        return {
            "source_following": follow,
            "scorable": True,
            "harness_valid": True,
            "reason": "query_constraint_value_in_answer",
            "metric": "directional",
        }

    if condition_id == "source_null":
        removed_pid = str(expected.get("source_removed") or "")
        old_snip = _manifest_old_snippet(edit_manifest)
        old_count = orig_excerpt.count(removed_pid) if removed_pid else 0
        new_count = new_excerpt.count(removed_pid) if removed_pid else 0
        snippet_gone = bool(old_snip) and _contains_value(orig_excerpt, old_snip[:120]) and not _contains_value(
            new_excerpt, old_snip[:120]
        )
        follow = snippet_gone or (old_count > new_count)
        return {
            "source_following": follow,
            "scorable": True,
            "harness_valid": True,
            "reason": "nulled_source_facts_reduced",
            "metric": "directional",
        }

    return {
        "source_following": None,
        "scorable": False,
        "harness_valid": True,
        "reason": f"unscored_condition:{condition_id}",
    }


def aggregate_following_stats(effects: list[dict[str, Any]]) -> dict[str, Any]:
    scorable = [e for e in effects if e.get("scorable") and e.get("source_following") is not None]
    harness_invalid = [e for e in effects if e.get("harness_valid") is False]
    n_follow = sum(1 for e in scorable if e.get("source_following"))
    rate = n_follow / len(scorable) if scorable else None
    by_cohort: dict[str, Any] = {}
    for e in scorable:
        c = str(e.get("cohort") or "?")
        by_cohort.setdefault(c, {"n": 0, "follow": 0})
        by_cohort[c]["n"] += 1
        if e.get("source_following"):
            by_cohort[c]["follow"] += 1
    for c in by_cohort:
        n = by_cohort[c]["n"]
        by_cohort[c]["rate"] = by_cohort[c]["follow"] / n if n else None
    by_condition: dict[str, Any] = {}
    for e in scorable:
        key = f"{e.get('cohort')}:{e.get('condition_id')}"
        by_condition.setdefault(key, {"n": 0, "follow": 0})
        by_condition[key]["n"] += 1
        if e.get("source_following"):
            by_condition[key]["follow"] += 1
    for k in by_condition:
        n = by_condition[k]["n"]
        by_condition[k]["rate"] = by_condition[k]["follow"] / n if n else None
    return {
        "n_effects_total": len(effects),
        "n_scorable": len(scorable),
        "n_harness_invalid": len(harness_invalid),
        "n_follow": n_follow,
        "source_following_rate": rate,
        "by_cohort": by_cohort,
        "by_condition": by_condition,
    }
