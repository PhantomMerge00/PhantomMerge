"""Shopping trajectory normalization to CCER schema (§3)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from ccer.io_utils import sha256_json, sha256_text, write_jsonl
from ccer.paths import (
    ANNOTATION_VERSION,
    EXPERT_INSTANCES,
    NORMALIZED_SHOPPING,
    SHOPPING_ROLLOUT,
    TRAJECTORY_SHEET,
)
from ccer.schema.validate import validate_trajectory

# Reuse existing digest helpers
_OPS = Path("${PHANTOM_MERGE_ROOT}/scripts")
if str(_OPS) not in sys.path:
    sys.path.insert(0, str(_OPS))
from human_pack_items_lib import (  # noqa: E402
    COMPARED_PID_RE,
    SELECTED_PID_RE,
    extract_shopping_rollout_digest,
    index_shopping_rollouts,
    shopping_final_response_text,
    shopping_query_from_steps,
    shopping_recommend_pids,
    shopping_tool_digest,
)

_SHOP_AGENT = Path("${PHANTOM_MERGE_ROOT}/data/benchmarks/shoppingbench/src/agent")
if str(_SHOP_AGENT) not in sys.path:
    sys.path.insert(0, str(_SHOP_AGENT))
from util.rollout_completeness import rebuild_history_messages  # noqa: E402

HARD_REQ_RE = re.compile(r"Hard user requirement:\s*([^.]+)\.", re.I)
USER_TAG_RE = re.compile(r"<user>(.*?)</user>", re.DOTALL | re.I)
PID_IN_TEXT_RE = re.compile(r"product_id[=:\s\"']*(\d{6,14})", re.I)
TOOL_CALL_ID_RE = re.compile(r'"tool_call_id"\s*:\s*"([^"]+)"')

VERDICT_TO_LEGACY = {
    "correct_binding": "CB",
    "cross_object_merge": "CEM",
    "constraint_projection": "CAP",
    "anchored_hallucination": "AH",
    "pure_hallucination": "PH",
    "evidence_gap": "EG",
}

PM_CORE_VERDICTS = frozenset({"cross_object_merge", "constraint_projection", "anchored_hallucination"})


def find_final_step_index(steps: list[dict[str, Any]]) -> int | None:
    for i, step in enumerate(steps):
        content = str((step.get("completion") or {}).get("content") or "")
        if "<response>" in content.lower():
            return i
    return None


def extract_actions(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    order = 0
    for st in steps or []:
        if not isinstance(st, dict):
            continue
        msg = (st.get("completion") or {}).get("message") or {}
        obs_list = msg.get("obs") or []
        for tc, ob in zip(msg.get("tool_call") or [], obs_list):
            order += 1
            name = str((tc or {}).get("name") or "")
            params = (tc or {}).get("parameters") or {}
            results = (ob or {}).get("results")
            success = results is not None and name not in {"", "terminate"}
            pid = str(params.get("product_id") or params.get("product_ids") or "").split(",")[0].strip()
            actions.append(
                {
                    "tool_call_id": str((tc or {}).get("tool_call_id") or f"step{order}"),
                    "order": order,
                    "tool_name": name,
                    "pid": pid or None,
                    "parameters": params,
                    "success": success,
                    "status": "success" if success else "failed_or_empty",
                    "results_summary": _summarize_results(name, results),
                }
            )
    return actions


def _summarize_results(tool_name: str, results: Any) -> dict[str, Any]:
    if results is None:
        return {}
    if isinstance(results, str):
        try:
            results = json.loads(results)
        except json.JSONDecodeError:
            return {"raw_len": len(results)}
    if not isinstance(results, dict):
        return {"type": type(results).__name__}
    out: dict[str, Any] = {}
    if tool_name == "find_product":
        products = results.get("products") or []
        out["n_products"] = len(products)
        out["product_ids"] = [str(p.get("product_id")) for p in products if p.get("product_id")][:10]
    elif tool_name == "view_product_information":
        out["product_ids"] = results.get("product_ids") or []
    elif tool_name == "recommend_product":
        out["product_ids"] = results.get("product_ids") or []
    return out


def build_commitment(steps: list[dict[str, Any]], digest: dict[str, Any]) -> dict[str, Any]:
    recommend_pids = shopping_recommend_pids(steps)
    action_anchor = recommend_pids[-1] if recommend_pids else None
    effective_ids = [a["pid"] for a in extract_actions(steps) if a.get("tool_name") == "recommend_product" and a.get("pid")]
    effective_action = effective_ids[-1] if effective_ids else action_anchor
    ambiguity: list[str] = []
    if digest.get("selected_pid") and effective_action and digest["selected_pid"] != effective_action:
        ambiguity.append("text_selected_pid_differs_from_last_recommend")
    if len(recommend_pids) > 1:
        ambiguity.append("multiple_recommend_calls")
    relation = "consistent"
    txt = digest.get("selected_pid")
    if txt and effective_action:
        if txt != effective_action:
            relation = "mismatch"
        else:
            relation = "consistent"
    elif txt and not effective_action:
        relation = "unobservable_action"
    elif effective_action and not txt:
        relation = "unobservable_text"
    else:
        relation = "ambiguous"
    return {
        "action_anchor": effective_action,
        "effective_action_ids": effective_ids,
        "recommend_pids": recommend_pids,
        "action_anchor_source": "last_successful_recommend_product",
        "commitment_relation": relation,
        "ambiguity": ambiguity,
    }


def build_text_anchor(digest: dict[str, Any]) -> dict[str, Any]:
    selected = digest.get("selected_pid")
    compared = digest.get("compared_pid")
    pids: list[str] = []
    if selected:
        pids.append(str(selected))
    return {
        "selected_pid": selected,
        "compared_pid": compared,
        "declared_pids": pids,
    }


def _norm_slot(raw: str) -> str:
    return " ".join(str(raw or "").strip().split())


def extract_evidence_from_final_user(
    user_content: str,
    *,
    query: str,
    anchor_pid: str | None,
    rival_pids: list[str],
    tool_digest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    eid = 0

    # User query requirement clauses
    for m in HARD_REQ_RE.finditer(query or ""):
        eid += 1
        clause = m.group(0)
        val_part = m.group(1).strip()
        evidence.append(
            {
                "evidence_id": f"ev_{eid}",
                "message_id": "final_call_user",
                "tool_call_id": None,
                "source_id": ["query"],
                "entity_ids": ["query"],
                "slot_raw": val_part.split()[0] if val_part else "requirement",
                "slot_norm": _norm_slot(val_part.split(":")[0] if ":" in val_part else val_part),
                "value_raw": val_part,
                "value_norm": val_part,
                "scope": "query_constraint",
                "polarity": "positive",
                "time_order": 0,
                "tool_status": "n/a",
                "char_span": [m.start(), m.end()],
                "token_span": None,
                "evidence_kind": "user_requirement",
            }
        )

    # Soft user utterance (non-hard)
    um = USER_TAG_RE.search(user_content or "")
    if um:
        eid += 1
        evidence.append(
            {
                "evidence_id": f"ev_{eid}",
                "message_id": "final_call_user",
                "tool_call_id": None,
                "source_id": ["user"],
                "entity_ids": ["user"],
                "slot_raw": "user_utterance",
                "slot_norm": "user_utterance",
                "value_raw": um.group(1).strip(),
                "value_norm": um.group(1).strip(),
                "scope": "whole_request",
                "polarity": "neutral",
                "time_order": 0,
                "tool_status": "n/a",
                "char_span": [um.start(), um.end()],
                "token_span": None,
                "evidence_kind": "user_factual_assertion",
            }
        )

    # Tool observations visible in final-call user string
    seen_pids: set[str] = set()
    for pid in PID_IN_TEXT_RE.findall(user_content or ""):
        seen_pids.add(pid)
    for td in tool_digest:
        results = td.get("results")
        if isinstance(results, dict):
            for p in results.get("products") or []:
                if p.get("product_id"):
                    seen_pids.add(str(p["product_id"]))
            for p in results.get("product_ids") or []:
                seen_pids.add(str(p))

    order = 1
    for pid in sorted(seen_pids):
        entity_role = "anchor" if anchor_pid and pid == str(anchor_pid) else (
            "rival" if pid in {str(r) for r in rival_pids} else "candidate"
        )
        # Extract title snippet around pid in user_content
        idx = (user_content or "").find(pid)
        snippet = ""
        if idx >= 0:
            snippet = user_content[max(0, idx - 20) : idx + 120]
        eid += 1
        evidence.append(
            {
                "evidence_id": f"ev_{eid}",
                "message_id": "final_call_user",
                "tool_call_id": None,
                "source_id": [pid],
                "entity_ids": [pid],
                "slot_raw": "product_record",
                "slot_norm": "product_record",
                "value_raw": snippet or pid,
                "value_norm": snippet or pid,
                "scope": "whole_product",
                "polarity": "positive",
                "time_order": order,
                "tool_status": "success",
                "char_span": [idx, idx + len(pid)] if idx >= 0 else None,
                "token_span": None,
                "evidence_kind": "tool_observation",
                "entity_role": entity_role,
            }
        )
        order += 1

    # Service filter evidence from find_product calls
    for td in tool_digest:
        if td.get("tool") != "find_product":
            continue
        svc = (td.get("params") or {}).get("service")
        if not svc:
            continue
        eid += 1
        evidence.append(
            {
                "evidence_id": f"ev_{eid}",
                "message_id": "final_call_user",
                "tool_call_id": None,
                "source_id": ["filter"],
                "entity_ids": [],
                "slot_raw": "service_filter",
                "slot_norm": "service_filter",
                "value_raw": str(svc),
                "value_norm": str(svc),
                "scope": "retrieval_filter",
                "polarity": "positive",
                "time_order": order,
                "tool_status": "success",
                "char_span": None,
                "token_span": None,
                "evidence_kind": "filter_result",
            }
        )
        order += 1
    return evidence


def build_claims(
    instances: list[dict[str, Any]],
    commitment: dict[str, Any],
    text_anchor: dict[str, Any],
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for inst in instances:
        legacy = str(inst.get("gold_verdict") or "")
        claims.append(
            {
                "claim_id": f"{inst.get('trajectory_id')}:{inst.get('instance_index')}",
                "response_span": inst.get("response_quote"),
                "referent_ids": [text_anchor.get("selected_pid")] if text_anchor.get("selected_pid") else [],
                "slot_norm": _norm_slot(inst.get("slot") or ""),
                "value": inst.get("value"),
                "scope": inst.get("claim_scope") or "anchor",
                "mode": "assertion",
                "anchor_support": None,
                "supporting_evidence_ids": [],
                "supporting_owner_set": [],
                "candidate_source_types": _candidate_sources(legacy),
                "legacy_label": legacy,
                "normalized_axes": {
                    "pm_binary": inst.get("pm_binary"),
                    "legacy_short": VERDICT_TO_LEGACY.get(legacy, legacy),
                    "commitment_relation": commitment.get("commitment_relation"),
                },
                "human_status": "annotated",
            }
        )
    return claims


def _candidate_sources(legacy: str) -> list[str]:
    if legacy == "cross_object_merge":
        return ["rival"]
    if legacy == "constraint_projection":
        return ["query"]
    if legacy == "anchored_hallucination":
        return ["anchor_contradiction"]
    return []


def compute_eligible(
    *,
    has_final_step: bool,
    stored_prompt_ok: bool,
    corrupt: bool,
    gold_present: bool,
) -> tuple[dict[str, bool], list[str]]:
    reasons: list[str] = []
    if corrupt:
        reasons.append("corrupt_or_overflow")
    if not has_final_step:
        reasons.append("missing_final_response_step")
    if not stored_prompt_ok:
        reasons.append("stored_prompt_unavailable")
    if not gold_present:
        reasons.append("missing_gold_label")

    semantic = gold_present and not corrupt and has_final_step
    native = semantic and stored_prompt_ok
    counterfactual = native
    patching = native
    return (
        {
            "semantic_eval": semantic,
            "native_replay": native,
            "counterfactual": counterfactual,
            "patching": patching,
        },
        reasons,
    )


def normalize_shopping_trajectory(
    tid: str,
    steps: list[dict[str, Any]],
    traj_gold: dict[str, Any] | None,
    instances: list[dict[str, Any]],
    *,
    raw_ref: dict[str, Any],
    model_manifest_id: str = "qwen3-32b_shopping_v1",
) -> dict[str, Any]:
    fidx = find_final_step_index(steps)
    exclusion: list[str] = []
    corrupt = False
    for st in steps:
        ei = (st.get("extra_info") or {}) if isinstance(st, dict) else {}
        if ei.get("error") in ("context_overflow", "protocol_rejected"):
            corrupt = True
            exclusion.append("corrupt_or_overflow")

    digest = extract_shopping_rollout_digest(steps)
    commitment = build_commitment(steps, digest)
    text_anchor = build_text_anchor(digest)
    actions = extract_actions(steps)
    tool_digest = shopping_tool_digest(steps)

    messages_final_call: list[dict[str, str]] = []
    messages_raw: dict[str, Any] = {"steps": []}
    stored_prompt_ok = False
    user_hash_match = False
    replay_fidelity = "not_reconstructable"

    if fidx is not None:
        final_step = steps[fidx]
        stored_messages = final_step.get("prompt") or []
        if len(stored_messages) >= 2:
            messages_final_call = [
                {"role": "system", "content": str(stored_messages[0].get("content") or "")},
                {"role": "user", "content": str(stored_messages[1].get("content") or "")},
            ]
            stored_prompt_ok = True
            replay_fidelity = "message_exact_but_original_tokens_unavailable"
            try:
                history = rebuild_history_messages(steps[:fidx])
                rebuilt_user = (
                    "# Dialogue Records History\n" + "\n\n".join(history) if history else messages_final_call[1]["content"]
                )
                user_hash_match = sha256_text(messages_final_call[1]["content"]) == sha256_text(rebuilt_user)
                if not user_hash_match:
                    replay_fidelity = "stored_prompt_exact_rebuild_from_steps_mismatch"
                    exclusion.append("stored_rebuild_user_mismatch")
            except Exception:
                exclusion.append("rebuild_history_failed")
        else:
            exclusion.append("missing_stored_final_prompt")
    else:
        exclusion.append("missing_final_response_step")

    for i, st in enumerate(steps):
        messages_raw["steps"].append(
            {
                "step_index": i,
                "prompt": st.get("prompt"),
                "completion": st.get("completion"),
                "extra_info": st.get("extra_info"),
            }
        )

    query = shopping_query_from_steps(steps)
    rival_pids = [p for p in digest.get("recommend_pids") or [] if str(p) != str(commitment.get("action_anchor"))]
    compared = text_anchor.get("compared_pid")
    if compared:
        rival_pids = list(dict.fromkeys(rival_pids + [str(compared)]))

    user_content = messages_final_call[1]["content"] if len(messages_final_call) > 1 else ""
    evidence = extract_evidence_from_final_user(
        user_content,
        query=query,
        anchor_pid=commitment.get("action_anchor"),
        rival_pids=rival_pids,
        tool_digest=tool_digest,
    )
    claims = build_claims(instances, commitment, text_anchor)

    eligible, elig_reasons = compute_eligible(
        has_final_step=fidx is not None,
        stored_prompt_ok=stored_prompt_ok,
        corrupt=corrupt,
        gold_present=traj_gold is not None,
    )
    exclusion.extend(elig_reasons)
    exclusion = list(dict.fromkeys(exclusion))

    input_hash = sha256_json(messages_final_call) if messages_final_call else None
    outcome = (traj_gold or {}).get("trajectory_outcome") or "unknown"
    is_calibration = str(tid).endswith("_cal") or "calibration" in str((traj_gold or {}).get("notes") or "").lower()

    row: dict[str, Any] = {
        "root_id": tid,
        "trajectory_id": tid,
        "raw_ref": raw_ref,
        "split": "unassigned",
        "group_id": tid,
        "annotation_version": ANNOTATION_VERSION,
        "domain": "shopping",
        "schema_version": "ccer_v1",
        "is_calibration": is_calibration,
        "trajectory_outcome": outcome,
        "pm_core_count": (traj_gold or {}).get("pm_core_count", 0),
        "messages_raw": messages_raw,
        "messages_final_call": messages_final_call,
        "model_manifest_id": model_manifest_id,
        "input_hash": input_hash,
        "actions": actions,
        "commitment": commitment,
        "text_anchor": text_anchor,
        "evidence": evidence,
        "claims": claims,
        "eligible": eligible,
        "exclusion_reason": exclusion,
        "metadata": {
            "query": query,
            "final_answer": digest.get("final_answer"),
            "replay_fidelity_level": replay_fidelity,
            "user_prompt_hash_match": user_hash_match,
            "final_step_index": fidx,
            "prompt_char_len": len(user_content),
        },
    }
    return row


def run_normalize_shopping(
    *,
    rollout_path: Path = SHOPPING_ROLLOUT,
    out_path: Path = NORMALIZED_SHOPPING,
) -> dict[str, Any]:
    rollout_index = index_shopping_rollouts([rollout_path])
    traj_gold = {r["trajectory_id"]: r for r in _load_jsonl(TRAJECTORY_SHEET)}
    inst_by_tid: dict[str, list[dict[str, Any]]] = {}
    for inst in _load_jsonl(EXPERT_INSTANCES):
        inst_by_tid.setdefault(inst["trajectory_id"], []).append(inst)

    rows: list[dict[str, Any]] = []
    missing_rollout = 0
    for tid, tg in traj_gold.items():
        steps = rollout_index.get(tid)
        if not steps:
            missing_rollout += 1
            continue
        raw_ref = {"rollout_file": str(rollout_path), "trajectory_id": tid}
        row = normalize_shopping_trajectory(
            tid,
            steps,
            tg,
            inst_by_tid.get(tid, []),
            raw_ref=raw_ref,
        )
        validate_trajectory(row)
        rows.append(row)

    write_jsonl(out_path, rows)
    stats = {
        "n_normalized": len(rows),
        "n_gold_trajectories": len(traj_gold),
        "missing_rollout": missing_rollout,
        "eligible_native_replay": sum(1 for r in rows if r["eligible"]["native_replay"]),
    }
    return stats


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    from ccer.io_utils import load_jsonl

    return load_jsonl(path)


if __name__ == "__main__":
    stats = run_normalize_shopping()
    print(json.dumps(stats, indent=2))
