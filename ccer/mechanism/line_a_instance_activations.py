"""Line A: instance-aligned activation extraction (quote span → token indices)."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ccer.io_utils import load_jsonl
from ccer.mechanism.activation_store import save_activation_npz
from ccer.mechanism.supervised_probe import (
    CLEAN_VERDICT,
    PM_VERDICTS,
    ProbeDataset,
    line_a_skip_trajectory_ids,
)
from ccer.paths import INCREMENTAL_ADJUDICATION_JSONL, LINE_A_INSTANCE_ACTIVATIONS, NORMALIZED_SHOPPING
from ccer.replay.answer_utils import extract_answer, normalize_final_synthesis_text
from ccer.replay.hf_forward import probe_layer_indices
from ccer.replay.position_registry import char_span_to_token_indices

LINE_A_INSTANCE_POSITIONS = ("prompt_end", "claim_onset", "pre_value")


def _load_splits() -> dict[str, str]:
    from ccer.paths import SPLIT_MANIFEST_JSON

    payload = json.loads(SPLIT_MANIFEST_JSON.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in (payload.get("splits") or {}).items()}


def instance_npz_path(instance_audit_key: str, trajectory_id: str) -> Path:
    safe = hashlib.sha1(instance_audit_key.encode("utf-8")).hexdigest()[:16]
    tid = trajectory_id.replace("/", "_")
    return LINE_A_INSTANCE_ACTIVATIONS / tid / f"{safe}.npz"


def resolve_instance_quote_tokens(
    *,
    serialized_text: str,
    offset_mapping: list[tuple[int, int]],
    answer_body: str,
    quote_start: int,
    quote_end: int,
    response_quote: str,
) -> dict[str, int | None]:
    """Map adjudication quote char span → token indices (Legible Failures / Line D style)."""
    positions: dict[str, int | None] = {p: None for p in LINE_A_INSTANCE_POSITIONS}

    abs_start = serialized_text.find(answer_body)
    if abs_start < 0:
        abs_start = serialized_text.lower().find(answer_body.lower())
    if abs_start < 0:
        return positions

    q0 = abs_start + int(quote_start)
    q1 = abs_start + int(quote_end)
    quote_idx = char_span_to_token_indices(offset_mapping, q0, q1)
    if quote_idx:
        positions["claim_onset"] = quote_idx[0]

    quote_text = str(response_quote or "")
    if ":" in quote_text:
        value = quote_text.split(":", 1)[1].strip()
        if value:
            rel = quote_text.lower().find(value.lower())
            if rel >= 0:
                val_start = q0 + rel
                pre_idx = char_span_to_token_indices(offset_mapping, q0, val_start)
                if pre_idx:
                    positions["pre_value"] = pre_idx[-1]

    return positions


@dataclass
class LineAInstanceRow:
    instance_audit_key: str
    trajectory_id: str
    split: str
    y: int
    gold_verdict: str
    response_quote: str
    final_answer: str
    quote_start: int
    quote_end: int


def line_a_adjudication_instance_rows(*, require_activation: bool = False) -> list[LineAInstanceRow]:
    splits = _load_splits()
    skips = line_a_skip_trajectory_ids()
    traj_meta = {r["trajectory_id"]: r for r in load_jsonl(NORMALIZED_SHOPPING)}
    rows: list[LineAInstanceRow] = []

    for row in load_jsonl(INCREMENTAL_ADJUDICATION_JSONL):
        if not row.get("commitment_eligible"):
            continue
        verdict = row.get("gold_verdict")
        if verdict not in PM_VERDICTS and verdict != CLEAN_VERDICT:
            continue
        tid = str(row["trajectory_id"])
        if tid in skips:
            continue
        split = splits.get(tid)
        if split not in ("train", "dev", "test"):
            continue
        y = 1 if verdict in PM_VERDICTS else 0
        traj = traj_meta.get(tid, {})
        final_answer = str((traj.get("metadata") or {}).get("final_answer") or "")
        if not final_answer.strip():
            continue
        iak = str(row["instance_audit_key"])
        path = instance_npz_path(iak, tid)
        if require_activation and not path.is_file():
            continue
        qs = row.get("quote_start")
        qe = row.get("quote_end")
        if qs is None or qe is None:
            continue
        rows.append(
            LineAInstanceRow(
                instance_audit_key=iak,
                trajectory_id=tid,
                split=split,
                y=y,
                gold_verdict=str(verdict),
                response_quote=str(row.get("response_quote") or ""),
                final_answer=final_answer,
                quote_start=int(qs),
                quote_end=int(qe),
            )
        )
    return rows


def group_instances_by_trajectory(rows: list[LineAInstanceRow]) -> dict[str, list[LineAInstanceRow]]:
    grouped: dict[str, list[LineAInstanceRow]] = defaultdict(list)
    for r in rows:
        grouped[r.trajectory_id].append(r)
    return dict(grouped)


def extract_line_a_instance_trajectory(
    *,
    model: Any,
    tokenizer: Any,
    trajectory: dict[str, Any],
    instances: list[LineAInstanceRow],
    layer_indices: list[int],
) -> dict[str, Any]:
    """One teacher-forced forward per trajectory; one npz per instance (quote-aligned tokens)."""
    from ccer.replay.hf_forward import teacher_force_on_ids, tokenize_ccer_messages
    from ccer.replay.position_registry import resolve_positions

    output_text = str((trajectory.get("metadata") or {}).get("final_answer") or "")
    messages = trajectory.get("messages_final_call") or []
    tok_pack = tokenize_ccer_messages(messages, tokenizer, output_text=output_text)
    pos_info = resolve_positions(
        trajectory=trajectory,
        serialized_text=tok_pack["serialized_text"],
        offset_mapping=tok_pack["offset_mapping"],
        prompt_token_count=tok_pack["prompt_token_count"],
        full_token_count=len(tok_pack["full_ids"]),
    )
    positions_base = pos_info["positions"]
    prompt_end_idx = positions_base.get("prompt_end")

    fwd = teacher_force_on_ids(
        model,
        tokenizer,
        tok_pack["full_ids"],
        prompt_token_count=tok_pack["prompt_token_count"],
        output_ids=tok_pack["output_ids"],
    )
    hidden_states = fwd["hidden_states"]
    answer_body = extract_answer(normalize_final_synthesis_text(output_text))

    def _vec_at(idx: int, li: int) -> np.ndarray | None:
        if idx is None or li + 1 >= len(hidden_states):
            return None
        return hidden_states[li + 1][0, int(idx)].detach().cpu().float().numpy().astype(np.float16)

    written = 0
    skipped = 0
    errors: list[str] = []
    for inst in instances:
        out_path = instance_npz_path(inst.instance_audit_key, inst.trajectory_id)
        if out_path.is_file():
            skipped += 1
            continue
        inst_positions = resolve_instance_quote_tokens(
            serialized_text=tok_pack["serialized_text"],
            offset_mapping=tok_pack["offset_mapping"],
            answer_body=answer_body,
            quote_start=inst.quote_start,
            quote_end=inst.quote_end,
            response_quote=inst.response_quote,
        )
        positions: dict[str, int | None] = {
            "prompt_end": prompt_end_idx,
            "claim_onset": inst_positions.get("claim_onset"),
            "pre_value": inst_positions.get("pre_value"),
        }
        vectors: dict[str, dict[str, np.ndarray]] = {}
        for pos in LINE_A_INSTANCE_POSITIONS:
            idx = positions.get(pos)
            if idx is None:
                continue
            layer_map: dict[str, np.ndarray] = {}
            for li in layer_indices:
                v = _vec_at(idx, li)
                if v is not None:
                    layer_map[f"layer_{li}"] = v
            if layer_map:
                vectors[pos] = layer_map

        if not vectors.get("claim_onset"):
            errors.append(f"missing_claim_onset:{inst.instance_audit_key}")
            continue

        save_activation_npz(
            out_path,
            trajectory_id=inst.trajectory_id,
            condition_id="line_a_instance",
            cohort="line_a_instance",
            layer_indices=layer_indices,
            positions={k: v for k, v in positions.items() if v is not None},
            vectors=vectors,
            metadata={
                "instance_audit_key": inst.instance_audit_key,
                "quote_start": inst.quote_start,
                "quote_end": inst.quote_end,
                "response_quote": inst.response_quote,
                "gold_verdict": inst.gold_verdict,
            },
        )
        written += 1

    return {
        "n_instances": len(instances),
        "n_written": written,
        "n_skipped_existing": skipped,
        "errors": errors,
        "layer_indices": layer_indices,
        "n_layers": len(layer_indices),
    }


def build_adjudication_instance_dataset(*, require_activation: bool = True) -> ProbeDataset:
    rows = line_a_adjudication_instance_rows(require_activation=require_activation)
    activation_layers: dict[str, list[int]] = {}
    instances: list[dict[str, Any]] = []

    for r in rows:
        path = instance_npz_path(r.instance_audit_key, r.trajectory_id)
        if require_activation and not path.is_file():
            continue
        loaded = None
        if path.is_file():
            from ccer.mechanism.activation_store import load_activation_npz

            loaded = load_activation_npz(path)
            activation_layers[r.instance_audit_key] = list(loaded["layer_indices"])
        instances.append(
            {
                "instance_audit_key": r.instance_audit_key,
                "trajectory_id": r.trajectory_id,
                "split": r.split,
                "y": r.y,
                "gold_verdict": r.gold_verdict,
                "response_quote": r.response_quote,
                "final_answer": r.final_answer,
                "has_activation": loaded is not None,
            }
        )

    manifest = {
        "schema": "line_a_instance_aligned_adjudication",
        "n_instances": len(instances),
        "n_trajectories": len({r["trajectory_id"] for r in instances}),
        "require_activation": require_activation,
        "split_counts": {
            sp: sum(1 for r in instances if r["split"] == sp) for sp in ("train", "dev", "test")
        },
        "label_counts": {
            "pm": sum(1 for r in instances if r["y"] == 1),
            "clean": sum(1 for r in instances if r["y"] == 0),
        },
        "positions": list(LINE_A_INSTANCE_POSITIONS),
        "reference_library": "third_party/sonde_linear_probes (layer/position sweep protocol)",
    }
    return ProbeDataset(instances=instances, activation_layers=activation_layers, manifest=manifest)


def activation_matrix_for_instances(
    instances: list[dict[str, Any]],
    *,
    position: str,
    layer: int,
) -> tuple[np.ndarray | None, list[int]]:
    from ccer.mechanism.activation_store import load_activation_npz

    rows: list[np.ndarray] = []
    valid_idx: list[int] = []
    cache: dict[str, dict[str, Any]] = {}
    for i, inst in enumerate(instances):
        iak = str(inst["instance_audit_key"])
        tid = inst["trajectory_id"]
        path = instance_npz_path(iak, tid)
        if not path.is_file():
            continue
        if iak not in cache:
            cache[iak] = load_activation_npz(path)
        vec = (cache[iak].get("vectors") or {}).get(position, {}).get(f"layer_{layer}")
        if vec is None:
            continue
        rows.append(np.asarray(vec, dtype=np.float32))
        valid_idx.append(i)
    if not rows:
        return None, []
    return np.stack(rows, axis=0), valid_idx
