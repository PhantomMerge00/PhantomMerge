"""Persist activation vectors from P3-0 extraction.

Note: Line A supervised probes must use instance-aligned activations under
``results/line_a/instance_activations/`` (see ``line_a_instance_activations``).
Do not use ``activation_path(..., "original")`` for Line A feature extraction.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np

from ccer.paths import P3_ACTIVATIONS


def _parse_metadata_blob(raw: Any) -> dict[str, Any]:
    text = str(raw).strip()
    if not text or text == "None":
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def activation_path(trajectory_id: str, condition_id: str) -> Path:
    safe_tid = trajectory_id.replace("/", "_")
    return P3_ACTIVATIONS / safe_tid / f"{condition_id}.npz"


def save_activation_npz(
    path: Path,
    *,
    trajectory_id: str,
    condition_id: str,
    cohort: str,
    layer_indices: list[int],
    positions: dict[str, int | None],
    vectors: dict[str, dict[str, np.ndarray]],
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "layer_indices": np.array(layer_indices, dtype=np.int32),
        "trajectory_id": np.array(trajectory_id),
        "condition_id": np.array(condition_id),
        "cohort": np.array(cohort),
    }
    for pos_name, layer_map in vectors.items():
        for layer_key, vec in layer_map.items():
            arrays[f"{pos_name}__{layer_key}"] = vec
    for pos_name, idx in positions.items():
        if idx is not None:
            arrays[f"posidx__{pos_name}"] = np.array(int(idx), dtype=np.int32)
    if metadata:
        arrays["meta_json"] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)


def load_activation_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    layer_indices = [int(x) for x in data["layer_indices"].tolist()]
    positions: dict[str, int] = {}
    vectors: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, Any] = {}
    for key in data.files:
        if key.startswith("posidx__"):
            positions[key.replace("posidx__", "", 1)] = int(data[key])
        elif key == "meta_json":
            metadata = _parse_metadata_blob(data[key])
        elif "__layer_" in key:
            pos_name, layer_key = key.split("__", 1)
            vectors.setdefault(pos_name, {})[layer_key] = data[key].astype(np.float32)
    return {
        "layer_indices": layer_indices,
        "positions": positions,
        "vectors": vectors,
        "metadata": metadata,
        "trajectory_id": str(data["trajectory_id"]),
        "condition_id": str(data["condition_id"]),
        "cohort": str(data["cohort"]),
    }


def fast_npz_roi_ok(
    trajectory_id: str,
    *,
    position: str,
    layer: int,
    condition_id: str = "original",
    require_line_d_v3: bool = True,
) -> tuple[bool, str]:
    """Check ROI keys/metadata without loading full activation vectors (fast pool audit)."""
    path = activation_path(trajectory_id, condition_id)
    if not path.is_file():
        return False, f"missing_npz:{trajectory_id}"
    vec_key = f"{position}__layer_{layer}"
    pos_key = f"posidx__{position}"
    with np.load(path, allow_pickle=True) as data:
        if vec_key not in data.files:
            return False, f"missing_vector:{trajectory_id}:{position}:L{layer}"
        if pos_key not in data.files:
            return False, f"missing_position_idx:{trajectory_id}:{position}"
        if int(data[pos_key]) < 0:
            return False, f"missing_position_idx:{trajectory_id}:{position}"
        if require_line_d_v3:
            if "meta_json" not in data.files:
                return False, f"line_d_v3_gate_fail:{trajectory_id}"
            meta = _parse_metadata_blob(data["meta_json"])
            if not meta.get("refreshed_line_d_v3_symmetric"):
                return False, f"line_d_v3_gate_fail:{trajectory_id}"
            if str(meta.get("claim_anchor_mode") or "") != "symmetric":
                return False, f"line_d_v3_gate_fail:{trajectory_id}"
    return True, "ok"


def get_vector(
    loaded: dict[str, Any],
    *,
    position: str,
    layer: int,
) -> np.ndarray | None:
    layer_key = f"layer_{layer}"
    vec = (loaded.get("vectors") or {}).get(position, {}).get(layer_key)
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32)


def missing_layers(path: Path, required_layers: list[int]) -> list[int]:
    """Return subset of ``required_layers`` absent from an on-disk activation file."""
    if not path.is_file():
        return list(required_layers)
    loaded = load_activation_npz(path)
    stored = set(int(x) for x in loaded.get("layer_indices") or [])
    return [int(li) for li in required_layers if int(li) not in stored]


def merge_activation_npz(
    path: Path,
    new_act: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Merge newly extracted layer vectors into an existing activation NPZ."""
    existing = load_activation_npz(path) if path.is_file() else None
    layer_indices = sorted(
        set(int(x) for x in (existing.get("layer_indices") if existing else []))
        | set(int(x) for x in (new_act.get("layer_indices") or []))
    )
    vectors: dict[str, dict[str, np.ndarray]] = {}
    if existing:
        for pos_name, layer_map in (existing.get("vectors") or {}).items():
            vectors[pos_name] = dict(layer_map)
    for pos_name, layer_map in (new_act.get("vectors") or {}).items():
        vectors.setdefault(pos_name, {})
        for layer_key, vec in layer_map.items():
            vectors[pos_name][layer_key] = np.asarray(vec, dtype=np.float16)
    positions = dict(existing.get("positions") or {}) if existing else {}
    for pos_name, idx in (new_act.get("positions") or {}).items():
        if idx is not None:
            positions[pos_name] = int(idx)
    merged_meta = dict((existing or {}).get("metadata") or {})
    merged_meta.update(metadata or {})
    merged_meta.setdefault("merged_roi_layers", True)
    save_activation_npz(
        path,
        trajectory_id=str(new_act.get("trajectory_id") or (existing or {}).get("trajectory_id") or path.parent.name),
        condition_id=str(new_act.get("condition_id") or (existing or {}).get("condition_id") or "original"),
        cohort=str(new_act.get("cohort") or (existing or {}).get("cohort") or "CEM"),
        layer_indices=layer_indices,
        positions=positions,
        vectors=vectors,
        metadata=merged_meta,
    )
