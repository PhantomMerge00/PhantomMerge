"""Batch J-lens anchor-value mass cache for AGR Round 3."""
from __future__ import annotations

import json
import math
from typing import Any

from ccer.io_utils import load_jsonl, write_jsonl
from ccer.mechanism.activation_store import get_vector, load_activation_npz
from ccer.mechanism.anchor_value_resolver import resolve_anchor_value
from ccer.mechanism.bind_surprise import compute_anchor_value_workspace_mass
from ccer.mechanism.jspace_readout import readout_topk
from ccer.mechanism.line_a_cohort_v3 import build_probe_dataset_v3
from ccer.mechanism.line_a_instance_activations import instance_npz_path
from ccer.paths import AGR_ANCHOR_VALUE_MASS_CACHE, INCREMENTAL_ADJUDICATION_JSONL
from ccer.mechanism.pair_select import load_trajectory_index
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

LINE_A_LAYER = 49
LINE_A_POSITION = "claim_onset"


def _load_adjudication_index() -> dict[str, dict[str, Any]]:
    if not INCREMENTAL_ADJUDICATION_JSONL.is_file():
        return {}
    return {str(r["instance_audit_key"]): r for r in load_jsonl(INCREMENTAL_ADJUDICATION_JSONL)}


def build_anchor_mass_cache_rows(
    *,
    lens: JacobianLens,
    jlens_model: LensModel,
    tokenizer: Any,
    topk: int = 50,
    layer: int = LINE_A_LAYER,
) -> list[dict[str, Any]]:
    ds = build_probe_dataset_v3(require_activation=True)
    adj = _load_adjudication_index()
    traj_index = load_trajectory_index()
    rows: list[dict[str, Any]] = []

    for inst in ds.instances:
        iak = str(inst["instance_audit_key"])
        tid = str(inst["trajectory_id"])
        path = instance_npz_path(iak, tid)
        if not path.is_file():
            continue
        loaded = load_activation_npz(path)
        h = get_vector(loaded, position=LINE_A_POSITION, layer=layer)
        if h is None:
            continue

        resolution = resolve_anchor_value(
            y_pm=int(inst["y"]),
            response_quote=str(inst.get("response_quote") or ""),
            adjudication_row=adj.get(iak),
            traj=traj_index.get(tid),
        )

        z2_available = resolution.v_anchor is not None
        log_mass: float | None = None
        mass_payload: dict[str, Any] = {
            "anchor_value_mass": None,
            "anchor_value_hits": [],
            "anchor_value_rank": None,
        }
        if resolution.v_anchor:
            topk_rows = readout_topk(
                lens,
                jlens_model,
                tokenizer,
                h,
                layer=layer,
                k=topk,
                use_jacobian=True,
            )
            mass_payload = compute_anchor_value_workspace_mass(topk_rows, resolution.v_anchor)
            raw = mass_payload.get("log_anchor_value_mass")
            if raw is not None and math.isfinite(float(raw)):
                log_mass = float(raw)

        rows.append(
            {
                "instance_audit_key": iak,
                "trajectory_id": tid,
                "y_pm": int(inst["y"]),
                "anchor_value": resolution.v_anchor,
                "anchor_pid": resolution.anchor_pid,
                "anchor_provenance": resolution.provenance,
                "anchor_miss_reason": resolution.miss_reason,
                "slot_norm": resolution.slot_norm,
                "z2_available": z2_available,
                "log_anchor_value_mass": log_mass,
                "log_anchor_value_mass_raw": mass_payload.get("log_anchor_value_mass"),
                "anchor_value_mass": (
                    float(mass_payload.get("anchor_value_mass") or 0.0)
                    if z2_available
                    else None
                ),
                "anchor_value_hits": mass_payload.get("anchor_value_hits") or [],
                "anchor_value_rank": mass_payload.get("anchor_value_rank"),
                "topk": topk,
                "layer": layer,
            }
        )
    return rows


def write_anchor_mass_cache(
    rows: list[dict[str, Any]],
    *,
    path: Any | None = None,
) -> None:
    out = path or AGR_ANCHOR_VALUE_MASS_CACHE
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)


def load_anchor_mass_cache_index() -> dict[str, dict[str, Any]]:
    if not AGR_ANCHOR_VALUE_MASS_CACHE.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in AGR_ANCHOR_VALUE_MASS_CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[str(r["instance_audit_key"])] = r
    return out
