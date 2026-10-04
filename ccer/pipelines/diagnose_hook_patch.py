"""ROUND4 step-1/4: verify interchange hook changes hidden states and logits."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.io_utils import load_json, write_json
from ccer.mechanism.activation_store import activation_path, get_vector, load_activation_npz
from ccer.mechanism.controls import build_control_specs
from ccer.mechanism.interchange import InterchangeSpec, _patch_hidden
from ccer.mechanism.pair_select import cem_valid_pair_ids, load_trajectory_index, messages_for_condition
from ccer.paths import P3_BINDING_ROI, P3_U_OWNER_NPZ, REPORTS
from ccer.replay.hf_forward import _model_layers, load_hf_model, tokenize_ccer_messages


def _capture_layer_hidden(
    model: Any,
    input_ids: torch.Tensor,
    *,
    layer: int,
    token_idx: int,
    spec: InterchangeSpec | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hidden_before, hidden_after) at layer output for token_idx."""
    captured: dict[str, torch.Tensor] = {}
    handle = None

    def _hook(_module, _inp, output):
        out = output.clone()
        captured["before"] = out[0, token_idx, :].detach().clone()
        if spec is not None:
            out[0, token_idx, :] = _patch_hidden(out[0, token_idx, :], spec)
        captured["after"] = out[0, token_idx, :].detach().clone()
        return out

    handle = _model_layers(model)[layer].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            logits = model(input_ids, use_cache=False).logits
    finally:
        handle.remove()
    return captured["before"], captured["after"], logits[0, -1].detach().clone()


def diagnose_pair(
    *,
    model: Any,
    tokenizer: Any,
    roi: dict[str, Any],
    u: np.ndarray,
    pm_tid: str,
    clean_tid: str,
    layer: int,
    position: str,
) -> dict[str, Any]:
    rows_index = load_trajectory_index()
    pm_traj = rows_index[str(pm_tid)]
    clean_traj = rows_index[clean_tid]
    messages = messages_for_condition(pm_traj, "original", track="CEM")

    pm_npz = load_activation_npz(activation_path(pm_tid, "original"))
    clean_npz = load_activation_npz(activation_path(clean_tid, "original"))
    pos_idx_npz = int((pm_npz.get("positions") or {}).get(position) or -1)
    pm_vec = get_vector(pm_npz, position=position, layer=layer)
    clean_vec = get_vector(clean_npz, position=position, layer=layer)
    if pm_vec is None or clean_vec is None:
        return {"error": "missing_activation_vectors", "pm_tid": pm_tid}

    tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
    prompt_len = int(tok["prompt_token_count"])
    pos_idx_live = prompt_len - 1 if position == "prompt_end" else pos_idx_npz
    seq = tok["full_ids"][:prompt_len]
    device = next(model.parameters()).device
    input_ids = torch.tensor([seq], device=device)

    spec_none = build_control_specs(
        control_id="no_intervention",
        roi={**roi, "U_owner": u.tolist()},
        donor_vec=clean_vec,
        recipient_vec=pm_vec,
        position_token_idx=pos_idx_live,
        alpha=1.0,
    )
    spec_target = build_control_specs(
        control_id="target_interchange",
        roi={**roi, "U_owner": u.tolist()},
        donor_vec=clean_vec,
        recipient_vec=pm_vec,
        position_token_idx=pos_idx_live,
        alpha=1.0,
    )
    spec_full_replace = InterchangeSpec(
        control_id="full_replace",
        layer=layer,
        position_token_idx=pos_idx_live,
        alpha=1.0,
        U=np.eye(len(pm_vec), dtype=np.float32),
        h_donor=clean_vec.astype(np.float32),
        h_recipient=pm_vec.astype(np.float32),
        mode="interchange",
    )

    b0, a0, log0 = _capture_layer_hidden(model, input_ids, layer=layer, token_idx=pos_idx_live, spec=None)
    bn, an, logn = _capture_layer_hidden(model, input_ids, layer=layer, token_idx=pos_idx_live, spec=spec_none)
    bt, at, logt = _capture_layer_hidden(model, input_ids, layer=layer, token_idx=pos_idx_live, spec=spec_target)
    bf, af, logf = _capture_layer_hidden(model, input_ids, layer=layer, token_idx=pos_idx_live, spec=spec_full_replace)

    def _norms(b, a):
        d = (a - b).float()
        return {
            "l2_before_after": float(torch.linalg.norm(d).item()),
            "cosine_before_after": float(
                torch.nn.functional.cosine_similarity(b.float().unsqueeze(0), a.float().unsqueeze(0)).item()
            ),
        }

    log_diff = (logt - log0).float()
    return {
        "pm_trajectory_id": pm_tid,
        "clean_trajectory_id": clean_tid,
        "layer": layer,
        "position": position,
        "pos_idx_npz": pos_idx_npz,
        "pos_idx_live": pos_idx_live,
        "pos_idx_match": pos_idx_npz == pos_idx_live,
        "prompt_token_count": prompt_len,
        "seq_len": len(seq),
        "delta_vec_l2": float(np.linalg.norm(clean_vec - pm_vec)),
        "no_intervention_hidden": _norms(b0, a0),
        "target_interchange_hidden": _norms(bt, at),
        "full_replace_hidden": _norms(bf, af),
        "logits_l2_ni_vs_target": float(torch.linalg.norm(log_diff).item()),
        "logits_argmax_flip": int(torch.argmax(log0).item()) != int(torch.argmax(logt).item()),
        "logits_argmax_flip_full_replace": int(torch.argmax(log0).item()) != int(torch.argmax(logf).item()),
        "mid_layer_sanity_layer": max(1, layer // 2),
    }


def run_diagnosis(*, device_map: str = "auto", limit: int = 3) -> dict[str, Any]:
    roi = load_json(P3_BINDING_ROI)
    primary = roi.get("primary_roi") or {}
    layer = int(primary.get("layer") or 63)
    position = str(primary.get("position") or "prompt_end")
    u = np.load(P3_U_OWNER_NPZ, allow_pickle=True)["U_pca"]

    selection = cem_valid_pair_ids(include_matched_clean=True)
    pairs = selection["pm_clean_cross_pairs"][:limit]

    model, tokenizer, n_layers = load_hf_model(device_map=device_map)

    pair_rows: list[dict[str, Any]] = []
    for cp in pairs:
        pair_rows.append(
            diagnose_pair(
                model=model,
                tokenizer=tokenizer,
                roi=roi,
                u=u,
                pm_tid=str(cp["pm_trajectory_id"]),
                clean_tid=str(cp["clean_trajectory_id"]),
                layer=layer,
                position=position,
            )
        )

    mid_layer = max(1, n_layers // 2)
    if pairs:
        cp = pairs[0]
        rows_index = load_trajectory_index()
        pm_traj = rows_index[str(cp["pm_trajectory_id"])]
        messages = messages_for_condition(pm_traj, "original", track="CEM")
        tok = tokenize_ccer_messages(messages, tokenizer, output_text=None)
        pos_idx = int(tok["prompt_token_count"]) - 1
        pm_npz = load_activation_npz(activation_path(str(cp["pm_trajectory_id"]), "original"))
        clean_npz = load_activation_npz(activation_path(cp["clean_trajectory_id"], "original"))
        pm_vec = get_vector(pm_npz, position=position, layer=mid_layer)
        clean_vec = get_vector(clean_npz, position=position, layer=mid_layer)
        input_ids = torch.tensor([tok["full_ids"][: tok["prompt_token_count"]]], device=next(model.parameters()).device)
        spec_mid = InterchangeSpec(
            control_id="full_replace",
            layer=mid_layer,
            position_token_idx=pos_idx,
            alpha=1.0,
            U=np.eye(len(pm_vec), dtype=np.float32),
            h_donor=clean_vec.astype(np.float32),
            h_recipient=pm_vec.astype(np.float32),
        )
        _, _, log_mid = _capture_layer_hidden(
            model, input_ids, layer=mid_layer, token_idx=pos_idx, spec=spec_mid
        )
        _, _, log_base = _capture_layer_hidden(model, input_ids, layer=mid_layer, token_idx=pos_idx, spec=None)
        exaggerated = {
            "pm_trajectory_id": cp["pm_trajectory_id"],
            "layer": mid_layer,
            "position": position,
            "logits_argmax_flip": int(torch.argmax(log_base).item()) != int(torch.argmax(log_mid).item()),
            "logits_l2": float(torch.linalg.norm((log_mid - log_base).float()).item()),
        }
    else:
        exaggerated = {}

    summary = {
        "schema_version": "ccer_hook_diagnosis_v1",
        "n_layers": n_layers,
        "primary_roi": primary,
        "label_source_note": "CEM cohort from cem_ah_final_adjudication via cohort_manifest; P3b does not read shopping_incremental_adjudication.jsonl",
        "incremental_overlap_all_22_pm_in_incremental": True,
        "pairs": pair_rows,
        "exaggerated_positive_control": exaggerated,
        "verdict_hints": {
            "hidden_changes_on_target": sum(
                1 for r in pair_rows if (r.get("target_interchange_hidden") or {}).get("l2_before_after", 0) > 1e-6
            ),
            "logits_flip_on_target": sum(1 for r in pair_rows if r.get("logits_argmax_flip")),
            "logits_flip_full_replace_at_roi": sum(1 for r in pair_rows if r.get("logits_argmax_flip_full_replace")),
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "p3_hook_diagnosis.json"
    write_json(out, summary)
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()
    print(json.dumps(run_diagnosis(device_map=args.device_map, limit=args.limit), indent=2))
