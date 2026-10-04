"""Activation NPZ metadata round-trip (Line D v3 gate dependency)."""
from __future__ import annotations

import numpy as np

from ccer.mechanism.activation_store import load_activation_npz, save_activation_npz
from ccer.replay.live_position import LINE_D_REFRESH_TAG, activation_passes_line_d_v3_gate


def test_load_activation_npz_reads_metadata_and_v3_gate(tmp_path):
    path = tmp_path / "shop_D_test" / "original.npz"
    meta = {
        LINE_D_REFRESH_TAG: True,
        "claim_anchor_mode": "symmetric",
        "merged_roi_layers": True,
    }
    save_activation_npz(
        path,
        trajectory_id="shop_D_test",
        condition_id="original",
        cohort="CEM",
        layer_indices=[49],
        positions={"claim_onset": 12},
        vectors={"claim_onset": {"layer_49": np.zeros(8, dtype=np.float16)}},
        metadata=meta,
    )
    loaded = load_activation_npz(path)
    assert loaded["metadata"] == meta
    assert activation_passes_line_d_v3_gate(loaded, position="claim_onset")


def test_load_activation_npz_parses_legacy_str_metadata(tmp_path):
    path = tmp_path / "legacy" / "original.npz"
    np.savez_compressed(
        path,
        layer_indices=np.array([32], dtype=np.int32),
        trajectory_id=np.array("legacy"),
        condition_id=np.array("original"),
        cohort=np.array("CEM"),
        posidx__claim_onset=np.array(5, dtype=np.int32),
        claim_onset__layer_32=np.zeros(4, dtype=np.float16),
        meta_json=np.array("{'merged_roi_layers': True, 'refreshed_line_d_v3_symmetric': True, 'claim_anchor_mode': 'symmetric'}"),
    )
    loaded = load_activation_npz(path)
    assert loaded["metadata"]["merged_roi_layers"] is True
    assert activation_passes_line_d_v3_gate(loaded, position="claim_onset")
