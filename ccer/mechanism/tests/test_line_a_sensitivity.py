"""Tests for Line A sensitivity dataset builders."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("${PHANTOM_MERGE_ROOT}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccer.mechanism.line_a_sensitivity import (
    build_sensitivity_dataset,
    collect_data_diagnostics,
    mixed_trajectory_ids,
    quote_label_conflict_keys,
)


def test_quote_conflicts_non_empty():
    keys = quote_label_conflict_keys()
    assert len(keys) >= 1


def test_mixed_trajectories_subset_of_cohort():
    mixed = mixed_trajectory_ids(require_activation=False)
    assert 0 < len(mixed) < 500


def test_trajectory_dataset_one_row_per_tid():
    ds = build_sensitivity_dataset(label_mode="trajectory_any_pm", require_activation=False)
    tids = [r["trajectory_id"] for r in ds.instances]
    assert len(tids) == len(set(tids))


def test_exclude_mixed_reduces_instances():
    full = build_sensitivity_dataset(label_mode="instance", require_activation=False)
    filt = build_sensitivity_dataset(
        label_mode="instance",
        exclude_mixed=True,
        require_activation=False,
    )
    assert filt.manifest["n_instances"] < full.manifest["n_instances"]


def test_diagnostics_schema():
    d = collect_data_diagnostics(require_activation=False)
    assert "bow_test_auroc_response_quote" in d
    assert d["n_instances_with_activation"] > 100
