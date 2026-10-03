# tests/test_synthetic_dual_stream.py
import pytest
import torch
from sg_ttrans.data.synthetic_dual_stream import SyntheticDualStreamDataset

def test_synthetic_dataset_batching():
    dataset = SyntheticDualStreamDataset(num_samples=10, num_objects=5)
    sample = dataset[0]
    assert "gaze" in sample
    assert "nodes" in sample
    assert "mask" in sample
    assert "target_action" in sample
    assert sample["gaze"].shape == (1, 5)
    assert sample["nodes"].shape == (5, 8)
    assert sample["mask"].shape == (5,)

def test_synthetic_scenario_types():
    # Scenario A (Handover): Gaze directed at hazard
    ds_a = SyntheticDualStreamDataset(num_samples=5, scenario="aligned")
    sample_a = ds_a[0]
    assert sample_a["target_action"] == 0

    # Scenario B (Spatial Cue): Gaze away from hazard
    ds_b = SyntheticDualStreamDataset(num_samples=5, scenario="blindness")
    sample_b = ds_b[0]
    assert sample_b["target_action"] == 1
