# tests/test_scene_graph.py
import pytest
import numpy as np
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.scene_graph.ttc_calculator import compute_ttc, compute_hazard_weight
from sg_ttrans.scene_graph.graph_builder import DynamicSceneGraphBuilder

def test_ttc_calculation():
    # Approaching at 5 m/s from 10m away -> TTC = 2.0s
    assert np.isclose(compute_ttc(distance=10.0, relative_velocity=-5.0), 2.0)
    # Moving away (+3 m/s) -> TTC = infinity
    assert compute_ttc(distance=10.0, relative_velocity=3.0) == float("inf")
    # Zero velocity -> TTC = infinity
    assert compute_ttc(distance=10.0, relative_velocity=0.0) == float("inf")

def test_hazard_weight_sigmoid():
    cfg = STHGSTConfig()
    # At critical horizon (TTC = 2.5s) -> H = 0.5
    h_crit = compute_hazard_weight(ttc=2.5, cfg=cfg)
    assert np.isclose(h_crit, 0.5, atol=0.02)
    # Imminent collision (TTC = 1.0s) -> H close to 1.0
    h_imm = compute_hazard_weight(ttc=1.0, cfg=cfg)
    assert h_imm > 0.90
    # Far obstacle (TTC = 10.0s) -> H close to 0.0
    h_far = compute_hazard_weight(ttc=10.0, cfg=cfg)
    assert h_far < 0.05

def test_dynamic_scene_graph_builder():
    builder = DynamicSceneGraphBuilder()
    detections = [
        {"id": 1, "class_id": 0, "bbox": (0.2, 0.4, 0.1, 0.2), "distance": 12.0, "rel_vel": -6.0},
        {"id": 2, "class_id": 1, "bbox": (0.6, 0.3, 0.2, 0.3), "distance": 25.0, "rel_vel": -2.0},
    ]
    graph = builder.build(detections)
    assert len(graph.nodes) == 2
    assert graph.nodes[0].ttc == 2.0
    assert graph.nodes[0].hazard_weight > 0.60
    assert graph.edge_index.shape[0] == 2
