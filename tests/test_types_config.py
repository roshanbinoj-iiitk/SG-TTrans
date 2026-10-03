# tests/test_types_config.py
import pytest
import numpy as np
import torch
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.types import (
    SceneNode,
    SceneGraph,
    GazeVector,
    CognitiveState,
    ArbitrationDecision,
    TakeoverAction,
)

def test_config_initialization():
    cfg = STHGSTConfig()
    assert cfg.tau_crit == 2.5
    assert cfg.tau_cog == 0.25
    assert cfg.theta_comp == 0.60
    assert cfg.theta_sacc == 0.20
    assert cfg.eag_handover_thresh == 0.20
    assert cfg.eag_cue_thresh == 0.65

def test_scene_node_contract():
    node = SceneNode(
        track_id=1,
        class_id=0,
        bbox=(0.1, 0.2, 0.3, 0.4),
        velocity=(0.0, -2.5),
        ttc=1.8,
        hazard_weight=0.82,
    )
    assert node.track_id == 1
    assert node.ttc == 1.8
    assert node.hazard_weight == 0.82

def test_gaze_vector_normalization():
    raw_vec = np.array([1.0, 1.0, 1.0])
    gaze = GazeVector.from_raw(raw_vec, pitch=0.1, yaw=-0.2)
    assert np.isclose(np.linalg.norm(gaze.unit_vector), 1.0, atol=1e-5)
    assert gaze.pitch == 0.1
    assert gaze.yaw == -0.2

def test_arbitration_decision_contract():
    dec = ArbitrationDecision(
        action=TakeoverAction.SAFE_HANDOVER,
        eag=0.08,
        causal_hazard_id=None,
        action_description="Safe torque handover authorized",
    )
    assert dec.action == TakeoverAction.SAFE_HANDOVER
    assert dec.eag < 0.20
