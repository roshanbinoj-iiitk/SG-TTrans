"""
End-to-end integration test verifying the entire paper pipeline:
Algorithm 1: Real-Time Dual-Stream SG-TTrans with Risk-Aware Safety Monitoring.
"""

import numpy as np
import torch
import pytest
from sg_ttrans.config import SGTransConfig
from sg_ttrans.inference.stream_engine import StreamEngine
from sg_ttrans.risk_engine.adas_controller import ADASLevel

def test_full_pipeline_algorithm_1_integration():
    """
    Verifies full computational pipeline from Algorithm 1:
    Video Stream -> FaceMesh + MobileNetV4 -> Cross-Modal Projection -> FIFO Buffer ->
    L-layer TDDA Transformer -> MeanPool -> Softmax Classification -> Persistence ->
    RSI(t) -> ADAS Level L(t).
    """
    cfg = SGTransConfig(
        sequence_length=15,
        d_vis=256,
        d_geom=16,
        d_model=256,
        num_heads=4,
        num_layers=2,
        fps=30.0
    )
    engine = StreamEngine(config=cfg, device="cuda" if torch.cuda.is_available() else "cpu")

    # Generate synthetic video stream of 20 frames
    frames = [np.full((480, 640, 3), 50, dtype=np.uint8) for _ in range(20)]

    for t, frame in enumerate(frames):
        res, hud_frame = engine.process_frame(frame, velocity=85.0, ttc=4.0)

    # After 20 frames (> 15), sequence buffer is fully populated and model has run
    assert engine.buffer.is_ready()
    assert res["driver_state"] in ["Alert", "Drowsy", "Microsleep", "Yawn", "Distracted"]
    assert 0.0 <= res["rsi"] <= 1.0
    assert res["adas_level"] in [ADASLevel.LEVEL_0_NOMINAL, ADASLevel.LEVEL_1_VISUAL, ADASLevel.LEVEL_2_AUDIO_HAPTIC, ADASLevel.LEVEL_3_CRITICAL_AEB]
    assert hud_frame.shape == (480, 640, 3)

def test_full_suite_numerical_coherence():
    cfg = SGTransConfig()
    # Confirm global parameters match paper
    assert cfg.sequence_length == 60
    assert cfg.gamma_init == 0.15
    assert cfg.w1 == 0.50
    assert cfg.w2 == 0.35
    assert cfg.w3 == 0.15
    assert cfg.tau0 == 1.5
    assert cfg.v_max == 130.0
    assert cfg.delta1 == 0.35
    assert cfg.delta2 == 0.60
    assert cfg.delta3 == 0.80
