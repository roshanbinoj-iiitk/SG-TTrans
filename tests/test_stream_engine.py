import numpy as np
import pytest
from sg_ttrans.config import SGTransConfig
from sg_ttrans.inference.stream_engine import StreamEngine

def test_stream_engine_processing():
    cfg = SGTransConfig(sequence_length=5, d_model=128, num_layers=1)
    engine = StreamEngine(cfg)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Push 5 frames to fill buffer
    for i in range(5):
        result, hud_frame = engine.process_frame(dummy_frame, velocity=80.0, ttc=8.0)

    assert result is not None
    assert "rsi" in result
    assert "driver_state" in result
    assert "adas_level" in result
    assert "fatigue_prob" in result
    assert "tau_persist" in result
    assert hud_frame.shape == (480, 640, 3)

def test_stream_engine_hud_drawing():
    cfg = SGTransConfig(sequence_length=2, d_model=128, num_layers=1)
    engine = StreamEngine(cfg)
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

    # Two frames to become ready
    engine.process_frame(dummy_frame, velocity=90.0, ttc=3.5)
    result, hud_frame = engine.process_frame(dummy_frame, velocity=90.0, ttc=3.5)

    # Resulting HUD frame should be modified with drawn text and gauge
    assert not np.array_equal(hud_frame, dummy_frame)
