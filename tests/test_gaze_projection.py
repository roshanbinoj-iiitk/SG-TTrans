# tests/test_gaze_projection.py
import pytest
import numpy as np
from sg_ttrans.types import GazeVector
from sg_ttrans.cabin.gaze_estimator import (
    project_gaze_to_windshield,
    generate_fixation_cone,
    GazeProjector,
)

def test_gaze_projection_center():
    projector = GazeProjector()
    # Looking directly along z-axis (pitch=0, yaw=0)
    gaze = GazeVector(unit_vector=np.array([0.0, 0.0, 1.0]), pitch=0.0, yaw=0.0)
    pt = projector.project(gaze)
    assert 0.0 <= pt[0] <= 1.0
    assert 0.0 <= pt[1] <= 1.0
    # Optical center should project near (0.5, 0.5)
    assert np.isclose(pt[0], 0.5, atol=0.05)
    assert np.isclose(pt[1], 0.5, atol=0.05)

def test_gaze_projection_clamping():
    projector = GazeProjector()
    # Extreme gaze looking far right/up
    gaze = GazeVector(unit_vector=np.array([0.9, 0.9, 0.1]), pitch=1.2, yaw=1.2)
    pt = projector.project(gaze)
    assert 0.0 <= pt[0] <= 1.0
    assert 0.0 <= pt[1] <= 1.0

def test_fixation_cone_density():
    center = (0.5, 0.5)
    cone = generate_fixation_cone(center, sigma=0.08, grid_size=(64, 64))
    assert cone.shape == (64, 64)
    assert np.isclose(cone.sum(), 1.0, atol=1e-3)
    # Peak density should be at grid center (31 or 32 for 64-length grid)
    peak_y, peak_x = np.unravel_index(np.argmax(cone), cone.shape)
    assert abs(peak_y - 31.5) <= 0.5
    assert abs(peak_x - 31.5) <= 0.5

