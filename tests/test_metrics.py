import numpy as np
import pytest
from sg_ttrans.geometry.metrics import compute_ear, compute_mar, compute_perclos

def test_compute_ear_open_and_closed():
    # Canonical open eye coordinates: p1=(-1,0), p4=(1,0), p2=(-0.5, 0.5), p3=(0.5, 0.5), p5=(0.5, -0.5), p6=(-0.5, -0.5)
    open_eye = np.array([
        [-1.0, 0.0, 0.0],
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, -0.5, 0.0]
    ])
    ear_open = compute_ear(open_eye)
    assert pytest.approx(ear_open, 0.01) == 0.50

    # Closed eye: vertical distance drops to 0
    closed_eye = open_eye.copy()
    closed_eye[:, 1] = 0.0
    ear_closed = compute_ear(closed_eye)
    assert pytest.approx(ear_closed, 0.01) == 0.0

def test_compute_mar_yawn():
    # p12, p13, p14, p15, p16, p17, p18
    # Horizontal width = 2.0 (p12=(-1,0), p16=(1,0))
    # Vertical height = 1.0 (p14/p18 and p15/p17 dist = 1.0)
    mouth = np.array([
        [-1.0, 0.0, 0.0],  # p12
        [0.0, 0.0, 0.0],   # p13
        [-0.5, 0.5, 0.0],  # p14
        [0.5, 0.5, 0.0],   # p15
        [1.0, 0.0, 0.0],   # p16
        [0.5, -0.5, 0.0],  # p17
        [-0.5, -0.5, 0.0]  # p18
    ])
    mar = compute_mar(mouth)
    assert pytest.approx(mar, 0.01) == 0.50

def test_compute_perclos():
    # 60 frames: 45 frames closed (< 0.20), 15 frames open
    history = [0.10] * 45 + [0.35] * 15
    perclos = compute_perclos(history, threshold=0.20)
    assert pytest.approx(perclos, 0.001) == 45.0 / 60.0

def test_zero_division_guard():
    # Coincident points: width = 0
    coincident_eye = np.zeros((6, 3))
    ear = compute_ear(coincident_eye)
    assert ear == 0.0
