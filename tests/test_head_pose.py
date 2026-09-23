import numpy as np
import pytest
from sg_ttrans.geometry.head_pose import HeadPoseEstimator, build_kinematic_token

def test_head_pose_frontal():
    estimator = HeadPoseEstimator()
    # 2D image points matching standard 3D anthropometric face model
    # Image center (320, 240) in a 640x480 frame
    pts_2d = np.array([
        [320.0, 240.0],  # Nose tip
        [320.0, 330.0],  # Chin
        [250.0, 190.0],  # Left eye corner
        [390.0, 190.0],  # Right eye corner
        [270.0, 290.0],  # Left mouth corner
        [370.0, 290.0],  # Right mouth corner
    ], dtype=np.float64)
    yaw, pitch, roll = estimator.estimate(pts_2d, img_w=640, img_h=480)
    assert abs(yaw) < 20.0
    assert abs(pitch) < 20.0
    assert abs(roll) < 20.0

def test_build_kinematic_token():
    token = build_kinematic_token(
        ear=0.25, mar=0.30, perclos=0.10,
        euler=np.array([5.0, -2.0, 1.0]),
        prev_kinematics=None,
        ear_l=0.25, ear_r=0.25, dt=1.0 / 30.0
    )
    assert token.shape == (16,)
    assert token[0] == pytest.approx(0.25)
    assert token[1] == pytest.approx(0.30)
    assert token[2] == pytest.approx(0.10)
    assert token[3] == pytest.approx(5.0)
    assert token[4] == pytest.approx(-2.0)
    assert token[5] == pytest.approx(1.0)
    # First frame derivative should be 0.0
    assert token[6] == 0.0
    assert token[7] == 0.0

def test_build_kinematic_token_with_derivatives():
    prev = np.zeros(16, dtype=np.float32)
    prev[0] = 0.20  # prev ear
    prev[1] = 0.20  # prev mar
    prev[3:6] = np.array([0.0, 0.0, 0.0])  # prev euler

    curr_euler = np.array([3.0, 0.0, 0.0])
    dt = 0.1
    token = build_kinematic_token(
        ear=0.30, mar=0.25, perclos=0.15,
        euler=curr_euler,
        prev_kinematics=prev,
        ear_l=0.30, ear_r=0.30, dt=dt
    )
    # dEAR = (0.30 - 0.20) / 0.1 = 1.0
    assert pytest.approx(token[6], 0.01) == 1.0
    # dMAR = (0.25 - 0.20) / 0.1 = 0.5
    assert pytest.approx(token[7], 0.01) == 0.5
    # dYaw = (3.0 - 0.0) / 0.1 = 30.0
    assert pytest.approx(token[8], 0.01) == 30.0
