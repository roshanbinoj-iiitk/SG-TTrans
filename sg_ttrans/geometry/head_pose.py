"""
3D Head Pose Euler angle estimation via Perspective-n-Point (cv2.solvePnP)
and 16-D kinematic token assembly.
"""

import cv2
import numpy as np

# Standard 3D anthropometric facial model points (in millimeters)
# Coordinate system matches image coordinates: +X right, +Y down, +Z away from camera
MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],          # Nose tip
    [0.0, 330.0, -65.0],      # Chin (+Y down)
    [-225.0, -170.0, -135.0], # Left eye corner (-Y up)
    [225.0, -170.0, -135.0],  # Right eye corner (-Y up)
    [-150.0, 150.0, -125.0],  # Left mouth corner (+Y down)
    [150.0, 150.0, -125.0]    # Right mouth corner (+Y down)
], dtype=np.float64)

class HeadPoseEstimator:
    """
    Estimates 3D Head Pose Euler angles (Yaw, Pitch, Roll) using Perspective-n-Point (PnP).
    """
    def __init__(self, model_pts: np.ndarray | None = None):
        self.model_pts = model_pts if model_pts is not None else MODEL_POINTS_3D

    def estimate(self, pts_2d: np.ndarray, img_w: int, img_h: int) -> tuple[float, float, float]:
        """
        Calculates Yaw, Pitch, Roll in degrees.
        Args:
            pts_2d: (6, 2) array matching nose, chin, left eye, right eye, left mouth, right mouth.
            img_w: Frame width in pixels.
            img_h: Frame height in pixels.
        Returns:
            tuple[float, float, float]: (phi_yaw, theta_pitch, psi_roll) in degrees.
        """
        focal_length = float(img_w)
        center = (float(img_w) / 2.0, float(img_h) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.model_pts, pts_2d.astype(np.float64), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return (0.0, 0.0, 0.0)

        rmat, _ = cv2.Rodrigues(rvec)
        # RQ decomposition yields Euler angles in degrees (pitch, yaw, roll)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles[0], angles[1], angles[2]

        return (float(yaw), float(pitch), float(roll))

def build_kinematic_token(
    ear: float,
    mar: float,
    perclos: float,
    euler: np.ndarray,
    prev_kinematics: np.ndarray | None = None,
    ear_l: float = 0.0,
    ear_r: float = 0.0,
    dt: float = 1.0 / 30.0
) -> np.ndarray:
    """
    Constructs the 16-D kinematic token z_geom^t according to Equation (5):
    [EAR, MAR, PERCLOS, phi, theta, psi, dEAR/dt, dMAR/dt, dphi/dt, dtheta/dt, dpsi/dt,
     EAR_L, EAR_R, ||euler||, ||d_euler/dt||, blink_flag]
    """
    if prev_kinematics is not None and len(prev_kinematics) == 16:
        d_ear = float((ear - prev_kinematics[0]) / dt)
        d_mar = float((mar - prev_kinematics[1]) / dt)
        d_euler = (euler - prev_kinematics[3:6]) / dt
    else:
        d_ear = 0.0
        d_mar = 0.0
        d_euler = np.zeros(3, dtype=np.float32)

    euler_norm = float(np.linalg.norm(euler))
    d_euler_norm = float(np.linalg.norm(d_euler))
    blink_flag = 1.0 if ear < 0.20 else 0.0

    token = np.array([
        ear,
        mar,
        perclos,
        float(euler[0]), float(euler[1]), float(euler[2]),
        d_ear,
        d_mar,
        float(d_euler[0]), float(d_euler[1]), float(d_euler[2]),
        ear_l,
        ear_r,
        euler_norm,
        d_euler_norm,
        blink_flag
    ], dtype=np.float32)

    return token
