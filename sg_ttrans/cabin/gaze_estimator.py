"""
3D Gaze vector estimation and camera-to-windshield projection.
"""

from typing import Tuple
import numpy as np
from sg_ttrans.types import GazeVector

class GazeProjector:
    """Projects 3D in-cabin gaze vectors to normalized 2D road camera windshield coordinates."""
    def __init__(self, homography: np.ndarray = None):
        if homography is None:
            # Default canonical projection mapping center gaze [0, 0, 1] to (0.5, 0.5)
            self.H = np.array([
                [0.6, 0.0, 0.5],
                [0.0, 0.6, 0.5],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
        else:
            self.H = homography.astype(np.float32)

    def project(self, gaze: GazeVector) -> Tuple[float, float]:
        v = gaze.unit_vector
        z = max(float(v[2]), 1e-4)
        x_proj = float(v[0]) / z
        y_proj = float(v[1]) / z
        
        # Homogeneous transformation
        p_homo = self.H @ np.array([x_proj, y_proj, 1.0], dtype=np.float32)
        denom = float(p_homo[2]) if abs(p_homo[2]) > 1e-6 else 1.0
        u = float(np.clip(p_homo[0] / denom, 0.0, 1.0))
        v_coord = float(np.clip(p_homo[1] / denom, 0.0, 1.0))
        gaze.windshield_point = (u, v_coord)
        return (u, v_coord)

def project_gaze_to_windshield(gaze: GazeVector, H: np.ndarray = None) -> Tuple[float, float]:
    return GazeProjector(H).project(gaze)

def generate_fixation_cone(
    center: Tuple[float, float],
    sigma: float = 0.08,
    grid_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Generates normalized 2D Gaussian fixation cone over image plane."""
    h, w = grid_size
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    
    dist_sq = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    gaussian = np.exp(-dist_sq / (2.0 * (sigma ** 2)))
    sum_val = float(gaussian.sum())
    if sum_val > 1e-8:
        gaussian /= sum_val
    return gaussian.astype(np.float32)
