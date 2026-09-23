"""
Physiological fatigue metrics: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
and Percentage of Eyelid Closure (PERCLOS).
"""

from collections.abc import Sequence
import numpy as np

def compute_ear(eye_pts: np.ndarray, eps: float = 1e-7) -> float:
    """
    Computes Eye Aspect Ratio (EAR) according to Equation (1):
        EAR = (||p2 - p6||_2 + ||p3 - p5||_2) / (2 * ||p1 - p4||_2)
    Args:
        eye_pts: (6, 2) or (6, 3) array of eye landmark coordinates.
                 p1, p4 are horizontal eye corners; p2, p3, p5, p6 are upper/lower eyelids.
        eps: Epsilon guard against division by zero.
    Returns:
        float: EAR value.
    """
    if len(eye_pts) < 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = eye_pts[:6]
    vert1 = float(np.linalg.norm(p2 - p6))
    vert2 = float(np.linalg.norm(p3 - p5))
    horiz = float(np.linalg.norm(p1 - p4))
    if horiz < eps:
        return 0.0
    return float((vert1 + vert2) / (2.0 * horiz))

def compute_mar(mouth_pts: np.ndarray, eps: float = 1e-7) -> float:
    """
    Computes Mouth Aspect Ratio (MAR) according to Equation (2):
        MAR = (||p14 - p18||_2 + ||p15 - p17||_2) / (2 * ||p12 - p16||_2)
    Args:
        mouth_pts: (7, 2) or (7, 3) or (20, 2) array of inner mouth landmarks.
                   Indices mapped to: p12 (left corner), p14 (upper left), p15 (upper right),
                   p16 (right corner), p17 (lower right), p18 (lower left).
        eps: Epsilon guard against division by zero.
    Returns:
        float: MAR value.
    """
    if len(mouth_pts) < 7:
        return 0.0
    # In standard 7-point slice: [p12, p13, p14, p15, p16, p17, p18]
    p12 = mouth_pts[0]
    p14 = mouth_pts[2]
    p15 = mouth_pts[3]
    p16 = mouth_pts[4]
    p17 = mouth_pts[5]
    p18 = mouth_pts[6]
    
    vert1 = float(np.linalg.norm(p14 - p18))
    vert2 = float(np.linalg.norm(p15 - p17))
    horiz = float(np.linalg.norm(p12 - p16))
    if horiz < eps:
        return 0.0
    return float((vert1 + vert2) / (2.0 * horiz))

def compute_perclos(ear_history: Sequence[float] | np.ndarray, threshold: float = 0.20) -> float:
    """
    Computes PERCLOS_W according to Equation (3):
        PERCLOS_W(t) = (1 / W) * sum_{tau=t-W+1}^t I(EAR(tau) < delta_EAR)
    Args:
        ear_history: List or 1D array of recent EAR values in sliding window.
        threshold: Eye closure threshold delta_EAR (default 0.20).
    Returns:
        float: Fraction of frames where eyes were closed.
    """
    if len(ear_history) == 0:
        return 0.0
    closed_frames = sum(1 for e in ear_history if e < threshold)
    return float(closed_frames / len(ear_history))
