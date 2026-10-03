"""
Time-to-Collision (TTC) and continuous collision hazard scoring.
"""

import numpy as np
from sg_ttrans.config import STHGSTConfig

def compute_ttc(distance: float, relative_velocity: float) -> float:
    """Computes Time-to-Collision (TTC) in seconds. Negative relative_velocity indicates approach."""
    if relative_velocity < -1e-4:
        return float(distance / (-relative_velocity))
    return float("inf")

def compute_hazard_weight(ttc: float, cfg: STHGSTConfig = None) -> float:
    """Calculates continuous intrinsic hazard weight H(v_i) in [0, 1]."""
    if cfg is None:
        cfg = STHGSTConfig()
    if ttc == float("inf") or ttc <= 0:
        return 0.0
    val = (ttc - cfg.tau_crit) / cfg.lambda_scale
    sig = 1.0 / (1.0 + np.exp(val))
    return float(np.clip(sig, 0.0, 1.0))
