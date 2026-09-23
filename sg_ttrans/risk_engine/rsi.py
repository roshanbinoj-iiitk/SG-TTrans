"""
Dynamic Risk-Aware Safety Index (RSI) computation according to Equation (14).
"""

import numpy as np
from sg_ttrans.config import SGTransConfig

def compute_rsi(
    fatigue_prob: float,
    tau_persist: float,
    velocity: float,
    ttc: float,
    pitch: float,
    yaw: float,
    config: SGTransConfig | None = None
) -> float:
    """
    Computes continuous Risk-Aware Safety Index (RSI):
        RSI(t) = w1 * y_fatigue(t) * (1 - exp(-tau_persist / tau0))
               + w2 * (v(t) / v_max) * (1 / (1 + max(0, TTC(t))))
               + w3 * sigma_head(t)
    Where:
        y_fatigue = P(Drowsy) + P(Microsleep)
        sigma_head = min(1.0, (|pitch| + |yaw|) / 45 deg)
        w1=0.50, w2=0.35, w3=0.15
        tau0=1.5s, v_max=130 km/h
    Returns:
        float: RSI(t) in [0.0, 1.0]
    """
    cfg = config if config is not None else SGTransConfig()

    # 1. Visual fatigue persistence factor
    persistence_factor = 1.0 - np.exp(-tau_persist / cfg.tau0)
    term1 = cfg.w1 * fatigue_prob * persistence_factor

    # 2. Vehicular dynamic risk factor (velocity & time-to-collision)
    v_norm = velocity / cfg.v_max
    ttc_clamped = max(0.0, float(ttc))
    ttc_factor = 1.0 / (1.0 + ttc_clamped)
    term2 = cfg.w2 * v_norm * ttc_factor

    # 3. Head posture deviation factor
    sigma_head = min(1.0, (abs(pitch) + abs(yaw)) / 45.0)
    term3 = cfg.w3 * sigma_head

    rsi = term1 + term2 + term3
    return float(np.clip(rsi, 0.0, 1.0))
