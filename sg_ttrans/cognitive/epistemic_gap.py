"""
Epistemic Attention Gap (EAG) quantification.
"""

from typing import Optional, Tuple
import numpy as np

def compute_eag(
    hazard_weights: np.ndarray,
    accumulated_cognition: np.ndarray,
    theta_comp: float = 0.60,
) -> Tuple[float, Optional[int]]:
    """
    Computes Epistemic Attention Gap (EAG) in [0, 1] and identifies causal hazard index.
    EAG = sum(H_i * (1 - min(1, C_i / theta_comp))) / max(sum(H_i), 1e-4)
    """
    h = np.asarray(hazard_weights, dtype=np.float32)
    c = np.asarray(accumulated_cognition, dtype=np.float32)
    if len(h) == 0:
        return 0.0, None

    comprehension_ratio = np.clip(c / max(theta_comp, 1e-4), 0.0, 1.0)
    uncomprehended_gap = 1.0 - comprehension_ratio
    hazard_gaps = h * uncomprehended_gap
    
    total_hazard = float(h.sum())
    if total_hazard < 1e-4:
        return 0.0, None

    eag = float(np.clip(hazard_gaps.sum() / max(total_hazard, 1.0), 0.0, 1.0))
    top_hazard_idx = int(np.argmax(hazard_gaps))
    return eag, top_hazard_idx
