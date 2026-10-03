"""
Neuro-Visual Leaky Cognitive Accumulator modeling human visual perception latency.
"""

from typing import Optional
import numpy as np
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.types import CognitiveState, FixationState

class LeakyCognitiveAccumulator:
    """Models human neuro-visual cognitive processing latency (tau_cog = 250 ms)."""
    def __init__(self, cfg: Optional[STHGSTConfig] = None):
        self.cfg = cfg or STHGSTConfig()
        # beta = exp(-delta_t / tau_cog)
        self.beta = float(np.exp(-self.cfg.delta_t / self.cfg.tau_cog))
        self.c_prev: Optional[np.ndarray] = None

    def reset(self):
        self.c_prev = None

    def step(self, attention_weights: np.ndarray) -> CognitiveState:
        weights = np.asarray(attention_weights, dtype=np.float32)
        if self.c_prev is None or len(self.c_prev) != len(weights):
            self.c_prev = np.zeros_like(weights)

        # Leaky accumulation: C_i(t) = beta * C_i(t-1) + (1 - beta) * alpha_i(t)
        c_t = self.beta * self.c_prev + (1.0 - self.beta) * weights
        c_t = np.clip(c_t, 0.0, 1.0)
        self.c_prev = c_t

        states = []
        for val in c_t:
            if val >= self.cfg.theta_comp:
                states.append(FixationState.COMPREHENDED)
            elif val >= self.cfg.theta_sacc:
                states.append(FixationState.SACCADIC_GLANCE)
            else:
                states.append(FixationState.UNSEEN)

        return CognitiveState(
            attention_weights=weights,
            accumulated_cognition=c_t,
            fixation_states=states,
        )
