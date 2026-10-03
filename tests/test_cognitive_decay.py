# tests/test_cognitive_decay.py
import pytest
import numpy as np
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.types import FixationState
from sg_ttrans.cognitive.leaky_accumulator import LeakyCognitiveAccumulator

def test_transient_saccade_rejection():
    cfg = STHGSTConfig(fps=30.0, tau_cog=0.25, theta_comp=0.60)
    accumulator = LeakyCognitiveAccumulator(cfg)
    
    # 3 frames of high glance (100 ms)
    state = None
    for _ in range(3):
        state = accumulator.step(attention_weights=np.array([0.9, 0.1]))
        
    # Must NOT reach Comprehended state in only 100 ms
    assert state.accumulated_cognition[0] < cfg.theta_comp
    assert state.fixation_states[0] != FixationState.COMPREHENDED

def test_sustained_fixation_comprehension():
    cfg = STHGSTConfig(fps=30.0, tau_cog=0.25, theta_comp=0.60)
    accumulator = LeakyCognitiveAccumulator(cfg)
    
    # 10 frames of sustained gaze (333 ms)
    state = None
    for _ in range(10):
        state = accumulator.step(attention_weights=np.array([0.95, 0.05]))
        
    # Must comfortably reach Comprehended state
    assert state.accumulated_cognition[0] >= cfg.theta_comp
    assert state.fixation_states[0] == FixationState.COMPREHENDED

def test_cognitive_decay_when_attention_shifts():
    cfg = STHGSTConfig(fps=30.0, tau_cog=0.25)
    accumulator = LeakyCognitiveAccumulator(cfg)
    # Establish fixation on node 0
    for _ in range(10):
        accumulator.step(attention_weights=np.array([1.0, 0.0]))
    
    # Attention shifts away to node 1 for 15 frames (500 ms)
    state = None
    for _ in range(15):
        state = accumulator.step(attention_weights=np.array([0.0, 1.0]))
        
    # Node 0 must decay below Comprehension
    assert state.accumulated_cognition[0] < cfg.theta_comp
    assert state.fixation_states[0] != FixationState.COMPREHENDED
