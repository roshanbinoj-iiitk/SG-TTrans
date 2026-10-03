# tests/test_eag_arbitration.py
import pytest
import numpy as np
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.types import TakeoverAction
from sg_ttrans.cognitive.epistemic_gap import compute_eag
from sg_ttrans.cognitive.takeover_arbitrator import TakeoverArbitrator

def test_scenario_a_aligned_attention():
    # Hazard is high (H=0.9), driver is fully comprehending it (C=0.85)
    eag, top_hazard = compute_eag(
        hazard_weights=np.array([0.9, 0.1]),
        accumulated_cognition=np.array([0.85, 0.05]),
        theta_comp=0.60,
    )
    assert eag < 0.20
    arbitrator = TakeoverArbitrator()
    decision = arbitrator.arbitrate(eag=eag, min_ttc=2.2, top_hazard_idx=top_hazard)
    assert decision.action == TakeoverAction.SAFE_HANDOVER

def test_scenario_b_inattentional_blindness():
    # Hazard is high (H=0.9), driver has partial/peripheral glance (C=0.35 on hazard, 0.65 on safe road)
    eag, top_hazard = compute_eag(
        hazard_weights=np.array([0.9, 0.0]),
        accumulated_cognition=np.array([0.35, 0.65]),
        theta_comp=0.60,
    )
    assert 0.20 <= eag < 0.65
    assert top_hazard == 0
    arbitrator = TakeoverArbitrator()
    decision = arbitrator.arbitrate(eag=eag, min_ttc=2.0, top_hazard_idx=top_hazard)
    assert decision.action == TakeoverAction.TARGETED_SPATIAL_CUE
    assert decision.causal_hazard_id == 0


def test_scenario_c_imminent_collision_override():
    # Even if EAG is moderate, imminent collision (TTC = 1.0s < 1.2s) must force Level 2 MRM
    arbitrator = TakeoverArbitrator()
    decision = arbitrator.arbitrate(eag=0.15, min_ttc=1.0, top_hazard_idx=0)
    assert decision.action == TakeoverAction.MINIMUM_RISK_MANEUVER
