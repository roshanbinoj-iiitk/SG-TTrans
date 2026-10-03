# tests/test_end_to_end_st_hgst.py
import pytest
import torch
import numpy as np
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.models.st_hgst_net import STHGSTNetwork
from sg_ttrans.cognitive.leaky_accumulator import LeakyCognitiveAccumulator
from sg_ttrans.cognitive.epistemic_gap import compute_eag
from sg_ttrans.cognitive.takeover_arbitrator import TakeoverArbitrator
from sg_ttrans.types import TakeoverAction

def test_full_pipeline_flow():
    cfg = STHGSTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STHGSTNetwork(embed_dim=128, num_heads=4).to(device)
    accumulator = LeakyCognitiveAccumulator(cfg)
    arbitrator = TakeoverArbitrator(cfg)

    # 1. Inputs: Gaze at optical center (0.5, 0.5)
    gaze = torch.tensor([[[0.0, 0.0, 1.0, 0.5, 0.5]]], device=device, dtype=torch.float32)
    # 2. Exterior Scene: Node 0 (hazard ahead at 0.5, 0.5), Node 1 (far car)
    nodes = torch.tensor([[[0.0, 0.5, 0.5, 0.1, 0.2, 0.0, -4.0, 2.0],
                           [1.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 10.0]]], device=device, dtype=torch.float32)
    
    # 3. Model inference
    model.eval()
    with torch.no_grad():
        out = model(gaze, nodes)
    attn_weights = out["attention_weights"].squeeze(0).cpu().numpy()
    assert attn_weights.shape == (2,)
    assert np.isclose(attn_weights.sum(), 1.0, atol=1e-5)
    
    # 4. Cognitive accumulation over 10 consecutive frames
    cog_state = None
    for _ in range(10):
        cog_state = accumulator.step(attn_weights)
    
    # 5. EAG & Takeover Decision
    hazard_weights = np.array([0.8, 0.05], dtype=np.float32)
    eag, top_hazard = compute_eag(hazard_weights, cog_state.accumulated_cognition)
    decision = arbitrator.arbitrate(eag, min_ttc=2.0, top_hazard_idx=top_hazard)
    
    assert decision.action in (
        TakeoverAction.SAFE_HANDOVER,
        TakeoverAction.TARGETED_SPATIAL_CUE,
        TakeoverAction.MINIMUM_RISK_MANEUVER,
    )
    assert 0.0 <= decision.eag <= 1.0

def test_empty_scene_handling():
    cfg = STHGSTConfig()
    arbitrator = TakeoverArbitrator(cfg)
    # Empty road (N=0)
    eag, top_hazard = compute_eag(hazard_weights=np.array([]), accumulated_cognition=np.array([]))
    assert eag == 0.0
    assert top_hazard is None
    decision = arbitrator.arbitrate(eag=eag, min_ttc=float("inf"), top_hazard_idx=top_hazard)
    assert decision.action == TakeoverAction.SAFE_HANDOVER
