import torch
import pytest
from sg_ttrans.config import SGTransConfig
from sg_ttrans.models.sg_ttrans_net import SGTransNet

def test_sg_ttrans_net_forward():
    cfg = SGTransConfig(sequence_length=10, d_model=256, num_layers=2)
    model = SGTransNet(cfg)
    B, T = 2, 10
    frames = torch.randn(B, T, 3, 224, 224)
    kinematics = torch.randn(B, T, 16)

    outputs = model(frames, kinematics)
    assert "logits" in outputs
    assert "probs" in outputs
    assert "fatigue_prob" in outputs
    assert "z_bar" in outputs

    assert outputs["logits"].shape == (B, 5)
    assert outputs["probs"].shape == (B, 5)
    assert outputs["fatigue_prob"].shape == (B,)
    assert outputs["z_bar"].shape == (B, 256)

    # Check fatigue_prob = probs[Drowsy] + probs[Microsleep]
    expected_fatigue = outputs["probs"][:, 1] + outputs["probs"][:, 2]
    assert torch.allclose(outputs["fatigue_prob"], expected_fatigue, atol=1e-5)

def test_sg_ttrans_net_backward_gradient_flow():
    cfg = SGTransConfig(sequence_length=4, d_model=256, num_layers=1)
    model = SGTransNet(cfg)
    B, T = 2, 4
    frames = torch.randn(B, T, 3, 224, 224, requires_grad=True)
    kinematics = torch.randn(B, T, 16, requires_grad=True)

    outputs = model(frames, kinematics)
    loss = outputs["logits"].sum()
    loss.backward()

    assert frames.grad is not None
    assert kinematics.grad is not None
