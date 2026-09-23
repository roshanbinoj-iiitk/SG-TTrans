import torch
import pytest
from sg_ttrans.models.backbone import MobileNetV4SpatialBackbone
from sg_ttrans.models.fusion import CrossModalFusion

def test_backbone_and_fusion_shapes():
    B, T = 2, 4
    dummy_frames = torch.randn(B, T, 3, 224, 224)
    dummy_kinematics = torch.randn(B, T, 16)

    backbone = MobileNetV4SpatialBackbone(out_dim=256)
    z_vis = backbone(dummy_frames)
    assert z_vis.shape == (B, T, 256)

    fusion = CrossModalFusion(d_vis=256, d_geom=16, d_model=256, max_seq_len=60)
    z_seq = fusion(z_vis, dummy_kinematics)
    assert z_seq.shape == (B, T, 256)

def test_backbone_single_frame():
    B = 2
    single_frame = torch.randn(B, 3, 224, 224)
    backbone = MobileNetV4SpatialBackbone(out_dim=256)
    out = backbone(single_frame)
    assert out.shape == (B, 256)
