# tests/test_cross_attention.py
import pytest
import torch
from sg_ttrans.models.gaze_cross_attention import BipartiteGazeSceneAttention
from sg_ttrans.models.st_hgst_net import STHGSTNetwork

def test_cross_attention_normalization():
    model = BipartiteGazeSceneAttention(embed_dim=128, num_heads=4)
    B, N = 2, 5
    # Gaze token: [gx, gy, gz, ug, vg] -> (B, 1, 5)
    gaze_tokens = torch.randn(B, 1, 5)
    # Node features: [class, x, y, w, h, vx, vy, ttc] -> (B, N, 8)
    node_tokens = torch.randn(B, N, 8)
    
    attn_weights, out_nodes = model(gaze_tokens, node_tokens)
    assert attn_weights.shape == (B, N)
    # Must sum to 1.0 along object dimension
    sums = attn_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_cross_attention_with_padding_mask():
    model = BipartiteGazeSceneAttention(embed_dim=128, num_heads=4)
    B, N = 1, 4
    gaze_tokens = torch.randn(B, 1, 5)
    node_tokens = torch.randn(B, N, 8)
    # Mask out the last 2 nodes (valid length = 2)
    mask = torch.tensor([[True, True, False, False]])
    
    attn_weights, _ = model(gaze_tokens, node_tokens, mask=mask)
    # Masked nodes must receive exactly zero attention
    assert torch.allclose(attn_weights[0, 2:], torch.zeros(2), atol=1e-6)
    assert torch.isclose(attn_weights[0, :2].sum(), torch.tensor(1.0), atol=1e-5)

def test_end_to_end_network_forward_pass():
    net = STHGSTNetwork(embed_dim=128, num_heads=4)
    gaze = torch.randn(2, 1, 5)
    nodes = torch.randn(2, 6, 8)
    out = net(gaze, nodes)
    assert "attention_weights" in out
    assert "node_embeddings" in out
    assert out["attention_weights"].shape == (2, 6)
