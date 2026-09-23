import torch
import pytest
import numpy as np
from sg_ttrans.models.tdda_attention import TDDAAttention
from sg_ttrans.models.transformer import TemporalTransformerEncoder

def test_tdda_attention_shapes_and_probabilities():
    B, T, d_model = 2, 60, 256
    x = torch.randn(B, T, d_model)
    tdda = TDDAAttention(d_model=d_model, num_heads=4, gamma_init=0.15)
    out, attn = tdda(x, return_attention=True)
    assert out.shape == (B, T, d_model)
    assert attn.shape == (B, 4, T, T)
    # Check probabilities sum to 1.0 along last dimension
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, 4, T), atol=1e-5)

def test_tdda_proposition_1_mathematical_decay():
    # Verify cumulative attention ratio for 6-frame blink vs 30-frame microsleep
    # From Equations (11) and (12) in paper:
    # W(S_blink) <= (1 - e^(-0.15*6)) / (1 - e^(-0.15)) = (1 - 0.4066) / 0.1393 ~= 4.26
    # W(S_micro) >= (1 - e^(-0.15*30)) / (1 - e^(-0.15)) = (1 - 0.0111) / 0.1393 ~= 7.10
    gamma = 0.15
    m_blink = 6
    m_micro = 30
    w_blink_bound = (1.0 - np.exp(-gamma * m_blink)) / (1.0 - np.exp(-gamma))
    w_micro_bound = (1.0 - np.exp(-gamma * m_micro)) / (1.0 - np.exp(-gamma))

    assert pytest.approx(w_blink_bound, abs=0.05) == 4.26
    assert pytest.approx(w_micro_bound, abs=0.05) == 7.10
    assert w_blink_bound < w_micro_bound

def test_tdda_learnable_gamma_gradient():
    B, T, d_model = 2, 10, 256
    x = torch.randn(B, T, d_model, requires_grad=True)
    tdda = TDDAAttention(d_model=d_model, num_heads=4, gamma_init=0.15)
    out = tdda(x)
    loss = out.sum()
    loss.backward()
    assert tdda.raw_gamma.grad is not None
    assert x.grad is not None

def test_transformer_encoder():
    B, T, d_model = 2, 60, 256
    encoder = TemporalTransformerEncoder(d_model=d_model, num_heads=4, num_layers=4, gamma_init=0.15)
    x = torch.randn(B, T, d_model)
    out = encoder(x)
    assert out.shape == (B, T, d_model)
