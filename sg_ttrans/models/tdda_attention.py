"""
Temporal Dynamic Decay Attention (TDDA) module for SG-TTrans.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TDDAAttention(nn.Module):
    """
    Temporal Dynamic Decay Attention (TDDA) mechanism (Equation 8):
        A_{i,j} = exp( (Q_i K_j^T)/sqrt(d_k) - gamma * |i - j| ) / sum_k(...)
    Where gamma > 0 is a learnable temporal decay regularization parameter.
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        gamma_init: float = 0.15,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize raw_gamma such that F.softplus(raw_gamma) == gamma_init
        # softplus(x) = ln(1 + e^x) => x = ln(e^gamma_init - 1)
        inv_softplus = math.log(math.exp(gamma_init) - 1.0)
        self.raw_gamma = nn.Parameter(torch.tensor(inv_softplus, dtype=torch.float32))

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.raw_gamma)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence [B, T, d_model].
            return_attention: If True, returns (output, attention_weights).
        Returns:
            Output tensor [B, T, d_model] (and optional attention weights [B, num_heads, T, T]).
        """
        B, T, _ = x.shape

        # Linear projections & split into heads: [B, num_heads, T, d_k]
        Q = self.w_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product: [B, num_heads, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Temporal distance matrix D_{i,j} = |i - j|: [T, T]
        indices = torch.arange(T, device=x.device, dtype=torch.float32)
        dist_matrix = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))  # [T, T]

        # Apply exponential distance decay regularization: - gamma * |i - j|
        decay_penalty = self.gamma * dist_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        logits = scores - decay_penalty

        attn_weights = F.softmax(logits, dim=-1)
        attn_dropped = self.dropout(attn_weights)

        # Context output: [B, num_heads, T, d_k] -> [B, T, d_model]
        context = torch.matmul(attn_dropped, V)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.w_o(context)

        if return_attention:
            return out, attn_weights
        return out
