"""
Temporal Transformer Encoder architecture with TDDA blocks for SG-TTrans.
"""

import torch
import torch.nn as nn
from sg_ttrans.models.tdda_attention import TDDAAttention

class TransformerBlock(nn.Module):
    """
    Single Pre-LayerNorm Temporal Transformer block with TDDA attention kernel:
        Z'_l = LayerNorm(Z_{l-1} + TDDA(Z_{l-1}))
        Z_l  = LayerNorm(Z'_l + FFN(Z'_l))
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        gamma_init: float = 0.15,
        d_ff: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TDDAAttention(
            d_model=d_model,
            num_heads=num_heads,
            gamma_init=gamma_init,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual connection
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TemporalTransformerEncoder(nn.Module):
    """
    Stack of L Transformer encoder blocks modulated by TDDA attention.
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        gamma_init: float = 0.15,
        d_ff: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                gamma_init=gamma_init,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [B, T, d_model].
        Returns:
            torch.Tensor: Output sequence [B, T, d_model].
        """
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
