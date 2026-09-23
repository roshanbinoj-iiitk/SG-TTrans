"""
Cross-Modal Fusion and Positional Tokenization for SG-TTrans.
"""

import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    """
    Fuses spatial appearance tokens with geometric kinematics according to Equation (6):
        z_t = W_v * z_vis^t + W_g * z_geom^t + b_f
    And adds learnable 1D temporal positional embeddings according to Equation (7):
        Z_0 = [z_1, z_2, ..., z_T]^T + E_pos
    """
    def __init__(
        self,
        d_vis: int = 256,
        d_geom: int = 16,
        d_model: int = 256,
        max_seq_len: int = 60
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.w_v = nn.Linear(d_vis, d_model, bias=False)
        self.w_g = nn.Linear(d_geom, d_model, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_model))
        
        # Learnable 1D temporal positional embeddings E_pos
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, z_vis: torch.Tensor, z_geom: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_vis: [B, T, d_vis] appearance tokens from MobileNetV4.
            z_geom: [B, T, d_geom] geometric kinematics tokens.
        Returns:
            torch.Tensor: [B, T, d_model] cross-modal sequence with positional embeddings.
        """
        B, T, _ = z_vis.shape
        # Equation (6): z_t = W_v z_vis^t + W_g z_geom^t + b_f
        z_t = self.w_v(z_vis) + self.w_g(z_geom) + self.bias
        
        # Equation (7): Z_0 = [z_1, ..., z_T]^T + E_pos
        z_seq = z_t + self.pos_embedding[:, :T, :]
        return self.layer_norm(z_seq)
