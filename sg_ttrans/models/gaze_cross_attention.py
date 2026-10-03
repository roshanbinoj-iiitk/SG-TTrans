"""
Bipartite Multi-Head Gaze-Scene Cross-Attention Kernel.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

class BipartiteGazeSceneAttention(nn.Module):
    """Bipartite multi-head cross-attention between driver 3D gaze query and exterior scene node keys."""
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.gaze_proj = nn.Linear(5, embed_dim)
        self.node_proj = nn.Linear(8, embed_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        gaze_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gaze_tokens: (B, 1, 5)
            node_tokens: (B, N, 8)
            mask: (B, N) boolean mask where True indicates valid node, False is padding.
        Returns:
            attn_weights: (B, N)
            updated_nodes: (B, N, embed_dim)
        """
        B, N, _ = node_tokens.shape
        q = self.gaze_proj(gaze_tokens)      # (B, 1, embed_dim)
        k = self.node_proj(node_tokens)      # (B, N, embed_dim)
        v = k
        
        key_padding_mask = ~mask if mask is not None else None
        out, raw_weights = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        # raw_weights has shape (B, 1, N)
        attn_weights = raw_weights.squeeze(1)  # (B, N)
        
        # Guarantee numerical simplex constraint (weights sum to 1.0)
        if mask is not None:
            attn_weights = attn_weights * mask.float()
        sums = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_weights = attn_weights / sums
            
        updated_nodes = self.norm(k)
        return attn_weights, updated_nodes

