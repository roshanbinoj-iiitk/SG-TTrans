"""
Unified Spatio-Temporal Hypergraph Gaze-Scene Transformer (ST-HGST) Network.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from sg_ttrans.models.gaze_cross_attention import BipartiteGazeSceneAttention

class STHGSTNetwork(nn.Module):
    """End-to-end ST-HGST network coupling driver 3D gaze and dynamic road scene graph."""
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = BipartiteGazeSceneAttention(embed_dim, num_heads, dropout)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        gaze: torch.Tensor,
        nodes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        attn_weights, node_embeds = self.attention(gaze, nodes, mask)
        node_embeds = node_embeds + self.temporal_mlp(node_embeds)
        return {
            "attention_weights": attn_weights,
            "node_embeddings": node_embeds,
        }
