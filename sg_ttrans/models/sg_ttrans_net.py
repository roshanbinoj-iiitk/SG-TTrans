"""
End-to-End SG-TTrans (Spatial-Geometric Temporal Transformer) neural architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sg_ttrans.config import SGTransConfig
from sg_ttrans.models.backbone import MobileNetV4SpatialBackbone
from sg_ttrans.models.fusion import CrossModalFusion
from sg_ttrans.models.transformer import TemporalTransformerEncoder
from sg_ttrans.models.heads import ClassificationHead

class SGTransNet(nn.Module):
    """
    Spatial-Geometric Temporal Transformer (SG-TTrans).
    Fuses spatial appearance from MobileNetV4 with 3D facial landmark kinematics,
    models temporal dependencies with TDDA Attention, and outputs driver state
    probabilities and continuous fatigue persistence metrics.
    """
    def __init__(self, config: SGTransConfig | None = None):
        super().__init__()
        self.config = config if config is not None else SGTransConfig()

        # 1. Spatial Appearance Stream (MobileNetV4)
        self.backbone = MobileNetV4SpatialBackbone(out_dim=self.config.d_vis)

        # 2. Cross-Modal Fusion & Positional Tokenization
        self.fusion = CrossModalFusion(
            d_vis=self.config.d_vis,
            d_geom=self.config.d_geom,
            d_model=self.config.d_model,
            max_seq_len=self.config.sequence_length
        )

        # 3. Temporal Transformer Encoder with TDDA Attention
        self.transformer = TemporalTransformerEncoder(
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            gamma_init=self.config.gamma_init,
            d_ff=self.config.d_ff
        )

        # 4. Multi-Task Driver State Classifier Head
        self.classifier = ClassificationHead(
            d_model=self.config.d_model,
            num_classes=self.config.num_classes
        )

    def forward(
        self,
        frames: torch.Tensor,
        kinematics: torch.Tensor,
        return_attention: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Forward execution:
            frames: RGB frames [B, T, 3, 224, 224]
            kinematics: 16-D kinematic tokens [B, T, 16]
        Returns:
            dict containing:
                - "logits": [B, 5]
                - "probs": [B, 5]
                - "fatigue_prob": [B] continuous fatigue score P(Drowsy) + P(Microsleep)
                - "z_bar": [B, d_model] pooled temporal representation
        """
        # 1. Spatial feature extraction: [B, T, d_vis]
        z_vis = self.backbone(frames)

        # 2. Cross-modal projection & 1D temporal positional embeddings: [B, T, d_model]
        z_seq = self.fusion(z_vis, kinematics)

        # 3. Temporal Transformer Encoder with TDDA: [B, T, d_model]
        z_trans = self.transformer(z_seq)

        # 4. Temporal mean pooling: \bar{z} = MeanPool(Z_L): [B, d_model]
        z_bar = z_trans.mean(dim=1)

        # 5. Driver state logits and probabilities
        logits = self.classifier(z_bar)
        probs = F.softmax(logits, dim=-1)

        # Fatigue probability: Class 1 (Drowsy) + Class 2 (Microsleep)
        fatigue_prob = probs[:, 1] + probs[:, 2]

        return {
            "logits": logits,
            "probs": probs,
            "fatigue_prob": fatigue_prob,
            "z_bar": z_bar
        }
