"""
Multi-Task Loss Formulation for PC-CSG Real-World Training.

Combines:
1. Continuous CRT Risk Loss (Smooth L1 + BCE)
2. Five-Level Preemptive Intervention Classification Loss (CrossEntropy)
3. Physics-Informed Kinematic Regularization Loss (PIKV Coulomb penalty)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pc_csg.config import PCCSGConfig


class PCCSGLoss(nn.Module):
    """
    Unified multi-task loss for training PC-CSG on real-world driving data.
    """

    def __init__(
        self,
        config: Optional[PCCSGConfig] = None,
        w_crt: float = 1.0,
        w_level: float = 0.5,
        w_physics: float = 0.2,
    ):
        super().__init__()
        self.config = config or PCCSGConfig()
        self.w_crt = w_crt
        self.w_level = w_level
        self.w_physics = w_physics

        self.smooth_l1 = nn.SmoothL1Loss(beta=0.1)
        self.ce_loss = nn.CrossEntropyLoss()

        # Intervention classification head maps scene embedding to 5 levels
        self.level_classifier = nn.Sequential(
            nn.Linear(self.config.d_model, 64),
            nn.GELU(),
            nn.Linear(64, 5),
        )

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        crt_targets: torch.Tensor,
        level_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.

        Args:
            model_outputs: dict from PCCSGNetwork.forward
            crt_targets: (B,) ground truth continuous CRT scalar in [0, 1]
            level_targets: (B,) ground truth intervention level in {0, 1, 2, 3, 4}

        Returns:
            dict with 'loss', 'loss_crt', 'loss_level', 'loss_physics'
        """
        device = crt_targets.device
        pred_crt = model_outputs["crt_scalar"]  # (B,)

        # 1. Continuous CRT regression loss
        loss_crt = self.smooth_l1(pred_crt, crt_targets)

        # 2. Intervention Level classification loss
        scene_embeds = model_outputs["scene_embeddings"]  # (B, N, D)
        pooled_embed = scene_embeds.mean(dim=1)           # (B, D)
        pred_logits = self.level_classifier(pooled_embed) # (B, 5)
        loss_level = self.ce_loss(pred_logits, level_targets)

        # 3. Physics violation penalty
        # Penalizes model when predicted trajectories violate kinematic bicycle limits
        violation_scores = model_outputs["violation_scores"]  # (B, N, K)
        loss_physics = violation_scores.mean()

        # Total weighted loss
        total_loss = (
            self.w_crt * loss_crt
            + self.w_level * loss_level
            + self.w_physics * loss_physics
        )

        return {
            "loss": total_loss,
            "loss_crt": loss_crt,
            "loss_level": loss_level,
            "loss_physics": loss_physics,
            "pred_level_logits": pred_logits,
        }
