"""
Multi-Class Focal Loss for handling severe class imbalance in driver fatigue states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: list[float] | torch.Tensor | None = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, list):
                self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            else:
                self.register_buffer("alpha", alpha.clone().detach().to(dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] unnormalized class predictions.
            targets: [B] integer class ground truths in [0, C-1].
        Returns:
            Scalar loss.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather probability of true classes: [B]
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Modulating factor (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - target_probs, self.gamma)

        # Apply class weights if specified
        if self.alpha is not None:
            alpha_device = self.alpha.to(logits.device)
            alpha_t = alpha_device[targets]
            focal_weight = focal_weight * alpha_t

        loss = -focal_weight * target_log_probs

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
