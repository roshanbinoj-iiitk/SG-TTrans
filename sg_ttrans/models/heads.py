"""
Multi-class Driver State Classification Head for SG-TTrans.
"""

import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    Maps pooled temporal representations z_bar to 5 driver state logits:
        y_hat = Softmax(W_c * z_bar + b_c)
    Classes:
        0: Alert
        1: Drowsy
        2: Microsleep
        3: Yawn
        4: Distracted
    """
    def __init__(self, d_model: int = 256, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, z_bar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_bar: [B, d_model] pooled temporal vector.
        Returns:
            torch.Tensor: [B, num_classes] raw class logits.
        """
        return self.fc(z_bar)
