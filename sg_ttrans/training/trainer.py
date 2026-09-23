"""
Training engine and evaluation loop for SG-TTrans.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

class SGTransTrainer:
    """
    Manages optimization, loss calculation, and performance metrics for SG-TTrans.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device | str = "cpu",
        use_amp: bool = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.model.to(self.device)

    def train_step(
        self,
        frames: torch.Tensor,
        kinematics: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Executes a single forward-backward optimization step.
        """
        self.model.train()
        self.optimizer.zero_grad()

        frames = frames.to(self.device)
        kinematics = kinematics.to(self.device)
        labels = labels.to(self.device)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            outputs = self.model(frames, kinematics)
            logits = outputs["logits"]
            loss = self.criterion(logits, labels)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == labels).float().mean().item()

        return {
            "loss": float(loss.item()),
            "accuracy": float(accuracy)
        }

    def evaluate_step(
        self,
        frames: torch.Tensor,
        kinematics: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Runs validation step without gradients.
        """
        self.model.eval()
        frames = frames.to(self.device)
        kinematics = kinematics.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            outputs = self.model(frames, kinematics)
            logits = outputs["logits"]
            loss = self.criterion(logits, labels)
            preds = torch.argmax(logits, dim=-1)

        return {
            "loss": float(loss.item()),
            "preds": preds.cpu(),
            "labels": labels.cpu(),
            "probs": outputs["probs"].cpu()
        }
