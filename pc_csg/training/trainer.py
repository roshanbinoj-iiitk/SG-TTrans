"""
Trainer for PC-CSG on Real-World Autonomous Traffic Datasets.

Executes end-to-end multi-task training with mixed precision,
validation monitoring, and checkpoint persistence.
"""

import json
import os
import time
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from pc_csg.config import PCCSGConfig
from pc_csg.models.pc_csg_net import PCCSGNetwork
from pc_csg.training.losses import PCCSGLoss


def collate_graph_batch(batch):
    """
    Collate function for PC-CSG scene graph samples.
    """
    node_features = torch.stack([b["node_features"] for b in batch])
    hazard_weights = torch.stack([b["hazard_weights"] for b in batch])
    node_mask = torch.stack([b["node_mask"] for b in batch])
    crt_label = torch.stack([b["crt_label"] for b in batch])
    level_label = torch.stack([b["level_label"] for b in batch])

    # For graph edges, use first sample or average for homogeneous graph size
    edge_index = batch[0]["edge_index"]
    edge_attr = torch.stack([b["edge_attr"] for b in batch]) if batch[0]["edge_attr"].shape[0] > 0 else torch.zeros(
        (len(batch), 0, 6), dtype=torch.float32
    )

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "hazard_weights": hazard_weights,
        "node_mask": node_mask,
        "crt_label": crt_label,
        "level_label": level_label,
    }


class RealWorldPCCSGTrainer:
    """
    High-performance trainer for PC-CSG on real driving datasets.
    """

    def __init__(
        self,
        model: PCCSGNetwork,
        config: Optional[PCCSGConfig] = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        save_dir: str = "checkpoints",
    ):
        self.config = config or PCCSGConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.loss_fn = PCCSGLoss(self.config).to(self.device)
        # Train both model and loss head (level classifier)
        params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_crt_mae": [],
            "val_acc": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_crt_loss = 0.0
        total_phys_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            node_feat = batch["node_features"].to(self.device)
            edge_idx = batch["edge_index"].to(self.device)
            edge_attr = batch["edge_attr"].to(self.device)
            hazard = batch["hazard_weights"].to(self.device)
            mask = batch["node_mask"].to(self.device)
            crt_targets = batch["crt_label"].to(self.device)
            level_targets = batch["level_label"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                outputs = self.model(
                    node_features=node_feat,
                    edge_index=edge_idx,
                    edge_attr=edge_attr,
                    hazard_weights=hazard,
                    node_mask=mask,
                )
                loss_dict = self.loss_fn(outputs, crt_targets, level_targets)
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_crt_loss += loss_dict["loss_crt"].item()
            total_phys_loss += loss_dict["loss_physics"].item()
            num_batches += 1

        return {
            "loss": total_loss / max(1, num_batches),
            "loss_crt": total_crt_loss / max(1, num_batches),
            "loss_phys": total_phys_loss / max(1, num_batches),
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        correct_levels = 0
        total_samples = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                node_feat = batch["node_features"].to(self.device)
                edge_idx = batch["edge_index"].to(self.device)
                edge_attr = batch["edge_attr"].to(self.device)
                hazard = batch["hazard_weights"].to(self.device)
                mask = batch["node_mask"].to(self.device)
                crt_targets = batch["crt_label"].to(self.device)
                level_targets = batch["level_label"].to(self.device)

                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    outputs = self.model(
                        node_features=node_feat,
                        edge_index=edge_idx,
                        edge_attr=edge_attr,
                        hazard_weights=hazard,
                        node_mask=mask,
                    )
                    loss_dict = self.loss_fn(outputs, crt_targets, level_targets)

                total_loss += loss_dict["loss"].item()
                pred_crt = outputs["crt_scalar"]
                total_mae += (pred_crt - crt_targets).abs().sum().item()

                preds = loss_dict["pred_level_logits"].argmax(dim=-1)
                correct_levels += (preds == level_targets).sum().item()
                total_samples += len(level_targets)
                num_batches += 1

        return {
            "val_loss": total_loss / max(1, num_batches),
            "val_crt_mae": total_mae / max(1, total_samples),
            "val_accuracy": (correct_levels / max(1, total_samples)) * 100.0,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        checkpoint_name: str = "pc_csg_real_best.pt",
    ) -> Dict[str, list]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        best_val_loss = float("inf")
        best_checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        print(f"Starting PC-CSG training on {self.device} for {epochs} epochs...")
        print("=" * 68)
        print(f" {'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'CRT MAE':^9} | {'Val Acc %':^9} | {'Time':^7}")
        print("=" * 68)

        start_train_time = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            scheduler.step()

            ep_time = time.time() - t0
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_crt_mae"].append(val_metrics["val_crt_mae"])
            self.history["val_acc"].append(val_metrics["val_accuracy"])

            print(
                f" {epoch:^7d} | {train_metrics['loss']:^12.4f} | {val_metrics['val_loss']:^10.4f} | "
                f"{val_metrics['val_crt_mae']:^9.4f} | {val_metrics['val_accuracy']:^9.1f}% | {ep_time:^6.1f}s"
            )

            # Checkpoint saving on improvement
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "loss_head_state_dict": self.loss_fn.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.config,
                        "val_loss": val_metrics["val_loss"],
                        "val_accuracy": val_metrics["val_accuracy"],
                        "val_crt_mae": val_metrics["val_crt_mae"],
                    },
                    best_checkpoint_path,
                )

        total_time = time.time() - start_train_time
        print("=" * 68)
        print(f"Training completed in {total_time:.1f}s. Best model saved to: {best_checkpoint_path}")

        # Save training history JSON
        history_path = os.path.join(self.save_dir, "training_history.json")
        history_records = [
            {
                "epoch": ep + 1,
                "train_loss": float(self.history["train_loss"][ep]),
                "val_loss": float(self.history["val_loss"][ep]),
                "val_crt_mae": float(self.history["val_crt_mae"][ep]),
                "val_acc": float(self.history["val_acc"][ep]),
            }
            for ep in range(len(self.history["train_loss"]))
        ]
        with open(history_path, "w") as f:
            json.dump(history_records, f, indent=2)

        self._plot_curves()
        return self.history

    def _plot_curves(self):
        """Save loss and accuracy curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.history["train_loss"], label="Train Loss", color="royalblue", lw=2)
        axes[0].plot(self.history["val_loss"], label="Val Loss", color="crimson", lw=2)
        axes[0].set_title("Multi-Task Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(self.history["val_crt_mae"], label="CRT MAE", color="darkorange", lw=2)
        axes[1].set_title("Counterfactual Risk Error (MAE)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Error")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(self.history["val_acc"], label="Intervention Accuracy", color="forestgreen", lw=2)
        axes[2].set_title("Graduated Intervention Accuracy (%)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy %")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plot_path = os.path.join(self.save_dir, "real_training_curves.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Training curves saved to: {plot_path}")
