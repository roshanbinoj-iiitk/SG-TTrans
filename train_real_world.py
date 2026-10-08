"""
PC-CSG Real-World Training Script.

Trains the Physics-Constrained Counterfactual Scene Graph Transformer
on real-world traffic accident, anomaly, and vehicle interaction data.

Usage:
    /home/roshanbinoj/Documents/BTP/venv/bin/python train_real_world.py --epochs 10 --batch-size 16
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from pc_csg.config import PCCSGConfig
from pc_csg.data.real_traffic_dataset import RealTrafficAccidentDataset
from pc_csg.models.pc_csg_net import PCCSGNetwork
from pc_csg.training.trainer import RealWorldPCCSGTrainer, collate_graph_batch


def main():
    parser = argparse.ArgumentParser(description="Train PC-CSG on Real-World Driving Datasets")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 0.0003)")
    parser.add_argument("--max-samples", type=int, default=3000, help="Max real dataset samples to use (default: 3000)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Output checkpoint directory")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = PCCSGConfig()

    print("=" * 68)
    print(" PC-CSG Real-World Autonomous Traffic Anomaly Training")
    print("=" * 68)
    print(f" Device:         {device}")
    if device.type == "cuda":
        print(f" GPU Name:       {torch.cuda.get_device_name(0)}")
    print(f" Max Samples:    {args.max_samples}")
    print(f" Batch Size:     {args.batch_size}")
    print(f" Learning Rate:  {args.lr}")
    print(f" Target Epochs:  {args.epochs}")
    print("=" * 68)

    # 1. Load real-world datasets
    print("Loading real-world traffic dataset (Hugging Face traffic accident detection)...")
    train_dataset = RealTrafficAccidentDataset(
        split="train",
        val_ratio=0.2,
        config=config,
        max_samples=int(args.max_samples * 0.8),
    )
    val_dataset = RealTrafficAccidentDataset(
        split="val",
        val_ratio=0.2,
        config=config,
        max_samples=int(args.max_samples * 0.2),
    )

    print(f"  [+] Train set size: {len(train_dataset)} real traffic scenarios")
    print(f"  [+] Val set size:   {len(val_dataset)} real traffic scenarios")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graph_batch,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_batch,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Instantiate Model and Trainer
    model = PCCSGNetwork(config)
    trainer = RealWorldPCCSGTrainer(
        model=model,
        config=config,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir,
    )

    # 3. Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_name="pc_csg_real_best.pt",
    )


if __name__ == "__main__":
    main()
