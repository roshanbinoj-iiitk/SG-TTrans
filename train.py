#!/usr/bin/env python3
"""
SG-TTrans: Training Pipeline for Driver Drowsiness Detection.
Trains the MobileNetV4 spatial stream, Cross-Modal Fusion, and TDDA Transformer
on the actual dataset present in data/ (Driver Drowsiness Dataset DDD).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from sg_ttrans.config import SGTransConfig
from sg_ttrans.models.sg_ttrans_net import SGTransNet
from sg_ttrans.data.dataset import DDDImageSequenceDataset
from sg_ttrans.training.losses import MultiClassFocalLoss
from sg_ttrans.training.trainer import SGTransTrainer

def main():
    parser = argparse.ArgumentParser(description="Train SG-TTrans on actual DDD dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset directory containing Drowsy and Non Drowsy folders")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (optimized for 4GB RTX 3050 GPU)")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence window length T (frames)")
    parser.add_argument("--stride", type=int, default=16, help="Frame stride for sequence slicing")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Optional subset limit for fast training (0 for all)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("="*65)
    print(f" SG-TTrans Training Engine")
    print(f" Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Dataset Directory: {args.data_dir}")
    print(f" Sequence Window: T={args.seq_len} frames | Stride: {args.stride}")
    print("="*65)

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Dataset & DataLoaders with Subject-Independent Split
    print("\n[1/4] Indexing subject sequences from dataset...")
    train_dataset = DDDImageSequenceDataset(
        data_dir=args.data_dir,
        sequence_length=args.seq_len,
        stride=args.stride,
        split="train",
        train_ratio=0.8
    )
    val_dataset = DDDImageSequenceDataset(
        data_dir=args.data_dir,
        sequence_length=args.seq_len,
        stride=args.stride,
        split="val",
        train_ratio=0.8
    )

    if len(train_dataset) == 0:
        print(f"Error: No sequences found in '{args.data_dir}'. Make sure 'Drowsy' and 'Non Drowsy' exist.")
        sys.exit(1)

    print(f" >> Training Sequences:   {len(train_dataset)} (disjoint subjects)")
    print(f" >> Validation Sequences: {len(val_dataset)} (disjoint subjects)")

    if args.max_train_samples > 0:
        train_dataset.sequences = train_dataset.sequences[:args.max_train_samples]
        val_dataset.sequences = val_dataset.sequences[:max(20, args.max_train_samples // 4)]
        print(f" >> Limited to {len(train_dataset)} train and {len(val_dataset)} val samples for fast run.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 2. Model Initialization (2 classes: Alert=0, Drowsy=1)
    print("\n[2/4] Instantiating SG-TTrans neural architecture...")
    config = SGTransConfig(
        sequence_length=args.seq_len,
        num_classes=2,
        d_vis=256,
        d_geom=16,
        d_model=256,
        num_heads=4,
        num_layers=2, # 2 layers fits easily in 4GB GPU with high throughput
        gamma_init=0.15
    )
    model = SGTransNet(config).to(device)

    # 3. Loss & Optimizer
    criterion = MultiClassFocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    trainer = SGTransTrainer(model=model, optimizer=optimizer, criterion=criterion, device=device, use_amp=True)

    # 4. Training Loop
    print("\n[3/4] Starting training loop with Focal Loss & AMP...")
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_losses = []
        train_accs = []

        model.train()
        for b_idx, batch in enumerate(train_loader):
            frames = batch["frames"]
            kinematics = batch["kinematics"]
            labels = batch["label"]

            metrics = trainer.train_step(frames, kinematics, labels)
            train_losses.append(metrics["loss"])
            train_accs.append(metrics["accuracy"])

            if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  Epoch [{epoch}/{args.epochs}] Step [{b_idx+1:3d}/{len(train_loader)}] "
                      f"Loss: {metrics['loss']:.4f} | Batch Acc: {metrics['accuracy']*100:.1f}%", flush=True)

        scheduler.step()

        # Validation phase
        val_losses = []
        all_preds = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"]
                kinematics = batch["kinematics"]
                labels = batch["label"]
                val_res = trainer.evaluate_step(frames, kinematics, labels)
                val_losses.append(val_res["loss"])
                all_preds.extend(val_res["preds"].tolist())
                all_labels.extend(val_res["labels"].tolist())

        epoch_time = time.time() - t0
        avg_train_loss = float(np.mean(train_losses))
        avg_train_acc = float(np.mean(train_accs))
        avg_val_loss = float(np.mean(val_losses))
        avg_val_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))

        # Compute Macro F1 score
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        f1_scores = []
        for c in range(2):
            tp = np.sum((preds_arr == c) & (labels_arr == c))
            fp = np.sum((preds_arr == c) & (labels_arr != c))
            fn = np.sum((preds_arr != c) & (labels_arr == c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_scores.append(f1)
        macro_f1 = float(np.mean(f1_scores))

        print(f"\n>> Epoch {epoch:02d}/{args.epochs:02d} ({epoch_time:.1f}s) | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc*100:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc*100:.2f}%, Val F1: {macro_f1*100:.2f}%")

        epoch_stat = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc,
            "val_f1": macro_f1
        }
        history.append(epoch_stat)

        # Save best model checkpoint
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_path = os.path.join(args.save_dir, "sg_ttrans_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_acc": best_val_acc,
                "val_f1": macro_f1
            }, save_path)
            print(f"   [+] Saved new best model checkpoint to '{save_path}' (Val Acc: {best_val_acc*100:.2f}%)")

    # Save training history
    history_file = os.path.join(args.save_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*65)
    print(f" Training Complete!")
    print(f" Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f" Checkpoint: {os.path.join(args.save_dir, 'sg_ttrans_best.pt')}")
    print(f" History:    {history_file}")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
