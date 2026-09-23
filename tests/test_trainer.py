import torch
import pytest
from sg_ttrans.config import SGTransConfig
from sg_ttrans.training.losses import MultiClassFocalLoss
from sg_ttrans.data.synthetic_generator import SyntheticSequenceGenerator
from sg_ttrans.models.sg_ttrans_net import SGTransNet
from sg_ttrans.training.trainer import SGTransTrainer

def test_focal_loss_computation():
    criterion = MultiClassFocalLoss(gamma=2.0)
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    loss = criterion(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None

def test_synthetic_data_generation():
    gen = SyntheticSequenceGenerator(seed=42)
    frames, kinematics, labels = gen.generate_batch(batch_size=4, seq_len=10)
    assert frames.shape == (4, 10, 3, 224, 224)
    assert kinematics.shape == (4, 10, 16)
    assert labels.shape == (4,)
    # Verify label range is [0, 4]
    assert (labels >= 0).all() and (labels < 5).all()

def test_trainer_single_step():
    cfg = SGTransConfig(sequence_length=4, d_model=128, num_layers=1)
    model = SGTransNet(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = MultiClassFocalLoss(gamma=2.0)
    trainer = SGTransTrainer(model=model, optimizer=optimizer, criterion=criterion)

    gen = SyntheticSequenceGenerator(seed=123)
    frames, kinematics, labels = gen.generate_batch(batch_size=2, seq_len=4)
    metrics = trainer.train_step(frames, kinematics, labels)

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"] > 0.0
