---
name: pytorch
description: >-
  Use this skill when developing, optimizing, training, or debugging PyTorch deep learning models,
  custom nn.Modules, Datasets/DataLoaders, mixed-precision training loops (AMP), CUDA GPU acceleration,
  and PyTorch 2.x inference pipelines.
---

# PyTorch Engineering & Deep Learning Guidelines

This skill provides guidelines and patterns for building robust, high-performance PyTorch applications.

---

## 1. Device Agnosticism & Reproducibility

### Automatic Device Selection
Support CUDA, Apple Silicon (MPS), and fallback to CPU gracefully:

```python
import os
import random
import numpy as np
import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

---

## 2. `nn.Module` Architecture Best Practices

* **Register Buffers vs Parameters**: Use `self.register_buffer("name", tensor)` for tensors that are part of module state (e.g. running statistics, masks, anchors) but should not receive gradients.
* **Avoid in-place modifications**: Avoid `x += y` or in-place activations (`inplace=False` by default) when autograd computation graphs are involved to prevent silently corrupting backpropagation.
* **Weight Initialization**: Apply explicit weight initialization in `__init__` or using `self.apply(_init_weights)`.

```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
```

---

## 3. High-Throughput `DataLoader` Optimization

Maximize GPU utilization by eliminating CPU I/O bottlenecks in data loading:

```python
from torch.utils.data import DataLoader, Dataset

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    is_train: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=is_train,
    )
```

---

## 4. Modern Training Loop with Mixed Precision (AMP)

Always leverage Automatic Mixed Precision (`torch.amp`) for 2x–3x training speedup and memory reduction on modern GPUs:

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    criterion: nn.Module,
    device: torch.device,
    max_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device=device_type, enabled=(device_type == "cuda"))

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 1. Zero gradients with set_to_none=True (faster & less memory)
        optimizer.zero_grad(set_to_none=True)

        # 2. Forward pass with autocast
        with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # 3. Backward pass scaled
        scaler.scale(loss).backward()

        # 4. Unscale & Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        # 5. Optimizer step & update scaler
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()
    return total_loss / len(dataloader)
```

---

## 5. High-Speed Inference & PyTorch 2.x Compilation

For evaluation and deployment:
1. Use `torch.inference_mode()` (significantly faster with lower overhead than `torch.no_grad()`).
2. Use `torch.compile(model)` on Linux/CUDA for JIT graph optimization.

```python
@torch.inference_mode()
def run_inference(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(inputs)

# Optional: compile for PyTorch 2.x
# compiled_model = torch.compile(model, mode="reduce-overhead")
```

---

## 6. Memory Management & Troubleshooting

* **CUDA Out of Memory (OOM)**:
  - Call `optimizer.zero_grad(set_to_none=True)`.
  - Delete unused tensors and call `torch.cuda.empty_cache()` if needed.
  - Store loss metrics as float primitives: `total_loss += loss.item()` (never `total_loss += loss`, which retains the autograd computation graph).
* **Detecting NaN / Inf Gradients**:
  ```python
  torch.autograd.set_detect_anomaly(True)  # Use only during debugging (slow)
  ```
* **Shape Debugging**:
  Always inspect `.shape`, `.dtype`, and `.device` when encountering dimension errors.
