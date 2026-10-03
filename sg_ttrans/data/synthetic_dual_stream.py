"""
Synthetic Dual-Stream Data Generator for ST-HGST CI/CD and Benchmarking.
"""

from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticDualStreamDataset(Dataset):
    """Generates synthetic synchronized dual-stream data for CI/CD and unit testing."""
    def __init__(self, num_samples: int = 100, num_objects: int = 5, scenario: str = "random"):
        self.num_samples = num_samples
        self.num_objects = num_objects
        self.scenario = scenario

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.RandomState(idx)
        n = self.num_objects
        nodes = np.zeros((n, 8), dtype=np.float32)
        
        # Node 0 is the critical hazard
        hazard_x, hazard_y = 0.5, 0.4
        nodes[0] = [0.0, hazard_x, hazard_y, 0.15, 0.25, 0.0, -5.0, 1.8]
        for i in range(1, n):
            nodes[i] = [1.0, float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.8)), 0.1, 0.1, 0.0, 0.0, 8.0]

        if self.scenario == "aligned" or (self.scenario == "random" and idx % 3 == 0):
            # Gaze aligned on hazard
            gaze = np.array([hazard_x - 0.5, hazard_y - 0.5, 1.0, hazard_x, hazard_y], dtype=np.float32)
            target = 0
        elif self.scenario == "blindness" or (self.scenario == "random" and idx % 3 == 1):
            # Gaze away on right side
            gaze = np.array([0.4, 0.0, 1.0, 0.9, 0.5], dtype=np.float32)
            target = 1
        else:
            # Distracted/sleep gaze looking down inside cabin
            gaze = np.array([0.0, -0.8, 0.2, 0.5, 0.95], dtype=np.float32)
            nodes[0, 7] = 1.0  # Imminent TTC
            target = 2

        norm = float(np.linalg.norm(gaze[:3]))
        gaze[:3] /= max(norm, 1e-4)

        return {
            "gaze": torch.from_numpy(gaze).unsqueeze(0),
            "nodes": torch.from_numpy(nodes),
            "mask": torch.ones(n, dtype=torch.bool),
            "target_action": target,
        }
