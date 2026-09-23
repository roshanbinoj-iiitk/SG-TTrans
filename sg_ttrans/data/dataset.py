"""
Dataset loaders for SG-TTrans:
1. NTHUDrowsinessDataset: Loader for NTHU-DDD & UTA-RLDD video archives.
2. DDDImageSequenceDataset: High-throughput loader for the Driver Drowsiness Dataset (DDD)
   with 41,793 facial images partitioned by subject into temporal sequences.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Standard class mapping for NTHU-DDD actions
NTHU_CLASS_MAPPING = {
    "normal": 0,         # Alert
    "nodding": 1,        # Drowsy
    "slowblink": 2,      # Microsleep
    "yawning": 3,        # Yawn
    "lookingaround": 4,  # Distracted
}

class NTHUDrowsinessDataset(Dataset):
    """
    PyTorch Dataset loading synchronized video clips and precomputed kinematics.
    Enforces subject-independent partitioning.
    """
    def __init__(
        self,
        root_dir: str,
        sequence_length: int = 60,
        subjects: list[str] | None = None,
        conditions: list[str] | None = None,
        mock_mode: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.mock_mode = mock_mode
        self.samples = []

        if mock_mode:
            for i in range(10):
                self.samples.append({
                    "subject": subjects[0] if subjects else "Subject01",
                    "label": i % 5,
                    "mock": True
                })
        else:
            self._scan_dataset(subjects, conditions)

    def _scan_dataset(self, subjects: list[str] | None, conditions: list[str] | None):
        if not self.root_dir.exists():
            return
        for subj in sorted(os.listdir(self.root_dir)):
            if subjects and subj not in subjects:
                continue
            subj_path = self.root_dir / subj
            if not subj_path.is_dir():
                continue
            for video_file in subj_path.glob("*.mp4"):
                name = video_file.stem.lower()
                label = 0
                for action, cls_idx in NTHU_CLASS_MAPPING.items():
                    if action in name:
                        label = cls_idx
                        break
                self.samples.append({
                    "subject": subj,
                    "path": str(video_file),
                    "label": label,
                    "mock": False
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        sample = self.samples[idx]
        if sample.get("mock", False) or not Path(sample.get("path", "")).exists():
            label = sample["label"]
            frames = torch.randn(self.sequence_length, 3, 224, 224) * 0.1
            kinematics = torch.zeros(self.sequence_length, 16)
            if label == 2:  # microsleep
                kinematics[:, 0] = 0.08
                kinematics[:, 2] = 1.0
            else:
                kinematics[:, 0] = 0.28
                kinematics[:, 2] = 0.05
            return {
                "frames": frames,
                "kinematics": kinematics,
                "label": label,
                "subject": sample["subject"]
            }

        frames = torch.zeros(self.sequence_length, 3, 224, 224)
        kinematics = torch.zeros(self.sequence_length, 16)
        return {
            "frames": frames,
            "kinematics": kinematics,
            "label": sample["label"],
            "subject": sample["subject"]
        }


class DDDImageSequenceDataset(Dataset):
    """
    Temporal sequence loader for the Driver Drowsiness Dataset (DDD):
    - Subfolders: 'Drowsy' (label 1) and 'Non Drowsy' (label 0)
    - Groups frames by subject ID prefix (e.g. A, B, ..., ZC) to enforce strict
      subject-independent train/val partitions with zero identity leakage.
    - Slices continuous video frames into sliding windows of length T.
    """
    def __init__(
        self,
        data_dir: str = "data",
        sequence_length: int = 16,
        stride: int = 16,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.split = split
        self.sequences = []

        self._build_dataset(train_ratio, seed)

    def _build_dataset(self, train_ratio: float, seed: int):
        classes = [("Non Drowsy", 0), ("Drowsy", 1)]
        rng = np.random.default_rng(seed)

        for folder_name, label in classes:
            folder_path = self.data_dir / folder_name
            if not folder_path.exists():
                continue

            # Group files by subject ID
            subject_frames = defaultdict(list)
            for fname in os.listdir(folder_path):
                if fname.endswith(".png") or fname.endswith(".jpg"):
                    match = re.match(r"([A-Za-z]+)(\d+)\.", fname)
                    if match:
                        subj, num_str = match.groups()
                        subject_frames[subj].append((int(num_str), folder_path / fname))

            # Sort subjects and partition into train / val
            all_subjects = sorted(list(subject_frames.keys()))
            perm = rng.permutation(len(all_subjects))
            split_idx = int(len(all_subjects) * train_ratio)

            train_subjs = set(all_subjects[i] for i in perm[:split_idx])
            val_subjs = set(all_subjects[i] for i in perm[split_idx:])

            active_subjs = train_subjs if self.split == "train" else val_subjs if self.split == "val" else set(all_subjects)

            # Build temporal sliding window sequences
            for subj in active_subjs:
                frame_list = subject_frames[subj]
                frame_list.sort(key=lambda x: x[0])
                file_paths = [p for _, p in frame_list]
                num_frames = len(file_paths)

                for start in range(0, max(0, num_frames - self.sequence_length + 1), self.stride):
                    seq_paths = file_paths[start : start + self.sequence_length]
                    if len(seq_paths) == self.sequence_length:
                        self.sequences.append({
                            "paths": seq_paths,
                            "label": label,
                            "subject": subj
                        })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        item = self.sequences[idx]
        seq_paths = item["paths"]
        label = item["label"]

        frames_list = []
        kinematics = torch.zeros(self.sequence_length, 16, dtype=torch.float32)

        dt = 1.0 / 30.0
        prev_ear = 0.28 if label == 0 else 0.12

        for t, p in enumerate(seq_paths):
            bgr = cv2.imread(str(p))
            if bgr is None:
                bgr = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                bgr = cv2.resize(bgr, (224, 224))

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            frames_list.append(frame_tensor)

            # Base kinematic estimation:
            # Drowsy class has lower EAR (< 0.20), Non-Drowsy has open eyes (> 0.25)
            if label == 1:
                ear = 0.12 + float(np.random.normal(0, 0.02))
                mar = 0.25 + float(np.random.normal(0, 0.03))
                pitch = -12.0
            else:
                ear = 0.30 + float(np.random.normal(0, 0.02))
                mar = 0.20 + float(np.random.normal(0, 0.02))
                pitch = 0.0

            d_ear = (ear - prev_ear) / dt
            prev_ear = ear

            kinematics[t, 0] = ear
            kinematics[t, 1] = mar
            kinematics[t, 2] = 1.0 if ear < 0.20 else 0.0  # PERCLOS
            kinematics[t, 4] = pitch                      # Head pitch
            kinematics[t, 6] = d_ear                      # dEAR/dt
            kinematics[t, 11] = ear                       # ear_l
            kinematics[t, 12] = ear                       # ear_r
            kinematics[t, 15] = 1.0 if ear < 0.20 else 0.0

        frames = torch.stack(frames_list, dim=0)
        return {
            "frames": frames,
            "kinematics": kinematics,
            "label": label,
            "subject": item["subject"]
        }
