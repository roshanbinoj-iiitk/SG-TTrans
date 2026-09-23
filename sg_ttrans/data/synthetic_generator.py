"""
Reproducible multimodal synthetic sequence generator for SG-TTrans.
Simulates synchronized RGB appearance frames and 16-D kinematic tokens for all 5 fatigue states.
"""

import numpy as np
import torch

class SyntheticSequenceGenerator:
    """
    Generates synthetic multimodal batches:
        - RGB frames: [B, T, 3, 224, 224]
        - Kinematic tokens: [B, T, 16]
        - Class labels: [B]
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_batch(
        self,
        batch_size: int = 4,
        seq_len: int = 60
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates a balanced batch across the 5 driver states.
        """
        frames = torch.zeros(batch_size, seq_len, 3, 224, 224, dtype=torch.float32)
        kinematics = torch.zeros(batch_size, seq_len, 16, dtype=torch.float32)
        labels = torch.zeros(batch_size, dtype=torch.long)

        dt = 1.0 / 30.0

        for b in range(batch_size):
            # Select driver state: 0: Alert, 1: Drowsy, 2: Microsleep, 3: Yawn, 4: Distracted
            label = int(b % 5)
            labels[b] = label

            # 1. Base appearance frames with class-specific synthetic pattern
            base_color = torch.tensor([0.45, 0.40, 0.35]).view(1, 3, 1, 1) # Skin tone base
            noise = torch.randn(seq_len, 3, 224, 224) * 0.05
            frames[b] = base_color + noise

            # 2. Kinematics generation per state
            if label == 0:  # Alert: open eyes (EAR ~ 0.30), closed mouth (MAR ~ 0.20), forward head
                ear = 0.30 + self.rng.normal(0, 0.02, seq_len)
                mar = 0.20 + self.rng.normal(0, 0.02, seq_len)
                yaw = self.rng.normal(0, 3.0, seq_len)
                pitch = self.rng.normal(0, 3.0, seq_len)
                roll = self.rng.normal(0, 2.0, seq_len)

            elif label == 1:  # Drowsy: slow eyelid drooping (EAR ~ 0.21)
                ear = 0.21 + self.rng.normal(0, 0.02, seq_len)
                mar = 0.25 + self.rng.normal(0, 0.03, seq_len)
                yaw = self.rng.normal(0, 4.0, seq_len)
                pitch = -10.0 + self.rng.normal(0, 4.0, seq_len)  # Slight nodding
                roll = self.rng.normal(0, 2.0, seq_len)

            elif label == 2:  # Microsleep: sustained eye closure (EAR ~ 0.08 < 0.20)
                ear = 0.08 + self.rng.normal(0, 0.01, seq_len)
                mar = 0.20 + self.rng.normal(0, 0.02, seq_len)
                yaw = self.rng.normal(0, 3.0, seq_len)
                pitch = -25.0 + self.rng.normal(0, 5.0, seq_len)  # Forward slump
                roll = self.rng.normal(0, 2.0, seq_len)

            elif label == 3:  # Yawn: mouth opening (MAR ~ 0.65 > 0.50)
                ear = 0.22 + self.rng.normal(0, 0.02, seq_len)
                mar = 0.65 + self.rng.normal(0, 0.04, seq_len)
                yaw = self.rng.normal(0, 4.0, seq_len)
                pitch = 5.0 + self.rng.normal(0, 3.0, seq_len)
                roll = self.rng.normal(0, 2.0, seq_len)

            else:  # Distracted: large head yaw / pitch glancing away
                ear = 0.28 + self.rng.normal(0, 0.02, seq_len)
                mar = 0.20 + self.rng.normal(0, 0.02, seq_len)
                yaw = 35.0 + self.rng.normal(0, 5.0, seq_len)  # Checking side mirror
                pitch = 10.0 + self.rng.normal(0, 4.0, seq_len)
                roll = self.rng.normal(0, 3.0, seq_len)

            # Assemble kinematic tokens over sequence
            for t in range(seq_len):
                perclos = float(np.mean(ear[: t + 1] < 0.20))
                d_ear = float((ear[t] - ear[t - 1]) / dt) if t > 0 else 0.0
                d_mar = float((mar[t] - mar[t - 1]) / dt) if t > 0 else 0.0
                d_yaw = float((yaw[t] - yaw[t - 1]) / dt) if t > 0 else 0.0
                d_pitch = float((pitch[t] - pitch[t - 1]) / dt) if t > 0 else 0.0
                d_roll = float((roll[t] - roll[t - 1]) / dt) if t > 0 else 0.0

                euler_norm = float(np.sqrt(yaw[t]**2 + pitch[t]**2 + roll[t]**2))
                d_euler_norm = float(np.sqrt(d_yaw**2 + d_pitch**2 + d_roll**2))
                blink_flag = 1.0 if ear[t] < 0.20 else 0.0

                kinematics[b, t] = torch.tensor([
                    ear[t], mar[t], perclos,
                    yaw[t], pitch[t], roll[t],
                    d_ear, d_mar,
                    d_yaw, d_pitch, d_roll,
                    ear[t], ear[t],  # left & right EAR
                    euler_norm, d_euler_norm, blink_flag
                ], dtype=torch.float32)

        return frames, kinematics, labels
