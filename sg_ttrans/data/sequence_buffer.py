"""
Sliding-window FIFO buffer for streaming real-time frames and kinematic tokens.
"""

import torch

class SequenceBuffer:
    """
    Maintains a FIFO queue of length T (default 60 frames = 2.0s @ 30 FPS).
    """
    def __init__(self, max_len: int = 60):
        self.max_len = max_len
        self.frames: list[torch.Tensor] = []
        self.kinematics: list[torch.Tensor] = []

    def push(self, frame: torch.Tensor, kinematic: torch.Tensor) -> None:
        """
        Pushes a new frame and kinematic token into the buffer, evicting the oldest.
        """
        self.frames.append(frame)
        self.kinematics.append(kinematic)
        if len(self.frames) > self.max_len:
            self.frames.pop(0)
            self.kinematics.pop(0)

    def is_ready(self) -> bool:
        """
        Returns True once buffer has accumulated max_len frames.
        """
        return len(self.frames) == self.max_len

    def get_sequence(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns stacked tensors:
            frames: [T, 3, 224, 224]
            kinematics: [T, 16]
        """
        if not self.frames:
            raise ValueError("Buffer is empty")
        return torch.stack(self.frames, dim=0), torch.stack(self.kinematics, dim=0)

    def clear(self) -> None:
        self.frames.clear()
        self.kinematics.clear()

    def __len__(self) -> int:
        return len(self.frames)
