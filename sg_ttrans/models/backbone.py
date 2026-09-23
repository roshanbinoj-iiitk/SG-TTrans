"""
Lightweight MobileNetV4 Spatial Appearance stream backbone.
"""

import torch
import torch.nn as nn

class UniversalInvertedBottleneck(nn.Module):
    """
    Universal Inverted Bottleneck (UIB) block inspired by MobileNetV4.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand_ratio: int = 2):
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        mid_ch = in_ch * expand_ratio

        self.conv = nn.Sequential(
            # Expansion
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            # Projection
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV4SpatialBackbone(nn.Module):
    """
    Lightweight MobileNetV4 architecture extracting dense 256-D facial tokens.
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 224 -> 112
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            UniversalInvertedBottleneck(32, 48, stride=2, expand_ratio=2),      # 112 -> 56
            UniversalInvertedBottleneck(48, 48, stride=1, expand_ratio=2),
        )
        self.stage2 = nn.Sequential(
            UniversalInvertedBottleneck(48, 96, stride=2, expand_ratio=2),      # 56 -> 28
            UniversalInvertedBottleneck(96, 96, stride=1, expand_ratio=2),
        )
        self.stage3 = nn.Sequential(
            UniversalInvertedBottleneck(96, 160, stride=2, expand_ratio=2),     # 28 -> 14
            UniversalInvertedBottleneck(160, 160, stride=1, expand_ratio=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(160, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handles both sequence [B, T, C, H, W] and single frame [B, C, H, W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            feat = self.pool(self.stage3(self.stage2(self.stage1(self.stem(x_flat)))))
            feat = feat.view(B * T, -1)
            out = self.proj(feat)
            return out.view(B, T, -1)
        else:
            feat = self.pool(self.stage3(self.stage2(self.stage1(self.stem(x))))).view(x.size(0), -1)
            return self.proj(feat)
