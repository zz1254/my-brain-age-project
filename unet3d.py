"""3D U-Net style architecture for brain age regression."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3d(nn.Module):
    """(convolution => [BN] => ReLU) * 2 block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down3d(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool3d(2), DoubleConv3d(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up3d(nn.Module):
    """Upscaling then double conv with skip connections."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = DoubleConv3d(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Pad x to match skip connection size in case of odd dimensions
        diff_depth = skip.size(2) - x.size(2)
        diff_height = skip.size(3) - x.size(3)
        diff_width = skip.size(4) - x.size(4)
        x = F.pad(
            x,
            [diff_width // 2, diff_width - diff_width // 2,
             diff_height // 2, diff_height - diff_height // 2,
             diff_depth // 2, diff_depth - diff_depth // 2],
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv3d(nn.Module):
    """Final 1x1x1 convolution to produce output map."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net producing voxel-wise outputs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        channels: List[int] = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        self.inc = DoubleConv3d(in_channels, channels[0])
        self.down1 = Down3d(channels[0], channels[1])
        self.down2 = Down3d(channels[1], channels[2])
        self.down3 = Down3d(channels[2], channels[3])

        self.bottleneck = DoubleConv3d(channels[3], channels[3] * 2)

        self.up1 = Up3d(channels[3] * 2 + channels[3], channels[3])
        self.up2 = Up3d(channels[3] + channels[2], channels[2])
        self.up3 = Up3d(channels[2] + channels[1], channels[1])
        self.up4 = Up3d(channels[1] + channels[0], channels[0])

        self.outc = OutConv3d(channels[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class BrainAgeUNet3D(nn.Module):
    """Wrapper around UNet3D to output scan-level age predictions."""

    def __init__(self, dropout: float = 0.0, base_channels: int = 32):
        super().__init__()
        self.unet = UNet3D(in_channels=1, out_channels=1, base_channels=base_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        age_map = self.unet(x)
        if self.dropout is not None:
            age_map = self.dropout(age_map)
        age_pred = age_map.mean(dim=[1, 2, 3, 4])
        return age_pred


__all__ = [
    "BrainAgeUNet3D",
    "UNet3D",
    "DoubleConv3d",
    "Down3d",
    "Up3d",
    "OutConv3d",
]
