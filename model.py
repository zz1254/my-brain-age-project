"""Model definitions for MRI brain age prediction."""
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

from unet3d import BrainAgeUNet3D


class BrainAgeResNet3D(nn.Module):
    """3D ResNet-18 backbone with regression head for brain age prediction."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.backbone = r3d_18(weights=None)
        self._adapt_first_conv()
        in_features = self.backbone.fc.in_features
        head = [nn.Linear(in_features, 1)]
        if dropout > 0:
            head.insert(0, nn.Dropout(dropout))
        self.backbone.fc = nn.Sequential(*head)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _adapt_first_conv(self) -> None:
        """Modify the first convolution to accept single-channel MRI volumes."""
        first_conv: nn.Conv3d = self.backbone.stem[0]  # type: ignore[assignment]
        new_conv = nn.Conv3d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(first_conv.weight.sum(dim=1, keepdim=True))
            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)
        self.backbone.stem[0] = new_conv  # type: ignore[index]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns a single age prediction per sample."""
        x = self.backbone.stem(x)
        x = self.backbone.layer1(self.backbone.maxpool(x))
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x.squeeze(-1)


def build_model(
    backbone: str = "unet",
    dropout: float = 0.0,
    base_channels: int = 32,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Utility to create and move the model to the appropriate device."""

    if backbone == "resnet3d":
        model: nn.Module = BrainAgeResNet3D(dropout=dropout)
    elif backbone == "unet":
        model = BrainAgeUNet3D(dropout=dropout, base_channels=base_channels)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    if device is not None:
        model.to(device)
    return model


__all__ = ["BrainAgeResNet3D", "BrainAgeUNet3D", "build_model"]
