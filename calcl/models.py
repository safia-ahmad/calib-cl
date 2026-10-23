"""The two architectures. Both start from ImageNet weights and output 2 logits."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class Resize(nn.Module):
    """Upsamples 96x96 patches to the input size a ViT expects."""

    def __init__(self, net: nn.Module, size: int = 224):
        super().__init__()
        self.net, self.size = net, size

    def forward(self, x):
        return self.net(F.interpolate(x, size=self.size, mode="bilinear", align_corners=False))


def make_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "resnet18":
        import torchvision
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m
    if name == "vit_s16":
        import timm
        return Resize(timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=2))
    raise ValueError(f"unknown model {name}")
