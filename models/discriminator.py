"""Discriminator architectures used by GAN-DANet."""
from __future__ import annotations

import torch
from torch import nn


class SRGAND(nn.Module):
    """Patch-based discriminator inspired by SRGAN."""

    def __init__(self, dim: int = 64, in_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(dim * 2)
        self.conv3 = nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(dim * 4)
        self.conv4 = nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(dim * 8)
        self.conv5 = nn.Conv2d(dim * 8, dim * 16, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(dim * 16)
        self.conv6 = nn.Conv2d(dim * 16, dim * 32, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(dim * 32)
        self.conv7 = nn.Conv2d(dim * 32, dim * 16, kernel_size=1)
        self.bn6 = nn.BatchNorm2d(dim * 16)
        self.conv8 = nn.Conv2d(dim * 16, dim * 8, kernel_size=1)
        self.bn7 = nn.BatchNorm2d(dim * 8)
        self.conv9 = nn.Conv2d(dim * 8, dim * 2, kernel_size=1)
        self.bn8 = nn.BatchNorm2d(dim * 2)
        self.conv10 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(dim * 2)
        self.conv11 = nn.Conv2d(dim * 2, dim * 8, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(dim * 8)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim * 8, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.bn1(self.conv2(x)))
        x = self.activation(self.bn2(self.conv3(x)))
        x = self.activation(self.bn3(self.conv4(x)))
        x = self.activation(self.bn4(self.conv5(x)))
        x = self.activation(self.bn5(self.conv6(x)))
        x = self.activation(self.bn6(self.conv7(x)))
        x = self.activation(self.bn7(self.conv8(x)))
        residual = x
        x = self.activation(self.bn8(self.conv9(x)))
        x = self.activation(self.bn9(self.conv10(x)))
        x = self.activation(self.bn10(self.conv11(x)))
        x = x + residual
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class Discriminator1(nn.Module):
    """Lightweight discriminator with lazy linear projection."""

    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.LazyLinear(1024)
        self.fc2 = nn.Linear(1024, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        return self.fc2(x)


__all__ = ["SRGAND", "Discriminator1"]
