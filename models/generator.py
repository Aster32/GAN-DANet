"""Generator architectures for GAN-DANet."""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from torch import nn
import torch.nn.functional as F


class OriginalRelationshipLearner(nn.Module):
    """Learns relationships between low- and high-resolution inputs."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        channels = [64, 128, 256, 512, 1024]
        layers: List[nn.Module] = []
        in_channels = input_channels
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - simple wrapper
        return self.net(x)


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.avg_pool(x)
        attention = self.relu(self.fc1(attention))
        attention = self.sigmoid(self.fc2(attention))
        return x * attention


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.channel_attention = SqueezeExcitation(channels, reduction_ratio)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        attention = self.spatial_attention(torch.cat([max_out, avg_out], dim=1))
        return x * attention


class PAMModule(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced_channels = max(1, channels // 8)
        self.query = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        query = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key(x).view(b, -1, h * w)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value(x).view(b, -1, h * w)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return self.gamma * out + x


class CAMModule(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        query = x.view(b, c, -1)
        key = query.permute(0, 2, 1)
        energy = torch.bmm(query, key)
        energy_new = energy.max(dim=-1, keepdim=True)[0].expand_as(energy) - energy
        attention = torch.softmax(energy_new, dim=-1)
        value = x.view(b, c, -1)
        out = torch.bmm(attention, value).view(b, c, h, w)
        return self.gamma * out + x


class DANetAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.position_attention = PAMModule(channels)
        self.channel_attention = CAMModule(channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position = self.position_attention(x)
        channel = self.channel_attention(x)
        features = torch.cat([position, channel], dim=1)
        return self.fuse(features)


def _build_attention(attention_type: Optional[str], channels: int) -> Optional[nn.Module]:
    if attention_type is None or attention_type.lower() == "none":
        return None
    attention = attention_type.lower()
    if attention == "danet":
        return DANetAttention(channels)
    if attention in {"senet", "cbam"}:
        warnings.warn(
            f"Attention type '{attention_type}' currently aliases to 'danet'.",
            RuntimeWarning,
        )
        return DANetAttention(channels)
    raise ValueError(f"Unsupported attention type: {attention_type}")


class FlexibleUpsamplingModule(nn.Module):
    """Generator used for super-resolution in GAN-DANet."""

    def __init__(
        self,
        input_channels: int = 40,
        growth_rate: int = 24,
        num_blocks: int = 3,
        num_layers_per_block: int = 4,
        attention_type: Optional[str] = "danet",
    ) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        self.feature_channels: List[int] = []

        num_features = 64
        for block_idx in range(num_blocks):
            dense_block = DenseBlock(num_layers_per_block, num_features, growth_rate)
            self.dense_blocks.append(dense_block)
            num_features += num_layers_per_block * growth_rate

            attention = _build_attention(attention_type, num_features)
            self.attention_modules.append(attention)
            self.feature_channels.append(num_features)

            if block_idx != num_blocks - 1:
                transition = TransitionLayer(num_features, num_features // 2)
                self.transition_layers.append(transition)
                num_features //= 2

        self.channel_adjust = nn.ModuleList(
            [nn.Conv2d(ch, 64, kernel_size=1, bias=False) for ch in reversed(self.feature_channels)]
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=False),
        )

        self.final = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        skip_connections: List[torch.Tensor] = []

        for dense_block, attention in zip(self.dense_blocks, self.attention_modules):
            x = dense_block(x)
            if attention is not None:
                x = attention(x)
            skip_connections.append(x)
            if len(self.transition_layers) > len(skip_connections) - 1:
                x = self.transition_layers[len(skip_connections) - 1](x)

        x = self.upsample(x)
        for adjust, feature in zip(self.channel_adjust, reversed(skip_connections)):
            resized = F.interpolate(feature, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = x + adjust(resized)

        return self.final(x)


__all__ = [
    "OriginalRelationshipLearner",
    "FlexibleUpsamplingModule",
    "SqueezeExcitation",
    "CBAMBlock",
]
