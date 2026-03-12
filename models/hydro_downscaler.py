"""Advanced hydrology-aware downscaling models for TWSA."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _ensure_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x.unsqueeze(1)
    if x.dim() != 5:
        raise ValueError(f"Expected a 4D or 5D tensor, received shape {tuple(x.shape)}")
    return x


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        activate: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.activate = nn.GELU() if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block1 = ConvGNAct(in_channels, out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.block2 = ConvGNAct(out_channels, out_channels, activate=False)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = ConvGNAct(in_channels, out_channels, kernel_size=1, padding=0, activate=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        return self.act(x + residual)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.down = ConvGNAct(in_channels, out_channels, stride=2)
        self.refine = ResidualBlock(out_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(in_channels, out_channels),
        )
        self.refine = ResidualBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)


class TemporalContextEncoder(nn.Module):
    """Encodes a temporal window and learns attention weights over time."""

    def __init__(self, in_channels: int, embed_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, embed_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(_num_groups(embed_channels), embed_channels),
            nn.GELU(),
            nn.Conv3d(embed_channels, embed_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(_num_groups(embed_channels), embed_channels),
            nn.GELU(),
        )
        hidden = max(embed_channels // 2, 8)
        self.score = nn.Sequential(
            nn.Linear(embed_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_sequence(x).permute(0, 2, 1, 3, 4)
        features = self.proj(x)
        pooled = features.mean(dim=(-1, -2)).permute(0, 2, 1)
        scores = self.score(pooled).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(features * weights[:, None, :, None, None], dim=2)


class CrossResolutionFusion(nn.Module):
    def __init__(self, high_channels: int, low_channels: int, static_channels: int = 0) -> None:
        super().__init__()
        self.low_proj = nn.Conv2d(low_channels, high_channels, kernel_size=1, bias=False)
        gate_channels = high_channels * 2 + static_channels
        self.gate = nn.Sequential(
            nn.Conv2d(gate_channels, high_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.mix = ResidualBlock(high_channels * 2, high_channels)

    def forward(
        self,
        high_feat: torch.Tensor,
        low_feat: torch.Tensor,
        static_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        low_feat = self.low_proj(F.interpolate(low_feat, size=high_feat.shape[-2:], mode="bilinear", align_corners=False))
        gate_inputs = [high_feat, low_feat]
        if static_feat is not None:
            gate_inputs.append(F.interpolate(static_feat, size=high_feat.shape[-2:], mode="bilinear", align_corners=False))
        gate = self.gate(torch.cat(gate_inputs, dim=1))
        return self.mix(torch.cat([high_feat, low_feat * gate], dim=1))


class MultiDilationPyramid(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                    nn.GroupNorm(_num_groups(channels), channels),
                    nn.GELU(),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.merge = ResidualBlock(channels * 3, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pyramid = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.merge(pyramid)


class StaticConditionEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvGNAct(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class HydroTWSADownscaler(nn.Module):
    """Dual-path temporal downscaler for anomaly/trend-aware TWSA prediction."""

    def __init__(
        self,
        aux_channels: int,
        static_channels: int = 3,
        base_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if aux_channels <= 0:
            raise ValueError("aux_channels must be positive")
        if static_channels < 0 or static_channels >= aux_channels:
            raise ValueError("static_channels must be in [0, aux_channels)")

        self.aux_channels = aux_channels
        self.static_channels = static_channels
        self.dynamic_channels = aux_channels - static_channels
        self.base_channels = base_channels
        static_embed_channels = max(base_channels // 4, 8) if static_channels > 0 else 0

        self.low_temporal = TemporalContextEncoder(1, 24)
        self.trend_temporal = TemporalContextEncoder(1, 16)
        self.dynamic_temporal = TemporalContextEncoder(self.dynamic_channels, 32)

        self.dynamic_stem = nn.Sequential(
            ConvGNAct(self.dynamic_channels, base_channels),
            ResidualBlock(base_channels, base_channels, dropout=dropout),
        )

        if static_channels > 0:
            self.static_encoder = StaticConditionEncoder(static_channels, static_embed_channels)
        else:
            self.static_encoder = None

        fusion_channels = 1 + 1 + 24 + 16 + base_channels + 32 + static_embed_channels
        self.high_stem = nn.Sequential(
            ConvGNAct(fusion_channels, base_channels),
            ResidualBlock(base_channels, base_channels, dropout=dropout),
        )

        low_feature_channels = 1 + 1 + 24 + 16
        self.low_mid_proj = nn.Sequential(
            ConvGNAct(low_feature_channels, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2, dropout=dropout),
        )
        self.low_coarse_proj = nn.Sequential(
            ConvGNAct(low_feature_channels, base_channels * 3),
            ResidualBlock(base_channels * 3, base_channels * 3, dropout=dropout),
        )

        self.enc_mid = DownsampleBlock(base_channels, base_channels * 2, dropout=dropout)
        self.cross_mid = CrossResolutionFusion(base_channels * 2, base_channels * 2, static_embed_channels)
        self.enc_coarse = DownsampleBlock(base_channels * 2, base_channels * 3, dropout=dropout)
        self.cross_coarse = CrossResolutionFusion(base_channels * 3, base_channels * 3, static_embed_channels)
        self.bottleneck = MultiDilationPyramid(base_channels * 3)

        self.up_mid = UpsampleBlock(base_channels * 3, base_channels * 2, base_channels * 2, dropout=dropout)
        self.up_high = UpsampleBlock(base_channels * 2, base_channels, base_channels, dropout=dropout)

        trend_head_channels = base_channels + 24 + 16 + 1 + static_embed_channels
        self.trend_refiner = nn.Sequential(
            ConvGNAct(trend_head_channels, base_channels),
            ResidualBlock(base_channels, base_channels, dropout=dropout),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )
        self.anomaly_head = nn.Sequential(
            ResidualBlock(base_channels, base_channels, dropout=dropout),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )
        self.logvar_head = nn.Sequential(
            ResidualBlock(base_channels, base_channels, dropout=dropout),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )

    def forward(
        self,
        lr_anomaly_seq: torch.Tensor,
        aux_seq: torch.Tensor,
        lr_trend_seq: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        lr_anomaly_seq = _ensure_sequence(lr_anomaly_seq)
        aux_seq = _ensure_sequence(aux_seq)

        if lr_anomaly_seq.shape[1] != aux_seq.shape[1]:
            raise ValueError("lr_anomaly_seq and aux_seq must have the same temporal window length")

        center_idx = lr_anomaly_seq.shape[1] // 2
        high_size = aux_seq.shape[-2:]

        current_anomaly_low = lr_anomaly_seq[:, center_idx]
        current_anomaly_high = F.interpolate(current_anomaly_low, size=high_size, mode="bicubic", align_corners=False)
        low_context = self.low_temporal(lr_anomaly_seq)
        low_context_high = F.interpolate(low_context, size=high_size, mode="bilinear", align_corners=False)

        if lr_trend_seq is None:
            lr_trend_seq = torch.zeros_like(lr_anomaly_seq)
        else:
            lr_trend_seq = _ensure_sequence(lr_trend_seq)
        current_trend_low = lr_trend_seq[:, center_idx]
        current_trend_high = F.interpolate(current_trend_low, size=high_size, mode="bicubic", align_corners=False)
        trend_context = self.trend_temporal(lr_trend_seq)
        trend_context_high = F.interpolate(trend_context, size=high_size, mode="bilinear", align_corners=False)

        current_aux = aux_seq[:, center_idx]
        if self.static_channels > 0:
            dynamic_seq = aux_seq[:, :, : self.dynamic_channels]
            static_aux = current_aux[:, self.dynamic_channels :]
            static_feat = self.static_encoder(static_aux) if self.static_encoder is not None else None
        else:
            dynamic_seq = aux_seq
            static_feat = None

        dynamic_current = dynamic_seq[:, center_idx]
        dynamic_feat = self.dynamic_stem(dynamic_current)
        dynamic_context = self.dynamic_temporal(dynamic_seq)

        fusion_parts = [
            current_anomaly_high,
            current_trend_high,
            low_context_high,
            trend_context_high,
            dynamic_feat,
            dynamic_context,
        ]
        if static_feat is not None:
            fusion_parts.append(static_feat)
        high = self.high_stem(torch.cat(fusion_parts, dim=1))

        low_features = torch.cat([current_anomaly_low, current_trend_low, low_context, trend_context], dim=1)
        high_skip = high

        mid = self.enc_mid(high)
        mid = self.cross_mid(mid, self.low_mid_proj(low_features), static_feat)
        mid_skip = mid

        coarse = self.enc_coarse(mid)
        coarse_inputs = F.avg_pool2d(low_features, kernel_size=2, stride=2)
        coarse = self.cross_coarse(coarse, self.low_coarse_proj(coarse_inputs), static_feat)
        coarse = self.bottleneck(coarse)

        decoded = self.up_mid(coarse, mid_skip)
        decoded = self.up_high(decoded, high_skip)

        anomaly = current_anomaly_high + self.anomaly_head(decoded)
        trend_inputs = [decoded, low_context_high, trend_context_high, current_trend_high]
        if static_feat is not None:
            trend_inputs.append(static_feat)
        trend = current_trend_high + self.trend_refiner(torch.cat(trend_inputs, dim=1))

        mean = anomaly + trend
        logvar = self.logvar_head(decoded).clamp(min=-6.0, max=3.0)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[-2:] != mean.shape[-2:]:
                mask = F.interpolate(mask.float(), size=mean.shape[-2:], mode="nearest")
            mean = mean * mask
            anomaly = anomaly * mask
            trend = trend * mask

        coarse_prediction = F.avg_pool2d(mean, kernel_size=2, stride=2)
        return {
            "mean": mean,
            "anomaly": anomaly,
            "trend": trend,
            "logvar": logvar,
            "coarse": coarse_prediction,
        }


__all__ = ["HydroTWSADownscaler"]
