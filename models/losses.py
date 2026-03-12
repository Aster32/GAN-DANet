"""Loss functions for GAN-DANet."""
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Set

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss with optional offline weights."""

    def __init__(
        self,
        feature_layers: Sequence[int] = (1, 6, 11, 20),
        weights_path: Optional[str] = None,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.feature_layers: Set[int] = set(feature_layers)
        if not self.feature_layers:
            raise ValueError("feature_layers must contain at least one index")
        max_layer = max(self.feature_layers)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        weights_enum = None
        if weights_path is None and pretrained:
            try:
                weights_enum = models.VGG19_Weights.DEFAULT  # type: ignore[attr-defined]
            except AttributeError:
                weights_enum = None

        try:
            vgg_features = models.vgg19(weights=weights_enum).features
        except Exception:  # pragma: no cover - dependent on runtime availability
            warnings.warn(
                "Falling back to randomly initialised VGG19 features. "
                "Pass pretrained=False or provide weights_path to silence this warning.",
                RuntimeWarning,
            )
            vgg_features = models.vgg19(weights=None).features

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location=device)
            missing, unexpected = vgg_features.load_state_dict(state_dict, strict=False)
            if unexpected:
                warnings.warn(f"Unexpected keys when loading VGG weights: {unexpected}", RuntimeWarning)
            if missing:
                warnings.warn(f"Missing keys when loading VGG weights: {missing}", RuntimeWarning)

        self.vgg = nn.Sequential(*list(vgg_features)[: max_layer + 1]).to(device)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_features = x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)
        y_features = y if y.shape[1] == 3 else y.repeat(1, 3, 1, 1)

        loss = torch.tensor(0.0, device=self.device)
        for idx, layer in enumerate(self.vgg):
            x_features = layer(x_features)
            y_features = layer(y_features)
            if idx in self.feature_layers:
                loss = loss + F.l1_loss(x_features, y_features)
        return loss


class TVLoss(nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h_tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum()
        w_tv = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).sum()
        count_h = x[:, :, 1:, :].numel()
        count_w = x[:, :, :, 1:].numel()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class SSIM(nn.Module):
    def __init__(self, window_size: int = 11, size_average: bool = True) -> None:
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer("window", self._create_window(window_size, self.channel))

    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32)
        gauss = torch.exp(-((coords - window_size // 2) ** 2) / (2 * sigma ** 2))
        return (gauss / gauss.sum()).unsqueeze(1)

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1d = self._gaussian(window_size, 1.5)
        _2d = _1d @ _1d.t()
        window = _2d.float().unsqueeze(0).unsqueeze(0)
        return window.expand(channel, 1, window_size, window_size).contiguous()

    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        channel: int,
        size_average: bool = True,
    ) -> torch.Tensor:
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        if size_average:
            return ssim_map.mean()
        return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.size(0) == channel:
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            self.register_buffer("window", window)
            self.channel = channel
        window = window.to(img1.device)
        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


def _masked_mean(
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is None:
        mask_term = torch.ones_like(value)
    else:
        if mask.dim() == value.dim() - 1:
            mask = mask.unsqueeze(1)
        mask_term = mask.to(value.device, dtype=value.dtype)
        while mask_term.dim() < value.dim():
            mask_term = mask_term.unsqueeze(1)

    if weight is not None:
        if weight.dim() == value.dim() - 1:
            weight = weight.unsqueeze(1)
        weight_term = weight.to(value.device, dtype=value.dtype)
        while weight_term.dim() < value.dim():
            weight_term = weight_term.unsqueeze(1)
        mask_term = mask_term * weight_term

    weighted = value * mask_term
    denom = mask_term.sum().clamp_min(1.0)
    return weighted.sum() / denom


class HeteroscedasticGaussianLoss(nn.Module):
    """Gaussian NLL for uncertainty-aware regression heads."""

    def forward(
        self,
        mean: torch.Tensor,
        target: torch.Tensor,
        logvar: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = 0.5 * (torch.exp(-logvar) * (mean - target).pow(2) + logvar)
        return _masked_mean(loss, mask)


class ConservationLoss(nn.Module):
    """Encourages fine-scale predictions to remain consistent with coarse observations."""

    def __init__(self, scale_factor: int = 2, reduction: str = "l1") -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        coarse_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        coarse_prediction = F.avg_pool2d(prediction, kernel_size=self.scale_factor, stride=self.scale_factor)
        if self.reduction == "l2":
            loss = (coarse_prediction - coarse_target).pow(2)
        else:
            loss = (coarse_prediction - coarse_target).abs()
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            coarse_mask = F.avg_pool2d(mask.float(), kernel_size=self.scale_factor, stride=self.scale_factor)
        else:
            coarse_mask = None
        return _masked_mean(loss, coarse_mask)


class SpatialGradientLoss(nn.Module):
    """Preserves spatial gradients and boundaries in predicted hydrologic fields."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        terrain_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
        pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        target_dy = target[..., 1:, :] - target[..., :-1, :]

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask_x = mask[..., :, 1:] * mask[..., :, :-1]
            mask_y = mask[..., 1:, :] * mask[..., :-1, :]
        else:
            mask_x = None
            mask_y = None

        if terrain_weight is not None:
            if terrain_weight.dim() == 3:
                terrain_weight = terrain_weight.unsqueeze(1)
            terrain_x = 0.5 * (terrain_weight[..., :, 1:] + terrain_weight[..., :, :-1])
            terrain_y = 0.5 * (terrain_weight[..., 1:, :] + terrain_weight[..., :-1, :])
        else:
            terrain_x = None
            terrain_y = None

        loss_x = _masked_mean((pred_dx - target_dx).abs(), mask_x, terrain_x)
        loss_y = _masked_mean((pred_dy - target_dy).abs(), mask_y, terrain_y)
        return 0.5 * (loss_x + loss_y)


class HydrologyLossBundle(nn.Module):
    """Composite loss for hydrology-aware TWSA downscaling."""

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        ssim_weight: float = 0.15,
        conservation_weight: float = 0.35,
        gradient_weight: float = 0.2,
        uncertainty_weight: float = 0.05,
        anomaly_weight: float = 0.2,
        trend_weight: float = 0.2,
        tv_weight: float = 1e-4,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.ssim_weight = ssim_weight
        self.conservation_weight = conservation_weight
        self.gradient_weight = gradient_weight
        self.uncertainty_weight = uncertainty_weight
        self.anomaly_weight = anomaly_weight
        self.trend_weight = trend_weight
        self.tv_weight = tv_weight

        self.ssim = SSIM(window_size=11, size_average=True)
        self.tv = TVLoss(weight=1.0)
        self.consistency = ConservationLoss(scale_factor=scale_factor)
        self.gradient = SpatialGradientLoss()
        self.uncertainty = HeteroscedasticGaussianLoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        coarse_target: torch.Tensor,
        target_anomaly: Optional[torch.Tensor] = None,
        target_trend: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        terrain_weight: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        mean = outputs["mean"]
        reconstruction = _masked_mean((mean - target).abs(), mask)
        ssim_loss = 1.0 - self.ssim(mean, target)
        conservation = self.consistency(mean, coarse_target, mask)
        gradient = self.gradient(mean, target, mask, terrain_weight=terrain_weight)
        tv = self.tv(mean)

        total = (
            self.reconstruction_weight * reconstruction
            + self.ssim_weight * ssim_loss
            + self.conservation_weight * conservation
            + self.gradient_weight * gradient
            + self.tv_weight * tv
        )

        uncertainty = torch.tensor(0.0, device=mean.device)
        if "logvar" in outputs and self.uncertainty_weight > 0:
            uncertainty = self.uncertainty(mean, target, outputs["logvar"], mask)
            total = total + self.uncertainty_weight * uncertainty

        anomaly = torch.tensor(0.0, device=mean.device)
        if target_anomaly is not None and "anomaly" in outputs:
            anomaly = _masked_mean((outputs["anomaly"] - target_anomaly).abs(), mask)
            total = total + self.anomaly_weight * anomaly

        trend = torch.tensor(0.0, device=mean.device)
        if target_trend is not None and "trend" in outputs:
            trend = _masked_mean((outputs["trend"] - target_trend).abs(), mask)
            total = total + self.trend_weight * trend

        return {
            "loss": total,
            "reconstruction": reconstruction,
            "ssim": ssim_loss,
            "conservation": conservation,
            "gradient": gradient,
            "uncertainty": uncertainty,
            "anomaly": anomaly,
            "trend": trend,
            "tv": tv,
        }


__all__ = [
    "PerceptualLoss",
    "TVLoss",
    "SSIM",
    "HeteroscedasticGaussianLoss",
    "ConservationLoss",
    "SpatialGradientLoss",
    "HydrologyLossBundle",
]
