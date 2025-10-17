"""Utility helpers for GAN-DANet models."""
from __future__ import annotations

from torch import nn


def weights_init_normal(module: nn.Module) -> None:
    """Initialize common layers with Kaiming/Xavier schemes."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Parameter):  # pragma: no cover - rare
        nn.init.constant_(module, 0)


__all__ = ["weights_init_normal"]
