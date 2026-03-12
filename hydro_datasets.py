"""UTF-8 dataset helpers for research-grade hydrology downscaling."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import load_data
from repro import worker_init_fn


@dataclass
class HydrologySplit:
    train: Dataset
    val: Dataset
    aux_channels: int


@dataclass
class RollingOriginFold:
    name: str
    train: Dataset
    val: Dataset
    aux_channels: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int


class HydrologyWindowDataset(Dataset):
    """Temporal-window dataset for anomaly/trend-aware TWSA downscaling."""

    def __init__(
        self,
        lr_anomaly: np.ndarray,
        hr_anomaly: np.ndarray,
        hr_aux: np.ndarray,
        lr_trend: Optional[np.ndarray] = None,
        hr_trend: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        window_size: int = 5,
        augment: bool = False,
        static_channels: int = 3,
        valid_indices: Optional[np.ndarray] = None,
        context_min_index: int = 0,
        context_max_index: Optional[int] = None,
    ) -> None:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if len(lr_anomaly) != len(hr_anomaly) or len(hr_anomaly) != len(hr_aux):
            raise ValueError("Input arrays must share the same temporal dimension")
        if lr_trend is not None and len(lr_trend) != len(lr_anomaly):
            raise ValueError("lr_trend must share the same temporal dimension as lr_anomaly")
        if hr_trend is not None and len(hr_trend) != len(hr_anomaly):
            raise ValueError("hr_trend must share the same temporal dimension as hr_anomaly")

        self.lr_anomaly = lr_anomaly.astype(np.float32)
        self.hr_anomaly = hr_anomaly.astype(np.float32)
        self.hr_aux = hr_aux.astype(np.float32)
        self.lr_trend = lr_trend.astype(np.float32) if lr_trend is not None else None
        self.hr_trend = hr_trend.astype(np.float32) if hr_trend is not None else None
        self.mask = mask.astype(np.float32) if mask is not None else None
        self.window_size = window_size
        self.half_window = window_size // 2
        self.augment = augment
        self.static_channels = static_channels
        if valid_indices is None:
            self.valid_indices = np.arange(len(self.lr_anomaly))
        else:
            self.valid_indices = np.asarray(valid_indices, dtype=np.int64)
        self.context_min_index = context_min_index
        self.context_max_index = len(self.lr_anomaly) - 1 if context_max_index is None else context_max_index
        self.terrain_weight = self._build_terrain_weight()

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _window_indices(self, idx: int) -> np.ndarray:
        raw = np.arange(idx - self.half_window, idx + self.half_window + 1)
        return np.clip(raw, self.context_min_index, self.context_max_index)

    def _apply_augmentation(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        spatial_keys = [
            "lr_anomaly_seq",
            "aux_seq",
            "target_anomaly",
            "coarse_target",
            "target",
            "terrain_weight",
        ]
        if "lr_trend_seq" in sample:
            spatial_keys.append("lr_trend_seq")
        if "target_trend" in sample:
            spatial_keys.append("target_trend")
        if "mask" in sample:
            spatial_keys.append("mask")

        if random.random() > 0.5:
            for key in spatial_keys:
                sample[key] = torch.flip(sample[key], dims=(-1,))
        if random.random() > 0.5:
            for key in spatial_keys:
                sample[key] = torch.flip(sample[key], dims=(-2,))
        if random.random() > 0.5:
            k = random.choice((1, 2, 3))
            for key in spatial_keys:
                sample[key] = torch.rot90(sample[key], k=k, dims=(-2, -1))
        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        center_idx = int(self.valid_indices[idx])
        indices = self._window_indices(center_idx)
        lr_anomaly_seq = torch.from_numpy(self.lr_anomaly[indices]).float().unsqueeze(1)
        aux_seq = torch.from_numpy(self.hr_aux[indices]).float().permute(0, 3, 1, 2)
        target_anomaly = torch.from_numpy(self.hr_anomaly[center_idx]).float().unsqueeze(0)

        sample: Dict[str, torch.Tensor] = {
            "lr_anomaly_seq": lr_anomaly_seq,
            "aux_seq": aux_seq,
            "target_anomaly": target_anomaly,
        }

        if self.lr_trend is not None:
            lr_trend_seq = torch.from_numpy(self.lr_trend[indices]).float().unsqueeze(1)
            sample["lr_trend_seq"] = lr_trend_seq
            coarse_target = lr_anomaly_seq[self.half_window] + lr_trend_seq[self.half_window]
        else:
            coarse_target = lr_anomaly_seq[self.half_window]

        if self.hr_trend is not None:
            target_trend = torch.from_numpy(self.hr_trend[center_idx]).float().unsqueeze(0)
            sample["target_trend"] = target_trend
            target = target_anomaly + target_trend
        else:
            target = target_anomaly

        sample["coarse_target"] = coarse_target
        sample["target"] = target
        sample["terrain_weight"] = self.terrain_weight.clone()

        if self.mask is not None:
            sample["mask"] = torch.from_numpy(self.mask).float().unsqueeze(0)

        if self.augment:
            sample = self._apply_augmentation(sample)

        return sample

    def _build_terrain_weight(self) -> torch.Tensor:
        if self.static_channels <= 0:
            h, w = self.hr_aux.shape[1:3]
            return torch.ones(1, h, w, dtype=torch.float32)

        static_slice = self.hr_aux[0, :, :, -self.static_channels :]
        dem = static_slice[..., -1].astype(np.float32)
        gy, gx = np.gradient(dem)
        slope = np.sqrt(gx ** 2 + gy ** 2)
        slope = slope - slope.min()
        scale = slope.max() if slope.max() > 0 else 1.0
        normalized = slope / scale
        return torch.from_numpy(1.0 + normalized).float().unsqueeze(0)


def build_hydrology_splits(
    window_size: int = 5,
    val_fraction: float = 0.2,
    augment_train: bool = True,
    mask: Optional[np.ndarray] = None,
    static_channels: int = 3,
) -> HydrologySplit:
    [lr_anomaly, trend05], [hr_anomaly, trend25], hr_aux, *_ = load_data()

    n = len(lr_anomaly)
    split_index = max(window_size, int(n * (1.0 - val_fraction)))
    train_indices = np.arange(0, max(split_index - window_size // 2, 1))
    val_indices = np.arange(split_index, n)

    train = HydrologyWindowDataset(
        lr_anomaly=lr_anomaly,
        hr_anomaly=hr_anomaly,
        hr_aux=hr_aux,
        lr_trend=trend05,
        hr_trend=trend25,
        mask=mask,
        window_size=window_size,
        augment=augment_train,
        static_channels=static_channels,
        valid_indices=train_indices,
        context_max_index=split_index - 1,
    )
    val = HydrologyWindowDataset(
        lr_anomaly=lr_anomaly,
        hr_anomaly=hr_anomaly,
        hr_aux=hr_aux,
        lr_trend=trend05,
        hr_trend=trend25,
        mask=mask,
        window_size=window_size,
        augment=False,
        static_channels=static_channels,
        valid_indices=val_indices,
        context_max_index=n - 1,
    )
    return HydrologySplit(train=train, val=val, aux_channels=hr_aux.shape[-1])


def build_hydrology_dataloaders(
    batch_size: int = 4,
    window_size: int = 5,
    val_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    static_channels: int = 3,
) -> tuple[DataLoader, DataLoader, int]:
    splits = build_hydrology_splits(
        window_size=window_size,
        val_fraction=val_fraction,
        static_channels=static_channels,
    )
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        splits.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        splits.val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, splits.aux_channels


def build_rolling_origin_folds(
    window_size: int = 5,
    initial_train_size: int = 72,
    validation_size: int = 12,
    step_size: int = 12,
    mask: Optional[np.ndarray] = None,
    static_channels: int = 3,
) -> List[RollingOriginFold]:
    [lr_anomaly, trend05], [hr_anomaly, trend25], hr_aux, *_ = load_data()
    n = len(lr_anomaly)
    folds: List[RollingOriginFold] = []

    if initial_train_size < window_size:
        raise ValueError("initial_train_size must be at least as large as window_size")

    fold_idx = 0
    train_end = initial_train_size
    while train_end + validation_size <= n:
        val_end = train_end + validation_size
        train_indices = np.arange(0, train_end)
        val_indices = np.arange(train_end, val_end)
        train = HydrologyWindowDataset(
            lr_anomaly=lr_anomaly,
            hr_anomaly=hr_anomaly,
            hr_aux=hr_aux,
            lr_trend=trend05,
            hr_trend=trend25,
            mask=mask,
            window_size=window_size,
            augment=False,
            static_channels=static_channels,
            valid_indices=train_indices,
            context_max_index=train_end - 1,
        )
        val = HydrologyWindowDataset(
            lr_anomaly=lr_anomaly,
            hr_anomaly=hr_anomaly,
            hr_aux=hr_aux,
            lr_trend=trend05,
            hr_trend=trend25,
            mask=mask,
            window_size=window_size,
            augment=False,
            static_channels=static_channels,
            valid_indices=val_indices,
            context_max_index=val_end - 1,
        )
        folds.append(
            RollingOriginFold(
                name=f"fold_{fold_idx:02d}",
                train=train,
                val=val,
                aux_channels=hr_aux.shape[-1],
                train_start=0,
                train_end=train_end,
                val_start=train_end,
                val_end=val_end,
            )
        )
        train_end += step_size
        fold_idx += 1

    return folds


def build_rolling_origin_dataloaders(
    batch_size: int = 4,
    window_size: int = 5,
    initial_train_size: int = 72,
    validation_size: int = 12,
    step_size: int = 12,
    num_workers: int = 0,
    seed: int = 42,
    static_channels: int = 3,
) -> List[tuple[str, DataLoader, DataLoader, int]]:
    folds = build_rolling_origin_folds(
        window_size=window_size,
        initial_train_size=initial_train_size,
        validation_size=validation_size,
        step_size=step_size,
        static_channels=static_channels,
    )
    generator = torch.Generator().manual_seed(seed)
    outputs: List[tuple[str, DataLoader, DataLoader, int]] = []
    for fold in folds:
        train_loader = DataLoader(
            fold.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        val_loader = DataLoader(
            fold.val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )
        outputs.append((fold.name, train_loader, val_loader, fold.aux_channels))
    return outputs


__all__ = [
    "HydrologySplit",
    "RollingOriginFold",
    "HydrologyWindowDataset",
    "build_hydrology_splits",
    "build_hydrology_dataloaders",
    "build_rolling_origin_folds",
    "build_rolling_origin_dataloaders",
]
