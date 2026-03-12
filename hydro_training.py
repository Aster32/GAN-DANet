"""Training helpers for the advanced hydrology-aware TWSA downscaler."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from benchmark import run_benchmark_from_files
from hydro_datasets import build_hydrology_dataloaders, build_rolling_origin_dataloaders
from models import HydrologyLossBundle, HydroTWSADownscaler
from reporting import run_report_from_files
from uncertainty import run_uncertainty_from_files


@dataclass
class HydroTrainingConfig:
    batch_size: int = 2
    window_size: int = 5
    aux_channels: int = 40
    static_channels: int = 3
    base_channels: int = 64
    dropout: float = 0.1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    val_fraction: float = 0.2
    num_workers: int = 0
    seed: int = 42
    epochs: int = 20
    early_stopping_patience: int = 6
    output_dir: str = "outputs/hydro"
    initial_train_size: int = 72
    validation_size: int = 12
    step_size: int = 12
    reconstruction_weight: float = 1.0
    ssim_weight: float = 0.15
    conservation_weight: float = 0.35
    gradient_weight: float = 0.2
    uncertainty_weight: float = 0.05
    anomaly_weight: float = 0.2
    trend_weight: float = 0.2
    tv_weight: float = 1e-4


def load_hydro_config(path: str) -> HydroTrainingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    dataset = raw.get("dataset", {})
    model = raw.get("model", {})
    optimization = raw.get("optimization", {})
    loss = raw.get("loss", {})
    training = raw.get("training", {})

    return HydroTrainingConfig(
        seed=raw.get("seed", 42),
        batch_size=dataset.get("batch_size", 2),
        window_size=dataset.get("window_size", 5),
        val_fraction=dataset.get("val_fraction", 0.2),
        static_channels=dataset.get("static_channels", 3),
        aux_channels=model.get("aux_channels", 40),
        base_channels=model.get("base_channels", 64),
        dropout=model.get("dropout", 0.1),
        learning_rate=optimization.get("learning_rate", 2e-4),
        weight_decay=optimization.get("weight_decay", 1e-4),
        epochs=training.get("epochs", 20),
        early_stopping_patience=training.get("early_stopping_patience", 6),
        output_dir=training.get("output_dir", "outputs/hydro"),
        initial_train_size=training.get("initial_train_size", 72),
        validation_size=training.get("validation_size", 12),
        step_size=training.get("step_size", 12),
        reconstruction_weight=loss.get("reconstruction_weight", 1.0),
        ssim_weight=loss.get("ssim_weight", 0.15),
        conservation_weight=loss.get("conservation_weight", 0.35),
        gradient_weight=loss.get("gradient_weight", 0.2),
        uncertainty_weight=loss.get("uncertainty_weight", 0.05),
        anomaly_weight=loss.get("anomaly_weight", 0.2),
        trend_weight=loss.get("trend_weight", 0.2),
        tv_weight=loss.get("tv_weight", 1e-4),
    )


def synthetic_hydro_batch(
    batch_size: int = 2,
    window_size: int = 5,
    aux_channels: int = 40,
    static_channels: int = 3,
    device: str | torch.device = "cpu",
) -> Dict[str, torch.Tensor]:
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if static_channels >= aux_channels:
        raise ValueError("static_channels must be smaller than aux_channels")

    device = torch.device(device)
    lr_anomaly_seq = torch.randn(batch_size, window_size, 1, 44, 90, device=device)
    lr_trend_seq = torch.randn(batch_size, window_size, 1, 44, 90, device=device) * 0.25
    aux_seq = torch.randn(batch_size, window_size, aux_channels, 88, 180, device=device)

    center = window_size // 2
    coarse_target = lr_anomaly_seq[:, center] + lr_trend_seq[:, center]
    target_anomaly = torch.randn(batch_size, 1, 88, 180, device=device)
    target_trend = torch.randn(batch_size, 1, 88, 180, device=device) * 0.25
    mask = torch.ones(batch_size, 1, 88, 180, device=device)
    terrain_weight = torch.ones(batch_size, 1, 88, 180, device=device)

    return {
        "lr_anomaly_seq": lr_anomaly_seq,
        "lr_trend_seq": lr_trend_seq,
        "aux_seq": aux_seq,
        "target_anomaly": target_anomaly,
        "target_trend": target_trend,
        "target": target_anomaly + target_trend,
        "coarse_target": coarse_target,
        "mask": mask,
        "terrain_weight": terrain_weight,
    }


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_criterion(config: HydroTrainingConfig) -> HydrologyLossBundle:
    return HydrologyLossBundle(
        reconstruction_weight=config.reconstruction_weight,
        ssim_weight=config.ssim_weight,
        conservation_weight=config.conservation_weight,
        gradient_weight=config.gradient_weight,
        uncertainty_weight=config.uncertainty_weight,
        anomaly_weight=config.anomaly_weight,
        trend_weight=config.trend_weight,
        tv_weight=config.tv_weight,
    )


def build_hydro_model(config: HydroTrainingConfig) -> HydroTWSADownscaler:
    return HydroTWSADownscaler(
        aux_channels=config.aux_channels,
        static_channels=config.static_channels,
        base_channels=config.base_channels,
        dropout=config.dropout,
    )


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_hydrology_loss(
    model: nn.Module,
    criterion: HydrologyLossBundle,
    batch: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    outputs = model(
        lr_anomaly_seq=batch["lr_anomaly_seq"],
        aux_seq=batch["aux_seq"],
        lr_trend_seq=batch.get("lr_trend_seq"),
        mask=batch.get("mask"),
    )
    losses = criterion(
        outputs=outputs,
        target=batch["target"],
        coarse_target=batch["coarse_target"],
        target_anomaly=batch.get("target_anomaly"),
        target_trend=batch.get("target_trend"),
        mask=batch.get("mask"),
        terrain_weight=batch.get("terrain_weight"),
    )
    return outputs, losses


def run_hydro_smoke(config: HydroTrainingConfig) -> Dict[str, object]:
    device = _resolve_device(config.device)
    model = build_hydro_model(config).to(device)
    criterion = _build_criterion(config).to(device)

    batch = synthetic_hydro_batch(
        batch_size=config.batch_size,
        window_size=config.window_size,
        aux_channels=config.aux_channels,
        static_channels=config.static_channels,
        device=device,
    )
    outputs, losses = compute_hydrology_loss(model, criterion, batch)

    return {
        "device": str(device),
        "mean_shape": tuple(outputs["mean"].shape),
        "trend_shape": tuple(outputs["trend"].shape),
        "coarse_shape": tuple(outputs["coarse"].shape),
        "loss": float(losses["loss"].detach().cpu()),
        "conservation": float(losses["conservation"].detach().cpu()),
        "gradient": float(losses["gradient"].detach().cpu()),
    }


def run_hydro_train_step(config: HydroTrainingConfig) -> Dict[str, float]:
    device = _resolve_device(config.device)
    model = build_hydro_model(config).to(device)
    criterion = _build_criterion(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    batch = synthetic_hydro_batch(
        batch_size=config.batch_size,
        window_size=config.window_size,
        aux_channels=config.aux_channels,
        static_channels=config.static_channels,
        device=device,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, losses = compute_hydrology_loss(model, criterion, batch)
    losses["loss"].backward()
    optimizer.step()

    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: HydrologyLossBundle,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        _, losses = compute_hydrology_loss(model, criterion, batch)
        losses["loss"].backward()
        optimizer.step()

        count += 1
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: HydrologyLossBundle,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        _, losses = compute_hydrology_loss(model, criterion, batch)
        count += 1
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    collected: Dict[str, list[np.ndarray]] = {
        "target": [],
        "mean": [],
        "std": [],
        "coarse_target": [],
        "coarse_pred": [],
        "anomaly": [],
        "trend": [],
    }

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            lr_anomaly_seq=batch["lr_anomaly_seq"],
            aux_seq=batch["aux_seq"],
            lr_trend_seq=batch.get("lr_trend_seq"),
            mask=batch.get("mask"),
        )
        collected["target"].append(batch["target"].detach().cpu().numpy())
        collected["mean"].append(outputs["mean"].detach().cpu().numpy())
        collected["std"].append(torch.exp(0.5 * outputs["logvar"]).detach().cpu().numpy())
        collected["coarse_target"].append(batch["coarse_target"].detach().cpu().numpy())
        collected["coarse_pred"].append(outputs["coarse"].detach().cpu().numpy())
        collected["anomaly"].append(outputs["anomaly"].detach().cpu().numpy())
        collected["trend"].append(outputs["trend"].detach().cpu().numpy())

    return {key: np.concatenate(value, axis=0) for key, value in collected.items()}


def _save_prediction_bundle(predictions: Dict[str, np.ndarray], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for key, value in predictions.items():
        path = output_dir / f"{key}.npy"
        np.save(path, value)
        paths[key] = str(path)
    return paths


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: HydroTrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": asdict(config),
        },
        path,
    )


def _artifact_summary(metrics: Dict[str, Any], history: list[Dict[str, Any]], output_dir: Path) -> str:
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps({"metrics": metrics, "history": history}, indent=2),
        encoding="utf-8",
    )
    return str(summary_path)


def _train_with_loaders(
    config: HydroTrainingConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    run_name: str,
) -> Dict[str, Any]:
    model = build_hydro_model(config).to(device)
    criterion = _build_criterion(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))

    best_val_loss = float("inf")
    best_epoch = -1
    patience = 0
    history: list[Dict[str, Any]] = []
    best_artifacts: Dict[str, Any] = {}

    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        epoch_record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            patience = 0

            _save_checkpoint(output_dir / f"{run_name}_best.pt", model, optimizer, best_epoch, best_val_loss, config)

            predictions = predict_loader(model, val_loader, device)
            prediction_paths = _save_prediction_bundle(predictions, output_dir / "predictions")
            benchmark_outputs = run_benchmark_from_files(
                target_path=prediction_paths["target"],
                prediction_paths={run_name: prediction_paths["mean"]},
                baseline=run_name,
                output_dir=str(output_dir / "benchmark"),
                seed=config.seed,
            )
            uncertainty_outputs = run_uncertainty_from_files(
                target_path=prediction_paths["target"],
                mean_path=prediction_paths["mean"],
                std_path=prediction_paths["std"],
                output_dir=str(output_dir / "uncertainty"),
            )
            report_outputs = run_report_from_files(
                target_path=prediction_paths["target"],
                pred_path=prediction_paths["mean"],
                metrics_csv=benchmark_outputs["metrics_csv"],
                output_dir=str(output_dir / "report"),
                model_name=run_name,
                uncertainty_summary_json=uncertainty_outputs["summary_json"],
            )
            best_artifacts = {
                "checkpoint": str(output_dir / f"{run_name}_best.pt"),
                "predictions": prediction_paths,
                "benchmark": benchmark_outputs,
                "uncertainty": uncertainty_outputs,
                "report": report_outputs,
            }
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    _save_checkpoint(output_dir / f"{run_name}_last.pt", model, optimizer, len(history), best_val_loss, config)

    summary = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history_path": str(history_path),
        "artifacts": best_artifacts,
    }
    summary["summary_json"] = _artifact_summary(summary, history, output_dir)
    return summary


def fit_hydro_model(config: HydroTrainingConfig) -> Dict[str, Any]:
    train_loader, val_loader, aux_channels = build_hydrology_dataloaders(
        batch_size=config.batch_size,
        window_size=config.window_size,
        val_fraction=config.val_fraction,
        num_workers=config.num_workers,
        seed=config.seed,
        static_channels=config.static_channels,
    )
    config.aux_channels = aux_channels
    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return _train_with_loaders(config, train_loader, val_loader, device, output_dir, "hydro_twsa")


def run_rolling_origin_experiment(config: HydroTrainingConfig) -> Dict[str, Any]:
    device = _resolve_device(config.device)
    fold_loaders = build_rolling_origin_dataloaders(
        batch_size=config.batch_size,
        window_size=config.window_size,
        initial_train_size=config.initial_train_size,
        validation_size=config.validation_size,
        step_size=config.step_size,
        num_workers=config.num_workers,
        seed=config.seed,
        static_channels=config.static_channels,
    )

    output_dir = Path(config.output_dir) / "rolling_origin"
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_summaries: list[Dict[str, Any]] = []
    for fold_name, train_loader, val_loader, aux_channels in fold_loaders:
        fold_config = HydroTrainingConfig(**asdict(config))
        fold_config.aux_channels = aux_channels
        fold_config.output_dir = str(output_dir / fold_name)
        fold_summary = _train_with_loaders(
            fold_config,
            train_loader,
            val_loader,
            device,
            Path(fold_config.output_dir),
            run_name=fold_name,
        )
        fold_summaries.append(fold_summary)

    best_losses = [fold["best_val_loss"] for fold in fold_summaries]
    aggregate = {
        "folds": fold_summaries,
        "mean_best_val_loss": float(np.mean(best_losses)) if best_losses else float("nan"),
        "std_best_val_loss": float(np.std(best_losses)) if best_losses else float("nan"),
    }
    summary_path = output_dir / "rolling_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    aggregate["summary_json"] = str(summary_path)
    return aggregate


def build_real_data_training_objects(
    config: HydroTrainingConfig,
) -> tuple[HydroTWSADownscaler, HydrologyLossBundle, torch.optim.Optimizer, tuple]:
    train_loader, val_loader, aux_channels = build_hydrology_dataloaders(
        batch_size=config.batch_size,
        window_size=config.window_size,
        val_fraction=config.val_fraction,
        num_workers=config.num_workers,
        seed=config.seed,
        static_channels=config.static_channels,
    )
    config.aux_channels = aux_channels
    device = _resolve_device(config.device)
    model = build_hydro_model(config).to(device)
    criterion = _build_criterion(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    return model, criterion, optimizer, (train_loader, val_loader, device)


def load_hydro_checkpoint(
    checkpoint_path: str,
    device: Optional[str | torch.device] = None,
) -> tuple[HydroTWSADownscaler, HydroTrainingConfig, Dict[str, Any]]:
    resolved_device = torch.device(device) if device is not None else _resolve_device("cuda")
    payload = torch.load(checkpoint_path, map_location=resolved_device)
    config_dict = payload.get("config", {})
    config = HydroTrainingConfig(**config_dict)
    config.device = str(resolved_device)
    model = build_hydro_model(config).to(resolved_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config, payload


__all__ = [
    "HydroTrainingConfig",
    "build_hydro_model",
    "build_real_data_training_objects",
    "compute_hydrology_loss",
    "evaluate_one_epoch",
    "fit_hydro_model",
    "load_hydro_checkpoint",
    "load_hydro_config",
    "predict_loader",
    "run_hydro_smoke",
    "run_hydro_train_step",
    "run_rolling_origin_experiment",
    "synthetic_hydro_batch",
    "train_one_epoch",
]
