"""Notebook-friendly hydrology workflows that preserve the original train/test entrypoints."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence

import h5py
from netCDF4 import Dataset, date2num
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, zoom
import torch

from benchmark import run_benchmark_from_files
from hydro_datasets import build_full_hydrology_dataloader, load_hydrology_arrays
from hydro_training import HydroTrainingConfig, fit_hydro_model, load_hydro_checkpoint, load_hydro_config, predict_loader
from reporting import run_report_from_files
from repro import seed_everything
from uncertainty import run_uncertainty_from_files


def load_notebook_config(
    config_path: str = "hydro_experiment_config.yml",
    output_dir: str = "outputs/notebooks/train",
) -> HydroTrainingConfig:
    config = load_hydro_config(config_path)
    config.output_dir = output_dir
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    return config


def _save_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _load_mask(*candidate_paths: str) -> np.ndarray | None:
    for candidate in candidate_paths:
        path = Path(candidate)
        if path.exists():
            return np.load(path)
    return None


def _squeeze_spatial(arr: np.ndarray) -> np.ndarray:
    squeezed = np.asarray(arr)
    if squeezed.ndim == 4 and squeezed.shape[1] == 1:
        squeezed = squeezed[:, 0]
    return squeezed


def _inverse_standardize(arr: np.ndarray, scaler: object) -> np.ndarray:
    original_shape = arr.shape
    flat = np.asarray(arr, dtype=np.float32).reshape(-1, 1)
    restored = scaler.inverse_transform(flat)
    return restored.reshape(original_shape)


def _scale_from_standardizer(scaler: object) -> float:
    return float(np.asarray(scaler.scale_).reshape(-1)[0])


def _monthly_time_index(length: int) -> pd.DatetimeIndex:
    return pd.date_range(start="2002-08-01", periods=length, freq="MS")


def _member_alias(index: int) -> str:
    return f"model1{index}_hydro_downscaler.pt"


def _write_grid_netcdf(
    path: Path,
    data: np.ndarray,
    uncertainty: np.ndarray | None,
    resolution_deg: float,
    lat_start: float,
    lon_start: float,
    units: str,
    description: str,
    model_name: str,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    time_values = list(_monthly_time_index(data.shape[0]).to_pydatetime())
    lat_values = lat_start + np.arange(data.shape[1], dtype=np.float32) * resolution_deg
    lon_values = lon_start + np.arange(data.shape[2], dtype=np.float32) * resolution_deg

    with Dataset(path, "w", format="NETCDF4") as handle:
        handle.createDimension("time", data.shape[0])
        handle.createDimension("lat", data.shape[1])
        handle.createDimension("lon", data.shape[2])

        times = handle.createVariable("time", "f4", ("time",))
        lats = handle.createVariable("lat", "f4", ("lat",))
        lons = handle.createVariable("lon", "f4", ("lon",))
        values = handle.createVariable("data", "f4", ("time", "lat", "lon"), zlib=True)

        times[:] = date2num(time_values, units="days since 2002-08-01", calendar="standard")
        lats[:] = lat_values
        lons[:] = lon_values
        values[:] = data.astype(np.float32)

        if uncertainty is not None:
            unc_var = handle.createVariable("uncertainty", "f4", ("time", "lat", "lon"), zlib=True)
            unc_var[:] = uncertainty.astype(np.float32)
            unc_var.units = units
            unc_var.description = "Predictive uncertainty"

        times.units = "days since 2002-08-01"
        times.calendar = "standard"
        lats.units = "degrees_north"
        lons.units = "degrees_east"
        values.units = units
        values.description = description
        handle.model = model_name
        handle.date_created = datetime.utcnow().strftime("%Y-%m-%d")

    return str(path)


def train_notebook_ensemble(
    config_path: str = "hydro_experiment_config.yml",
    output_dir: str = "outputs/notebooks/train",
    seeds: Sequence[int] = (42, 26),
    epochs: int | None = None,
) -> dict[str, Any]:
    base_config = load_notebook_config(config_path=config_path, output_dir=output_dir)
    if epochs is not None:
        base_config.epochs = epochs

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    members: list[dict[str, Any]] = []
    for member_index, seed in enumerate(seeds, start=1):
        member_config = HydroTrainingConfig(**asdict(base_config))
        member_config.seed = int(seed)
        member_config.output_dir = str(output_root / f"member_{member_index:02d}_seed{seed}")
        seed_everything(seed=member_config.seed, deterministic=True)
        member_summary = fit_hydro_model(member_config)

        checkpoint_path = Path(member_summary["artifacts"]["checkpoint"])
        alias_path = output_root / _member_alias(member_index)
        shutil.copy2(checkpoint_path, alias_path)

        members.append(
            {
                "member_index": member_index,
                "seed": member_config.seed,
                "checkpoint": str(checkpoint_path),
                "compat_checkpoint": str(alias_path),
                "summary": member_summary,
            }
        )

    manifest = {
        "workflow": "GAN-DANet notebook ensemble training",
        "config_path": config_path,
        "base_config": asdict(base_config),
        "members": members,
    }
    manifest_path = output_root / "ensemble_manifest.json"
    manifest["manifest_path"] = _save_json(manifest_path, manifest)
    return manifest


def load_notebook_manifest(manifest_path: str = "outputs/notebooks/train/ensemble_manifest.json") -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def _checkpoint_list_from_manifest(manifest_path: str | None, checkpoints: Sequence[str] | None) -> list[str]:
    if checkpoints:
        return [str(Path(path)) for path in checkpoints]
    if manifest_path is None:
        raise ValueError("Either manifest_path or checkpoints must be provided")
    manifest = load_notebook_manifest(manifest_path)
    return [member.get("compat_checkpoint", member["checkpoint"]) for member in manifest["members"]]


def _ensemble_predict(checkpoints: Sequence[str], batch_size: int = 4) -> tuple[dict[str, np.ndarray], HydroTrainingConfig]:
    model, config, _ = load_hydro_checkpoint(checkpoints[0])
    loader, _, _ = build_full_hydrology_dataloader(
        batch_size=batch_size,
        window_size=config.window_size,
        static_channels=config.static_channels,
    )
    first_predictions = predict_loader(model, loader, torch.device(config.device))

    member_means = [_squeeze_spatial(first_predictions["mean"])]
    member_variances = [_squeeze_spatial(first_predictions["std"]) ** 2]
    target = _squeeze_spatial(first_predictions["target"])
    anomaly = [_squeeze_spatial(first_predictions["anomaly"])]
    trend = [_squeeze_spatial(first_predictions["trend"])]
    coarse_pred = [_squeeze_spatial(first_predictions["coarse_pred"])]
    coarse_target = _squeeze_spatial(first_predictions["coarse_target"])

    for checkpoint in checkpoints[1:]:
        member_model, member_config, _ = load_hydro_checkpoint(checkpoint, device=config.device)
        if member_config.window_size != config.window_size or member_config.static_channels != config.static_channels:
            raise ValueError("All ensemble members must share window_size and static_channels")
        predictions = predict_loader(member_model, loader, torch.device(config.device))
        member_means.append(_squeeze_spatial(predictions["mean"]))
        member_variances.append(_squeeze_spatial(predictions["std"]) ** 2)
        anomaly.append(_squeeze_spatial(predictions["anomaly"]))
        trend.append(_squeeze_spatial(predictions["trend"]))
        coarse_pred.append(_squeeze_spatial(predictions["coarse_pred"]))

    mean_stack = np.stack(member_means, axis=0)
    var_stack = np.stack(member_variances, axis=0)
    anomaly_stack = np.stack(anomaly, axis=0)
    trend_stack = np.stack(trend, axis=0)
    coarse_pred_stack = np.stack(coarse_pred, axis=0)

    outputs = {
        "mean": mean_stack.mean(axis=0),
        "std": np.sqrt(np.maximum(var_stack.mean(axis=0) + mean_stack.var(axis=0), 0.0)),
        "target": target,
        "anomaly": anomaly_stack.mean(axis=0),
        "trend": trend_stack.mean(axis=0),
        "coarse_pred": coarse_pred_stack.mean(axis=0),
        "coarse_target": coarse_target,
        "epistemic_std": mean_stack.std(axis=0),
    }
    return outputs, config


def run_notebook_stage1_inference(
    manifest_path: str = "outputs/notebooks/train/ensemble_manifest.json",
    checkpoints: Sequence[str] | None = None,
    output_dir: str = "outputs/notebooks/test_025",
    batch_size: int = 4,
) -> dict[str, Any]:
    checkpoint_paths = _checkpoint_list_from_manifest(manifest_path, checkpoints)
    arrays = load_hydrology_arrays()
    predictions, config = _ensemble_predict(checkpoint_paths, batch_size=batch_size)

    pred025_native = _inverse_standardize(predictions["mean"], arrays.grace_scaler_025)
    target025_native = _inverse_standardize(predictions["target"], arrays.grace_scaler_025)
    std025_native = predictions["std"] * _scale_from_standardizer(arrays.grace_scaler_025)

    coarse_native = _inverse_standardize(arrays.lr_anomaly + arrays.lr_trend, arrays.grace_scaler_05)
    bicubic025 = zoom(coarse_native, (1, 2, 2), order=3)

    mask025 = _load_mask("tpb_h.npy", "cache/tpb_h.npy")
    pred025_export = pred025_native
    target025_export = target025_native
    std025_export = std025_native
    bicubic025_export = bicubic025
    if mask025 is not None:
        pred025_export = np.where(mask025[None] > 0, pred025_export, np.nan)
        target025_export = np.where(mask025[None] > 0, target025_export, np.nan)
        std025_export = np.where(mask025[None] > 0, std025_export, np.nan)
        bicubic025_export = np.where(mask025[None] > 0, bicubic025_export, np.nan)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pred_path = output_root / "pred025_mean.npy"
    target_path = output_root / "target025.npy"
    std_path = output_root / "pred025_std.npy"
    bicubic_path = output_root / "bicubic025.npy"
    np.save(pred_path, pred025_native.astype(np.float32))
    np.save(target_path, target025_native.astype(np.float32))
    np.save(std_path, std025_native.astype(np.float32))
    np.save(bicubic_path, bicubic025.astype(np.float32))
    np.save(output_root / "ensemble_uncertainty.npy", std025_native.astype(np.float32))

    with h5py.File(output_root / "grace025.h5", "w") as handle:
        handle.create_dataset("data", data=target025_export.astype(np.float32))
    with h5py.File(output_root / "valid.h5", "w") as handle:
        handle.create_dataset("data", data=pred025_export.astype(np.float32))
    with h5py.File(output_root / "biash.h5", "w") as handle:
        handle.create_dataset("data", data=(target025_export - pred025_export).astype(np.float32))

    timeseries = pd.DataFrame(
        {
            "date": _monthly_time_index(len(pred025_native)),
            "grace025_mean": np.nanmean(target025_export, axis=(1, 2)),
            "bicubic025_mean": np.nanmean(bicubic025_export, axis=(1, 2)),
            "hydro025_mean": np.nanmean(pred025_export, axis=(1, 2)),
            "hydro025_std_mean": np.nanmean(std025_export, axis=(1, 2)),
        }
    )
    timeseries_path = output_root / "timeseries_025.csv"
    timeseries.to_csv(timeseries_path, index=False)

    benchmark_outputs = run_benchmark_from_files(
        target_path=str(target_path),
        prediction_paths={
            "bicubic025": str(bicubic_path),
            "hydro_ensemble_025": str(pred_path),
        },
        baseline="bicubic025",
        output_dir=str(output_root / "benchmark"),
        n_boot=400,
        seed=config.seed,
    )
    uncertainty_outputs = run_uncertainty_from_files(
        target_path=str(target_path),
        mean_path=str(pred_path),
        std_path=str(std_path),
        output_dir=str(output_root / "uncertainty"),
    )
    report_outputs = run_report_from_files(
        target_path=str(target_path),
        pred_path=str(pred_path),
        metrics_csv=benchmark_outputs["metrics_csv"],
        output_dir=str(output_root / "report"),
        model_name="HydroGAN-DANet Ensemble (0.25 deg)",
        uncertainty_summary_json=uncertainty_outputs["summary_json"],
    )

    netcdf_path = _write_grid_netcdf(
        path=output_root / "downscaled_grace_tws_025.nc",
        data=pred025_export,
        uncertainty=std025_export,
        resolution_deg=0.25,
        lat_start=24.125,
        lon_start=65.125,
        units="native",
        description="Hydrology-aware ensemble downscaled TWSA at 0.25 degree resolution",
        model_name="HydroGAN-DANet Ensemble",
    )

    summary = {
        "manifest_path": manifest_path,
        "checkpoints": checkpoint_paths,
        "prediction_path": str(pred_path),
        "target_path": str(target_path),
        "uncertainty_path": str(std_path),
        "bicubic_path": str(bicubic_path),
        "timeseries_csv": str(timeseries_path),
        "benchmark": benchmark_outputs,
        "uncertainty": uncertainty_outputs,
        "report": report_outputs,
        "netcdf": netcdf_path,
        "summary_json": str(output_root / "stage1_summary.json"),
    }
    _save_json(Path(summary["summary_json"]), summary)
    return summary


def terrain_guided_refinement(
    pred025: np.ndarray,
    hr_aux: np.ndarray,
    static_channels: int = 3,
    upscale_factor: int = 5,
    detail_weight: float = 0.2,
    smooth_sigma: float = 1.2,
) -> np.ndarray:
    refined_base = zoom(pred025, (1, upscale_factor, upscale_factor), order=3)

    aux_channels = hr_aux.shape[-1]
    if static_channels > 0:
        dynamic_aux = hr_aux[..., : aux_channels - static_channels]
        dem = hr_aux[0, :, :, -1]
    else:
        dynamic_aux = hr_aux
        dem = np.zeros(hr_aux.shape[1:3], dtype=np.float32)

    dem_high = zoom(dem, (upscale_factor, upscale_factor), order=3)
    dem_gy, dem_gx = np.gradient(dem_high)
    slope = np.sqrt(dem_gx ** 2 + dem_gy ** 2)
    slope = slope - np.nanmin(slope)
    slope = slope / max(float(np.nanmax(slope)), 1.0)

    dynamic_intensity = np.mean(np.abs(dynamic_aux), axis=-1)
    dynamic_high = zoom(dynamic_intensity, (1, upscale_factor, upscale_factor), order=3)
    dynamic_high = dynamic_high - np.nanmin(dynamic_high, axis=(1, 2), keepdims=True)
    dynamic_scale = np.nanmax(dynamic_high, axis=(1, 2), keepdims=True)
    dynamic_high = dynamic_high / np.maximum(dynamic_scale, 1e-6)

    anomaly_detail = refined_base - gaussian_filter(refined_base, sigma=(0.0, smooth_sigma, smooth_sigma))
    terrain_gate = 0.6 * slope[None] + 0.4 * dynamic_high
    refined = refined_base + detail_weight * terrain_gate * anomaly_detail

    base_mean = np.nanmean(refined_base, axis=(1, 2), keepdims=True)
    refined_mean = np.nanmean(refined, axis=(1, 2), keepdims=True)
    refined = refined + (base_mean - refined_mean)
    return refined.astype(np.float32)


def run_notebook_stage2_refinement(
    stage1_dir: str = "outputs/notebooks/test_025",
    output_dir: str = "outputs/notebooks/test_005",
    static_channels: int = 3,
    upscale_factor: int = 5,
    detail_weight: float = 0.2,
    smooth_sigma: float = 1.2,
    unit_scale: float = 10.0,
    unit_label: str = "cm",
) -> dict[str, Any]:
    stage1_root = Path(stage1_dir)
    pred025 = np.load(stage1_root / "pred025_mean.npy")
    std025 = np.load(stage1_root / "pred025_std.npy")
    target025 = np.load(stage1_root / "target025.npy")
    arrays = load_hydrology_arrays()

    refined005_native = terrain_guided_refinement(
        pred025=pred025,
        hr_aux=arrays.hr_aux,
        static_channels=static_channels,
        upscale_factor=upscale_factor,
        detail_weight=detail_weight,
        smooth_sigma=smooth_sigma,
    )
    uncertainty005_native = zoom(std025, (1, upscale_factor, upscale_factor), order=0)

    mask025 = _load_mask("tpb_h.npy", "cache/tpb_h.npy")
    mask005 = zoom(mask025, (upscale_factor, upscale_factor), order=0) if mask025 is not None else None
    if mask005 is not None:
        refined005_native = np.where(mask005[None] > 0, refined005_native, np.nan)
        uncertainty005_native = np.where(mask005[None] > 0, uncertainty005_native, np.nan)

    refined005_units = refined005_native * unit_scale
    uncertainty005_units = uncertainty005_native * unit_scale
    target025_units = target025 * unit_scale
    pred025_units = pred025 * unit_scale

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pred_path = output_root / "pred005_mean.npy"
    std_path = output_root / "pred005_std.npy"
    np.save(pred_path, refined005_units.astype(np.float32))
    np.save(std_path, uncertainty005_units.astype(np.float32))

    with h5py.File(output_root / "raw_downscaled.h5", "w") as handle:
        handle.create_dataset("data", data=refined005_units.astype(np.float32))
    with h5py.File(output_root / "downscaled.h5", "w") as handle:
        handle.create_dataset("data", data=refined005_units.astype(np.float32))

    timeseries = pd.DataFrame(
        {
            "date": _monthly_time_index(len(refined005_units)),
            "GRACE025": np.nanmean(target025_units, axis=(1, 2)),
            "Downscaled025": np.nanmean(pred025_units, axis=(1, 2)),
            "Downscaled005": np.nanmean(refined005_units, axis=(1, 2)),
            "uncertainty_025": np.nanmean(std025 * unit_scale, axis=(1, 2)),
            "uncertainty_005": np.nanmean(uncertainty005_units, axis=(1, 2)),
        }
    )
    timeseries_path = output_root / "timeseries_tp.csv"
    timeseries.to_csv(timeseries_path, index=False)

    netcdf_path = _write_grid_netcdf(
        path=output_root / "downscaled_grace_tws_005.nc",
        data=refined005_units,
        uncertainty=uncertainty005_units,
        resolution_deg=0.05,
        lat_start=24.025,
        lon_start=65.025,
        units=unit_label,
        description="Terrain-guided 0.05 degree TWSA refinement from the 0.25 degree hydrology ensemble",
        model_name="HydroGAN-DANet Ensemble",
    )

    summary = {
        "stage1_dir": str(stage1_root),
        "prediction_path": str(pred_path),
        "uncertainty_path": str(std_path),
        "timeseries_csv": str(timeseries_path),
        "netcdf": netcdf_path,
        "summary_json": str(output_root / "stage2_summary.json"),
    }
    _save_json(Path(summary["summary_json"]), summary)
    return summary


__all__ = [
    "load_notebook_config",
    "load_notebook_manifest",
    "run_notebook_stage1_inference",
    "run_notebook_stage2_refinement",
    "terrain_guided_refinement",
    "train_notebook_ensemble",
]
