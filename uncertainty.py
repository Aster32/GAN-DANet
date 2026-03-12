"""Uncertainty calibration utilities for spatial downscaling outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def gaussian_nll(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    sigma = np.maximum(np.asarray(std, dtype=np.float64).reshape(-1), eps)
    nll = 0.5 * np.log(2.0 * np.pi * sigma**2) + ((yt - mu) ** 2) / (2.0 * sigma**2)
    return float(np.mean(nll))


def picp_mpiw(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, level: float) -> Tuple[float, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    sigma = np.asarray(std, dtype=np.float64).reshape(-1)

    z_lookup = {
        0.80: 1.2815515655446004,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    z = z_lookup.get(round(float(level), 2), 1.959963984540054)

    lower = mu - z * sigma
    upper = mu + z * sigma
    covered = (yt >= lower) & (yt <= upper)

    picp = float(np.mean(covered))
    mpiw = float(np.mean(upper - lower))
    return picp, mpiw


def calibration_curve(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    levels: Iterable[float] = (0.80, 0.90, 0.95, 0.99),
) -> Dict[str, List[float]]:
    nominal: List[float] = []
    observed: List[float] = []
    widths: List[float] = []

    for lv in levels:
        picp, mpiw = picp_mpiw(y_true=y_true, mean=mean, std=std, level=float(lv))
        nominal.append(float(lv))
        observed.append(float(picp))
        widths.append(float(mpiw))

    return {
        "nominal": nominal,
        "observed": observed,
        "mpiw": widths,
    }


def interval_calibration_error(curve: Dict[str, List[float]]) -> float:
    nominal = np.asarray(curve["nominal"], dtype=np.float64)
    observed = np.asarray(curve["observed"], dtype=np.float64)
    return float(np.mean(np.abs(nominal - observed)))


def sharpness(std: np.ndarray) -> float:
    sigma = np.asarray(std, dtype=np.float64).reshape(-1)
    return float(np.mean(np.maximum(sigma, 1e-8)))


def save_calibration_plot(curve: Dict[str, List[float]], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    nominal = np.array(curve["nominal"])
    observed = np.array(curve["observed"])

    plt.figure(figsize=(6, 6))
    plt.plot(nominal, observed, marker="o", label="Model coverage")
    plt.plot([0, 1], [0, 1], "--", label="Ideal calibration")
    plt.xlim(0.75, 1.0)
    plt.ylim(0.75, 1.0)
    plt.xlabel("Nominal coverage")
    plt.ylabel("Observed coverage (PICP)")
    plt.title("Uncertainty calibration curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def save_sharpness_plot(curve: Dict[str, List[float]], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    nominal = np.array(curve["nominal"])
    widths = np.array(curve["mpiw"])

    plt.figure(figsize=(7, 4))
    plt.bar([f"{x:.2f}" for x in nominal], widths)
    plt.xlabel("Nominal coverage")
    plt.ylabel("MPIW")
    plt.title("Prediction interval width by coverage level")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def run_uncertainty_from_files(
    target_path: str,
    mean_path: str,
    std_path: str,
    output_dir: str = "outputs/uncertainty",
) -> Dict[str, str]:
    y_true = np.load(target_path)
    y_mean = np.load(mean_path)
    y_std = np.load(std_path)

    curve = calibration_curve(y_true=y_true, mean=y_mean, std=y_std)
    ice = interval_calibration_error(curve)
    nll = gaussian_nll(y_true=y_true, mean=y_mean, std=y_std)
    shp = sharpness(y_std)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    calib_plot_path = out / "calibration_curve.png"
    sharp_plot_path = out / "sharpness_curve.png"
    save_calibration_plot(curve, str(calib_plot_path))
    save_sharpness_plot(curve, str(sharp_plot_path))

    report_path = out / "uncertainty_report.md"
    lines = [
        "# Uncertainty Calibration Report",
        "",
        f"- Interval Calibration Error (ICE): {ice:.6f}",
        f"- Gaussian NLL: {nll:.6f}",
        f"- Sharpness (mean sigma): {shp:.6f}",
        "",
        "| Nominal | Observed (PICP) | MPIW |",
        "|---:|---:|---:|",
    ]
    for n, o, w in zip(curve["nominal"], curve["observed"], curve["mpiw"]):
        lines.append(f"| {n:.2f} | {o:.4f} | {w:.4f} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_json = out / "uncertainty_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "ice": ice,
                "gaussian_nll": nll,
                "sharpness": shp,
                "curve": curve,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "calibration_plot": str(calib_plot_path),
        "sharpness_plot": str(sharp_plot_path),
        "report_md": str(report_path),
        "summary_json": str(summary_json),
    }
