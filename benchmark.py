"""Benchmark and statistical comparison utilities for downscaling experiments."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class BenchmarkResult:
    model: str
    rmse: float
    mae: float
    r2: float
    bias: float
    nse: float
    kge: float


def _flatten(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _paired_valid_arrays(*arrays: np.ndarray) -> list[np.ndarray]:
    flattened = [_flatten(array) for array in arrays]
    valid = np.ones_like(flattened[0], dtype=bool)
    for array in flattened:
        valid &= np.isfinite(array)
    return [array[valid] for array in flattened]


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return num / den


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _paired_valid_arrays(a, b)
    if a.size == 0:
        return float("nan")
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    if std_a == 0.0 or std_b == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BenchmarkResult:
    yt, yp = _paired_valid_arrays(y_true, y_pred)
    if yt.size == 0:
        return BenchmarkResult(model="", rmse=float("nan"), mae=float("nan"), r2=float("nan"), bias=float("nan"), nse=float("nan"), kge=float("nan"))

    diff = yp - yt
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    sst = float(np.sum((yt - np.mean(yt)) ** 2))
    ssr = float(np.sum((yt - yp) ** 2))
    r2 = float(1.0 - _safe_div(ssr, sst))
    nse = float(1.0 - _safe_div(ssr, sst))

    std_t = float(np.std(yt))
    std_p = float(np.std(yp))
    corr = _corr(yt, yp)
    alpha = _safe_div(std_p, std_t)
    mean_t = float(np.mean(yt))
    mean_p = float(np.mean(yp))
    beta = _safe_div(mean_p, mean_t) if mean_t != 0.0 else float("nan")
    if np.isnan(corr) or np.isnan(alpha) or np.isnan(beta):
        kge = float("nan")
    else:
        kge = float(1.0 - np.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

    return BenchmarkResult(model="", rmse=rmse, mae=mae, r2=r2, bias=bias, nse=nse, kge=kge)


def bootstrap_mae_delta(
    y_true: np.ndarray,
    y_baseline: np.ndarray,
    y_candidate: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    max_points: int = 100000,
) -> Dict[str, float]:
    yt, yb, yc = _paired_valid_arrays(y_true, y_baseline, y_candidate)
    if yt.size == 0:
        return {
            "delta_mae_mean": float("nan"),
            "delta_mae_ci_low": float("nan"),
            "delta_mae_ci_high": float("nan"),
            "p_value_two_sided": float("nan"),
        }

    base_err = np.abs(yb - yt)
    cand_err = np.abs(yc - yt)

    rng = np.random.default_rng(seed)
    n = yt.shape[0]
    if n > max_points:
        subset = rng.choice(n, size=max_points, replace=False)
        base_err = base_err[subset]
        cand_err = cand_err[subset]
        n = max_points
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = float(np.mean(cand_err[idx] - base_err[idx]))

    p_two_sided = float(2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
    lo = float(np.quantile(deltas, 0.025))
    hi = float(np.quantile(deltas, 0.975))
    return {
        "delta_mae_mean": float(np.mean(cand_err - base_err)),
        "delta_mae_ci_low": lo,
        "delta_mae_ci_high": hi,
        "p_value_two_sided": p_two_sided,
    }


def bootstrap_corr_delta(
    y_true: np.ndarray,
    y_baseline: np.ndarray,
    y_candidate: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    max_points: int = 100000,
) -> Dict[str, float]:
    yt, yb, yc = _paired_valid_arrays(y_true, y_baseline, y_candidate)
    if yt.size == 0:
        return {
            "delta_corr_mean": float("nan"),
            "delta_corr_ci_low": float("nan"),
            "delta_corr_ci_high": float("nan"),
        }

    rng = np.random.default_rng(seed + 17)
    n = yt.shape[0]
    if n > max_points:
        subset = rng.choice(n, size=max_points, replace=False)
        yt = yt[subset]
        yb = yb[subset]
        yc = yc[subset]
        n = max_points
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = _corr(yt[idx], yc[idx]) - _corr(yt[idx], yb[idx])

    lo = float(np.quantile(deltas, 0.025))
    hi = float(np.quantile(deltas, 0.975))
    return {
        "delta_corr_mean": float(np.nanmean(deltas)),
        "delta_corr_ci_low": lo,
        "delta_corr_ci_high": hi,
    }


def evaluate_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    baseline: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[List[BenchmarkResult], Dict[str, Dict[str, float]]]:
    results: List[BenchmarkResult] = []
    stats: Dict[str, Dict[str, float]] = {}

    if baseline not in predictions:
        raise ValueError(f"Baseline '{baseline}' is not present in predictions")

    for model_name, y_pred in predictions.items():
        metric = regression_metrics(y_true, y_pred)
        metric.model = model_name
        results.append(metric)

        if model_name != baseline:
            mstats = bootstrap_mae_delta(
                y_true=y_true,
                y_baseline=predictions[baseline],
                y_candidate=y_pred,
                n_boot=n_boot,
                seed=seed,
            )
            mstats.update(
                bootstrap_corr_delta(
                    y_true=y_true,
                    y_baseline=predictions[baseline],
                    y_candidate=y_pred,
                    n_boot=n_boot,
                    seed=seed,
                )
            )
            stats[model_name] = mstats

    results.sort(key=lambda x: x.rmse)
    return results, stats


def _to_csv(results: Iterable[BenchmarkResult], output_csv: Path) -> None:
    lines = ["model,rmse,mae,r2,bias,nse,kge"]
    for row in results:
        lines.append(
            f"{row.model},{row.rmse:.6f},{row.mae:.6f},{row.r2:.6f},{row.bias:.6f},{row.nse:.6f},{row.kge:.6f}"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_markdown(results: Iterable[BenchmarkResult], output_md: Path) -> None:
    header = "| Model | RMSE | MAE | R2 | Bias | NSE | KGE |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in results:
        lines.append(
            f"| {row.model} | {row.rmse:.4f} | {row.mae:.4f} | {row.r2:.4f} | {row.bias:.4f} | {row.nse:.4f} | {row.kge:.4f} |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stats_to_markdown(stats: Dict[str, Dict[str, float]], output_md: Path) -> None:
    header = "| Model vs Baseline | Delta MAE | 95% CI(MAE) | p-value | Delta Corr | 95% CI(Corr) |"
    sep = "|---|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for model, vals in stats.items():
        ci_mae = f"[{vals['delta_mae_ci_low']:.4f}, {vals['delta_mae_ci_high']:.4f}]"
        ci_corr = f"[{vals['delta_corr_ci_low']:.4f}, {vals['delta_corr_ci_high']:.4f}]"
        lines.append(
            f"| {model} | {vals['delta_mae_mean']:.4f} | {ci_mae} | {vals['p_value_two_sided']:.4f} | {vals['delta_corr_mean']:.4f} | {ci_corr} |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_json(results: Sequence[BenchmarkResult], stats: Dict[str, Dict[str, float]], output_json: Path) -> None:
    payload = {
        "metrics": [asdict(r) for r in results],
        "significance": stats,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_npy(path: str) -> np.ndarray:
    return np.load(path)


def run_benchmark_from_files(
    target_path: str,
    prediction_paths: Dict[str, str],
    baseline: str,
    output_dir: str = "outputs/benchmark",
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, str]:
    y_true = load_npy(target_path)
    preds = {name: load_npy(path) for name, path in prediction_paths.items()}
    results, stats = evaluate_models(y_true, preds, baseline=baseline, n_boot=n_boot, seed=seed)

    out = Path(output_dir)
    metrics_csv = out / "metrics.csv"
    metrics_md = out / "metrics.md"
    stats_md = out / "significance.md"
    summary_json = out / "summary.json"

    _to_csv(results, metrics_csv)
    _to_markdown(results, metrics_md)
    _stats_to_markdown(stats, stats_md)
    _save_json(results, stats, summary_json)

    return {
        "metrics_csv": str(metrics_csv),
        "metrics_md": str(metrics_md),
        "significance_md": str(stats_md),
        "summary_json": str(summary_json),
    }
