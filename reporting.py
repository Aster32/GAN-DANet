"""Reporting helpers for manuscript-ready tables and figures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _paired_valid_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(yt) & np.isfinite(yp)
    return yt[valid], yp[valid]


def _load_csv(path: Path) -> List[Dict[str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split(",")
    rows: List[Dict[str, str]] = []
    for line in lines[1:]:
        vals = line.split(",")
        rows.append(dict(zip(header, vals)))
    return rows


def metrics_csv_to_latex(metrics_csv: str, output_tex: str) -> None:
    rows = _load_csv(Path(metrics_csv))
    out = Path(output_tex)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Benchmark metrics for GRACE downscaling}",
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Model & RMSE & MAE & R2 & Bias & NSE & KGE \\\\ ",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {float(row['rmse']):.4f} & {float(row['mae']):.4f} & "
            f"{float(row['r2']):.4f} & {float(row['bias']):.4f} & {float(row['nse']):.4f} & {float(row['kge']):.4f} \\\\ "
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_scatter_figure(y_true: np.ndarray, y_pred: np.ndarray, output_path: str, title: str) -> None:
    yt, yp = _paired_valid_arrays(y_true, y_pred)
    if yt.size == 0:
        return

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.hexbin(yt, yp, gridsize=40, mincnt=1)
    mn = float(min(np.min(yt), np.min(yp)))
    mx = float(max(np.max(yt), np.max(yp)))
    plt.plot([mn, mx], [mn, mx], "--", linewidth=1)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def save_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, output_path: str, title: str) -> None:
    yt, yp = _paired_valid_arrays(y_true, y_pred)
    if yt.size == 0:
        return
    residual = yp - yt
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(residual, bins=50)
    plt.xlabel("Residual (Pred - Obs)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def save_spatial_snapshot_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    title: str,
    time_index: int = 0,
) -> None:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim < 2 or yp.ndim < 2:
        return

    if yt.ndim == 3:
        idx = max(0, min(time_index, yt.shape[0] - 1))
        obs = yt[idx]
        pred = yp[idx]
    else:
        obs = yt
        pred = yp

    valid = np.isfinite(obs) & np.isfinite(pred)
    if not np.any(valid):
        return

    residual = pred - obs
    vmin = float(min(np.min(obs[valid]), np.min(pred[valid])))
    vmax = float(max(np.max(obs[valid]), np.max(pred[valid])))
    rmax = float(np.max(np.abs(residual[valid])))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    im0 = axes[0].imshow(obs, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("Observed")
    axes[0].axis("off")

    im1 = axes[1].imshow(pred, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("Predicted")
    axes[1].axis("off")

    im2 = axes[2].imshow(residual, vmin=-rmax, vmax=rmax, cmap="coolwarm")
    axes[2].set_title("Residual")
    axes[2].axis("off")

    fig.suptitle(title)
    fig.colorbar(im0, ax=axes[:2], fraction=0.045)
    fig.colorbar(im2, ax=axes[2], fraction=0.045)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)


def run_report_from_files(
    target_path: str,
    pred_path: str,
    metrics_csv: str,
    output_dir: str = "outputs/report",
    model_name: str = "GAN-DANet",
    uncertainty_summary_json: str = "",
) -> Dict[str, str]:
    y_true = np.load(target_path)
    y_pred = np.load(pred_path)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scatter = out / "figure_scatter.png"
    residual = out / "figure_residual_hist.png"
    spatial = out / "figure_spatial_snapshot.png"
    latex = out / "table_metrics.tex"

    save_scatter_figure(y_true, y_pred, str(scatter), title=f"{model_name} vs Observation")
    save_residual_hist(y_true, y_pred, str(residual), title=f"{model_name} residual distribution")
    save_spatial_snapshot_figure(y_true, y_pred, str(spatial), title=f"{model_name} spatial snapshot")
    metrics_csv_to_latex(metrics_csv, str(latex))

    rows = _load_csv(Path(metrics_csv))
    top_row = rows[0] if rows else {}

    manifest = out / "manifest.md"
    manifest.write_text(
        "\n".join(
            [
                "# Manuscript Artifact Manifest",
                "",
                f"- Scatter figure: {scatter}",
                f"- Residual histogram: {residual}",
                f"- Spatial snapshot: {spatial}",
                f"- LaTeX table: {latex}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary_md = out / "summary.md"
    lines = ["# Results Summary", "", f"Model focus: **{model_name}**", ""]
    if top_row:
        lines.extend(
            [
                "## Best benchmark row",
                "",
                f"- Model: {top_row.get('model', '')}",
                f"- RMSE: {float(top_row.get('rmse', 'nan')):.4f}",
                f"- MAE: {float(top_row.get('mae', 'nan')):.4f}",
                f"- NSE: {float(top_row.get('nse', 'nan')):.4f}",
                f"- KGE: {float(top_row.get('kge', 'nan')):.4f}",
                "",
            ]
        )

    if uncertainty_summary_json:
        upath = Path(uncertainty_summary_json)
        if upath.exists():
            u = json.loads(upath.read_text(encoding="utf-8"))
            lines.extend(
                [
                    "## Uncertainty summary",
                    "",
                    f"- ICE: {float(u.get('ice', float('nan'))):.4f}",
                    f"- Gaussian NLL: {float(u.get('gaussian_nll', float('nan'))):.4f}",
                    f"- Sharpness: {float(u.get('sharpness', float('nan'))):.4f}",
                    "",
                ]
            )

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "scatter_figure": str(scatter),
        "residual_hist": str(residual),
        "spatial_snapshot": str(spatial),
        "latex_table": str(latex),
        "manifest": str(manifest),
        "summary_md": str(summary_md),
    }
