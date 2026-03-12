"""Ablation utilities for hydrology-aware TWSA downscaling experiments."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, Iterable, List

from hydro_training import HydroTrainingConfig, fit_hydro_model


def build_ablation_variants(base: HydroTrainingConfig) -> List[HydroTrainingConfig]:
    variants: List[HydroTrainingConfig] = []

    def clone(name: str, **updates: object) -> HydroTrainingConfig:
        payload = asdict(base)
        payload.update(updates)
        payload["output_dir"] = str(Path(base.output_dir) / name)
        return HydroTrainingConfig(**payload)

    variants.append(clone("full_model"))
    variants.append(clone("window1_no_temporal", window_size=1))
    variants.append(clone("no_static_conditioning", static_channels=0))
    variants.append(clone("no_uncertainty", uncertainty_weight=0.0))
    variants.append(clone("no_conservation", conservation_weight=0.0))
    variants.append(clone("no_trend_supervision", trend_weight=0.0))
    return variants


def run_ablation_suite(base: HydroTrainingConfig, variants: Iterable[HydroTrainingConfig] | None = None) -> Dict[str, object]:
    if variants is None:
        variants = build_ablation_variants(base)

    summaries = []
    for variant in variants:
        summary = fit_hydro_model(variant)
        summaries.append(
            {
                "name": Path(variant.output_dir).name,
                "output_dir": variant.output_dir,
                "best_val_loss": summary["best_val_loss"],
                "best_epoch": summary["best_epoch"],
                "summary_json": summary["summary_json"],
            }
        )

    report = {"ablations": summaries}
    report_path = Path(base.output_dir) / "ablation_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["summary_json"] = str(report_path)
    return report


__all__ = ["build_ablation_variants", "run_ablation_suite"]
