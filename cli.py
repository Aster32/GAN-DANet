"""Project CLI for quick sanity checks and reproducible evaluation workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark import run_benchmark_from_files
from hydro_ablation import run_ablation_suite
from hydro_training import (
    HydroTrainingConfig,
    fit_hydro_model,
    load_hydro_config,
    run_hydro_smoke,
    run_hydro_train_step,
    run_rolling_origin_experiment,
)
from models import Discriminator1, FlexibleUpsamplingModule
from reporting import run_report_from_files
from repro import seed_everything
from uncertainty import run_uncertainty_from_files


def run_smoke_test(batch_size: int = 2, aux_channels: int = 40) -> None:
    """Run lightweight model forward-pass checks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = FlexibleUpsamplingModule(input_channels=aux_channels).to(device)
    discriminator = Discriminator1(input_channels=1).to(device)

    # The legacy generator expects quarter-scale inputs and upsamples by 4x.
    x = torch.randn(batch_size, aux_channels, 22, 45, device=device)
    y = generator(x)
    score = discriminator(y)

    print(f"device={device}")
    print(f"generator_out_shape={tuple(y.shape)}")
    print(f"discriminator_out_shape={tuple(score.shape)}")


def _parse_prediction_map(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid prediction mapping '{item}'. Expected name=path.npy")
        name, path = item.split("=", 1)
        mapping[name.strip()] = path.strip()
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAN-DANet utility CLI")
    parser.add_argument(
        "command",
        choices=[
            "smoke",
            "hydro-smoke",
            "hydro-train-step",
            "hydro-fit",
            "hydro-ablate",
            "hydro-rolling",
            "benchmark",
            "uncertainty",
            "report",
            "all",
        ],
    )
    parser.add_argument("--seed", type=int, default=42, help="Global seed")
    parser.add_argument("--non-deterministic", action="store_true", help="Disable deterministic torch behavior")

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--aux-channels", type=int, default=40)
    parser.add_argument("--static-channels", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")

    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--pred", type=str, action="append", default=[])
    parser.add_argument("--baseline", type=str, default="")

    parser.add_argument("--pred-mean", type=str, default="")
    parser.add_argument("--pred-std", type=str, default="")

    parser.add_argument("--benchmark-dir", type=str, default="outputs/benchmark")
    parser.add_argument("--uncertainty-dir", type=str, default="outputs/uncertainty")
    parser.add_argument("--report-dir", type=str, default="outputs/report")
    parser.add_argument("--model-name", type=str, default="GAN-DANet")

    return parser.parse_args()


def _require(value: str, name: str) -> None:
    if not value:
        raise ValueError(f"Missing required argument: --{name}")


def main() -> None:
    args = parse_args()
    seed_everything(seed=args.seed, deterministic=not args.non_deterministic)

    if args.command == "smoke":
        run_smoke_test(batch_size=args.batch_size, aux_channels=args.aux_channels)
        return

    if args.command in {"hydro-smoke", "hydro-train-step"}:
        config = HydroTrainingConfig(
            batch_size=args.batch_size,
            window_size=args.window_size,
            aux_channels=args.aux_channels,
            static_channels=args.static_channels,
            base_channels=args.base_channels,
            epochs=args.epochs or 20,
            seed=args.seed,
            output_dir=args.output_dir or "outputs/hydro",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if args.command == "hydro-smoke":
            print(json.dumps(run_hydro_smoke(config), indent=2))
        else:
            print(json.dumps(run_hydro_train_step(config), indent=2))
        return

    if args.command in {"hydro-fit", "hydro-ablate", "hydro-rolling"}:
        if args.config:
            config = load_hydro_config(args.config)
        else:
            config = HydroTrainingConfig(
                batch_size=args.batch_size,
                window_size=args.window_size,
                aux_channels=args.aux_channels,
                static_channels=args.static_channels,
                base_channels=args.base_channels,
                epochs=args.epochs or 20,
                seed=args.seed,
                output_dir=args.output_dir or "outputs/hydro",
            )
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        config.seed = args.seed
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.epochs > 0:
            config.epochs = args.epochs

        if args.command == "hydro-fit":
            print(json.dumps(fit_hydro_model(config), indent=2))
        elif args.command == "hydro-ablate":
            print(json.dumps(run_ablation_suite(config), indent=2))
        else:
            print(json.dumps(run_rolling_origin_experiment(config), indent=2))
        return

    if args.command == "benchmark":
        _require(args.target, "target")
        _require(args.baseline, "baseline")
        preds = _parse_prediction_map(args.pred)
        out = run_benchmark_from_files(
            target_path=args.target,
            prediction_paths=preds,
            baseline=args.baseline,
            output_dir=args.benchmark_dir,
            seed=args.seed,
        )
        print(json.dumps(out, indent=2))
        return

    if args.command == "uncertainty":
        _require(args.target, "target")
        _require(args.pred_mean, "pred-mean")
        _require(args.pred_std, "pred-std")
        out = run_uncertainty_from_files(
            target_path=args.target,
            mean_path=args.pred_mean,
            std_path=args.pred_std,
            output_dir=args.uncertainty_dir,
        )
        print(json.dumps(out, indent=2))
        return

    if args.command == "report":
        _require(args.target, "target")
        _require(args.pred_mean, "pred-mean")
        metrics_csv = str(Path(args.benchmark_dir) / "metrics.csv")
        out = run_report_from_files(
            target_path=args.target,
            pred_path=args.pred_mean,
            metrics_csv=metrics_csv,
            output_dir=args.report_dir,
            model_name=args.model_name,
            uncertainty_summary_json=str(Path(args.uncertainty_dir) / "uncertainty_summary.json"),
        )
        print(json.dumps(out, indent=2))
        return

    if args.command == "all":
        _require(args.target, "target")
        _require(args.baseline, "baseline")
        _require(args.pred_mean, "pred-mean")
        _require(args.pred_std, "pred-std")

        preds = _parse_prediction_map(args.pred)
        if args.model_name not in preds:
            preds[args.model_name] = args.pred_mean

        bench_out = run_benchmark_from_files(
            target_path=args.target,
            prediction_paths=preds,
            baseline=args.baseline,
            output_dir=args.benchmark_dir,
            seed=args.seed,
        )
        unc_out = run_uncertainty_from_files(
            target_path=args.target,
            mean_path=args.pred_mean,
            std_path=args.pred_std,
            output_dir=args.uncertainty_dir,
        )
        rep_out = run_report_from_files(
            target_path=args.target,
            pred_path=args.pred_mean,
            metrics_csv=bench_out["metrics_csv"],
            output_dir=args.report_dir,
            model_name=args.model_name,
            uncertainty_summary_json=unc_out["summary_json"],
        )

        payload = {
            "benchmark": bench_out,
            "uncertainty": unc_out,
            "report": rep_out,
        }
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
