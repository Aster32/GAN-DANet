# GAN-DANet

GAN-DANet for spatial downscaling of GRACE terrestrial water storage anomalies (TWSA), with upgraded reproducibility and safer data pipeline defaults.

## Repository layout

- `models/`: generator, discriminator, losses, initialization helpers
- `datasets.py`: data loading, preprocessing, augmentation
- `NC_READ.py`: NetCDF reader for ERA5 variables (deterministic file order)
- `utils.py`: plotting and evaluation utilities
- `cli.py`: deterministic smoke-test entrypoint
- `repro.py`: shared seed/determinism utilities
- `cache/`: precomputed scalers and small cached arrays for quick tests

## Environment

```bash
conda env create -f requirement.yml
conda activate gan-danet
```

## Quick validation

```bash
python cli.py smoke --seed 42
```

Expected output includes generator and discriminator output shapes.

The new research-grade hydro model also has dedicated validation commands:

```bash
python cli.py hydro-smoke --seed 42 --aux-channels 40 --window-size 5
python cli.py hydro-train-step --seed 42 --aux-channels 40 --window-size 5
```

For real experiments:

```bash
python cli.py hydro-fit --config hydro_experiment_config.yml
python cli.py hydro-ablate --config hydro_experiment_config.yml
python cli.py hydro-rolling --config hydro_experiment_config.yml
```

## Data and path configuration

The code supports environment-variable overrides in `datasets.py`:

- `PROJECT_DIR` (default `.`)
- `DATA_DIR` (default `/mnt/sdc/xy/data_op`)
- `ERA5_SUBDIR` (default `ERA5/11`)
- `QZ_*` file path overrides if needed

`utils.plot_results(...)` now accepts `cache_dir` so mask files can be loaded from `cache/` regardless of working directory.

## Reproducibility guidance

- Use `repro.seed_everything(seed, deterministic=True)` in train/eval scripts.
- Report seed values and deterministic mode in manuscript methods.
- Keep `requirement.yml` pinned and archive the exact environment with your submission.

## Scientific reporting checklist (recommended)

- Compare against strong baselines: bicubic, CNN/U-Net SR, and non-DL geostatistical or regression-based methods.
- Include ablations: attention on/off, GAN loss on/off, auxiliary variable subsets, detrending choices.
- Report uncertainty: spatially explicit confidence intervals and event-wise error behavior.
- Report temporal robustness: dry/wet years, extreme events, and rolling-origin validation.
- Provide reproducibility artifacts: code, config, seeds, and small public test subset.

## Advanced hydrology path

The upgraded research path adds:

- `models/hydro_downscaler.py`: temporal anomaly/trend-aware hydrology downscaler
- `models/losses.py`: conservation, gradient, uncertainty, and composite hydrology losses
- `hydro_datasets.py`: temporal window datasets and dataloaders
- `hydro_training.py`: synthetic checks, checkpointed fitting, prediction export, and rolling-origin experiments
- `hydro_notebooks.py`: notebook-preserving train/test workflows for ensemble fitting, 0.25 deg inference, and 0.05 deg refinement
- `hydro_experiment_config.yml`: baseline settings for the advanced model

## Notebook entrypoints

The original notebook front doors are preserved:

- `GAN_DANet_train.ipynb`: now trains a two-member hydrology ensemble, preserving the original `model1`/`model2` logic for epistemic uncertainty.
- `test.ipynb`: still runs the two-stage workflow, with Script 1 generating the 0.25 degree product and Script 2 producing the 0.05 degree refinement/export package.

These notebooks now call `hydro_notebooks.py` so the user-facing workflow stays recognizable while the underlying method is substantially upgraded.

## Evaluation CLI

Run benchmark metrics from `.npy` files:

```bash
python cli.py benchmark \
  --target path/to/obs.npy \
  --baseline bicubic \
  --pred bicubic=path/to/bicubic.npy \
  --pred unet=path/to/unet.npy \
  --pred GAN-DANet=path/to/gandanet.npy
```

Run uncertainty calibration analysis:

```bash
python cli.py uncertainty \
  --target path/to/obs.npy \
  --pred-mean path/to/gandanet_mean.npy \
  --pred-std path/to/gandanet_std.npy
```

Generate manuscript figures/tables:

```bash
python cli.py report \
  --target path/to/obs.npy \
  --pred-mean path/to/gandanet_mean.npy \
  --benchmark-dir outputs/benchmark \
  --report-dir outputs/report
```

Run the full publication artifact pipeline:

```bash
python cli.py all \
  --target path/to/obs.npy \
  --baseline bicubic \
  --pred bicubic=path/to/bicubic.npy \
  --pred unet=path/to/unet.npy \
  --pred GAN-DANet=path/to/gandanet_mean.npy \
  --pred-mean path/to/gandanet_mean.npy \
  --pred-std path/to/gandanet_std.npy
```
