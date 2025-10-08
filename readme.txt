# GAN-DANet GRACE Downscaling Toolbox — README

**Program name:** GAN-DANet GRACE Downscaling Toolbox
**Manuscript title:** Spatial Downscaling of GRACE Observations Using an Improved Generative Adversarial Network (GAN-DANet) in the Tibetan Plateau
**Authors:** YI Xiong
**Repository (if any):** https://github.com/Aster32/GAN-DANet

This package contains code and notebooks to train and evaluate **GAN-DANet** for downscaling GRACE TWSA. It ships with a **prebuilt cache of small test data** so reviewers can run the notebooks immediately without downloading large inputs.

---

## Contents (typical layout)

```
.
├─ GAN_DANet_train.ipynb        # TRAIN notebook
├─ test.ipynb                   # TEST/INFERENCE notebook
├─ model/                       # Model definitions (e.g., FlexibleUpsamplingModule, Discriminator1, losses)
├─ datasets/                    # Data I/O, preprocessing, augmentation, cache helpers
├─ utils/                       # Plotting and utilities
├─ cache/                       # **Already contains test data and cached arrays (see below)**
│  ├─ dataset_cache.npz
│  ├─ grace_scaler_05.joblib
│  ├─ grace_scaler_025.joblib
│  ├─ aux_scalers.joblib
│  ├─ tpb_l.npy                 # 44×90 mask (low-res)
│  ├─ tpb_h.npy                 # 88×180 mask (0.25° grid)
│  └─ ensemble_uncertainty.npy  # small synthetic uncertainty sample
└─ (outputs created by notebooks: *.h5, *.nc, *.png, *.pdf)
```

No executables are included (in line with **Computers & Geosciences** guidance).

---

## Requirements
Default packages could be found in requirement.yml
- **Python** 3.9–3.11
- **Recommended:** PyTorch 2.x (GPU optional)
- Packages:
  ```
  numpy scipy scikit-learn statsmodels h5py netCDF4 pandas matplotlib
  torch torchvision torchaudio        # match CUDA if using a GPU
  torchviz hiddenlayer opencv-python
  ```

Example (CPU-only) setup:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy scikit-learn statsmodels h5py netCDF4 pandas matplotlib             torchviz hiddenlayer opencv-python             torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Configuration (environment variables)

All defaults are **relative paths** for portability. You may override them via environment variables or at the top of the notebooks:

- `PROJECT_DIR=.`
- `DATA_DIR=./data_op`
- `ERA5_SUBDIR=ERA5/11`
- `CACHE_DIR=./cache`  (**already populated for testing**)
- `TEST_MODE=1`        (optional; enables synthetic data fallback if raw inputs are absent)
- `REBUILD_CACHE=0`    (set to `1` only if you want to reprocess and overwrite `./cache`)

> For review, keep `REBUILD_CACHE=0` to use the provided cache immediately.

---

## How to Use the Program

### 1) Train — `GAN_DANet_train.ipynb`
1. Open the notebook in Jupyter/Lab.
2. Ensure these variables are set (first cell or your environment):
   - `CACHE_DIR=./cache`
   - `REBUILD_CACHE=0`
3. Run all cells.
   - The notebook **loads arrays & scalers from the cache** (skipping heavy preprocessing).
   - It trains GAN-DANet and saves weights (e.g., `best_model.pth` or `model12_upsampling_module.pth`).

**Typical training outputs**
- `best_model.pth` (or your chosen filename)
- Optional training logs/plots

### 2) Test / Inference — `test.ipynb`
1. Open `test.ipynb`.
2. Point the **weights path** to the file produced by training (e.g., `model12_upsampling_module.pth`).
3. Run all cells. The notebook:
   - Loads cached arrays/scalers from `./cache/`
   - Runs 0.25° and 0.05° inference
   - Produces evaluation plots and exports

**Typical inference outputs**
- **HDF5:** `grace025.h5`, `grace05.h5`, `biash.h5`, `valid.h5`, `downscaled.h5`
- **NetCDF:** `downscaled_grace_tws_data_with_uncertainty_gan_danet.nc`, `grace_025.nc`
- **Plots:** `downscaled_vs_grace_tws_plot0.05.png/pdf`, `spatial_distribution_comparisondan.pdf`

> The test notebooks read `tpb_l.npy`, `tpb_h.npy`, and `ensemble_uncertainty.npy` from `./cache/`.

---

## Cache & Test Data (for reviewers)

To satisfy the **C&G** “small test data” requirement and enable instant runs:

- `./cache/` already contains:
  - `dataset_cache.npz` (arrays: `lr_grace_05`, `trend05`, `lr_grace_025`, `trend25`, `hr_aux`)
  - `grace_scaler_05.joblib`, `grace_scaler_025.joblib`, `aux_scalers.joblib`
  - Small masks (`tpb_l.npy`, `tpb_h.npy`) and uncertainty (`ensemble_uncertainty.npy`)

This lets you execute both notebooks without large downloads.

> If you ever need to regenerate the cache (not required for review), set `REBUILD_CACHE=1`. With `TEST_MODE=1`, the code falls back to synthetic ERA5-like data when raw inputs are missing.

---

## Expected Units & Shapes (quick reference)

- GRACE 0.5°: `(T, 44, 90)` (standardized/detrended in cache)
- GRACE 0.25°: `(T, 88, 180)` (standardized/detrended in cache)
- Aux stack `hr_aux`: `(T, 88, 180, C)` (standardized; smoothed GLDAS subset + ERA5-derived features)
- Masks: `tpb_l.npy` (44×90), `tpb_h.npy` (88×180)

---

## Reproducibility Notes

- Some GPU/cuDNN operations are nondeterministic.
- For strict reproducibility, fix seeds, pin package versions, and disable nondeterministic kernels where applicable.

---

## License & Citation

- **License:** _[MIT / Apache-2.0 / BSD-3 — choose one and state here]_
- **Citation:** Please cite the manuscript once published and the GRACE/GRACE-FO CSR Mascon products as appropriate.

---

## Contact

jayim@stu.cdut.edu.cn
---

**Version:** 1.0 · 
