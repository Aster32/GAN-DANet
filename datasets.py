import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import NC_READ
from scipy.ndimage import zoom
from scipy.stats import linregress
from statsmodels.tsa.seasonal import STL
from scipy.fftpack import fft, ifft
from pathlib import Path

# ========= Path Config (edit/override via env vars if needed) =========
# PROJECT_DIR: base folder for .npy inputs/outputs (default: current folder)
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", ".")).resolve()

# DATA_DIR: your preexisting default for ERA5 (kept as-is)
DATA_DIR = Path(os.getenv("DATA_DIR", "/media/xy/data_op/")).resolve()

# ERA5 subdirectory (kept default; overridable)
ERA5_SUBDIR = os.getenv("ERA5_SUBDIR", "ERA5/11")

# ---- Inputs (defaults preserved) ----
QZ_RLWE_05   = PROJECT_DIR / os.getenv("QZ_RLWE_05",   "qz_rlwe-05.npy")
QZ_RLWE_25   = PROJECT_DIR / os.getenv("QZ_RLWE_25",   "qz_rlwe-25.npy")
GLDAS25      = PROJECT_DIR / os.getenv("GLDAS25",      "gldas25.npy")
DEM_NPY      = PROJECT_DIR / os.getenv("DEM_NPY",      "dem.npy")

# ---- Intermediate/outputs (defaults preserved) ----
QZ_HT_01     = PROJECT_DIR / os.getenv("QZ_HT_01",     "qz_ht-01.npy")
QZ_HET_01    = PROJECT_DIR / os.getenv("QZ_HET_01",    "qz_het-01.npy")
QZ_HT_1      = PROJECT_DIR / os.getenv("QZ_HT_1",      "qz_ht-1.npy")
QZ_HET_1     = PROJECT_DIR / os.getenv("QZ_HET_1",     "qz_het-1.npy")
QZ_HP_1      = PROJECT_DIR / os.getenv("QZ_HP_1",      "qz_hp-1.npy")   # only used if you actually save it
QZ_HRO_1     = PROJECT_DIR / os.getenv("QZ_HRO_1",     "qz_hro-1.npy")
QZ_HSDE_1    = PROJECT_DIR / os.getenv("QZ_HSDE_1",    "qz_hsde-1.npy")
QZ_HTP_1     = PROJECT_DIR / os.getenv("QZ_HTP_1",     "qz_htp-1.npy")
QZ_ERA5_1    = PROJECT_DIR / os.getenv("QZ_ERA5_1",    "qz_era5-1.npy")
QZ_ET_1      = PROJECT_DIR / os.getenv("QZ_ET_1",      "qz_et-1.npy")

# ---- Tiny helpers so the rest of the code stays clean ----
def _np_load(pathlike):
    return np.load(str(pathlike))

def _np_save(pathlike, arr):
    Path(pathlike).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(pathlike), arr)

def _era5_dir():
    return DATA_DIR / ERA5_SUBDIR
# ======================================================================


def detrend_and_compare(data):
    """
    对输入的三维数据进行线性趋势分解，去除趋势后再还原，并进行对比。

    参数:
    - data: 3D numpy 数组，形状为 (time, space_x, space_y)

    返回:
    - trend_data: 提取的趋势项，形状与输入数据相同
    - detrended_data: 去除趋势后的数据，形状与输入数据相同
    - reconstructed_data: 还原后的数据，形状与输入数据相同
    - max_difference: 原始数据与还原数据之间的最大误差
    """

    def detrend_data(data):
        """
        对输入的三维数据进行线性趋势分解。

        参数:
        - data: 3D numpy 数组，形状为 (time, space_x, space_y)

        返回:
        - trend_data: 提取的趋势项，形状与输入数据相同
        - detrended_data: 去除趋势后的数据，形状与输入数据相同
        """
        # 获取数据的形状
        time_steps, space_x, space_y = data.shape

        # 初始化趋势和去趋势后的数据数组
        trend_data = np.zeros_like(data)
        detrended_data = np.zeros_like(data)

        # 对每个空间点的时间序列进行线性回归，提取趋势项
        for i in range(space_x):
            for j in range(space_y):
                y = data[:, i, j]

                # 使用STL进行趋势分解
                stl = STL(y, seasonal=13, period=12)  # 可根据需要调整 seasonal
                result = stl.fit()

                # 获取趋势项
                trend = result.trend

                # 保存趋势项和去除趋势后的数据
                trend_data[:, i, j] = trend
                detrended_data[:, i, j] = y - trend

        return trend_data, detrended_data

    # 拆分趋势和去趋势处理
    trend, detrended = detrend_data(data)

    # 还原数据
    reconstructed_data = detrended + trend

    # 对比原始数据和还原数据
    difference = np.abs(data - reconstructed_data)
    max_difference = np.max(difference)

    # 输出最大误差
    print(f'最大误差: {max_difference}')

    # 可视化比较（随机选择一个空间点）（保留为注释）
    time_steps, space_x, space_y = data.shape
    i, j = np.random.randint(space_x), np.random.randint(space_y)
    time = np.arange(time_steps)
    '''
    plt.figure(figsize=(12, 6))
    plt.plot(time, data[:, i, j], label='Original Data', color='blue')
    plt.plot(time, trend[:, i, j], label='Trend', color='red')
    plt.plot(time, detrended[:, i, j], label='Detrended Data', color='green')
    plt.plot(time, reconstructed_data[:, i, j], '--', label='Reconstructed Data', color='orange')
    plt.legend()
    plt.title(f'Comparison at Spatial Point ({i}, {j})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
    '''
    return trend, detrended, reconstructed_data, max_difference


class FlexibleStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=(0, 1, 2), keepdims=True)
        self.scale_ = X.std(axis=(0, 1, 2), keepdims=True)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        return (X * self.scale_) + self.mean_


class CustomDataset(Dataset):
    def __init__(self, lr_grace_05, lr_grace_025, hr_aux, augment=False):
        self.lr_grace_05 = torch.from_numpy(lr_grace_05).float().unsqueeze(1)  # Adding channel dimension
        self.lr_grace_025 = torch.from_numpy(lr_grace_025).float().unsqueeze(1)  # Adding channel dimension
        self.hr_aux = torch.from_numpy(hr_aux).float().permute(0, 3, 1, 2)  # Reformat to [N, C, H, W]
        self.augment = augment

    def __len__(self):
        return len(self.lr_grace_05)

    def __getitem__(self, idx):
        lr_grace_05 = self.lr_grace_05[idx]
        lr_grace_025 = self.lr_grace_025[idx]
        hr_aux = self.hr_aux[idx]

        if self.augment:
            lr_grace_05, lr_grace_025, hr_aux = self.apply_augmentation(lr_grace_05, lr_grace_025, hr_aux)

        return lr_grace_05, lr_grace_025, hr_aux

    # Placeholder version (kept intact; overridden by the next definition)
    def apply_augmentation(self, lr_grace_05, elr_grace_025, hr_aux, trend05, trend25):
        return lr_grace_05, elr_grace_025, hr_aux, trend05, trend25

    # Actual augmentation used
    def apply_augmentation(self, lr_grace_05, lr_grace_025, hr_aux):
        # Random horizontal flip
        if random.random() > 0.5:
            lr_grace_05 = torch.flip(lr_grace_05, [2])
            lr_grace_025 = torch.flip(lr_grace_025, [2])
            hr_aux = torch.flip(hr_aux, [2])

        # Random vertical flip
        if random.random() > 0.5:
            lr_grace_05 = torch.flip(lr_grace_05, [1])
            lr_grace_025 = torch.flip(lr_grace_025, [1])
            hr_aux = torch.flip(hr_aux, [1])

        # Random rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            lr_grace_05 = torch.rot90(lr_grace_05, k=angle // 90, dims=[1, 2])
            lr_grace_025 = torch.rot90(lr_grace_025, k=angle // 90, dims=[1, 2])
            hr_aux = torch.rot90(hr_aux, k=angle // 90, dims=[1, 2])

        # Adding random noise
        if random.random() > 0.5:
            noise = torch.randn_like(lr_grace_05) * 0.05
            lr_grace_05 = lr_grace_05 + noise
            noise = torch.randn_like(lr_grace_025) * 0.05
            lr_grace_025 = lr_grace_025 + noise

        return lr_grace_05, lr_grace_025, hr_aux


def fill_placeholder_with_mean(data, placeholder=-9999):
    # Iterate over each variable (last dimension)
    for i in range(data.shape[-1]):
        variable = data[..., i]
        # placeholder_mask = (variable <= placeholder)
        mean_value = np.mean(variable[:, 14:16, 12:14])
        # variable[placeholder_mask] = mean_value
        data[:, 0:14, 0:12, i] = mean_value
    return data


def fill_placeholder_with_nearest(data, placeholder=-9999, sigma=3):
    # Iterate over each variable (last dimension)
    for i in range(data.shape[-1]):
        variable = data[..., i]
        placeholder_mask = (variable <= placeholder)

        # Create a copy of the variable with placeholders replaced by zeros
        variable_filled = np.where(placeholder_mask, 0, variable)

        # Apply Gaussian filter
        smoothed_variable = gaussian_filter(variable_filled, sigma=sigma)

        # Create a mask for the valid points (1 for valid, 0 for placeholders)
        valid_mask = 1 - placeholder_mask.astype(float)

        # Smooth the mask
        mask_smoothed = gaussian_filter(valid_mask, sigma=sigma)

        # Avoid division by zero
        mask_smoothed[mask_smoothed == 0] = 1

        # Get the final filled data
        filled_data = smoothed_variable / mask_smoothed

        # Replace the placeholders in the original data
        variable[placeholder_mask] = filled_data[placeholder_mask]
        data[..., i] = variable

    return data


def read_era():
    lr_grace_05 = _np_load(QZ_RLWE_05)
    print(np.shape(lr_grace_05))

    era5_path = str(_era5_dir())
    era5r, timer = NC_READ.readdata(era5_path)
    era5 = np.array(era5r['t2m'])
    ET = np.array(era5r['e'])
    ro = np.array(era5r['ro'])
    sde = np.array(era5r['sde'])
    tp = np.array(era5r['tp'])
    ET = np.squeeze(ET)
    ro = np.squeeze(ro)
    sde = np.squeeze(sde)
    tp = np.squeeze(tp)
    et_s = ET[15:lr_grace_05.shape[0] + 15, :, :]
    ro_s = ro[15:lr_grace_05.shape[0] + 15, :, :]
    sde_s = sde[15:lr_grace_05.shape[0] + 15, :, :]
    tp_s = tp[15:lr_grace_05.shape[0] + 15, :, :]
    et_s = np.transpose(et_s, axes=(1, 2, 0))
    ro_s = np.transpose(ro_s, axes=(1, 2, 0))
    sde_s = np.transpose(sde_s, axes=(1, 2, 0))
    tp_s = np.transpose(tp_s, axes=(1, 2, 0))
    et_s = np.rot90(et_s, k=3, axes=(0, 1))
    ro_s = np.rot90(ro_s, k=3, axes=(0, 1))
    sde_s = np.rot90(sde_s, k=3, axes=(0, 1))
    tp_s = np.rot90(tp_s, k=3, axes=(0, 1))
    era5 = np.squeeze(era5)
    era5_s = era5[15:lr_grace_05.shape[0] + 15, :, :]
    era5_s = np.transpose(era5_s, axes=(1, 2, 0))
    era5_s = np.rot90(era5_s, k=3, axes=(0, 1))

    scale_factors = (1, 0.4, 0.4)
    qz_het = np.transpose(et_s, axes=(2, 0, 1))
    qz_hro = np.transpose(ro_s, axes=(2, 0, 1))
    qz_hsde = np.transpose(sde_s, axes=(2, 0, 1))
    qz_htp = np.transpose(tp_s, axes=(2, 0, 1))
    qz_ht = np.transpose(era5_s, axes=(2, 0, 1))
    _np_save(QZ_HT_01, qz_ht)
    _np_save(QZ_HET_01, qz_het)

    qz_het = zoom(qz_het, scale_factors, order=3, mode='nearest')
    qz_ht = zoom(qz_ht, scale_factors, order=3, mode='nearest')
    qz_hro = zoom(qz_hro, scale_factors, order=3, mode='nearest')
    qz_hsde = zoom(qz_hsde, scale_factors, order=3, mode='nearest')
    qz_htp = zoom(qz_htp, scale_factors, order=3, mode='nearest')

    scale_factors = (1, 0.25, 0.25)
    scale_factors = (0.1, 0.1, 1)
    era_5 = zoom(era5_s, scale_factors, order=3, mode='nearest')
    et_ = zoom(et_s, scale_factors, order=3, mode='nearest')
    era5_s = era_5
    qz_era5 = np.transpose(era5_s, axes=(2, 0, 1))
    qz_et = np.transpose(et_, axes=(2, 0, 1))

    _np_save(QZ_HT_1, qz_ht)
    _np_save(QZ_HET_1, qz_het)
    # If you actually generate qz_hp, also save it to QZ_HP_1 here.
    _np_save(QZ_HRO_1, qz_hro)
    _np_save(QZ_HSDE_1, qz_hsde)
    _np_save(QZ_HTP_1, qz_htp)
    _np_save(QZ_ERA5_1, qz_era5)
    _np_save(QZ_ET_1, qz_et)


def frequency_domain_augmentation(data, seasonal_freq, noise_level=0.1, axis=0):
    """
    Augments data using frequency-domain transformations.

    Args:
        data (numpy.ndarray): The data to augment.
        seasonal_freq (int): Dominant seasonal frequency for perturbation.
        noise_level (float): Scale of random noise to add in the frequency domain.
        axis (int): Axis along which to augment the data (e.g., 0 for time).

    Returns:
        numpy.ndarray: Augmented data with the same shape as the input.
    """
    # Compute the FFT along the specified axis
    freq_data = fft(data, axis=axis)

    # Add random noise to seasonal frequency components
    shape = freq_data.shape
    freq_indices = np.arange(-seasonal_freq, seasonal_freq + 1)
    random_perturbation = np.random.normal(scale=noise_level, size=shape)

    for idx in freq_indices:
        if 0 <= idx < freq_data.shape[axis]:
            slicing = [slice(None)] * len(shape)
            slicing[axis] = idx
            freq_data[tuple(slicing)] += random_perturbation[tuple(slicing)]

    # Reconstruct the data using the inverse FFT
    augmented_data = np.real(ifft(freq_data, axis=axis))
    return augmented_data


def load_data():
    read_era()
    lat05l = np.linspace(24.5, 45.5, 44)
    lon05l = np.linspace(65.5, 109.5, 90)
    lat05, lon05 = np.meshgrid(lat05l, lon05l)
    lon05 = np.expand_dims(lon05, axis=0)
    lat05 = np.expand_dims(lat05, axis=0)
    lon05 = np.repeat(lon05, 181, axis=0)
    lat05 = np.repeat(lat05, 181, axis=0)
    lon05 = np.expand_dims(lon05, axis=-1)
    lat05 = np.expand_dims(lat05, axis=-1)
    lat025l = np.linspace(24.5, 45.5, 88)
    lon025l = np.linspace(65.5, 109.5, 180)
    lat025, lon025 = np.meshgrid(lat025l, lon025l)
    lon025 = np.expand_dims(lon025, axis=0)
    lat025 = np.expand_dims(lat025, axis=0)
    lon025 = np.repeat(lon025, 181, axis=0)
    lat025 = np.repeat(lat025, 181, axis=0)
    lon025 = np.expand_dims(lon025, axis=-1)
    lat025 = np.expand_dims(lat025, axis=-1)

    lr_grace_05 = _np_load(QZ_RLWE_05)
    lr_grace_025 = _np_load(QZ_RLWE_25)
    lr_grace_025 = lr_grace_025[0:lr_grace_05.shape[0], :, :]

    gldas = _np_load(GLDAS25)
    gldas = gldas[19:, :, :, :]
    dem = np.expand_dims(_np_load(DEM_NPY), axis=-1)
    dem = np.repeat(dem[np.newaxis, :, :, :], 181, axis=0)

    qz_ht = np.expand_dims(_np_load(QZ_HT_1), axis=-1)
    qz_het = np.expand_dims(_np_load(QZ_HET_1), axis=-1)

    # If you actually saved qz_hp in read_era(), keep this; otherwise ensure file exists.
    qz_hp = np.expand_dims(_np_load(QZ_HP_1), axis=-1)
    print(np.shape(qz_hp))

    qz_hro = np.expand_dims(_np_load(QZ_HRO_1), axis=-1)
    qz_hsde = np.expand_dims(_np_load(QZ_HSDE_1), axis=-1)
    qz_htp = np.expand_dims(_np_load(QZ_HTP_1), axis=-1)
    qz_ht = fill_placeholder_with_nearest(qz_ht, placeholder=100)

    print(qz_ht[0, 0, 0])
    print(qz_hro[0, 0, 0])
    print(qz_hsde[0, 0, 0])
    print(qz_htp[0, 0, 0])

    # Combine auxiliary datasets along the last dimension
    hr_aux = np.concatenate((gldas, qz_ht, qz_het, qz_hp, qz_hro, qz_hsde, qz_htp, lat025, lon025, dem), axis=-1)
    print("Combined HR Aux Data Shape:", hr_aux.shape)  # Debugging line
    print(hr_aux[0, 0, 0, -1])
    print(hr_aux[0, 0, 0, -2])

    print("Sliced HR Aux Data Shape:", hr_aux.shape)  # Debugging line

    # Fill placeholder values with mean
    hr_aux = fill_placeholder_with_mean(hr_aux, placeholder=-9999)

    # Standardize the GRACE data
    grace_scaler_05 = StandardScaler()
    grace_scaler_025 = StandardScaler()
    lr_grace_05 = grace_scaler_05.fit_transform(lr_grace_05.reshape(-1, 1)).reshape(lr_grace_05.shape)
    lr_grace_025 = grace_scaler_025.fit_transform(lr_grace_025.reshape(-1, 1)).reshape(lr_grace_025.shape)

    grace_mean_05, grace_std_05 = grace_scaler_05.mean_, grace_scaler_05.scale_
    grace_mean_025, grace_std_025 = grace_scaler_025.mean_, grace_scaler_025.scale_

    # Standardize each variable in the auxiliary data separately
    hr_aux_standardized = np.empty_like(hr_aux)
    aux_scalers = []
    for i in range(hr_aux.shape[-1]):
        scaler = StandardScaler()
        standardized_var = scaler.fit_transform(hr_aux[..., i].reshape(-1, 1)).reshape(hr_aux[..., i].shape)
        hr_aux_standardized[..., i] = standardized_var
        aux_scalers.append(scaler)

    # Separate gldas and the other data
    gldas_data = hr_aux_standardized[:, :, :, :gldas.shape[-1]]
    other_data = hr_aux_standardized[:, :, :, gldas.shape[-1]:]

    # Apply Gaussian smoothing to gldas data
    smoothed_gldas = np.copy(gldas_data)
    for t in range(gldas_data.shape[0]):
        for ch in range(gldas_data.shape[-1]):
            smoothed_gldas[t, :, :, ch] = gaussian_filter(gldas_data[t, :, :, ch], sigma=3)

    # Reassemble hr_aux
    smoothed_hr_aux = np.concatenate((smoothed_gldas, other_data), axis=-1)
    print(np.min(smoothed_hr_aux))
    slice_data = smoothed_hr_aux[0, :, :, 35:42]
    print(np.shape(slice_data))

    trend, detrended, reconstructed, max_diff = detrend_and_compare(lr_grace_05)
    trend25, detrended25, reconstructed25, max_diff = detrend_and_compare(lr_grace_025)
    return [detrended, trend], [detrended25, trend25], smoothed_hr_aux, grace_scaler_05, grace_scaler_025, aux_scalers


def inverse_transform(data, scaler):
    return scaler.inverse_transform(data)


def load_data_with_augmentation():
    # Load existing datasets
    [detrended, trend], [detrended25, trend25], smoothed_hr_aux, grace_scaler_05, grace_scaler_025, aux_scalers = load_data()

    # Define augmentation parameters
    augmentation_factor = 2  # Number of augmentations
    seasonal_freq = 12       # Example: Monthly seasonality
    noise_level = 0.1        # Scale of frequency perturbations

    # Augment datasets
    augmented_detrended = []
    augmented_detrended25 = []
    augmented_smoothed_hr_aux = []

    for _ in range(augmentation_factor):
        # Augment detrended
        aug_detrended = frequency_domain_augmentation(
            detrended, seasonal_freq=seasonal_freq, noise_level=noise_level, axis=0
        )
        augmented_detrended.append(aug_detrended)

        # Augment detrended25
        aug_detrended25 = frequency_domain_augmentation(
            detrended25, seasonal_freq=seasonal_freq, noise_level=noise_level, axis=0
        )
        augmented_detrended25.append(aug_detrended25)

        # Augment smoothed_hr_aux
        aug_smoothed_hr_aux = frequency_domain_augmentation(
            smoothed_hr_aux, seasonal_freq=seasonal_freq, noise_level=noise_level, axis=0
        )
        augmented_smoothed_hr_aux.append(aug_smoothed_hr_aux)

    # Concatenate augmented datasets with the original
    detrended_augmented = np.concatenate([detrended] + augmented_detrended, axis=0)
    detrended25_augmented = np.concatenate([detrended25] + augmented_detrended25, axis=0)
    smoothed_hr_aux_augmented = np.concatenate([smoothed_hr_aux] + augmented_smoothed_hr_aux, axis=0)

    # Match trends with the augmented datasets
    trend_replicated = np.tile(trend, (1 + augmentation_factor, 1, 1))      # Match time dimension
    trend25_replicated = np.tile(trend25, (1 + augmentation_factor, 1, 1))  # Match time dimension

    # Return augmented datasets and trends
    return (
        [detrended_augmented, trend_replicated],
        [detrended25_augmented, trend25_replicated],
        smoothed_hr_aux_augmented,
        grace_scaler_05,
        grace_scaler_025,
        aux_scalers,
    )