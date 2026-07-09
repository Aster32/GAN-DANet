#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_生成预测文件_vectorized_gpu_fast.py (daily outputs)
- 只推理；强制 GPU
- 静/动全面向量化：静态一次提取所有点；动态每天/每变量只建一次 3D 滑窗，再按批索引
- 预先对整场做数值变换（缩放/对数/温度），避免批内重复计算
- H2D 传输：pin_memory + non_blocking（可选半精度上传）
- 仅在 4D 张量上设置 channels_last（避免 rank 错误）
- NetCDF 输出改为按日写文件，并保留压缩/分块
"""

import os
import glob
import gc
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import xarray as xr
import geopandas as gpd

import torch
import torch.nn as nn
from torch import amp

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw):
        return x

try:
    import rasterio
    from rasterio.transform import rowcol
except Exception as e:
    raise RuntimeError("需要 rasterio：pip install rasterio") from e

from numpy.lib.stride_tricks import sliding_window_view

xr.set_options(file_cache_maxsize=2)

# ===================== CONFIG =====================
CONFIG = {
    "start_date": "2015-01-01",
    "end_date": "2015-12-31",
    "time_window": 7,  # 时窗(天)；奇数
    "patch_size": 7,  # 空间 7×7；奇数
    # 动态源（与训练一致）
    "era5_tp_nc": r"D:\wht2\02-gpm-era5-0.0625-邻近基准点\era5\ERA5L_2015_daily_nearest_to_shp.nc",
    "era5_t2m_nc": r"D:\wht2\02-gpm-era5-0.0625-邻近基准点\era5\t2m_2015_daily_on_points_keepnames.nc",
    "imerg_nc": r"D:\wht2\02-gpm-era5-0.0625-邻近基准点\gpm\IMERG_2015_daily_nearest_to_shp.nc",
    "cldas_nc": r"D:\wht2\02-gpm-era5-0.0625-邻近基准点\cldas\CLDAS_2015_daily_sum.nc",
    # 静态
    "dem_tif": r"D:\wht2\02-gpm-era5-0.0625-邻近基准点\dem\dem_0p0625deg_centers.tif",
    "slope_tif": r"D:\wht2\02-gpm-era5-0.0625-邻近基准点\dem\Slope.tif",
    "aspect_tif": r"D:\wht2\02-gpm-era5-0.0625-邻近基准点\dem\aspect.tif",
    # 基准网点（Point Shapefile）
    "grid_points_shp": r"D:\wht2\01-0.625°基准点\cldas-point.shp",
    # 模型
    "ckpt_path": r"E:\pycharm2024.1.7\PycharmProjects\pythonProject\.venv\wenzhang2fuxian\ckpts_bmodel_opt\runs_AConvLSTM_v1\best.pth",
    # 输出目录（脚本将按日写入 pred_YYYYMMDD.nc 文件）
    "out_dir": r"D:\wht2\04end\pred_daily",
    # 批大小（可按显存上调：32768, 65536 ...）
    "batch_size": 32768,
    # 是否逐批严格检查动态 NaN（很慢；一般关闭）
    "strict_nan_check": False,
    # 上传半精度到 GPU（进一步减半 H2D 带宽；与 AMP 兼容）
    "half_upload": True,
    # 坐标名候选
    "lat_candidates": ["lat", "latitude", "LAT", "Latitude", "y"],
    "lon_candidates": ["lon", "longitude", "LON", "Longitude", "x"],
    "time_candidates": ["time", "valid_time"],
}
# ===================================================

# --------- helpers ---------
def _find_coord_name(ds, cands):
    for n in cands:
        if n in ds.coords:
            return n
        if n in ds.variables and ds[n].ndim == 1:
            return n
    return None


def _ensure_lon_units(point_lon, grid_lons):
    if (grid_lons.min() >= -1.0) and (grid_lons.max() > 180.0):
        return point_lon + 360.0 if point_lon < 0 else point_lon
    return point_lon - 360.0 if point_lon > 180.0 else point_lon


def _nearest_index(arr1d, value):
    return int(np.argmin(np.abs(arr1d - value)))


def _normalize_orientation_latlon(lat1d, lon1d, arr, lat_axis, lon_axis):
    # 上=北(纬度递减)、左=西(经度递增)
    if lat1d.size >= 2 and lat1d[0] < lat1d[-1]:
        arr = np.flip(arr, axis=lat_axis)
        lat1d = lat1d[::-1]
    if lon1d.size >= 2 and lon1d[0] > lon1d[-1]:
        arr = np.flip(arr, axis=lon_axis)
        lon1d = lon1d[::-1]
    return lat1d, lon1d, arr


def _build_all_days(s, e):
    s = datetime.strptime(s, "%Y-%m-%d")
    e = datetime.strptime(e, "%Y-%m-%d")
    out = []
    d = s
    while d <= e:
        out.append(d)
        d += timedelta(days=1)
    return out


def _dates_for_center(all_days, i, w):
    r = w // 2
    n = len(all_days)
    idxs = [max(0, min(n - 1, j)) for j in range(i - r, i + r + 1)]
    return [all_days[j] for j in idxs]


def _time_index_map(ds, tname):
    vals = [np.datetime_as_string(t, unit="D") for t in ds[tname].values]
    return {s.replace("-", ""): i for i, s in enumerate(vals)}


def _resolve_ckpt_path(p):
    p = os.path.normpath(p.strip())
    print("[DEBUG] ckpt_path =", p)
    if os.path.isfile(p):
        print("[DEBUG] exists = True")
        return p
    roots = [p if os.path.isdir(p) else os.path.dirname(p)]
    here = os.path.abspath(os.path.dirname(__file__))
    roots += [here]
    parent = here
    for _ in range(3):
        parent = os.path.dirname(parent)
        roots.append(parent)
    roots.append(os.getcwd())
    cands = []
    for r in dict.fromkeys(roots):
        if not r or not os.path.isdir(r):
            continue
        cands += glob.glob(os.path.join(r, "best.pth")) + glob.glob(os.path.join(r, "last.pth"))
        if not cands:
            cands += glob.glob(os.path.join(r, "**", "*.pth"), recursive=True)
    if not cands:
        raise FileNotFoundError("未找到 .pth；请把 ckpt_path 写为绝对路径。")
    cands.sort(key=lambda x: (0 if os.path.basename(x).lower() == "best.pth" else 1, -os.path.getmtime(x)))
    print("[INFO] 使用权重:", cands[0])
    return cands[0]


# --------- NC source (h5netcdf) ---------
class NCSource:
    def __init__(self, name, path, var, lat_c, lon_c, time_c):
        self.name = name
        self.path = path
        self.var = var
        self.ds = xr.open_dataset(self.path, engine="h5netcdf", decode_cf=True)
        self.lat_name = _find_coord_name(self.ds, lat_c)
        self.lon_name = _find_coord_name(self.ds, lon_c)
        self.time_name = _find_coord_name(self.ds, time_c)
        if any(v is None for v in (self.lat_name, self.lon_name, self.time_name)):
            raise ValueError(f"[{self.name}] 坐标识别失败。coords={list(self.ds.coords)}")
        self.lats = np.asarray(self.ds[self.lat_name].values)
        self.lons = np.asarray(self.ds[self.lon_name].values)
        self.tmap = _time_index_map(self.ds, self.time_name)
        self.lats_std = None
        self.lons_std = None

    def has_date(self, ymd):
        return ymd in self.tmap

    def get_stack7(self, ymds):
        # 返回 (T, Ny, Nx)，方向已标准化
        idxs = [self.tmap[y] for y in ymds]
        da = self.ds[self.var].isel({self.time_name: idxs}).transpose(self.time_name, self.lat_name, self.lon_name)
        arr = np.asarray(da.values, dtype=np.float32)
        latv = self.lats.copy()
        lonv = self.lons.copy()
        latv, lonv, arr = _normalize_orientation_latlon(latv, lonv, arr, 1, 2)
        self.lats_std = latv
        self.lons_std = lonv
        return arr


# --------- Model ----------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid, k=3):
        super().__init__()
        pad = k // 2
        self.hid = hid
        self.conv = nn.Conv2d(in_ch + hid, 4 * hid, k, padding=pad)

    def forward(self, x, h, c):
        if h is None:
            b, _, hgt, wid = x.shape
            h = torch.zeros(b, self.hid, hgt, wid, device=x.device, dtype=x.dtype)
            c = torch.zeros_like(h)
        i, f, g, o = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMLayer(nn.Module):
    def __init__(self, in_ch, hid, k=3):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hid, k)

    def forward(self, x):  # x: (B,T,C,H,W)
        b, t, _, h, w = x.shape
        h_state = None
        c_state = None
        for ti in range(t):
            xt = x[:, ti].contiguous().to(memory_format=torch.channels_last)  # 仅 4D 才能 channels_last
            h_state, c_state = self.cell(xt, h_state, c_state)
        return h_state


class ChannelAttention(nn.Module):
    def __init__(self, cin, r=8):
        super().__init__()
        mid = max(1, cin // r)
        self.mlp = nn.Sequential(
            nn.Conv2d(cin, mid, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid, cin, 1, bias=False),
        )

    def forward(self, x):
        avg = torch.mean(x, dim=[2, 3], keepdim=True)
        mx = torch.amax(x, dim=[2, 3], keepdim=True)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        avg = torch.mean(x, 1, True)
        mx = torch.amax(x, 1, True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))


class CBAM(nn.Module):
    def __init__(self, cin, r=8):
        super().__init__()
        self.ca = ChannelAttention(cin, r)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class DynAConvLSTMBranch(nn.Module):
    def __init__(self, hidden=(32, 16), k=3, r=8):
        super().__init__()
        self.l1 = ConvLSTMLayer(1, hidden[0], k)
        self.l2 = ConvLSTMLayer(hidden[0], hidden[1], k)
        self.cbam = CBAM(hidden[1], r)
        self.head = nn.AdaptiveAvgPool2d(1)
        self.out_dim = hidden[1]

    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1.unsqueeze(1))
        h2 = self.cbam(h2)
        return self.head(h2).flatten(1)


class StaticBranch(nn.Module):
    def __init__(self, cin, base=16, r=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, base, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(True),
        )
        self.cbam = CBAM(base, r)
        self.head = nn.AdaptiveAvgPool2d(1)
        self.out_dim = base

    def forward(self, x):
        x = x.contiguous().to(memory_format=torch.channels_last)
        return self.head(self.cbam(self.net(x))).flatten(1)


class AConvLSTMModel(nn.Module):
    def __init__(self, n_dyn=4, static_in=4, hidden=(32, 16), k=3, r=8, mlp=64, drop=0.3):
        super().__init__()
        self.dyns = nn.ModuleList([DynAConvLSTMBranch(hidden, k, r) for _ in range(n_dyn)])
        self.static = StaticBranch(static_in, base=16, r=r)
        total = sum(b.out_dim for b in self.dyns) + self.static.out_dim
        self.head = nn.Sequential(
            nn.Linear(total, mlp),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Linear(mlp, 1),
        )

    def forward(self, dyn_list, static):
        feats = [b(x) for b, x in zip(self.dyns, dyn_list)]
        feats.append(self.static(static))
        return self.head(torch.cat(feats, 1)).squeeze(1)


# --------- main ---------
def main():
    t0 = time.time()
    cfg = CONFIG
    k = cfg["patch_size"]
    w = cfg["time_window"]
    assert k % 2 == 1 and w % 2 == 1
    r = k // 2

    # 强制 GPU
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境（脚本强制用 GPU）")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # 数据源
    era5_tp = NCSource(
        "era5_tp",
        cfg["era5_tp_nc"],
        "tp",
        cfg["lat_candidates"],
        cfg["lon_candidates"],
        cfg["time_candidates"],
    )
    era5_t2m = NCSource(
        "era5_t2m",
        cfg["era5_t2m_nc"],
        "t2m",
        cfg["lat_candidates"],
        cfg["lon_candidates"],
        cfg["time_candidates"],
    )
    imerg = NCSource(
        "imerg",
        cfg["imerg_nc"],
        "precipitation",
        cfg["lat_candidates"],
        cfg["lon_candidates"],
        cfg["time_candidates"],
    )
    cldas = NCSource(
        "cldas",
        cfg["cldas_nc"],
        "PRCP_DAILY_SUM",
        cfg["lat_candidates"],
        cfg["lon_candidates"],
        cfg["time_candidates"],
    )

    # 基准点
    gdf = gpd.read_file(cfg["grid_points_shp"])
    if gdf.crs is not None and gdf.crs.to_string().upper() not in ("EPSG:4326", "WGS84"):
        gdf = gdf.to_crs(4326)
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    gdf = gdf.reset_index(drop=True)
    point_lon = gdf["lon"].to_numpy(np.float32)
    point_lat = gdf["lat"].to_numpy(np.float32)
    n_all = len(gdf)

    # 权重与训练期配置
    ckpt_path = _resolve_ckpt_path(cfg["ckpt_path"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_cfg = ckpt.get("cfg", {})
    dynamic_vars = train_cfg.get("DYNAMIC_VARS", ["era5_tp", "imerg", "cldas", "era5_t2m"])
    static_vars = train_cfg.get("STATIC_VARS", ["dem", "slope", "aspect"])
    precip_vars = set(train_cfg.get("PRECIP_VARS", ["era5_tp", "imerg", "cldas"]))
    scale_factors = train_cfg.get(
        "SCALE_FACTORS",
        {"era5_tp": 1.0, "imerg": 1.0, "cldas": 1.0, "era5_t2m": 1.0},
    )
    input_log1p_for_precip = bool(train_cfg.get("INPUT_LOG1P_FOR_PRECIP", True))
    t2m_to_celsius = bool(train_cfg.get("T2M_TO_CELSIUS", True))
    aspect_sincos = bool(train_cfg.get("ASPECT_SINCOS", True))
    static_in_channels = (len(static_vars) + 1) if ("aspect" in static_vars and aspect_sincos) else len(static_vars)

    # 模型
    model = AConvLSTMModel(
        n_dyn=len(dynamic_vars),
        static_in=static_in_channels,
        hidden=tuple(train_cfg.get("conv_lstm_hidden", [32, 16])),
        k=int(train_cfg.get("conv_lstm_kernel", 3)),
        r=int(train_cfg.get("cbam_reduction", 8)),
        mlp=64,
        drop=float(train_cfg.get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    amp_enabled = True

    # demo 天，拿标准化方向
    demo = next(iter(era5_tp.tmap.keys()))
    _ = era5_tp.get_stack7([demo] * 7)
    _ = era5_t2m.get_stack7([demo] * 7)
    _ = imerg.get_stack7([demo] * 7)
    _ = cldas.get_stack7([demo] * 7)

    # 静态覆盖边界检查
    def read_meta(p):
        with rasterio.open(p) as rdr:
            return rdr.height, rdr.width, rdr.transform

    h_dem, w_dem, tf_dem = read_meta(cfg["dem_tif"])
    h_slp, w_slp, tf_slp = read_meta(cfg["slope_tif"])
    h_asp, w_asp, tf_asp = read_meta(cfg["aspect_tif"])

    def ok_latlon(lat, lon, lats, lons):
        iy = _nearest_index(lats, lat)
        ix = _nearest_index(lons, _ensure_lon_units(lon, lons))
        return (iy - r >= 0) and (iy + r < len(lats)) and (ix - r >= 0) and (ix + r < len(lons))

    def ok_raster(lat, lon, hgt, wid, tf):
        rr, cc = rowcol(tf, lon, lat, op=round)
        return (rr - r >= 0) and (rr + r < hgt) and (cc - r >= 0) and (cc + r < wid)

    keep = []
    for row in gdf.itertuples(index=False):
        lat = float(row.lat)
        lon = float(row.lon)
        okd = (
            ok_latlon(lat, lon, era5_tp.lats_std, era5_tp.lons_std)
            and ok_latlon(lat, lon, era5_t2m.lats_std, era5_t2m.lons_std)
            and ok_latlon(lat, lon, imerg.lats_std, imerg.lons_std)
            and ok_latlon(lat, lon, cldas.lats_std, cldas.lons_std)
        )
        oks = (
            ok_raster(lat, lon, h_dem, w_dem, tf_dem)
            and ok_raster(lat, lon, h_slp, w_slp, tf_slp)
            and ok_raster(lat, lon, h_asp, w_asp, tf_asp)
        )
        keep.append(okd and oks)
    keep = np.array(keep, bool)
    gdf_use = gdf[keep].reset_index(drop=True)
    n_keep = len(gdf_use)
    print(f"[Filter] 基准点总数={n_all} | 可用(完整7×7且覆盖)={n_keep} | 其余写 NaN")

    # 预计算：动态 7×7 起点
    def pre_latlon_start(df, lats, lons):
        iy0 = np.empty(len(df), np.int32)
        ix0 = np.empty(len(df), np.int32)
        for i, row in enumerate(df.itertuples(index=False)):
            iy = _nearest_index(lats, float(row.lat))
            ix = _nearest_index(lons, _ensure_lon_units(float(row.lon), lons))
            iy0[i] = iy - r
            ix0[i] = ix - r
        return iy0, ix0

    iy0_tp, ix0_tp = pre_latlon_start(gdf_use, era5_tp.lats_std, era5_tp.lons_std)
    iy0_t2m, ix0_t2m = pre_latlon_start(gdf_use, era5_t2m.lats_std, era5_t2m.lons_std)
    iy0_im, ix0_im = pre_latlon_start(gdf_use, imerg.lats_std, imerg.lons_std)
    iy0_cl, ix0_cl = pre_latlon_start(gdf_use, cldas.lats_std, cldas.lons_std)

    # 静态：一次性向量化提取所有点
    def read_full(p):
        with rasterio.open(p) as rdr:
            arr = rdr.read(1).astype(np.float32)
            if rdr.nodata is not None:
                arr[np.isclose(arr, rdr.nodata)] = np.nan
            tf = rdr.transform
        return arr, tf

    dem_full, tf_dem = read_full(cfg["dem_tif"])
    slp_full, tf_slp = read_full(cfg["slope_tif"])
    asp_full, tf_asp = read_full(cfg["aspect_tif"])

    def rrcc_start(df, tf):
        rr0 = np.empty(len(df), np.int32)
        cc0 = np.empty(len(df), np.int32)
        for i, row in enumerate(df.itertuples(index=False)):
            rr, cc = rowcol(tf, float(row.lon), float(row.lat), op=round)
            rr0[i] = rr - r
            cc0[i] = cc - r
        return rr0, cc0

    rr0_dem, cc0_dem = rrcc_start(gdf_use, tf_dem)
    rr0_slp, cc0_slp = rrcc_start(gdf_use, tf_slp)
    rr0_asp, cc0_asp = rrcc_start(gdf_use, tf_asp)

    dem_win = sliding_window_view(dem_full, (k, k))
    slp_win = sliding_window_view(slp_full, (k, k))
    asp_win = sliding_window_view(asp_full, (k, k))

    dem_patches = dem_win[rr0_dem, cc0_dem]
    slp_patches = slp_win[rr0_slp, cc0_slp]
    if "aspect" in static_vars:
        a7 = asp_win[rr0_asp, cc0_asp].astype(np.float32, copy=False)
        a7 = np.where((a7 < 0) | (~np.isfinite(a7)), np.nan, a7)
        ang = np.deg2rad(a7)
        asp_sin = np.nan_to_num(np.sin(ang), nan=0.0).astype(np.float32, copy=False)
        asp_cos = np.nan_to_num(np.cos(ang), nan=0.0).astype(np.float32, copy=False)
        static_all = np.stack([dem_patches, slp_patches, asp_sin, asp_cos], axis=1)
        static_bad = (
            np.isnan(a7).any(axis=(1, 2))
            | np.isnan(dem_patches).any(axis=(1, 2))
            | np.isnan(slp_patches).any(axis=(1, 2))
        )
    else:
        static_all = np.stack([dem_patches, slp_patches], axis=1)
        static_bad = (
            np.isnan(dem_patches).any(axis=(1, 2))
            | np.isnan(slp_patches).any(axis=(1, 2))
        )

    del dem_full, slp_full, asp_full, dem_win, slp_win, asp_win
    gc.collect()

    # 时间/结果
    all_days = _build_all_days(cfg["start_date"], cfg["end_date"])
    global_ids = np.flatnonzero(keep)
    batch_size = cfg["batch_size"]

    # 动态源映射
    name2src = {
        "era5_tp": (era5_tp, iy0_tp, ix0_tp),
        "era5_t2m": (era5_t2m, iy0_t2m, ix0_t2m),
        "imerg": (imerg, iy0_im, ix0_im),
        "cldas": (cldas, iy0_cl, ix0_cl),
    }

    os.makedirs(cfg["out_dir"], exist_ok=True)

    print(f"[Run] device={device} | days={len(all_days)} | batch_size={batch_size}")
    for ti, day in enumerate(tqdm(all_days, desc="Days", ncols=100)):
        wdates = _dates_for_center(all_days, ti, w)
        ymds = [d.strftime("%Y%m%d") for d in wdates]
        if not all(all(src.has_date(y) for y in ymds) for src, _, _ in name2src.values()):
            continue

        # 读四源整场 + 预变换
        stacks = {}
        for nm, (src, _, _) in name2src.items():
            arrT = src.get_stack7(ymds)
            sc = float(scale_factors.get(nm, 1.0))
            if sc != 1.0:
                arrT *= sc
            if (nm in precip_vars) and input_log1p_for_precip:
                np.clip(arrT, 0.0, None, out=arrT)
                np.log1p(arrT, out=arrT)
            if (nm == "era5_t2m") and t2m_to_celsius:
                arrT -= 273.15
            stacks[nm] = arrT

        # 每变量只建一次 3D 滑窗 (T, Ny-k+1, Nx-k+1, 1, k, k)
        win3d_map = {nm: sliding_window_view(arrT, (1, k, k)) for nm, arrT in stacks.items()}

        day_out = np.full(n_all, np.nan, np.float32)

        for off in tqdm(
            range(0, n_keep, batch_size),
            desc=f"  Batches@{ti + 1:03d}",
            ncols=100,
            leave=False,
        ):
            sl = slice(off, min(off + batch_size, n_keep))
            nB = sl.stop - sl.start
            dyn_list = []
            bad_dyn = np.zeros(nB, bool)
            for nm, (_, iy0_all, ix0_all) in name2src.items():
                win3d = win3d_map[nm]
                iy0 = iy0_all[sl]
                ix0 = ix0_all[sl]
                patches = win3d[:, iy0, ix0, :, :, :]
                patches = np.moveaxis(patches, 0, 1).squeeze(2).astype(np.float32, copy=False)
                if cfg["strict_nan_check"]:
                    bad_dyn |= np.isnan(patches).any(axis=(1, 2, 3))
                dyn_list.append(torch.from_numpy(patches[:, :, None, :, :]))

            static_batch = torch.from_numpy(static_all[sl])
            bad_mask = static_bad[sl].copy()
            if cfg["strict_nan_check"]:
                bad_mask |= bad_dyn

            with torch.no_grad():
                if cfg["half_upload"]:
                    dyn_tensors = [
                        x.to(torch.float16).pin_memory().to(device, non_blocking=True) for x in dyn_list
                    ]
                    static_t = static_batch.to(torch.float16).pin_memory().to(device, non_blocking=True)
                else:
                    dyn_tensors = [x.pin_memory().to(device, non_blocking=True) for x in dyn_list]
                    static_t = static_batch.pin_memory().to(device, non_blocking=True)
                static_t = static_t.to(memory_format=torch.channels_last)
                with amp.autocast(device_type=device.type, enabled=amp_enabled):
                    pred_log = model(dyn_tensors, static_t)
                pred = torch.expm1(pred_log).clamp_min(0).float().cpu().numpy()

            cur_global = global_ids[sl]
            good = ~bad_mask
            day_out[cur_global[good]] = pred[good]

        del stacks, win3d_map
        gc.collect()

        # 写当日文件
        times = np.array([np.datetime64(day.strftime("%Y-%m-%d"))])
        point_ids = np.arange(n_all, dtype=np.int32)
        ds = xr.Dataset(
            {"precip_pred": (("time", "point"), day_out[None, :])},
            coords={
                "time": ("time", times),
                "point": ("point", point_ids),
                "lon": ("point", point_lon),
                "lat": ("point", point_lat),
            },
            attrs={
                "title": "A-ConvLSTM precipitation prediction (daily)",
                "desc": "Predict only points with complete 7x7 across all sources; others are NaN.",
                "units": "mm/day",
                "window": "spatial 7x7, temporal 7 days (±3)",
                "date": day.strftime("%Y-%m-%d"),
            },
        )
        ds["lon"].attrs["units"] = "degrees_east"
        ds["lat"].attrs["units"] = "degrees_north"

        encoding = {
            "precip_pred": {
                "zlib": True,
                "complevel": 4,
                "chunksizes": (1, min(8192, day_out.shape[0])),
            }
        }

        out_path = os.path.join(cfg["out_dir"], f"pred_{day.strftime('%Y%m%d')}.nc")
        ds.to_netcdf(out_path, encoding=encoding)
        print(f"[Save] 写入完成：{out_path}")

    print(f"[Time] total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
