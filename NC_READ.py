# -*- coding: utf-8 -*-
"""Utilities for reading ERA5 NetCDF variables into numpy arrays."""

import os
import numpy as np
import netCDF4 as nc


def readdata(directory_path='F:/ERA5/'):
    """Read and stack variables from NetCDF files in a deterministic order."""
    nc_files = sorted(
        file for file in os.listdir(directory_path)
        if file.endswith('.nc') or file.endswith('.nc4')
    )

    datasets_dict = {}
    time = []

    for file_name in nc_files:
        file_path = os.path.join(directory_path, file_name)
        with nc.Dataset(file_path) as ds:
            for var_name, var in ds.variables.items():
                if var_name == 'time':
                    time.append(var[:])
                    continue

                var_values = var[:]
                if len(var.dimensions) == 1:
                    reshaped = var_values.reshape((-1, 1))
                    if var_name in datasets_dict:
                        datasets_dict[var_name] = np.concatenate([datasets_dict[var_name], reshaped], axis=1)
                    else:
                        datasets_dict[var_name] = reshaped
                else:
                    expanded = np.expand_dims(var_values, axis=-1)
                    if var_name in datasets_dict:
                        datasets_dict[var_name] = np.concatenate([datasets_dict[var_name], expanded], axis=-1)
                    else:
                        datasets_dict[var_name] = expanded

    if len(datasets_dict) == 0:
        print('No valid datasets found in the NetCDF files.')
        return None, None

    for var_name, var_data in datasets_dict.items():
        datasets_dict[var_name] = np.ma.masked_invalid(var_data)

    if len(time) == 0:
        return datasets_dict, time

    return datasets_dict, np.concatenate(time)
