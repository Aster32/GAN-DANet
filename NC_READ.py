# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:56:40 2024

@author: 17689
"""

import os
import numpy as np
import xarray as xr
import netCDF4 as nc
import win
from scipy.ndimage import zoom

def readdata(directory_path='F:/ERA5/'):
    nc_files = [file for file in os.listdir(directory_path) if file.endswith('.nc') or file.endswith('.nc4')]

    # Initialize a dictionary to store datasets for each variable
    datasets_dict = {}
    time = []

    # Loop through each .nc file and read its content
    for file_name in nc_files:
        file_path = os.path.join(directory_path, file_name)
        
        # Open the NetCDF file using netCDF4
        ds = nc.Dataset(file_path)
        
        # Loop through variables in the dataset
        for var_name, var in ds.variables.items():
            # Skip variables that are coordinates
            if var_name == 'time':
                # Collect time values
                time.append(var[:])
                continue
            
            # Convert variables to numpy arrays
            var_values = var[:]
            
            # Check if the variable has only one dimension
            if len(var.dimensions) == 1:
                # For variables with one dimension, stack them along that dimension
                if var_name in datasets_dict:
                    datasets_dict[var_name] = np.concatenate([datasets_dict[var_name], var_values.reshape((-1, 1))], axis=1)
                else:
                    datasets_dict[var_name] = var_values.reshape((-1, 1))
            else:
                # For variables with more than one dimension, concatenate along the time dimension
                var_values = np.expand_dims(var_values, axis=-1)
                if var_name in datasets_dict:
                    datasets_dict[var_name] = np.concatenate([datasets_dict[var_name], var_values], axis=-1)
                else:
                    datasets_dict[var_name] = var_values
                
    if len(datasets_dict) == 0:
        print("No valid datasets found in the NetCDF files.")
        return None, None
    
    # Convert the dictionary to numpy arrays
    for var_name, var_data in datasets_dict.items():
        datasets_dict[var_name] = np.ma.masked_invalid(var_data)
    if len(time)==0:
        return datasets_dict, time
    return datasets_dict, np.concatenate(time)



