import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np
import csv
import os
import csv
import xarray as xr
#filename = 'F:/data_op/file_path.csv'
def readdata(directory_path = 'F:/data_op/TWSA/'):
    nc_files = [file for file in os.listdir(directory_path) if file.endswith('.nc') or file.endswith('.nc4')]

    # Initialize an empty list to store the datasets
    datasets = []
    times=[]
    # Loop through each .nc file and read its content
    if directory_path=='F:/data_op/TWSA/':
        for file_name in nc_files:
            file_path = os.path.join(directory_path, file_name)
        
            # Open the NetCDF file using xarray
            ds = xr.open_dataset(file_path)
            twsa_data = ds['twsan']
            time = ds['time']
            # Append the dataset to the list
           # datasets.append(ds)
            #twsa_data=np.array(twsa_data)
            datasets.append(twsa_data)
            times.append(time)
        # Stack the datasets along the first dimension
        stacked_data = xr.concat(datasets, dim='time')
        stacked_time=xr.concat(times,dim='time')
        numpy_array = stacked_data.values
        stacked_time = stacked_time.values
    else:
        for file_name in nc_files:
            file_path = os.path.join(directory_path, file_name)
        
            # Open the NetCDF file using xarray
            ds = xr.open_dataset(file_path)
            dss=nc.Dataset(file_path)
            keys=dss.variables.keys()
            for key in keys:
                dd=dss.variables[key]
                cd=np.array(dd)
               # print(np.max(cd))
                #print(np.min(cd))
                if cd.size>10000:
                    break
            keys=ds.keys()
            #print(keys)
           # print(keys)
            for key in keys:
                if cd.size>100:
                    break
                dd=ds[key].data[6]
                cd=np.array(dd)
                #print(np.max(cd))
                #print(np.min(cd))
                if cd.size>10000:
                    break
            twsa_data = cd
            try:
                time = ds['time']
            except:
                print('no time info detected')
                datasets.append(twsa_data)
                datasets=xr.DataArray(datasets)
                #print(np.max(datasets))
                stacked_data = xr.concat(datasets, dim='time')
                numpy_array = stacked_data.values
                #print(np.max(numpy_array))
                stacked_time = []
                return numpy_array, stacked_time
            # Append the dataset to the list
           # datasets.append(ds)
            #twsa_data=np.array(twsa_data)
            datasets.append(twsa_data)
            times.append(time)
            
        # Stack the datasets along the first dimension
        datasets=xr.DataArray(datasets)
        #print(np.max(datasets))
        stacked_data = xr.concat(datasets, dim='time')
        stacked_time=xr.concat(times,dim='time')
        numpy_array = stacked_data.values
        #print(np.max(numpy_array))
        stacked_time = stacked_time.values
    return numpy_array, stacked_time
#ncfile = Dataset("DATA/GRD-3_2002095-2002120_GRAC_GFZOP_BA01_0600_LND_v04.nc")
#dd=readdata()
#print(dd)

#ncfile
#print(ncfile.variables.keys())
'''
import matplotlib.pyplot as plt
long = ncfile["lon"][:]
long
lat = ncfile["lat"][:]
lat
lwe=ncfile["lwe_thickness"][:]

lwe.shape
long.shape
lat.shape
lwe_m=np.reshape(lwe,(180,360))
#print(lwe_m[0,:])

plt.contourf(lwe[0])
plt.colorbar(label="lwe_thickness", orientation="horizontal")
#print(long)
#print(lat)
plt.show()
'''