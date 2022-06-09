'''
Author: You-Ting Wu @ NTUAS
Date: 2021.10.27 
'''
from inspect import TPFLAGS_IS_ABSTRACT
import numpy as np
import math as m
from numpy.core.fromnumeric import sort
import pandas as pd
import datetime as dt
import glob
import xarray as xr

# initial data setting
lat_arctic_lower = 70
lat_arctic_upper = 90
lon_arctic_lower = 0
lon_arctic_upper = 360
run_num = 30
ref_year = 1951
year_interval = [2006, 2100]
water_density = 997 #[kg/m^3]
smile_name = ['CESM1-CAM5']
smile_folder_name = ['cesm_lens']


# fx. of grouping the data (into 12 months) and calculating running mean
def group_running_mean(data):

    # groupby based on month and calculate moving data
    data_moving = data.groupby('time.month')[1].rolling(time = run_num, center=True).mean()
    for i in range(11):
        tmp = data.groupby('time.month')[i+2].rolling(time = run_num, center=True).mean()
        data_moving = xr.concat([data_moving, tmp], dim = 'month', join = 'override')

    # revise dimensions 
    data_moving = data_moving.rename({'time': 'year'})
    year_interval = [data_moving['year.year'][0].values, data_moving['year.year'][-1].values]
    data_moving['year'] = np.arange(year_interval[0] - int(run_num/2), year_interval[1] - int(run_num/2) + 1)
    data_moving['month'] = np.arange(1, 13)

     # drop nan value
    data_moving = data_moving.sel(year = slice(year_interval[0], year_interval[1] - int(run_num/2) - int((run_num-1)/2) ))
    
    return(data_moving)

# fx. of calculating multi-ensemble ref. mean of given variable 
def reference_mean(variable):

    # read multi-ensemble data
    file_init = glob.glob('/work6/L.yuchiao/cesm1_le/%s/b.e11.B20TRC5CNBDRD.*185001-200512.nc'%(variable))
    file_list = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/%s/b.e11.B20TRC5CNBDRD.*192001-200512.nc'%(variable)))
    data = xr.open_mfdataset(file_list, concat_dim = 'ensemble',\
                    combine='nested', join = 'override', parallel = True)
    data_other = xr.open_dataset(file_init[0])

    # revise time shifting problem and get data in time interval we want 
    data['time'] = data.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
    data_other['time'] = data_other.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
    data = data['%s'%(variable)][:, data['time.year'] >= ref_year, :, :]
    data_other = data_other['%s'%(variable)][data_other['time.year'] >= ref_year, :, :]
    ens_data = xr.concat([data, data_other], dim = 'ensemble', join = 'override')

    # running mean
    ref_mean = group_running_mean(ens_data)

    return(ref_mean)

# fx. of calculating multi-ensemble mean of given variable (after 2006)
def projection_mean(variable):

    # read multi-ensemble data w/ seperate data
    file_list_1 = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/%s/b.e11.BRCP85C5CNBDRD.*200601-208012.nc'%(variable)))
    file_list_2 = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/%s/b.e11.BRCP85C5CNBDRD.*208101-210012.nc'%(variable)))    
    data1 = xr.open_mfdataset(file_list_1, concat_dim = 'ensemble', combine='nested', join = 'override', parallel = True)['%s'%(variable)]
    data2 = xr.open_mfdataset(file_list_2, concat_dim = 'ensemble', combine='nested', join = 'override', parallel = True)['%s'%(variable)]
    
    # revise time shifting problem and merge
    data1['time'] = data1.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
    data2['time'] = data2.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
    data = xr.concat([data1, data2], dim = 'time', join = 'override')

    # read data w/ integrated data and merge
    file_list_3 = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/%s/b.e11.BRCP85C5CNBDRD.*200601-210012.nc'%(variable)))
    for i in range(len(file_list_3)):
        d = xr.open_dataset(file_list_3[i])['%s'%(variable)]
        d['time'] = d.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
        data = xr.concat([data, d], dim = 'ensemble', join = 'override')

    # running mean
    proj_data = group_running_mean(data)

    return(proj_data)


file_PS = sorted(glob.glob('/work6/YCL.youtingw/CS2022/PS/*.nc'))
t_kernel = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/t.kernel.nc')
hyam = t_kernel['hyam']
hybm = t_kernel['hybm']
p0 = t_kernel['P0']

for i in range(40):
    ps = xr.open_dataset(file_PS[i])['PS']
    p_simga = (hyam * p0 + hybm * ps).transpose('year', 'month', 'lat', 'lon', 'lev')
    filename = '%s_%s.nc'%('P_sigma', i)
    print('output to the file: %s\n'%(filename))
    p_simga.to_dataset(name ='pmid').to_netcdf('/work6/YCL.youtingw/CS2022/%s/%s'%('P_sigma', filename))
