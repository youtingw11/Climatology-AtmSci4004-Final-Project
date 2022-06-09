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
print('\n--------------------------------------------------------------------------')
print('merge data')
print('start time: %s'%(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
print('--------------------------------------------------------------------------\n')


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


def ensemble_mean_proj(variable):

    file_list = sorted(glob.glob('/home/youtingw/radiative_kernel/cesm_cam5_kernel/%s_proj_arctic/*.nc'%(variable)))
    data = xr.open_mfdataset(file_list, concat_dim = 'ensemble', combine='nested', join = 'override', parallel = True)['%s_proj'%(variable)]
    data.mean('ensemble').to_dataset(name ='%s'%(variable)).to_netcdf('%s_proj_arctic.nc'%(variable))

    return()

# fx. of merging data and calculating running mean
def ensemble_merge_mean(variable):

    # read multi-ensemble data w/ seperate data
    file_init_hist = glob.glob('/work6/L.yuchiao/cesm1_le/%s/b.e11.B20TRC5CNBDRD.*185001-200512.nc'%(variable))
    file_list_hist = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/%s/b.e11.B20TRC5CNBDRD.*192001-200512.nc'%(variable)))
    file_list_1 = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/%s/b.e11.BRCP85C5CNBDRD.*200601-208012.nc'%(variable)))
    file_list_2 = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/%s/b.e11.BRCP85C5CNBDRD.*208101-210012.nc'%(variable)))
    file_list_3 = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/%s/b.e11.BRCP85C5CNBDRD.*200601-210012.nc'%(variable)))    
   

    for i in range(40):
        
        # get historical data
        if i == 0:
            data_hist = xr.open_dataset(file_init_hist[0])['%s'%(variable)]
            print(file_init_hist[0])
        else:
            data_hist = xr.open_dataset(file_list_hist[i-1])['%s'%(variable)]
            print(file_list_hist[i-1])
        
        data_hist['time'] = data_hist.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
        if len(data_hist.dims) == 4:
            data_hist = data_hist[data_hist['time.year'] >= ref_year, :, :]
        else:
            data_hist = data_hist[data_hist['time.year'] >= ref_year, :]

        # get projection data and merge
        if i < len(file_list_1):
            data1 = xr.open_dataset(file_list_1[i])['%s'%(variable)]
            data2 = xr.open_dataset(file_list_2[i])['%s'%(variable)]
            print(file_list_1[i])
            print(file_list_2[i])
            data1['time'] = data1.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
            data2['time'] = data2.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
            data = xr.concat([data_hist, data1, data2], dim = 'time', join = 'override')
        else:
            data3 = xr.open_dataset(file_list_3[i-33])['%s'%(variable)]
            print(file_list_3[i-33])
            data3['time'] = data3.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
            data = xr.concat([data_hist, data3], dim = 'time', join = 'override')
        
        # running mean
        print('running mean ... ')
        running_mean_data = group_running_mean(data)

        # mask Arctic data and output to .nc file
        lat = running_mean_data['lat']
        lon = running_mean_data['lon']
        
        if len(running_mean_data.dims) == 5:
            running_mean_data = running_mean_data.transpose('year', 'month', 'lat', 'lon', 'lev')
        else:
            running_mean_data = running_mean_data.transpose('year', 'month', 'lat', 'lon')

        filename = '%s_%s.nc'%(variable, i)
        print('output to the file: %s\n'%(filename))
        running_mean_data.to_dataset(name ='%s'%(variable)).to_netcdf('/work6/YCL.youtingw/CS2022/%s/%s'%(variable, filename))
        
        
    return()


# 3D
'''
file_PS = sorted(glob.glob('/work6/YCL.youtingw/CS2022/PS/*.nc'))
t_kernel = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/t.kernel.nc')
hyam = t_kernel['hyam']
hybm = t_kernel['hybm']
p0 = t_kernel['P0']

def p_sigma(file_PS):
    for i in range(40):
        ps = xr.open_dataset(file_PS[i])['PS']
        p_simga = (hyam * p0 + hybm * ps).transpose('year', 'month', 'lat', 'lon', 'lev')
        filename = '%s_%s.nc'%('P_sigma', i)
        p_simga.to_dataset(name ='pmid').to_netcdf('/work6/YCL.youtingw/CS2022/%s/%s'%('P_sigma', filename))
        print('output to the file: %s\n'%(filename))
    return()
'''

#p_sigma(file_PS)
#ensemble_merge_mean('T')
#ensemble_merge_mean('Q')

# 2D
#ensemble_merge_mean('PS')
#ensemble_merge_mean('TS')
#ensemble_merge_mean('FSNS')
#ensemble_merge_mean('FSDS')
# ensemble_merge_mean('FSNT')
# ensemble_merge_mean('FSNTC')
# ensemble_merge_mean('FLNT')
# ensemble_merge_mean('FLNTC')
# ensemble_merge_mean('FLNS')
# ensemble_merge_mean('LHFLX')
# ensemble_merge_mean('SHFLX')
# ensemble_merge_mean('LANDFRAC')

def data_zonal(var):
    url = '/work6/YCL.youtingw/CS2022'
    file_list = glob.glob('%s/%s/*.nc'%(url,var))

    for i in range(40):
        if var == 'P_sigma':
            data = xr.open_dataset(file_list[i])['pmid']
        else:
            data = xr.open_dataset(file_list[i])[var]
        data_zonal = data.mean(dim = 'lon', skipna = True)

        filename = '%s_zonal_%s.nc'%(var, i)
        data_zonal.to_dataset(name ='%s_zonal'%(var)).to_netcdf('/work6/YCL.youtingw/CS2022/%s_zonal/%s'%(var, filename))
        print('output to the file: %s\n'%(filename))


    return()

data_zonal('P_sigma')
data_zonal('T')
data_zonal('Q')

print('--------------------------------------------------------------------------')
print('finish time: %s'%(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
print('--------------------------------------------------------------------------\n')
