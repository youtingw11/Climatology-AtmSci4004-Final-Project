'''
Author: You-Ting Wu @ NTUAS
Date: 2021.11.10
'''
import datetime as dt
import glob
import math as m
from inspect import TPFLAGS_IS_ABSTRACT

import numpy as np
import pandas as pd
from six import reraise
import xarray as xr
from netCDF4 import Dataset
from numpy.core.fromnumeric import sort

print('\n--------------------------------------------------------------------------')
print('ocean heat uptake: calculating ocean heat transport in Arctic in CESM-CAM5')
print('start time: %s'%(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
print('--------------------------------------------------------------------------\n')


######################### initial data setting ###############################
# initial data setting
lat_arctic_lower = 70
lat_arctic_upper = 90
lon_arctic_lower = 0
lon_arctic_upper = 360
Cp = 3996. # specific heat for seawater [J/kg/K]
rho_sw = 1026.
run_num = 30
ref_year = 1951
year_interval = [2006, 2100]
water_density = 997 #[kg/m^3]
smile_name = ['CESM1-CAM5']
smile_folder_name = ['cesm_lens']
days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
TAREA = xr.open_dataarray('TAREA.nc')
#############################################################################


########################### calculating function #############################
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
##############################################################################


############################ feedback function ###############################
# fx. of calculating multi-ensemble mean of given variable (after 2006)
def projection_map(variable):

    # read multi-ensemble data w/ seperate data
    file_list_1 = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/ocn/%s/b.e11.BRCP85C5CNBDRD.*200601-208012.nc'%(variable)))
    file_list_2 = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/ocn/%s/b.e11.BRCP85C5CNBDRD.*208101-210012.nc'%(variable)))
    file_list_3 = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/ocn/%s/b.e11.BRCP85C5CNBDRD.*200601-210012.nc'%(variable)))    
    
    for i in range(len(file_list_1)):
        print(file_list_1[i])
        file1 = xr.open_dataset(file_list_1[i])
        t1 = file1['%s'%(variable)]

        print(file_list_2[i])
        file2 = xr.open_dataset(file_list_2[i])
        t2 = file2['%s'%(variable)]
        t_tmp = xr.concat([t1, t2], dim = 'time')

        print('Calculating OHC...')
        dz = file1['dz'] / 100.
        t_tmp = t_tmp.where(abs(t_tmp)<1.e20, np.nan)
        t_tmp = t_tmp + 273.15
        ohc = t_tmp.fillna(0).dot(dz) * Cp * rho_sw

        filename = 'OHC_proj_arctic/ohc_%s_200601-210012.nc'%(i)
        print('output to the file: %s'%(filename))
        ohc.to_dataset(name ='OHC').to_netcdf(filename)

    for i in range(len(file_list_3)):
        
        print(file_list_3[i])
        file3 = xr.open_dataset(file_list_3[i])
        t_tmp = file3['%s'%(variable)]

        print('Calculating OHC...')
        dz = file3['dz'] / 100.
        t_tmp = t_tmp.where(abs(t_tmp)<1.e20, np.nan)
        t_tmp = t_tmp + 273.15
        ohc = t_tmp.fillna(0).dot(dz) * Cp * rho_sw

        filename = 'OHC_proj_arctic/ohc_%s_200601-210012.nc'%(i+33)
        print('output to the file: %s'%(filename))
        ohc.to_dataset(name ='OHC').to_netcdf(filename)
        
    return()

# fx. of calculating multi-ensemble mean of given variable (before 2006)
def historical_map(variable):

    # read multi-ensemble data w/ seperate data
    file_init_hist = glob.glob('/work6/L.yuchiao/cesm1_le/ocn/%s/b.e11.B20TRC5CNBDRD.*185001-200512.nc'%(variable))
    file_list_hist = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/ocn/%s/b.e11.B20TRC5CNBDRD.*192001-200512.nc'%(variable)))

    for i in range(40):
        
        # get historical data
        if i == 0:
            file = xr.open_dataset(file_init_hist[0])
            t_tmp = file['%s'%(variable)]
            print(file_init_hist[0])
        else:
            file = xr.open_dataset(file_list_hist[i-1])
            t_tmp = file['%s'%(variable)]
            print(file_list_hist[i-1])
        
        t_tmp['time'] = t_tmp.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
        t_tmp = t_tmp[t_tmp['time.year'] >= ref_year, :, :, :]

        
        print('Calculating OHC...')
        dz = file['dz'] / 100.
        t_tmp = t_tmp.where(abs(t_tmp)<1.e20, np.nan)
        t_tmp = t_tmp + 273.15
        ohc = t_tmp.fillna(0).dot(dz) * Cp * rho_sw

        filename = 'ohc_%s_195101-200512.nc'%(i)
        print('output to the file: %s\n'%(filename))
        ohc.to_dataset(name ='OHC').to_netcdf('/work6/YCL.youtingw/OHC_hist/%s'%(filename))
        
        
    return(dz, t_tmp, ohc)

# fx. of calculating ocean heat transport
def ocean_heat_transport_calculating():

    # read file in one dataset
    file_hist = sorted(glob.glob('/work6/YCL.youtingw/OHC_hist/*.nc'))
    file_proj = sorted(glob.glob('/work6/YCL.youtingw/OHC_proj/*.nc'))

    for i in range(40):
        data_hist = xr.open_dataset(file_hist[i])['OHC']
        data_proj = xr.open_dataset(file_proj[i])['OHC']
        print(file_hist[i])
        print(file_proj[i])

        # revise time shifting problem and merge data
        data_proj['time'] = data_proj.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
        data = xr.concat([data_hist, data_proj], dim = 'time', join = 'override')

        # calculating dOHC/dt (by using central finite diff.)
        ohc1 = data.roll(time = 1, roll_coords = False)
        ohc2 = data.roll(time = -1, roll_coords = False)
        dohcdt = (ohc2 - ohc1) / 2
        dohcdt[0,:,:] = np.asarray(data[1,:,:]) - np.asarray(data[0,:,:])
        dohcdt[-1,:,:] = np.asarray(data[-1,:,:]) - np.asarray(data[-2,:,:])
    
        # convert time unit to second
        sec_per_month = xr.DataArray(np.tile(days_per_month, int(len(dohcdt['time'])/12)) * 86400, [ ('time', data['time']) ])
        dohcdt = dohcdt / sec_per_month
        
        # zonal mean
        dohcdt_zonal = dohcdt.fillna(0).mean(dim = 'nlon', skipna = True)
        
        # output to the flie
        filename = 'OHT/OHT_%s.nc'%(i)
        dohcdt_zonal.to_dataset(name ='OHT').to_netcdf(filename)

    return()

# fx. of calculating dOHT in running mean diff. metric
def dOHT_calculating():

    # read file
    file = sorted(glob.glob('OHT/*.nc'))

    # calculate dOHT (running mean diff.) and output
    for i in range(40):
        data = xr.open_dataset(file[i])
        print(file[i])
        running_mean_data = group_running_mean(data)
        dOHT = (running_mean_data- running_mean_data.sel(year = ref_year)).transpose('year', 'month','nlat').rename({'nlat': 'lat'})
        
        # output to the flie
        filename = 'dOHT/dOHT%s.nc'%(i)
        dOHT.to_netcdf(filename)

    return()

##############################################################################


# file = sorted(glob.glob('dOHT/*.nc'))
# data = xr.open_mfdataset(file, concat_dim = 'ensemble', combine='nested', join = 'override', parallel = True)['OHT']
# data.to_netcdf('dOHT.nc')

def ensemble_mean_proj(variable):

    file_list = sorted(glob.glob('/home/youtingw/radiative_kernel/cesm_cam5_kernel/%s_proj_arctic/*.nc'%(variable)))
    data = xr.open_mfdataset(file_list, concat_dim = 'ensemble', combine='nested', join = 'override', parallel = True)['%s_proj'%(variable)]
    data.mean('ensemble').to_dataset(name ='%s'%(variable)).to_netcdf('%s_proj_arctic.nc'%(variable))

    return()



################################# main part ###################################
#projection_map('TEMP')
#historical_map('TEMP')
# ocean_heat_transport_calculating()
# dOHT_calculating()
#############################################################################


print('--------------------------------------------------------------------------')
print('finish time: %s'%(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
print('--------------------------------------------------------------------------\n')


def zonal_annual(data, var):

    if var == 'TS':
        ylim = [0,20]
        ylabel = '[K]'
        title = 'Surface Temperature'

    elif var == 'albedo':
        ylim = [0,35]
        ylabel = '[W$m^{-2}$]'
        title = 'Albedo Feedback'
    
    elif var == 'albedo_parameter':
        ylim = [-15,15]
        ylabel = '[W$m^{-2}K^{-1}$]'
        title = 'Albedo Feedback Parameter'
    
    elif var == 'lapserate':
        ylim = [-10,30]
        ylabel = '[W$m^{-2}$]'
        title = 'Lapse-rate Feedback'

    elif var == 'planck':
        ylim = [-50,30]
        ylabel = '[W$m^{-2}$]'
        title = 'Planck Feedback'

    elif var == 'water_vapor':
        ylim = [-10,20]
        ylabel = '[W$m^{-2}$]'
        title = 'Water Vapor Feedback'
    
    elif var == 'lw_cloud':
        ylim = [-10,40]
        ylabel = '[W$m^{-2}$]'
        title = 'LW Cloud Feedback'

    elif var == 'sw_cloud':
        ylim = [-80,10]
        ylabel = '[W$m^{-2}$]'
        title = 'SW Cloud Feedback'

    elif var == 'cloud':
        ylim = [-40,10]
        ylabel = '[W$m^{-2}$]'
        title = 'Cloud Feedback'

    elif var == 'dH0':
        ylim = [-5,10]
        ylabel = '[W$m^{-2}$]'
        title = '$\Delta H_{0}$'
    
    elif var == 'dET':
        ylim = [-10,12]
        ylabel = '[W$m^{-2}$]'
        title = '$\Delta$Energy Convergence'

    elif var == 'dR':
        ylim = [-10,30]
        ylabel = '[W$m^{-2}$]'
        title = '$\Delta$R'


    year = [2011, 2041, 2071]
    color = ['#F5F469', '#86A8E7', '#D16BA5']
    data_annual = data.mean('month')

    
    # main plot
    fig,axes = plt.subplots(nrows = 1, ncols = 1,figsize = (10,6), dpi=300, constrained_layout=True)
    label_fontsize = 20

    ax = axes
    for i in range(3):
        # time series
        ax.plot(data_annual['lat'], data_annual.mean('ensemble').loc[year[i]], linewidth = 1, color = color[i])
        ax.fill_between(data_annual['lat'],\
            data_annual.min('ensemble').loc[year[i]],\
            data_annual.max('ensemble').loc[year[i]],\
            color = color[i], alpha = 0.3)
    
    ax.legend(['2011-2040','2041-2070','2071-2100'], fontsize = label_fontsize, loc = 'upper left')
    ax.set_ylim(ylim)
    ax.set_xlim([0.5, 12.5])
    ax.set_xticks(np.arange(-90,91,30))
    ax.set_xticklabels(['-90','-60','-30', 'EQ', '30', '60', '90'])
    ax.tick_params(axis="both", labelsize = label_fontsize)
    ax.set_ylabel(ylabel, fontsize = label_fontsize)
    ax.set_title('%s (annual mean)'%(title),fontsize = label_fontsize*1.5)


    plt.style.use("dark_background")
    plt.savefig('%s_zonal_annual.jpg'%(var))

    return()