'''
Author: You-Ting Wu @ NTUAS
Date: 2022.06.09
'''
from inspect import TPFLAGS_IS_ABSTRACT
from unittest import skip
import numpy as np
import math as m
from numpy.core.fromnumeric import sort
import pandas as pd
import datetime as dt
import glob
import xarray as xr
from xarray.core import variable
print('\n--------------------------------------------------------------------------')
print('feedback analysis: calculating partitioning radiative feedbacks using CESM-CAM5')
print('start time: %s'%(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
print('--------------------------------------------------------------------------\n')


######################### initial data setting ###############################
# initial data setting
run_num = 30
ref_year = 1951
year_interval = [2006, 2100]
water_density = 997 #[kg/m^3]
smile_name = ['CESM1-CAM5']
smile_folder_name = ['cesm_lens']
ens = ["001","002","003","004","005","006","007","008","009","010","011","012","013","014","015","016","017","018","019","020", "021","022","023","024","025","026","027","028","029","030","031","032","033","034","035","101","102","103","104","105",]
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

# fx. of claculating saturation specific humidity [g/kg]
def calcsatspechum(t,p):

    #T is temperature, P is pressure in hPa 

    # Formulae from Buck (1981):
    # saturation mixing ratio wrt liquid water [g/kg]
    es = (1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*(t-273.15)/(240.97+(t-273.15)))
    wsl = 0.622*es/(p-es)
    
    # saturation mixing ratio wrt ice [g/kg]
    es = (1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*(t-273.15)/(272.55+(t-273.15)))
    wsi = 0.622*es/(p-es)

    ws = wsl.where(t > 273.15, wsi)
    qs = ws/(1+ws)  # saturation specific humidity, g/kg

    return(qs)
##############################################################################


############################ feedback function ###############################

def data_merge(variable, ens):

    file_hist = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/%s/b.e11.B20TRC5CNBDRD.f09_g16.%s.*.nc'%(variable, ens)))
    file_proj = sorted(glob.glob('/work6/L.yuchiao/cesm1_le/%s/b.e11.BRCP85C5CNBDRD.f09_g16.%s.*.nc'%(variable, ens)))
    print(file_hist)
    print(file_proj)

    data_hist = xr.open_dataset(file_hist[0])['%s'%(variable)]
    data_hist['time'] = data_hist.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
    data_hist = data_hist[data_hist['time.year'] >= ref_year, :, :]

    if len(file_proj)>1:
        data1 = xr.open_dataset(file_proj[0])['%s'%(variable)]
        data2 = xr.open_dataset(file_proj[1])['%s'%(variable)]
        data1['time'] = data1.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
        data2['time'] = data2.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
        data = xr.concat([data_hist, data1, data2], dim='time', join = 'override')
    else:
        data3 = xr.open_dataset(file_proj[0])['%s'%(variable)]
        data3['time'] = data3.indexes['time'].to_datetimeindex(unsafe=True) - np.timedelta64(10*60*60*24, 's')
        data = xr.concat([data_hist, data3], dim='time', join = 'override')

    print('running mean...')
    data_mean = group_running_mean(data)

    if variable == 'PS':
        hyam = t_kernel['hyam']    
        hybm = t_kernel['hybm']
        p0 = t_kernel['P0']
        data_mean = (hyam * p0 + hybm * data_mean)

    return(data_mean)

def feedback_ensemble(type):
    file_list = sorted(glob.glob('%s/*.nc'%(type)))
    data = xr.open_mfdataset(file_list, concat_dim = 'ensemble', combine='nested', parallel = True)
    data.to_netcdf('%s.nc'%(type))
    return()

# calculating radiatvie feedback
def feedback_calculation(type):
    
    print('\n***  calculating %s feedback  ***'%(type))

    # Albedo feedback
    if type == 'albedo':
        # collect surface shortwave radiation fields for calculating albedo change
        # ref.mean (and ensemble mean)
        file_fsns = sorted(glob.glob('/work6/YCL.youtingw/CS2022/FSNS/*.nc'))
        file_fsds = sorted(glob.glob('/work6/YCL.youtingw/CS2022/FSDS/*.nc'))
        file_ts = sorted(glob.glob('/work6/YCL.youtingw/CS2022/TS/*.nc'))

        # read TOA albedo kernel 
        alb_kernel = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/alb.kernel.nc')['FSNT'].rename({'time':'month'}).transpose('month', 'lat', 'lon')
        alb_kernel['month'] = range(1,13)

        for i in range(40):
            SW_sfc_net_2 = xr.open_dataset(file_fsns[i])['FSNS']
            SW_sfc_down_2 = xr.open_dataset(file_fsds[i])['FSDS']
            ts = xr.open_dataset(file_ts[i])['TS']
            print(file_fsns[i])
            print(file_fsds[i])
            print(file_ts[i])

            SW_sfc_net_1 = SW_sfc_net_2.sel(year = ref_year)
            SW_sfc_down_1 = SW_sfc_down_2.sel(year = ref_year)

            # calculate albedo difference
            alb1 = (1 - SW_sfc_net_1/SW_sfc_down_1).fillna(0)
            alb2 = (1 - SW_sfc_net_2/SW_sfc_down_2).fillna(0)
            dalb = (alb2-alb1) * 100

            # calculate albedo feedback in Arctic
            dts = (ts - ts.sel(year = ref_year)).transpose('year', 'month', 'lat', 'lon')
            dSW_alb = (alb_kernel * dalb).transpose('year', 'month', 'lat', 'lon')
            alb_parameter = (dSW_alb / dts)
            dSW_alb_zonal = dSW_alb.mean(dim = 'lon', skipna = True)
            alb_parameter_zonal = alb_parameter.mean(dim = 'lon', skipna = True)
            filename = 'albedo/albedo_feedback_%s.nc'%(i)
            print('output to the file: %s'%(filename))
            file = alb_parameter_zonal.to_dataset(name ='parameter')
            file['energy'] = dSW_alb_zonal
            file.to_netcdf(filename)

            dts_zonal = dts.mean(dim = 'lon', skipna = True)
            filename = 'ts/ts_%s.nc'%(i)
            print('output to the file: %s'%(filename))
            file = dts_zonal.to_dataset(name ='dts_zonal').to_netcdf(filename)
    
    # calculating Lapse rate feedback
    elif type == 'lapse_rate_planck':

            # calculate air temperature change and mask stratosphere
            # ref.mean and projection mean (and ensemble mean)

        for i in range(1,40):

            p = data_merge('PS', ens[i]) / 100
            t = data_merge('T', ens[i])
            ts = data_merge('TS', ens[i])

            dta = (t - t.sel(year = ref_year)).transpose('year', 'month', 'lat', 'lon', 'lev')
            dts = (ts - ts.sel(year = ref_year)).transpose('year', 'month', 'lat', 'lon')
            
            lat = dts['lat']
            x = np.cos(np.radians(lat))
            p_tropopause_zonalmean = 300 - 200 * x
            p_tropopause, _ = xr.broadcast(p_tropopause_zonalmean, lon)
            p_tropopause, _ = xr.broadcast(p_tropopause, lev)
            p_tropopause, _ = xr.broadcast(p_tropopause, p['month'])
            p_tropopause, _ = xr.broadcast(p_tropopause, p['year'])
            p_tropopause = p_tropopause.transpose('year', 'month', 'lat', 'lon', 'lev')

            dta = dta * (p >= p_tropopause)
            dts3d, _ = xr.broadcast(dts, lev)
            dt_planck = dts3d * (p >= p_tropopause)
            dt_lapserate = (dta - dt_planck) * (p >= p_tropopause)

            # Convolve air temperature kernel with air temperature change
            ta_kernel = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/t.kernel.nc')['FLNT'].rename({'time':'month'}).transpose('month', 'lat', 'lon', 'lev')
            ta_kernel['month'] = range(1,13)
            ta_kernel['lat'] = dts['lat']
            ts_kernel = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/ts.kernel.nc')['FLNT'].rename({'time':'month'}).transpose('month', 'lat', 'lon')
            ts_kernel['month'] = range(1,13)
            ts_kernel['lat'] = dts['lat']

            # LW calculation
            dLW_ts = (dts * ts_kernel).transpose('year', 'month', 'lat', 'lon')
            dLW_planck = (dt_planck * ta_kernel).sum(dim = 'lev', skipna = True).transpose('year', 'month', 'lat', 'lon')
            dLW_lapserate = (dt_lapserate * ta_kernel).sum(dim = 'lev', skipna = True).transpose('year', 'month', 'lat', 'lon')
            
            # calculate Planck feedback in Arctic
            dLW_planck_zonal = ((- dLW_planck - dLW_ts)).mean(dim = 'lon', skipna = True)
            planck_feedback_zonal = ((- dLW_planck - dLW_ts)/ dts).mean(dim = 'lon', skipna = True)

            # output to the file
            filename = 'planck/planck_feedback_%s.nc'%(i)
            print('output to the file: %s'%(filename))
            file = planck_feedback_zonal.to_dataset(name ='parameter')
            file['energy'] = dLW_planck_zonal
            file.to_netcdf(filename)

            # calculate lapse rate feedbacks in Arctic
            dLW_lapserate_zonal = (-dLW_lapserate).mean(dim = 'lon', skipna = True)
            lapserate_feedback_zonal = (-dLW_lapserate / dts).mean(dim = 'lon', skipna = True)

            # output to the file
            filename = 'lapserate/lapserate_feedback_%s.nc'%(i)
            print('output to the file: %s'%(filename))
            file = lapserate_feedback_zonal.to_dataset(name ='parameter')
            file['energy'] = dLW_lapserate_zonal
            file.to_netcdf(filename)
    
    # calculating water vapor feedback
    elif type == 'water_vapor':

        file_dq1k = sorted(glob.glob('/work6/YCL.youtingw/CS2022/dq1k/*.nc'))
        file_q = sorted(glob.glob('/work6/YCL.youtingw/CS2022/Q/*.nc'))
        file_ts = sorted(glob.glob('/work6/YCL.youtingw/CS2022/TS/*.nc'))
        file_p_sigma = sorted(glob.glob('/work6/YCL.youtingw/CS2022/P_sigma/*.nc'))

        # Read q kernels 
        q_LW_kernel_raw = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/q.kernel.nc')['FLNT'].rename({'time':'month'}).transpose('month', 'lat', 'lon', 'lev')
        q_SW_kernel_raw = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/q.kernel.nc')['FSNT'].rename({'time':'month'}).transpose('month', 'lat', 'lon', 'lev')

        for i in range(40):
            # get q data
            q = xr.open_dataset(file_q[i])['Q']
            ts = xr.open_dataset(file_ts[i])['TS']
            dts = (ts - ts.sel(year = ref_year)).transpose('year', 'month', 'lat', 'lon')
            p = xr.open_dataset(file_p_sigma[i])['pmid'] / 100 #[hPa]

            lat = dts['lat']
            x = np.cos(np.radians(lat))
            p_tropopause_zonalmean = 300 - 200 * x
            p_tropopause, _ = xr.broadcast(p_tropopause_zonalmean, lon)
            p_tropopause, _ = xr.broadcast(p_tropopause, lev)
            p_tropopause, _ = xr.broadcast(p_tropopause, p['month'])
            p_tropopause, _ = xr.broadcast(p_tropopause, p['year'])
            p_tropopause = p_tropopause.transpose('year', 'month', 'lat', 'lon', 'lev')

            # Normalize kernels by the change in moisture for 1 K warming at constant RH (linear)
            dq1k = xr.open_dataset(file_dq1k[i])['dq1k']
            q_LW_kernel = q_LW_kernel_raw / dq1k
            q_SW_kernel = q_SW_kernel_raw / dq1k

            # mask out the stratosphere
            dq = (q - q.sel(year = ref_year)).transpose('year', 'month', 'lat', 'lon', 'lev')
            dq = dq * (p >= p_tropopause)

            # calculate water vapor feedback in zonal mean
            dLW_q = (q_LW_kernel * dq).sum(dim = 'lev', skipna = True)
            dSW_q = (q_SW_kernel * dq).sum(dim = 'lev', skipna = True)
            dR_q_zonal = (-dLW_q+dSW_q).mean(dim = 'lon', skipna = True)
            q_feedback_zonal = ((-dLW_q+dSW_q) / dts).mean(dim = 'lon', skipna = True)

            filename = 'water_vapor/water_vapor_%s.nc'%(i)
            file = q_feedback_zonal.to_dataset(name ='parameter')
            file['energy'] = dR_q_zonal
            file.to_netcdf(filename)
            print('output to the file: %s'%(filename))

    # calculating cloud feedaback
    elif type == 'cloud':
        # read kernel data

        ts_kernel = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/ts.kernel.nc')['FLNT'].rename({'time':'month'}).transpose('month', 'lat', 'lon').mean(dim = 'lon', skipna = True)
        ts_kernel_clearsky = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/ts.kernel.nc')['FLNTC'].rename({'time':'month'}).transpose('month', 'lat', 'lon').mean(dim = 'lon', skipna = True)
        ta_kernel = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/t.kernel.nc')['FLNT'].rename({'time':'month'}).transpose('month', 'lat', 'lon', 'lev').mean(dim = 'lon', skipna = True)
        ta_kernel_clearsky = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/t.kernel.nc')['FLNTC'].rename({'time':'month'}).transpose('month', 'lat', 'lon', 'lev').mean(dim = 'lon', skipna = True)
        alb_kernel = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/alb.kernel.nc')['FSNT'].rename({'time':'month'}).transpose('month', 'lat', 'lon').mean(dim = 'lon', skipna = True)
        alb_kernel_clearsky = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/alb.kernel.nc')['FSNTC'].rename({'time':'month'}).transpose('month', 'lat', 'lon').mean(dim = 'lon', skipna = True)
        q_LW_kernel_raw = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/q.kernel.nc')['FLNT'].rename({'time':'month'}).transpose('month', 'lat', 'lon', 'lev').mean(dim = 'lon', skipna = True)
        q_SW_kernel_raw = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/q.kernel.nc')['FSNT'].rename({'time':'month'}).transpose('month', 'lat', 'lon', 'lev').mean(dim = 'lon', skipna = True)
        q_LW_kernel_clearsky_raw = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/q.kernel.nc')['FLNTC'].rename({'time':'month'}).transpose('month', 'lat', 'lon', 'lev').mean(dim = 'lon', skipna = True)
        q_SW_kernel_clearsky_raw = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/q.kernel.nc')['FSNTC'].rename({'time':'month'}).transpose('month', 'lat', 'lon', 'lev').mean(dim = 'lon', skipna = True)

        # file paths
        file_q = sorted(glob.glob('/work6/YCL.youtingw/CS2022/Q_zonal/*.nc'))
        file_ts = sorted(glob.glob('/work6/YCL.youtingw/CS2022/TS/*.nc'))
        file_t = sorted(glob.glob('/work6/YCL.youtingw/CS2022/T_zonal/*.nc'))
        file_p_sigma = sorted(glob.glob('/work6/YCL.youtingw/CS2022/P_sigma_zonal/*.nc'))
        file_fsns = sorted(glob.glob('/work6/YCL.youtingw/CS2022/FSNS/*.nc'))
        file_fsds = sorted(glob.glob('/work6/YCL.youtingw/CS2022/FSDS/*.nc'))
        file_fsnt = sorted(glob.glob('/work6/YCL.youtingw/CS2022/FSNT/*.nc'))
        file_fsntc = sorted(glob.glob('/work6/YCL.youtingw/CS2022/FSNTC/*.nc'))
        file_flnt = sorted(glob.glob('/work6/YCL.youtingw/CS2022/FLNT/*.nc'))
        file_flntc = sorted(glob.glob('/work6/YCL.youtingw/CS2022/FLNTC/*.nc'))

        for i in range(40):
            # get basic variables 
            ts = xr.open_dataset(file_ts[i])['TS']
            dts = (ts - ts.sel(year = ref_year)).transpose('year', 'month', 'lat','lon').mean(dim = 'lon', skipna = True)
            t = xr.open_dataset(file_t[i])['T_zonal']
            dta = (t - t.sel(year = ref_year)).transpose('year', 'month', 'lat', 'lev')

            q = xr.open_dataset(file_q[i])['Q_zonal']
            p = xr.open_dataset(file_p_sigma[i])['P_sigma_zonal'] / 100 #[hPa]
            dq = (q - q.sel(year = ref_year)).transpose('year', 'month', 'lat', 'lev')

            lat = dts['lat']
            x = np.cos(np.radians(lat))
            p_tropopause_zonalmean = 300 - 200 * x
            p_tropopause, _ = xr.broadcast(p_tropopause_zonalmean, lev)
            p_tropopause, _ = xr.broadcast(p_tropopause, p['month'])
            p_tropopause, _ = xr.broadcast(p_tropopause, p['year'])
            p_tropopause = p_tropopause.transpose('year', 'month', 'lat', 'lev')
            
            # mask out the stratosphere
            dta = dta * (p >= p_tropopause)
            dq = dq * (p >= p_tropopause)

            # Read change in mixing ratio per degree warming at constant RH
            print('calculating q feedback')
            p['lat'] = t['lat'] 
            q['lat'] = t['lat']
            qs1 = calcsatspechum(t[0,:,:,:], p[0,:,:,:])
            qs2 = calcsatspechum(t, p)
            dqsdt = (qs2 - qs1) / dta
            rh = q[0,:,:,:] / qs1
            dqdt = rh * dqsdt

            # Normalize kernels by the change in moisture for 1 K warming at constant RH (linear)
            q_LW_kernel = q_LW_kernel_raw / dqdt
            q_SW_kernel = q_SW_kernel_raw / dqdt
            q_LW_kernel_clearsky = q_LW_kernel_clearsky_raw / dqdt
            q_SW_kernel_clearsky = q_SW_kernel_clearsky_raw / dqdt

            # Collect surface shortwave radiation fields for calculating albedo change
            print('calculating alb feedback')
            SW_sfc_net_2 = xr.open_dataset(file_fsns[i])['FSNS']
            SW_sfc_down_2 = xr.open_dataset(file_fsds[i])['FSDS']
            SW_sfc_net_1 = SW_sfc_net_2.sel(year = ref_year)
            SW_sfc_down_1 = SW_sfc_down_2.sel(year = ref_year)
            alb1 = (1 - SW_sfc_net_1/SW_sfc_down_1).fillna(0)
            alb2 = (1 - SW_sfc_net_2/SW_sfc_down_2).fillna(0)
            dalb = ((alb2-alb1) * 100).mean(dim = 'lon', skipna = True)

            # LW for sfc. temp. change
            print('calculating energy in y/n cloud')
            dLW_ts = ts_kernel * dts
            dLW_ts_cs = ts_kernel_clearsky * dts

            # LW for temp. change in each layer
            dLW_ta = (ta_kernel * dta).sum(dim = 'lev', skipna = True)
            dLW_ta_cs = (ta_kernel_clearsky * dta).sum(dim = 'lev', skipna = True)

            # SW for albedo change
            dSW_alb = alb_kernel * dalb
            dSW_alb_cs = alb_kernel_clearsky * dalb

            # LW & SW for moisture change
            dLW_q = (q_LW_kernel * dq).sum(dim = 'lev', skipna = True)
            dSW_q = (q_SW_kernel * dq).sum(dim = 'lev', skipna = True)
            dLW_q_cs = (q_LW_kernel_clearsky * dq).sum(dim = 'lev', skipna = True)
            dSW_q_cs = (q_SW_kernel_clearsky * dq).sum(dim = 'lev', skipna = True)

            # Change in Cloud Radiative Effect (CRE)
            print('calculating CRE')
            d_sw = xr.open_dataset(file_fsnt[i])['FSNT'] - xr.open_dataset(file_fsnt[i])['FSNT'].sel(year = ref_year)
            d_sw_cs = xr.open_dataset(file_fsntc[i])['FSNTC'] - xr.open_dataset(file_fsntc[i])['FSNTC'].sel(year = ref_year)
            d_lw = xr.open_dataset(file_flnt[i])['FLNT'] - xr.open_dataset(file_flnt[i])['FLNT'].sel(year = ref_year)
            d_lw_cs = xr.open_dataset(file_flntc[i])['FLNTC'] - xr.open_dataset(file_flntc[i])['FLNTC'].sel(year = ref_year)
            d_cre_sw = (d_sw_cs - d_sw).mean(dim = 'lon', skipna = True)
            d_cre_lw = (d_lw_cs - d_lw).mean(dim = 'lon', skipna = True)

            # Cloud masking of radiative forcing
            # GHG forcing
            ghg_sw = 0
            ghg_lw = 0
            # aerosol forcing
            aerosol_sw = 0
            aerosol_lw = 0
            cloud_masking_of_forcing_sw = aerosol_sw + ghg_sw
            cloud_masking_of_forcing_lw = aerosol_lw + ghg_lw

            # cloud feedback (CRE + cloud masking of radiative forcing + corrections for each feedback)
            dLW_cloud = -d_cre_lw + cloud_masking_of_forcing_lw + (dLW_q_cs-dLW_q) + (dLW_ta_cs-dLW_ta) + (dLW_ts_cs-dLW_ts)
            dSW_cloud = -d_cre_sw + cloud_masking_of_forcing_sw + (dSW_q_cs-dSW_q) + (dSW_alb_cs-dSW_alb)

            # Take the annual average and global area average 
            dLW_cloud_zonal = (-dLW_cloud)
            dSW_cloud_zonal = (dSW_cloud)

            # Divide by the global annual mean surface warming (units: W/m2/K)
            lw_cloud_feedback_zonal = (dLW_cloud / dts)
            sw_cloud_feedback_zonal = (dSW_cloud / dts)

            # output to the files
            filename_lw = 'lw_cloud/lw_cloud_%s.nc'%(i)
            file = lw_cloud_feedback_zonal.to_dataset(name ='parameter')
            file['energy'] = dLW_cloud_zonal
            file.to_netcdf(filename_lw)
            print('output to the file: %s'%(filename_lw))

            filename_sw = 'sw_cloud/sw_cloud_%s.nc'%(i)
            file = sw_cloud_feedback_zonal.to_dataset(name ='parameter')
            file['energy'] = dSW_cloud_zonal
            file.to_netcdf(filename_sw)
            print('output to the file: %s'%(filename_sw))


    elif type == 'dR':
        file_fsnt = sorted(glob.glob('/work6/YCL.youtingw/CS2022/FSNT/*.nc'))
        file_flnt = sorted(glob.glob('/work6/YCL.youtingw/CS2022/FLNT/*.nc'))

        fsnt = xr.open_mfdataset(file_fsnt, concat_dim = 'ensemble', combine='nested', parallel = True)['FSNT']
        flnt = xr.open_mfdataset(file_flnt, concat_dim = 'ensemble', combine='nested', parallel = True)['FLNT']

        dfsnt = fsnt - fsnt.sel(year = 1951)
        dflnt = flnt - flnt.sel(year = 1951)
        dR = (dfsnt - dflnt).mean(dim = 'lon', skipna = True)

        filename = 'dR.nc'
        print('output to the file: %s'%(filename))
        dR.to_dataset(name ='dR').to_netcdf(filename)

    return()
##############################################################################


############################### base field part ##############################
# coordinate info.
ps = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/PS.nc')
t_kernel = xr.open_dataset('/home/youtingw/radiative_kernel/cesm_cam5_kernel/kernels/t.kernel.nc')
lat = ps['lat']
lon = ps['lon']
lev = t_kernel['lev']
gw = t_kernel['gw']
#############################################################################


################################# main part ###################################
# feedback_calculation('albedo')
# feedback_calculation('lapse_rate_planck')
# feedback_calculation('water_vapor')
feedback_calculation('cloud')
# feedback_calculation('dR')

# feedback_ensemble('albedo')
# feedback_ensemble('lapse_rate_planck')
# feedback_ensemble('water_vapor')
feedback_ensemble('lw_cloud')
feedback_ensemble('sw_cloud')
# feedback_ensemble('LHF')
# feedback_ensemble('SHF')
#feedback_ensemble('dOHT')
#############################################################################


print('--------------------------------------------------------------------------')
print('finish time: %s'%(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
print('--------------------------------------------------------------------------\n')
