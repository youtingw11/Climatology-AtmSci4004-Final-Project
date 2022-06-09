'''
Author: You-Ting Wu @ NTUAS
Date: 2021.10.22 
'''
import numpy as np
import math as m
import pandas as pd
import datetime as dt
import glob
import xarray as xr

# fx. of claculating saturation specific humidity [g/kg]
def calcsatspechum(t,p):

    #T is temperature, P is pressure in hPa
    

    # Formulae from Buck (1981):
    # saturation mixing ratio wrt liquid water [kg/kg]
    es = (1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*(t-273.15)/(240.97+(t-273.15)))
    wsl = 0.622*es/(p-es)

    # saturation mixing ratio wrt ice [kg/kg]
    es = (1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*(t-273.15)/(272.55+(t-273.15)))
    wsi = 0.622*es/(p-es)
    

    ws = wsl.where(t > 273.15, wsi)
    qs = ws/(1+ws)  # saturation specific humidity, kg/kg

    return(qs)

# read files and get data
file_p_sigma = sorted(glob.glob('/work6/YCL.youtingw/CS2022/P_sigma/*.nc'))
file_t = sorted(glob.glob('/work6/YCL.youtingw/CS2022/T/*.nc'))
file_q = sorted(glob.glob('/work6/YCL.youtingw/CS2022/Q/*.nc'))


for i in range(40):
    pmid = xr.open_dataset(file_p_sigma[i])['pmid'][0,:,:,:,:] / 100 # hPa
    q = xr.open_dataset(file_q[i])['Q'][0,:,:,:,:]
    t = xr.open_dataset(file_t[i])['T'][0,:,:,:,:]
    pmid['lat'] = t['lat'] 
    q['lat'] = t['lat'] 

    # Calculate the moisture change in response to 1 K warming at constant relative humidity
    qs0 = calcsatspechum(t, pmid) # Initial saturation mixing ratio
    rh = q / qs0
    qs1k = calcsatspechum(t+1, pmid)
    q1k = rh * qs1k
    q1k = q1k.where(q1k>q, q)
    dq1k = q1k-q # Mixing ratio response for 1 K warming

    filename = '%s_%s.nc'%('dq1k', i)
    print('output to the file: %s\n'%(filename))
    dq1k.to_dataset(name ='dq1k').to_netcdf('/work6/YCL.youtingw/CS2022/%s/%s'%('dq1k', filename))