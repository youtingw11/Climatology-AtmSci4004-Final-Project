#%%
from crypt import methods
from operator import imod
import numpy as np
import netCDF4 as nc
from numpy.lib.function_base import append
import pandas as pd
import datetime as dt
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors 
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs
import matplotlib.path as mpath
from scipy.interpolate import griddata
from xarray.core.common import C
from matplotlib.colors import LinearSegmentedColormap
import copy
import os
import imageio

# initial data setting
month_abbr = ['Jan.','Feb.','Mar.','Apr.','May.','Jun.','Jul.','Aug.','Sep.','Oct.','Nov.','Dec.']
season_abbr = ['MAM', 'JJA', 'SON', 'DJF']
alph = ['(a)','(b)','(c)','(d)','(e)','(f)',
        '(g)','(h)','(i)','(j)','(k)','(l)',
        '(m)','(n)','(o)','(p)','(q)','(r)',
        '(s)','(t)','(u)','(v)','(w)','(x)']
run_num = 30
year_interval = [1951, 2100]


#%%
################################################################################################
alb = xr.open_dataset('data/albedo.nc')
planck = xr.open_dataset('data/planck.nc')
ts = xr.open_dataset('data/ts.nc')['dts_zonal']
lr = xr.open_dataset('data/lapserate.nc')
wr = xr.open_dataset('data/water_vapor.nc')
lw_cloud = xr.open_dataset('data/lw_cloud.nc')
sw_cloud = xr.open_dataset('data/sw_cloud.nc')
dR = xr.open_dataset('data/dR.nc')['dR']
dH0 = xr.open_dataset('data/dOHT.nc')['OHT']
dH0['lat'] = dH0['lat'] = dH0['lat']*180/384-90

dH0_regrid = dH0.interp(lat = dR['lat'], method = 'nearest')
dH0_regrid['lat'] = dR['lat']
dET = (dR - dH0_regrid).transpose('ensemble','year','month','lat')

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
    ax.set_ylim([-40,40])
    ax.set_xlim([0.5, 12.5])
    ax.set_xticks(np.arange(-90,91,30))
    ax.set_xticklabels(['-90','-60','-30', 'EQ', '30', '60', '90'])
    ax.tick_params(axis="both", labelsize = label_fontsize)
    ax.set_ylabel(ylabel, fontsize = label_fontsize)
    ax.set_title('%s (annual mean)'%(title),fontsize = label_fontsize*1.5)


    plt.style.use("dark_background")
    plt.savefig('%s_zonal_annual.jpg'%(var))

    return()


# zonal_annual(ts, var = 'TS')
# zonal_annual(alb['energy'], var = 'albedo')
# zonal_annual(lr['energy'], var = 'lapserate')
# zonal_annual(planck['energy'], var = 'planck')
# zonal_annual(wr['energy'].transpose('ensemble','year','month','lat'), var = 'water_vapor')
# # zonal_annual(lw_cloud['energy'], var = 'lw_cloud')
# # zonal_annual(sw_cloud['energy'], var = 'sw_cloud')
# zonal_annual(sw_cloud['energy']+lw_cloud['energy'], var = 'cloud')
# zonal_annual(dH0, var = 'dH0')
# zonal_annual(dET, var = 'dET')
# zonal_annual(dR, var = 'dR')

# fx. of producing animation in gif. file
def make_gif(gif_name):

    # build gif
    with imageio.get_writer('%s'%(gif_name), mode='I', fps = 8) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        
    # Remove files
    for filename in set(filenames):
        os.remove(filename)

    return()


data = [alb['energy'], lr['energy'], planck['energy'], wr['energy'].transpose('ensemble','year','month','lat'), sw_cloud['energy']+lw_cloud['energy'], dH0, dET]
def zonal_annual_gif(year):

    color = ['#F66C7D', '#F4BA78', '#F9F871', '#9FD17D', '#75B5EB', '#E25ED3','white']
    # year = 2071

    # main plot
    fig,axes = plt.subplots(nrows = 1, ncols = 1,figsize = (10,6), dpi=300, constrained_layout=True)
    label_fontsize = 20
    ylim = [-40,40]
    ylabel = '[W$m^{-2}$]'

    ax = axes
    for i in range(7):
        data_annual = data[i].mean('month')
        ax.plot(data_annual['lat'], data_annual.mean('ensemble').loc[year], linewidth = 1, color = color[i])
        ax.fill_between(data_annual['lat'],\
            data_annual.min('ensemble').loc[year],\
            data_annual.max('ensemble').loc[year],\
            color = color[i], alpha = 0.3)

    ax.legend(['Albdeo','Lapse-rate','Planck','Water Vapor','Cloud','$\Delta H_{0}$','$\Delta$ET'], fontsize = label_fontsize-5, loc = 'upper left',ncol = 3)
    # ax.legend(['Albdeo','Lapse-rate','Planck','Water Vapor','Cloud'], fontsize = label_fontsize-5, loc = 'upper left',ncol = 2)
    ax.set_ylim(ylim)
    ax.set_xlim([0.5, 12.5])
    ax.set_xticks(np.arange(-90,91,30))
    ax.set_xticklabels(['-90','-60','-30', 'EQ', '30', '60', '90'])
    ax.tick_params(axis="both", labelsize = label_fontsize)
    ax.set_ylabel(ylabel, fontsize = label_fontsize)
    ax.set_title('%s-%s mean'%(year, year+29),fontsize = label_fontsize*1.2)

    return()


def zonal_seasonal(data, var):

    if var == 'TS':
        unit = '[K]'
        title = 'Surface Temperature'
        bounds = np.arange(0,22, 2)
        cmap_color = 'Oranges'

    elif var == 'albedo':
        unit = '[W$m^{-2}$]'
        bounds = np.arange(-40,41, 8)
        title = 'Albedo Feedback'
        cmap_color = 'RdBu_r'
    
    elif var == 'lapserate':
        unit = '[W$m^{-2}$]'
        bounds = np.arange(-40,41, 8)
        title = 'Lapse-rate Feedback'
        cmap_color = 'RdBu_r'

    elif var == 'planck':
        unit = '[W$m^{-2}$]'
        bounds = np.arange(-40,41, 8)
        title = 'Planck Feedback'
        cmap_color = 'RdBu_r'

    data_mean = data.mean('ensemble').loc[2071,:,:]

    # main plot
    fig,axes = plt.subplots(nrows = 1, ncols = 1,figsize = (10,6), dpi=300, constrained_layout=True)
    label_fontsize = 20

    ax = axes

    cf = ax.contourf(data_mean['lat'], range(1,13), data_mean,
            cmap = cmap_color, levels = bounds, extend = 'both', alpha = 1)

    # axis setting
    ax.set_ylim([0.8, 12.2])
    ax.set_yticks(range(1,13))
    ax.set_yticklabels(month_abbr)
    ax.set_xlim([0.5, 12.5])
    ax.set_xticks(np.arange(-90,91,30))
    ax.set_xticklabels(['-90','-60','-30', 'EQ', '30', '60', '90'])
    ax.tick_params(axis="both", labelsize = label_fontsize)
    ax.set_title('%s'%(title),fontsize = label_fontsize*1.5)

    cbar1 = fig.colorbar(cf, ticks=bounds, aspect = 30, ax = fig.axes, orientation='horizontal')
    cbar1.ax.tick_params(labelsize = label_fontsize-6)
    cbar1.set_label('%s'%(unit), fontsize = label_fontsize-6)

    plt.style.use("dark_background")
    plt.savefig('%s_zonal_seasonal.jpg'%(var))

    return()


# zonal_annual_gif(year = dET['year'][-1].values )

# filenames = []
# for i in range(121):
#     zonal_annual_gif(year = dET['year'][i].values )
    
#     # save frame
#     filename = f'{i}.png'
#     filenames.append(filename)
#     plt.style.use("dark_background")
#     plt.savefig(filename, bbox_inches='tight')
#     plt.close() 

# make_gif('energy_budget.gif')

# zonal_seasonal(ts, var = 'TS')
# zonal_seasonal(alb['energy'], var = 'albedo')
# zonal_seasonal(lr['energy'][:,:,:,:-1], var = 'lapserate')
# zonal_seasonal(planck['energy'][:,:,:,:-1], var = 'planck')
###############################################################################################