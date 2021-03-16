'''
Script to plot the MMC for Isca data
'''

import numpy as np
import xarray as xr
import os, sys

import analysis_functions as funcs
import colorcet as cc
import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import (cm, colors,cycler)
import matplotlib.path as mpath
import matplotlib

import pandas as pd

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


if __name__ == "__main__":

    # Mars-specific!
    theta0 = 200. # reference temperature
    kappa = 0.25 # ratio of specific heats
    p0 = 610. # reference pressure
    omega = 7.08822e-05 # planetary rotation rate
    g = 3.72076 # gravitational acceleration
    rsphere = 3.3962e6 # mean planetary radius

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    colors = ['#006BA4', '#C85200', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',
              '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']


    ##### change parameters #####
    Lsmin = 180
    Lsmax = 360

    sh = False

    plev = 50
    year = [#"MY 28", 
    "Climatology"
    ]

    

    fig, axs = plt.subplots(2, 1, figsize = (10, 10))

    axs[0].set_xlim([Lsmin, Lsmax])
    axs[0].tick_params(length = 6, labelsize = 18)
    axs[1].set_xlim([Lsmin, Lsmax])
    axs[1].tick_params(length = 6, labelsize = 18)
    axs[0].set_xticklabels([])
    axs[1].set_xlabel('solar longitude (degrees)', fontsize = 20)
    axs[0].set_ylabel('latitude ($^{\circ}$ N)', fontsize = 20)
    axs[1].set_ylabel('maximum jet strength (ms$^{-1}$)', fontsize = 20)

    ##### get data #####
    exp = [
        #'soc_mars_mk36_per_value70.85_none_mld_2.0',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_cdod_clim_scenario_7.4e-05',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_cdod_clim_scenario_7.4e-05_lh_rel',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_lh_rel',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05_lh_rel',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY24_7.4e-05_lh_rel',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY25_7.4e-05_lh_rel',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY26_7.4e-05_lh_rel',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY27_7.4e-05_lh_rel',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY28_7.4e-05_lh_rel',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY29_7.4e-05_lh_rel',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY30_7.4e-05_lh_rel',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY31_7.4e-05_lh_rel',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY32_7.4e-05_lh_rel',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_all_years',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_lh_rel',
    ]
    name = [
        #'stand_',
        #'lh_',
        'scenario_',
        #'scenario_lh_',
        #'topo_',
        #'topo_lh_',
        #'topo_scenario_',
        #'topo_scenario_lh_',

    ]

    location = [
        #'triassic',
        #'triassic',
        'anthropocene',
        #'anthropocene',
        #'triassic',
        #'triassic',
        #'anthropocene',
        #'silurian',
        #'silurian',
        #'silurian',
        #'silurian',
        #'silurian',
        #'silurian',
        #'silurian', 
        #'silurian', 
        #'silurian', 
        #'silurian',
        #'anthropocene',
        #'triassic',
    ]
    start_file = [
        #33, 
        #33, 
        33, 
        #33, 
        #33, 
        #33, 
        #33, 
        #33,
        #33, 
        #33,
        #33, 
        #33, 
        #33, 
        #33, 
        #33, 
        #33, 
        #33,
        #21,
        #25,
    ]
    end_file = [
        #99,
        #99, 
        99, 
        #99,
        #99, 
        #99, 
        #99, 
        #139
        #80, 
        #99, 
        #88, 
        #99, 
        #222, 
        #96, 
        #99, 
        #99, 
        #88,
        #222,
        #99,
    ]

    freq = 'daily'

    p_file = 'atmos_'+freq+'_interp_new_height_temp_PV.nc'

    for i in range(len(start_file)):
        print(exp[i])

        filepath = '/export/' + location[i] + '/array-01/xz19136/Isca_data'
        figpath = 'Isca_figs/'+exp[i]+'/'
        start = start_file[i]
        end = end_file[i]

        _, _, i_files = funcs.filestrings(exp[i], filepath, start, end, p_file)

        d = xr.open_mfdataset(i_files, decode_times = False,
                              concat_dim = 'time', combine='nested')


        ##### reduce dataset #####
        d = d.astype('float32')
        d = d.sortby('time', ascending = True)
        d = d.sortby('lat', ascending = False)
        d = d.sortby('pfull', ascending = True)
        d = d.where(d.mars_solar_long != 354.3762, other=359.762)
        d = d.where(d.mars_solar_long != 354.37808, other = 359.762)
        d = d.where(d.mars_solar_long <= Lsmax, drop=True)
        d = d.where(Lsmin <= d.mars_solar_long, drop=True)
        d = d.mean(dim = 'lon', skipna = True).squeeze()
        d["pfull"] = d.pfull*100
        

        if sh == False:
            d = d.where(d.lat > 0, drop = True).squeeze()
        else:
            d = d.where(d.lat < 0, drop = True).squeeze()

        plev = d.pfull.sel(pfull = plev, method = "nearest").values
        
        d["mars_solar_long"] = d.mars_solar_long.sel(lat=5,method="nearest")

        x, index = funcs.assign_MY(d)
        dsr, N, n = funcs.make_coord_MY(x, index)

        if exp[i] == 'soc_mars_mk36_per_value70.85_none_mld_2.0_all_years':
            for j in range(len(n)):
                dsrj = dsr.sel(MY = dsr.MY[j], method = "nearest").squeeze()
                dsrj.to_netcdf(filepath + '/' + exp[i] + '/MY' + str(j) + '_' \
                     + str(Lsmin) + '-' \
                        + str(Lsmax) + '_dsr.nc')
                print(str(j))
            print("Done")
            continue

        else:
            dsr = dsr.mean(dim = "MY")
            dsr.to_netcdf('/export/silurian/array-01/xz19136/Isca_data/' \
                + 'Streamfn/' + name[i] + str(Lsmin) + '-' + str(Lsmax) \
                + '_dsr.nc')

            print('dsr saved.')
