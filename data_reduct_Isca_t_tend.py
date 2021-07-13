'''
Script to reduce volume of data needed for Roles_of_latent_heat_and_dust
'''


import numpy as np
import xarray as xr
import os, sys

import analysis_functions as funcs
import string

import pandas as pd

if __name__ == "__main__":

    # Mars-specific!
    theta0 = 200. # reference temperature
    kappa = 0.25 # ratio of specific heats
    p0 = 610. # reference pressure
    omega = 7.08822e-05 # planetary rotation rate
    g = 3.72076 # gravitational acceleration
    rsphere = 3.3962e6 # mean planetary radius

    ##### change parameters #####
    Lsmin = 0
    Lsmax = 360

    sh = False

    plev = 2

    roo = 'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_'

    ##### get data #####
    exp = [
        roo + 'scenario_7.4e-05_lh_rel',
        roo + 'MY24_7.4e-05_lh_rel',
        roo + 'MY25_7.4e-05_lh_rel',
        roo + 'MY26_7.4e-05_lh_rel',
        roo + 'MY27_7.4e-05_lh_rel',
        roo + 'MY28_7.4e-05_lh_rel',
        roo + 'MY29_7.4e-05_lh_rel',
        roo + 'MY30_7.4e-05_lh_rel',
        roo + 'MY31_7.4e-05_lh_rel',
        roo + 'MY32_7.4e-05_lh_rel',
    ]
    name = [
        'topo_dust_lh_',
        'MY24_',
        'MY25_',
        'MY26_',
        'MY27_',
        'MY28_',
        'MY29_',
        'MY30_',
        'MY31_',
        'MY32_',
    ]

    location = [
        'silurian',
        'silurian',
        'silurian',
        'silurian',
        'silurian',
        'silurian',
        'silurian', 
        'silurian', 
        'silurian', 
        'silurian',
    ]
    start_file = [
        33,
        33, 
        33,
        33, 
        33, 
        33, 
        33, 
        33, 
        33, 
        33,
    ]
    end_file = [
        139,
        80, 
        99, 
        88, 
        99, 
        222, 
        96, 
        99, 
        99, 
        88,
    ]

    freq = 'daily'

    p_file = 'atmos_'+freq+'_interp_new_height_temp.nc'

    for i in range(len(start_file)):
        print(exp[i])

        filepath = '/export/' + location[i] + '/array-01/xz19136/Isca_data'
        start = start_file[i]
        end = end_file[i]

        _, _, i_files = funcs.filestrings(exp[i], filepath, start, end, p_file)

        d = xr.open_mfdataset(i_files, decode_times = False,
                              concat_dim = 'time', combine='nested')


        ##### reduce dataset #####
        d = d.astype('float32')
        d = d.sortby('time', ascending = True)
        d = d.sortby('lat', ascending = False)
        d = d[["dt_tg_lh_condensation", "mars_solar_long"]]

        #d = d.sortby('pfull', ascending = True)
        d["mars_solar_long"] = d.mars_solar_long.where(d.mars_solar_long != 354.3762, other=359.762)
        d["mars_solar_long"] = d.mars_solar_long.where(d.mars_solar_long != 354.37808, other = 359.762)
        
        if exp[i] != 'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05_lh_rel':
            plev = d.pfull.sel(pfull = plev, method = "nearest").values
            d = d.sel(pfull = plev, method = "nearest")
        
        x, index = funcs.assign_MY(d)
        
        dsr, _, _ = funcs.make_coord_MY(x, index)
        dsr.to_netcdf('/export/anthropocene/array-01/xz19136/Data_Ball_etal' + \
                        '_2021/' + name[i] +'Ls'+ str(Lsmin) + '-' + str(Lsmax) \
            + '_dt_tg_zonal.nc')

        print('dsr saved.')
