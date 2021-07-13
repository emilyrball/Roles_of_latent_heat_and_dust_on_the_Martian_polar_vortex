'''
Script to reduce volume of data needed for Roles_of_latent_heat_and_dust
'''


import numpy as np
import xarray as xr
import os, sys

import analysis_functions as funcs
import string

import pandas as pd

def make_coord_MY_high(x, index):
    x24 = x.where(x.time <= index[0], drop=True)
    x = x.where(x.time > index[0], drop=True)
    #x = x.where(x.time <= index[-1], drop=True)

    N=int(np.max(x.MY))
    n = range(N)

    y = x.time[:len(x.time)//N]

    ind = pd.MultiIndex.from_product((n,y),names=('MY','new_time'))
    dsr = x.assign_coords({'time':ind}).unstack('time')
    dsr = dsr.squeeze()

    return dsr, N, n, x24

if __name__ == "__main__":

    # Mars-specific!
    theta0 = 200. # reference temperature
    kappa = 0.25 # ratio of specific heats
    p0 = 610. # reference pressure
    omega = 7.08822e-05 # planetary rotation rate
    g = 3.72076 # gravitational acceleration
    rsphere = 3.3962e6 # mean planetary radius

    ##### change parameters #####
    Lsmin = 270
    Lsmax = 300

    sh = False

    ilev = 300

    ##### get data #####
    exp = [
        'soc_mars_mk36_per_value70.85_none_mld_2.0',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_cdod_clim_scenario_7.4e-05',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_cdod_clim_scenario_7.4e-05_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY24_7.4e-05_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY25_7.4e-05_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY26_7.4e-05_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY27_7.4e-05_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY28_7.4e-05_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY29_7.4e-05_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY30_7.4e-05_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY31_7.4e-05_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY32_7.4e-05_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_all_years',
    ]
    name = [
        'control_',
        'lh_',
        'dust_',
        'dust_lh_',
        'topo_',
        'topo_lh_',
        'topo_dust_',
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
        'triassic',
        'triassic',
        'anthropocene',
        'anthropocene',
        'triassic',
        'triassic',
        'anthropocene',
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
        'anthropocene',
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
        30, 
        33,
        33, 
        33, 
        33, 
        33, 
        33, 
        33, 
        33,
        21,
    ]
    end_file = [
        99,
        99, 
        99, 
        99,
        99, 
        99, 
        99, 
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
        222,
    ]

    freq = 'daily'

    p_file = 'atmos_'+freq+'_interp_new_height_temp_isentropic.nc'

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
        d = d[["PV", "uwnd", "mars_solar_long"]]

        d["mars_solar_long"] = d.mars_solar_long.sel(lon=5,method="nearest")
        #d = d.sortby('pfull', ascending = True)
        d = d.where(d.mars_solar_long != 354.3762, other=359.762)
        d = d.where(d.mars_solar_long != 354.37808, other = 359.762)
        
        #d = d.mean(dim = 'lon', skipna = True).squeeze()
        #d["pfull"] = d.pfull*100
        

        if sh == False:
            d = d.where(d.lat > 0, drop = True).squeeze()
        else:
            d = d.where(d.lat < 0, drop = True).squeeze()

        #plev = d.pfull.sel(pfull = plev, method = "nearest").values
        ilev = d.ilev.sel(ilev = ilev, method = "nearest").values
        
        d["mars_solar_long"] = d.mars_solar_long.sel(lat=5,method="nearest")


        x, index = funcs.assign_MY(d)
        
        if exp[i] == 'soc_mars_mk36_per_value70.85_none_mld_2.0_all_years':
            x = x.where(x.mars_solar_long <= Lsmax, drop=True)
            x = x.where(Lsmin <= x.mars_solar_long, drop=True)
            dsr, N, n, x24 = make_coord_MY_high(x, index)

            x24 = x24.rename({"time":"new_time"}).squeeze()
            x24.to_netcdf('/export/anthropocene/array-01/xz19136/Data_Ball_etal' + \
                            '_2021/high_dust_MY24_Ls' + str(Lsmin) + '-' \
                            + str(Lsmax) + '_300K.nc')
            for j in range(len(n)):
                dsrj = dsr.sel(MY = dsr.MY[j], method = "nearest").squeeze()
                dsrj.to_netcdf('/export/anthropocene/array-01/xz19136/Data_Ball_etal' + \
                            '_2021/high_dust_MY' + str(j+25) + '_Ls' + str(Lsmin) \
                                + '-' + str(Lsmax) + '_300K.nc')
                print(str(j))
            print("Done")
            continue

        else:
            dsr, N, n = funcs.make_coord_MY(x, index)
            dsr = dsr.where(dsr.mars_solar_long <= Lsmax, drop=True)
            dsr = dsr.where(Lsmin <= dsr.mars_solar_long, drop=True)
            dsr.to_netcdf('/export/anthropocene/array-01/xz19136/Data_Ball_etal' + \
                            '_2021/' + name[i] +'Ls'+ str(Lsmin) + '-' + str(Lsmax) \
                + '_300K.nc')

            print('dsr saved.')
