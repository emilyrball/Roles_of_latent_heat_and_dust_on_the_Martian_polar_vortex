'''
Calculate streamfunction psi from dsr
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
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_cdod_clim_scenario_7.4e-05',
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
        'soc_mars_mk36_per_value70.85_none_mld_2.0_all_years',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_lh_rel',
    ]
    location = [
        #'triassic',
        #'triassic',
        #'anthropocene',
        #'anthropocene',
        #'triassic',
        #'triassic',
        #'anthropocene',
        #'silurian'
        #'silurian',
        #'silurian',
        #'silurian',
        #'silurian',
        #'silurian',
        #'silurian', 
        #'silurian', 
        #'silurian', 
        #'silurian',
        'anthropocene',
        #'triassic',
    ]
    

    for i in range(len(exp)):
        print(exp[i])

        filepath = '/export/' + location[i] + '/array-01/xz19136/Isca_data'
        figpath = 'Isca_figs/'+exp[i]+'/'
        
        for year in [6]:
            d = xr.open_mfdataset(filepath+'/'+exp[i]+'/MY'+str(year)+'*dsr.nc',
                                  decode_times = False,
                                  concat_dim = 'time', combine='nested')


            ##### reduce dataset #####
            d = d.astype('float32')
            d = d.sortby('new_time', ascending = True)
            d = d.sortby('lat', ascending = False)
            d = d.sortby('pfull', ascending = True)        

            plev = d.pfull.sel(pfull = plev, method = "nearest").values

            ls = d.mars_solar_long.squeeze()
            u = d.ucomp.squeeze()
            v = d.vcomp.squeeze()
            lat = d.lat
            pfull = d.pfull

            lat_max = []
            mag_max = []

            pfull_max = []
            psi_lat = []
            psi_check = []
            psi_i = []


            for j in range(ls.shape[0]):
                lsj = ls[j]
                vj = v.where(ls == lsj, drop = True).squeeze()
                psi_j = funcs.calc_streamfn(lat.load(), pfull.load(), vj.load(),
                                       radius = rsphere, g = g)

                psi_j = xr.DataArray(data = psi_j, dims = ["pfull", "lat"],
                                coords = dict(pfull = (["pfull"], pfull.values),
                                              lat   = (["lat"],   lat.values)),
                                attrs = dict(description="Meridional streamfunction",
                                             units="kg/s"))
                psi_j = psi_j.assign_coords({'time':lsj.values})
                psi_j = psi_j.rename("psi")
                psi_i.append(psi_j)

            psi = xr.concat(psi_i, dim='time')
            psi.to_netcdf(filepath + '/' + exp[i] + '/MY'+str(year)+'_' + str(Lsmin) + '-' \
                            + str(Lsmax) + '_psi.nc')

            print('Saved.')
