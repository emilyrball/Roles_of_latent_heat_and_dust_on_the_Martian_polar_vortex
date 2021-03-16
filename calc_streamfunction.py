'''
Script to calculate the MMC for OpenMARS data
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

    colors = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200',
              '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']


    ##### change parameters #####
    Lsmin = 0
    Lsmax = 75

    sh = False

    plev = 50

    

    fig, axs = plt.subplots(2, 1, figsize = (10, 10))

    axs[0].set_xlim([Lsmin, Lsmax])
    axs[0].tick_params(length = 6, labelsize = 18)
    axs[1].set_xlim([Lsmin, Lsmax])
    axs[1].tick_params(length = 6, labelsize = 18)
    axs[0].set_xticklabels([])
    axs[1].set_xlabel('solar longitude (degrees)', fontsize = 20)
    axs[0].set_ylabel('latitude ($^{\circ}$ N)', fontsize = 20)
    axs[1].set_ylabel('jet strength (ms$^{-1}$)', fontsize = 20)

    ##### get data #####
    PATH = '/export/anthropocene/array-01/xz19136/OpenMARS/Isobaric'
    infiles = '/isobaric*'
    figpath = 'OpenMARS_figs/'
    d = xr.open_mfdataset(PATH + infiles, decode_times = False,
                          concat_dim = 'time', combine = 'nested',
                          chunks = {'time' : 'auto'})

    ##### reduce dataset #####
    d = d.astype('float32')
    d = d.sortby('time', ascending =True)

    d = d.where(d.Ls <= Lsmax, drop=True)
    d = d.where(Lsmin <= d.Ls, drop=True)
    d = d.mean(dim = 'lon', skipna = True)

    if sh == False:
        d = d.where(d.lat > 0, drop = True)
    else:
        d = d.where(d.lat < 0, drop = True)


    plev = d.plev.sel(plev = plev, method = "nearest").values  

    for i in [24,25,26,28,29,30,31,32]:
        year = str(i)
        print(year)
        di = d.where(d.MY == i, drop = True)
        #di = di.sel(time = di.time[slice(None,None,3000)])

        #di = di.chunk({'time':'auto'})
        #di = di.rolling(time = 500, center = True)
        #di = di.mean().dropna("time")
        #print(di)
        #di = di.where(di != np.nan, drop = True)
        #print(di)

        u = di.uwnd
        v = di.vwnd
        lat = di.lat
        pfull = di.plev

        di["Ls"] = di.Ls.sel(lat = 5, method = 'nearest', drop = True)

        ls = di.Ls

        lat_max = []
        mag_max = []

        pfull_max = []
        psi_lat = []
        psi_check = []
        psi_i = []

        print(ls.shape[0])

        for j in range(ls.shape[0]):
            lsj = ls[j]
            vj = v.where(di.Ls == lsj, drop = True).squeeze()
            psi_j = funcs.calc_streamfn(lat.load(), pfull.load(), vj.load(),
                                   radius = rsphere, g = g)

            psi_j = xr.DataArray(data = psi_j, dims = ["pfull", "lat"],
                            coords = dict(pfull = (["pfull"], pfull.values),
                                          lat   = (["lat"],   lat.values),
                                          ),
                            attrs = dict(description="Meridional streamfunction",
                                         units="kg/s")
                                         )
            psi_j = psi_j.assign_coords({'time':lsj.values})
            psi_j = psi_j.rename("psi")
            psi_i.append(psi_j)
        
        psi = xr.concat(psi_i, dim='time')
        psi.to_netcdf('/export/anthropocene/array-01/xz19136/OpenMARS/' \
                        + 'Streamfn/MY' + year + '_' + str(Lsmin) + '-' \
                        + str(Lsmax) + '_psi.nc')
