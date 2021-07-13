'''
Script to calculate the MMC for OpenMARS data
'''

import numpy as np
import xarray as xr
import os, sys

import analysis_functions as funcs

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
    Lsmin = 180
    Lsmax = 360

    sh = False

    plev = 50

    ##### get data #####
    PATH = '/export/anthropocene/array-01/xz19136/Data_Ball_etal_2021/OpenMARS_'
    d = xr.open_mfdataset(PATH + 'Ls200-360_zonal.nc', decode_times = False,
                          concat_dim = 'time', combine = 'nested',
                          chunks = {'time' : 'auto'})

    ##### reduce dataset #####
    d = d.astype('float32')
    d = d.sortby('time', ascending =True)

    d = d.where(d.Ls <= Lsmax, drop=True)
    d = d.where(Lsmin <= d.Ls, drop=True)

    if sh == False:
        d = d.where(d.lat > 0, drop = True)
    else:
        d = d.where(d.lat < 0, drop = True)


    plev = d.plev.sel(plev = plev, method = "nearest").values

    for i in [24,25,26,28,29,30,31,32]:
        year = str(i)
        print(year)
        di = d.where(d.MY == i, drop = True)

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
        psi["MY"] = i

        psi.to_netcdf(PATH + 'MY'+str(i)+'_Ls200-360_psi.nc')
