'''
Calculates and plots eddy enstrophy for OpenMARS, Isca - all years and exps.
'''

import numpy as np
import xarray as xr
import os, sys

import analysis_functions as funcs

if __name__ == "__main__":

    theta0 = 200.
    kappa = 1/4.0
    p0 = 610.

    EMARS = True
    ilev = 350

    Lsmax = 360
    Lsmin = 0

    PATH = '/export/anthropocene/array-01/xz19136/EMARS'
    files = '/*isentropic*'
    reanalysis = 'EMARS'

    d = xr.open_mfdataset(PATH+files, decode_times=False, concat_dim='time',
                           combine='nested',chunks={'time':'auto'})

    d["Ls"] = d.Ls.expand_dims({"lat":d.lat})
    #d = d.rename({"pfull":"plev"})
    #d = d.rename({"t":"tmp"})
    smooth = 200
    yearmax = 32

    d = d.sel(ilev = ilev, method='nearest').squeeze()
    d = d[["PV", "MY", "Ls"]]
    d = d.where(d.lat > 0, drop = True)
    d = d.where(d.Ls <= Lsmax, drop = True)
    d = d.where(d.Ls >= Lsmin, drop = True)

    d.to_netcdf('/export/anthropocene/array-01/xz19136/Data_Ball_etal_2021/' \
        + 'EMARS_' + 'Ls' + str(Lsmin) + '-' + str(Lsmax) + '_350K.nc')
