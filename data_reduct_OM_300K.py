'''
Script to reduce volume of data needed for Roles_of_latent_heat_and_dust
'''

import numpy as np
import xarray as xr
import os, sys

import analysis_functions as funcs
import string

if __name__ == "__main__":


    Lsmin = 270
    Lsmax = 300

    theta0 = 200.
    kappa = 1/4.0
    p0 = 610.

    ilev = 300

    sh = False

    PATH = '/export/anthropocene/array-01/xz19136/OpenMARS/Isentropic'
    infiles = '/isentropic*'

    d1 = xr.open_mfdataset(PATH+infiles, decode_times=False, concat_dim='time',
                           combine='nested',chunks={'time':'auto'})

    # reduce dataset
    d = d1.astype('float32')
    d = d.sortby('time', ascending=True)
    d = d.where(d.Ls <= Lsmax, drop=True)
    d = d.where(Lsmin <= d.Ls, drop=True)
    d = d.sel(ilev=ilev, method='nearest')
    d = d[["PV", "uwnd", "MY", "Ls"]]
    
    if sh == False:
        d = d.where(d.lat > 0, drop = True)
    else:
        d = d.where(d.lat < 0, drop = True)

    d.to_netcdf('/export/anthropocene/array-01/xz19136/Data_Ball_etal_' + \
        '2021/OpenMARS_' +'Ls'+ str(Lsmin) + '-' + str(Lsmax) + '_300K.nc')

    ilev = 350
    Lsmin = 0
    Lsmax = 360

    d = d1.astype('float32')
    d = d.sortby('time', ascending=True)
    d = d.where(d.Ls <= Lsmax, drop=True)
    d = d.where(Lsmin <= d.Ls, drop=True)
    d = d.sel(ilev=ilev, method='nearest')
    d = d[["PV", "MY", "Ls"]]
    
    if sh == False:
        d = d.where(d.lat > 60, drop = True)
    else:
        d = d.where(d.lat < -60, drop = True)

    d.to_netcdf('/export/anthropocene/array-01/xz19136/Data_Ball_etal_' + \
        '2021/OpenMARS_' +'Ls'+ str(Lsmin) + '-' + str(Lsmax) + '_PV_350K.nc')

    

        