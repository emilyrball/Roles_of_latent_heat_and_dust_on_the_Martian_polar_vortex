'''
Calculate streamfunction psi from dsr
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
    year = [#"MY 28", 
    "Climatology"
    ]

    ##### get data #####
    exp = [
        'topo_dust_lh_',
        'MY28_',
        
    ]
    
    for i in range(len(exp)):
        print(exp[i])

        filepath = '/export/anthropocene/array-01/xz19136/Data_Ball_etal_2021/'
        
        d = xr.open_mfdataset(filepath+exp[i]+'Ls270-300_zonal.nc',
                              decode_times = False,
                              concat_dim = 'time', combine='nested')

        ##### reduce dataset #####

        d = d.astype('float32')

        d = d.sortby('new_time', ascending = True)

        d = d.sortby('lat', ascending = False)
        d = d.sortby('pfull', ascending = True)
        d["pfull"] = d.pfull*100

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
        psi.to_netcdf(filepath+exp[i]+'Ls270-300_psi.nc')
        print('Saved.')
