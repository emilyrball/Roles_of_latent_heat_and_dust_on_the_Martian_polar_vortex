'''
Script to plot the MMC for OpenMARS data
'''

import numpy as np
import xarray as xr
import os, sys

import calculate_PV as cPV
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


from calculate_PV_Isca_anthro import filestrings

from eddy_enstrophy_Isca_all_years import (assign_MY, make_coord_MY)

def calc_streamfn(lats, pfull, vz, **kwargs):
    '''
    Calculate meridional streamfunction from zonal mean meridional wind.
    
    Parameters
    ----------

    lats   : array-like, latitudes, units (degrees)
    pfull  : array-like, pressure levels, units (Pa)
    vz     : array-like, zonal mean meridional wind, dimensions (lat, pfull)
    radius : float, planetary radius, optional, default 3.39e6 m
    g      : float, gravity, optional, default 3.72 m s**-2

    Returns
    -------

    psi   : array-like, meridional streamfunction, dimensions (lat, pfull),
            units (kg/s)
    '''

    radius = kwargs.pop('radius', 3.39e6)
    g      = kwargs.pop('g', 3.72)

    coeff = 2 * np.pi * radius / g

    psi = np.empty_like(vz.values)
    for ilat in range(lats.shape[0]):
        psi[0, ilat] = coeff * np.cos(np.deg2rad(lats[ilat]))*vz[0, ilat] * pfull[0]
        for ilev in range(pfull.shape[0])[1:]:
            psi[ilev, ilat] = psi[ilev - 1, ilat] + coeff*np.cos(np.deg2rad(lats[ilat])) \
                              * vz[ilev, ilat] * (pfull[ilev] - pfull[ilev - 1])
    
    #psi = xr.DataArray(psi, coords = {"pfull" : pfull.values,
    #                                  "lat"   : lats.values})
    #psi.attrs['units'] = 'kg/s'

    return psi

def calc_jet_lat(u, lats, plot = False):
    '''
    Function to calculate location and strength of maximum given zonal wind
    u(lat) field

    Parameters
    ----------

    u    : array-like
    lats : array-like. Default use will be to calculate jet on a given pressure
           level, but this array may also represent pressure level.

    Returns
    -------

    jet_lat : latitude (pressure level) of maximum zonal wind
    jet_max : strength of maximum zonal wind
    '''

    # Restrict to 10 points around maximum

    u_max = np.where(u == np.ma.max(u.values))[0][0]

    u_near = u[u_max-1:u_max+2]
    lats_near = lats[u_max-1:u_max+2]
    # Quartic fit, with smaller lat spacing
    coefs = np.ma.polyfit(lats_near,u_near,2)
    fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
    quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
    # Find jet lat and max
    jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]
    jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2
    # Plot fit?
    if plot:
        print (jet_max)
        print (jet_lat)
        plt.plot(lats_near, u_near)
        plt.plot(fine_lats, quad)
        plt.show()

    return jet_lat, jet_max

def calc_Hadley_lat(u, lats, plot = False):
    '''
    Function to calculate location of 0 streamfunction.

    Parameters
    ----------

    u    : array-like
    lats : array-like. Default use will be to calculate jet on a given pressure
           level, but this array may also represent pressure level.

    Returns
    -------

    jet_lat : latitude (pressure level) of 0 streamfunction
    '''

    asign = np.sign(u)#.values)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    signchange[0] = 0

    

    for i in range(len(signchange)):
        if u[i] > 0 and i < len(signchange) - 4:
            continue
        signchange[i] = 0
    
    for i in range(len(signchange)):
        if signchange[i] == 0:
            continue
        u_0 = i
    
    if all(signchange[i] == 0 for i in range(len(signchange))):
        if u[0] > 0:
            u_0 = 0
        else:
            u_0 = 1

        #u_0 = np.where(u == np.ma.min(np.absolute(u)))[0][0]

    # Restrict to 10 points around maximum
    #u_0 = np.where(u == np.ma.min(np.absolute(u.values)))[0][0]
    if u_0 > 1:
        u_near = u[u_0-2:u_0+2]
        lats_near = lats[u_0-2:u_0+2]

        # Quartic fit, with smaller lat spacing
        coefs = np.ma.polyfit(lats_near,u_near,3)
        fine_lats = np.linspace(lats_near[0], lats_near[-1],300)
        quad = coefs[3]+coefs[2]*fine_lats+coefs[1]*fine_lats**2 \
                    +coefs[0]*fine_lats**3
        # Find jet lat and max
        #jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]

        minq = min(np.absolute(quad))
        jet_lat = fine_lats[np.where(np.absolute(quad) == minq)[0][0]]
        jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2
        # Plot fit?
        if plot:
            print (jet_max)
            print (jet_lat)
            plt.plot(lats_near, u_near)
            plt.plot(fine_lats, quad)
            plt.show()
    elif u_0 == 0:
        jet_lat = 90
        jet_max = u[-1]
    else:
        jet_lat = np.nan
        jet_max = np.nan

    

    return jet_lat, jet_max

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
    axs[1].set_ylabel('maximum jet strength (ms$^{-1}$)', fontsize = 20)

    ##### get data #####
    PATH = 'link-to-anthro/OpenMARS/Isobaric'
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

    print(d.plev.values)

    plev = d.plev.sel(plev = plev, method = "nearest").values
    print(plev)

    

    for i in [25,26,27,29,30,31,32]:
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
            #pfullj = pfull.where(pfull <= plev + 20, drop = True)

            #vj0 = vj.where(vj.plev <= plev + 20, drop = True)
            psi_j = calc_streamfn(lat.load(), pfull.load(), vj.load(),
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


        for j in range(ls.shape[0]):
            tmp_mag = []
            tmp_lat = []
            lsj = ls[j]
            uj = u.where(di.Ls == lsj, drop = True).sel(plev=plev,method="nearest")
            psi_j = psi.where(psi.time == lsj, drop = True).squeeze()

            latmax, u_p_max = calc_jet_lat(uj.squeeze().compute(), uj.lat)

            lat_max.append(latmax)
            mag_max.append(u_p_max)

            
            psi_j = psi_j.sel(pfull = plev, method = "nearest").squeeze()
            psi0_lat, _ = calc_Hadley_lat(psi_j.load(), psi_j.lat.load())
            psi_lat.append(psi0_lat)
                    
        

        ci_u = axs[0].plot(ls, lat_max, label = year,
                           color = colors[i-24], linestyle = '-')
        ci_psi = axs[0].plot(ls, psi_lat, label = "",
                           color = colors[i-24], linestyle = '--')

        ci_umax = axs[1].plot(ls, mag_max, label = year, color = colors[i-24],
                       linestyle = '-')

        axs[0].legend(fontsize = 14, loc = 'upper left')
        axs[1].legend(fontsize = 14, loc = 'upper left')

        fig.savefig(figpath+'Hadley_edge_max_jet_lats_'+str(plev)+'Pa.pdf',
                            bbox_inches='tight', pad_inches = 0.1)


    

    fig.savefig(figpath+'Hadley_edge_max_jet_lats_'+str(plev)+'Pa.pdf',
                bbox_inches='tight', pad_inches = 0.1)