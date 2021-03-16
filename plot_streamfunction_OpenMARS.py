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
from matplotlib.legend_handler import HandlerTuple

import pandas as pd

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


from calculate_PV_Isca_anthro import filestrings

from eddy_enstrophy_Isca_all_years import (assign_MY, make_coord_MY)

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

    if u_max == 0:
        jet_lat = 90
        jet_max = u[-1]
    else:
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
            u_0 = -1

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
        jet_max = coefs[3]+coefs[2]*jet_lat+coefs[1]*jet_lat**2 \
                    +coefs[0]*jet_lat**3

    elif u_0 == 0:
        jet_lat = 90
        jet_max = u[-1]

    elif u_0 == 1:
        u_near = u[u_0-1:u_0+2]
        lats_near = lats[u_0-1:u_0+2]

        # Quartic fit, with smaller lat spacing
        coefs = np.ma.polyfit(lats_near,u_near,2)
        fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
        quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2 
        # Find jet lat and max
        #jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]

        minq = min(np.absolute(quad))
        jet_lat = fine_lats[np.where(np.absolute(quad) == minq)[0][0]]
        jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2

    else:
        print(u.values)
        jet_lat = np.nan
        jet_max = np.nan

    return jet_lat, jet_max

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

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
    colors = ['#5F9ED1','#5F9ED1','#5F9ED1','#5F9ED1', '#C85200','#898989',
                '#898989','#898989','#898989',]
    linestyles = ['--', '--', '--', '-', '-.','-.','-.','-.',]
    labels = ["MY 24-26","MY 24-26","MY 24-26", "MY 28", "MY 29-32","MY 29-32",
                "MY 29-32","MY 29-32","MY 29-32"]


    ##### change parameters #####
    Lsmin = 210
    Lsmax = 360

    sh = False

    plev = 50

    

    fig, axs = plt.subplots(2, 1, figsize = (10, 10))

    axs[0].set_xlim([Lsmin, Lsmax])
    axs[0].set_ylim([46, 79])
    axs[0].text(-0.03, 1.02, 'a', size = 20,
                    transform = axs[0].transAxes, weight = 'bold')
    axs[1].text(-0.03, 1.02, 'b', size = 20,
                    transform = axs[1].transAxes, weight = 'bold')
    axs[0].tick_params(length = 6, labelsize = 18)
    axs[1].set_xlim([Lsmin, Lsmax])
    #axs[1].set_ylim([70, 183])
    axs[1].tick_params(length = 6, labelsize = 18)
    axs[0].set_xticklabels([])
    axs[1].set_xlabel('solar longitude (degrees)', fontsize = 20)
    axs[0].set_ylabel('latitude ($^{\circ}$N)', fontsize = 20)
    axs[1].set_ylabel('maximum jet strength (ms$^{-1}$)', fontsize = 20)

    plt.subplots_adjust(hspace = 0.1)

    ax2 = axs[1].twinx()
    ax2.tick_params(length = 6, labelsize = 18)
    ax2.set_ylim([2, 55])

    axs[0].plot(np.linspace(265,310,200),np.full(200, 50), color = 'k',
                 linewidth = '3.5',)
    ax2.plot(np.linspace(265,310,200),np.full(200,5), color = 'k',
                 linewidth = '3.5',)
    
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.set_ticks_position('right')
    ax2.set_ylabel('$\psi$ strength ($10^8$ kg/s)', fontsize = 20)

    ##### get data #####
    PATH = 'link-to-anthro/OpenMARS/Streamfn'
    
    figpath = 'OpenMARS_figs/Hadley_lats/'
    
    ci = []
    cimax = []
    cu = []
    cumax = []

    for i in [24,25,26,28,29,30,31,32]:
        year = 'MY'+str(i)
        infiles = '/'+year+'_180-360_psi.nc'
        d = xr.open_mfdataset(PATH + infiles, decode_times = False,
                          concat_dim = 'time', combine = 'nested',
                          chunks = {'time' : 'auto'})

        ##### reduce dataset #####
        d = d.astype('float32')
        d = d.sortby('time', ascending =True) / 10 ** 8

        d = d.where(d.time <= Lsmax, drop=True)
        #d = d.where(Lsmin - 10 <= d.time, drop=True)

        plev = d.pfull.sel(pfull = plev, method = "nearest").values
        print(plev)

        d = d.sel(time = d.time[slice(None,None,5)], method = "nearest")

        #d = d.chunk({'time':'auto'})
        #d = d.rolling(time = 10, center = True)
        #d = d.mean().dropna("time")

        #d = d.sel(time = d.time[slice(None,None,15)])
        
        #print(di)
        #di = di.where(di != np.nan, drop = True)
        #print(di)

        ls = d.time

        lat_max = []
        mag_max = []

        pfull_max = []
        psi_lat = []
        psi_mag = []



        print(ls.shape[0])


        for j in range(ls.shape[0]):
            lsj = ls[j]
            psi_j = d.where(d.time == lsj, drop = True).squeeze()
            psi_j = psi_j.to_array().squeeze()
                     
            
            # edge and strength of Hadley cell
            psi_j = psi_j.sel(pfull = plev, method = "nearest").squeeze()
            _, psi_max = calc_jet_lat(psi_j.compute(), psi_j.lat)
            psi0_lat, _ = calc_Hadley_lat(psi_j.load(), psi_j.lat.load())
            psi_lat.append(psi0_lat)
            psi_mag.append(psi_max)
            
        psi_new = xr.Dataset({
            "lat": (["time"], psi_lat),
            "mag": (["time"], psi_mag),
        },
        coords = {
            "time": (["time"], ls),
        },
        )

        psi_new = psi_new.where(psi_new.lat != np.nan, drop = True)
        psi_new = psi_new.where(psi_new.time > psi_new.time[9], drop = True)
        
        psi_new = psi_new.chunk({'time':'auto'})
        psi_new = psi_new.rolling(time = 70)
        psi_new = psi_new.mean().dropna("time")

        #ls = moving_average(ls, 10)
        #psi_lat = moving_average(psi_lat, 10)
        #psi_mag = moving_average(psi_mag, 10)
        
        ci_psi, = axs[0].plot(psi_new.time, psi_new.lat, label = "",
                           color = colors[i-24], linestyle = '--')

        ci_psimax, = ax2.plot(psi_new.time, psi_new.mag, label = "",
                                color = colors[i-24], linestyle = '--')

        ci.append(ci_psi)
        cimax.append(ci_psimax)


        fig.savefig(figpath+'Hadley_edge_max_jet_lats_'+str(plev)+'Pa_new.pdf',
                            bbox_inches='tight', pad_inches = 0.1)

    #ax2.set_yticklabels(["{:.1e}".format(i) for i in ax2.get_yticks()])
    fig.savefig(figpath+'Hadley_edge_max_jet_lats_'+str(plev)+'Pa_new.pdf',
                            bbox_inches='tight', pad_inches = 0.1)
    ##### get data #####
    PATH = 'link-to-anthro/OpenMARS/Isobaric'
    infiles = '/isobaric*'

    for i in [24,25,26,28,29,30,31,32]:
        d = xr.open_mfdataset(PATH+infiles+ str(i) + '*', decode_times = False,
                          concat_dim = 'time', combine = 'nested',
                          chunks = {'time' : 'auto'})

        ##### reduce dataset #####
        d = d.astype('float32')
        d = d.sortby('time', ascending =True)
        d = d.where(d.lat > 0, drop = True)

        d = d.where(d.Ls <= Lsmax, drop=True)
        #d = d.where(Lsmin - 10 <= d.Ls, drop=True)

        d = d.mean(dim = 'lon', skipna = True)
        d = d.sel(plev = plev, method = "nearest")

    
        year = str(i)
        print(year)
        di = d.where(d.MY == i, drop = True)
        
        di = di.sel(time = di.time[slice(None,None,5)], method = "nearest")
        u = di.uwnd
        lat = di.lat
        di["Ls"] = di.Ls.sel(lat = 5, method = 'nearest', drop = True).squeeze()
        ls = di.Ls

        lat_max = []
        mag_max = []

        for j in range(ls.shape[0]):
            lsj = ls[j]
            uj=u.where(di.Ls==lsj, drop=True)
            latmax, u_p_max = calc_jet_lat(uj.squeeze().compute(), uj.lat)
            lat_max.append(latmax)
            mag_max.append(u_p_max)

        u_new = xr.Dataset({
            "lat": (["time"], lat_max),
            "mag": (["time"], mag_max),
        },
        coords = {
            "time": (["time"], ls.values),
        },
        )

        u_new = u_new.where(u_new.lat != np.nan, drop = True)
        u_new = u_new.where(u_new.time > u_new.time[9], drop = True)
        u_new = u_new.chunk({'time':'auto'})
        u_new = u_new.rolling(time = 50)
        u_new = u_new.mean().dropna("time")


        ci_u, = axs[0].plot(u_new.time, u_new.lat, label = year,
                           color = colors[i-24], linestyle = '-')
        ci_umax, = axs[1].plot(u_new.time, u_new.mag, label = year,
                           color = colors[i-24], linestyle = '-')
        cu.append(ci_u)
        cumax.append(ci_umax)

        fig.savefig(figpath+'Hadley_edge_max_jet_lats_'+str(plev)+'Pa_new.pdf',
                bbox_inches='tight', pad_inches = 0.1)

    c0 = [[ci[j],cu[j]] for j in range(len(cu))]
    c1 = [[cimax[j],cumax[j]] for j in range(len(cumax))]

    axs[0].legend([tuple(c0[j]) for j in [2,3,4]], [i for i in labels[2:5]],
                 fontsize = 12,# loc = 'upper left',
            handler_map={tuple: HandlerTuple(ndivide=None)})
    axs[1].legend([tuple(c1[j]) for j in [2,3,4]], [i for i in labels[2:5]],
                 fontsize = 12,# loc = 'upper left',
            handler_map={tuple: HandlerTuple(ndivide=None)})

    fig.savefig(figpath+'Hadley_edge_max_jet_lats_'+str(plev)+'Pa_new.pdf',
                bbox_inches='tight', pad_inches = 0.1)