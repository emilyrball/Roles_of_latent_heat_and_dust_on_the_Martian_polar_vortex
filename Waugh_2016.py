import numpy as np
import xarray as xr
import os, sys
import glob
import analysis_functions as funcs
import PVmodule as PVmod

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import (cm, colors)
import matplotlib.path as mpath

import colorcet as cc
import string

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

if "__name__" == "__main__":
    Lsmin = 255
    Lsmax = 285
    
    thetalevs=[200., 250., 300., 350., 400., 450., 500., 550., 600., 650., 700., 750., 800., 850., 900., 950.]
    
    theta_0 = 200.
    kappa = 1/4.0
    p0 = 610.
    
    inpath_O = '/export/anthropocene/array-01/xz19136/OpenMARS/Isobaric/'
    
    figpath = 'Figs/'
    
    d_O1 = xr.open_mfdataset(inpath_O + '*mars_my*', decode_times=False, concat_dim='time',
                           combine='nested',chunks={'time':'auto'})
    
    p_o = d_O1.plev / 100
    lat_o = d_O1.lat

    nrow = 2
    ncol = 4

    vmin = -5
    vmax = 65
    step = 5

    fig, axs = plt.subplots(nrow, ncol, sharey=True,sharex=True,
                            figsize=(25, 8))

    
    plt.subplots_adjust(hspace=.2,wspace=.1)

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'


    boundaries, _, _, cmap, norm = funcs.make_colourmap(vmin, vmax, step,
                                            col = 'OrRd', extend = 'max')

    for i, ax in enumerate(fig.axes):
        
        ax.set_yscale('log')
        ax.set_xlim([0,90])
        ax.set_ylim([10,0.005])
        ax.text(-0.05, 1.05, string.ascii_lowercase[i], transform=ax.transAxes,
                size=22, weight='bold')

        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        if i < 3:
            my = i+24
        else:
            my = i+25
        print(my)
        d_O = d_O1.where(d_O1.MY==my, drop=True)
        d_O = d_O.where(Lsmin <= d_O.Ls, drop = True)
        d_O = d_O.where(d_O.Ls <= Lsmax, drop = True)
        d_O = d_O.where(d_O.lat >= 0, drop = True)
        
        lait_O = cPV.lait(d_O.PV, d_O.theta, theta_0, kappa=kappa)
        pv_o = lait_O.mean(dim='time').mean(dim='lon') *10**5
        t_o = d_O.theta.mean(dim='time').mean(dim='lon')
        u_o = d_O.uwnd.mean(dim='time').mean(dim='lon')

        lats_max = []
        arr = pv_o.load()
        for jlev in range(len(arr.plev)):
            marr = arr.sel(plev=arr.plev[jlev])
            marr_max = marr.max().values
            marr = marr.where(marr >= marr_max,drop=True)
            lat_max = marr.lat.values

            #lat_max, max_mag = calc_PV_max(marr[:,ilev], lait1.lat)
            lats_max.append(lat_max)

        ax.plot(lats_max, arr.plev/100, linestyle='-', color='blue',linewidth=2)

        ax.contourf(lat_o.where(lat_o.lat >= 0, drop=True), p_o, pv_o.transpose('plev','lat'),
                    levels = [-50]+boundaries+[150],norm=norm,cmap=cmap)
        ax.contour(lat_o.where(lat_o.lat >= 0, drop=True), p_o, t_o.transpose('plev','lat'),
                    levels=[200,300,400,500,600,700,800,900,1000,1100],
                    linestyles = '--', colors='black', linewidths=1)
        csi = ax.contour(lat_o.where(lat_o.lat >= 0, drop=True), p_o, u_o.transpose('plev','lat'),
                        levels=[0,50,100,150], colors='black',linewidths=1)

        csi.levels = [funcs.nf(val) for val in csi.levels]
        ax.clabel(csi, csi.levels, inline=1, fmt=fmt, fontsize=14)
        ax.tick_params(labelsize=18, length=8)
        ax.tick_params(length=4, which='minor')

        ax.set_title('MY '+str(my), fontsize = 22, weight = 'bold', y = 1.02)
        plt.savefig('OpenMARS_figs/Waugh_yearly.pdf', bbox_inches='tight',
                pad_inches=0.04)

    
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs,
                extend='max', ticks=boundaries[slice(None,None,1)],pad=0.01)

    cb.set_label(label='Lait-scaled PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)',
                 fontsize=20)
    cb.ax.tick_params(labelsize=18)


    axs[1,0].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[1,1].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[1,2].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[1,3].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[0,0].set_ylabel('pressure (hPa)',fontsize=20)
    axs[1,0].set_ylabel('pressure (hPa)',fontsize=20)

    plt.savefig('OpenMARS_figs/Waugh_yearly.pdf', bbox_inches='tight',
                pad_inches=0.04)
    
    
    
    
    
    
