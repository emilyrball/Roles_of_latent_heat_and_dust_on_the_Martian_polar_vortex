'''
Script to plot the MMC for OpenMARS data
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

import colorcet as cc
import string

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

    ##### change parameters #####
    Lsmin = 270
    Lsmax = 300

    vmin = -2
    vmax = 19
    step = 2

    vmin0 = -10
    vmax0 = 10.5
    step0 = 1

    yearlist = [
        24,25,26,
        28,
        29,30,31,32]

    figpath = 'Thesis/streamfunctions/'



    fig, axs = plt.subplots(2, 3, figsize = (24, 16))

    for i, ax in enumerate(fig.axes):
        ax.set_yscale('log')
        ax.tick_params(length = 6, labelsize = 18)
        ax.set_xlim([0,90])
        ax.set_ylim([10,0.005])
        ax.text(-0.06, 1.02, string.ascii_lowercase[i], transform=ax.transAxes,
                size=22, weight='bold')
        
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    axs[1,0].set_xlabel('latitude ($^{\circ}$N)', fontsize = 20)
    axs[1,1].set_xlabel('latitude ($^{\circ}$N)', fontsize = 20)
    axs[1,2].set_xlabel('latitude ($^{\circ}$N)', fontsize = 20)

    axs[0,1].set_yticklabels([])
    axs[0,2].set_yticklabels([])
    axs[1,1].set_yticklabels([])
    axs[1,2].set_yticklabels([])

    axs[0,0].set_xticklabels([])
    axs[0,1].set_xticklabels([])
    axs[0,2].set_xticklabels([])

    axs[0,0].set_ylabel('pressure (hPa)', fontsize = 20)
    axs[1,0].set_ylabel('pressure (hPa)', fontsize = 20)
    axs[0,0].text(-0.26, 0.3, "OpenMARS", transform=axs[0,0].transAxes, 
                        size=22, weight='bold',rotation='vertical')
    axs[1,0].text(-0.26, 0.39, "Model", transform=axs[1,0].transAxes, 
                        size=22, weight='bold',rotation='vertical')
    
    axs[0,0].set_title('MY 28', fontsize = 22, weight = 'bold', y = 1.02)
    axs[0,1].set_title('Climatology', fontsize = 22, weight = 'bold', y = 1.02)
    axs[0,2].set_title('Difference', fontsize = 22, weight = 'bold', y = 1.02)

    plt.subplots_adjust(hspace=.11,wspace=.09)


    boundaries, _, _, cmap, norm = funcs.make_colourmap(vmin, vmax, step,
                                        col = 'cet_CET_L12', extend = 'both')

    
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[:,0:2],
                extend='both', ticks=boundaries,pad=0.07,
                orientation = 'horizontal', aspect = 40)

    cb.set_label(label='$\psi$ ($10^{8}$ kg/s)',
                 fontsize=20)
    cb.ax.tick_params(labelsize=18)

    boundaries0, _, _, cmap0, norm0 = funcs.make_colourmap(vmin0, vmax0, step0,
                                        col = 'cet_CET_CBD1', extend = 'both')

    
    cb0 = fig.colorbar(cm.ScalarMappable(norm=norm0, cmap=cmap0),ax=axs[:,2],
                extend='both', ticks=boundaries0[slice(None,None,2)],pad=0.07,
                orientation = 'horizontal')

    cb0.set_label(label = '$\psi$ difference ($10^{8}$ kg/s)', fontsize = 20)
    cb0.ax.tick_params(labelsize=18)


    fig.savefig(figpath+'all_psi_cross-section_Ls' + str(Lsmin) + '-' \
            + str(Lsmax) + '.pdf', bbox_inches = 'tight', pad_inches = 0.06)
    
    ##### get data #####
    
    PATH = '/export/anthropocene/array-01/xz19136/OpenMARS/Streamfn'
    
    if Lsmax < 180:
        path = '_0-75'
    else:
        path = '_180-360'
    
    years = []

    for i in yearlist:
        year = 'MY'+str(i)
        infiles = '/'+year+path+'_psi.nc'
        d = xr.open_mfdataset(PATH + infiles, decode_times = False,
                          concat_dim = 'time', combine = 'nested',
                          chunks = {'time' : 'auto'})

        ##### reduce dataset #####
        d = d.astype('float32')

        d = d.where(d.time <= Lsmax, drop=True)
        d = d.where(Lsmin <= d.time, drop=True)
        d = d.mean(dim = "time", skipna = True)
        d = d.assign_coords({'MY' : i})
        
        years.append(d)
        
    d = xr.concat(years, dim = "MY") / 10 ** 8
    d["pfull"] = d.pfull / 100

    d28 = d.where(d.MY == 28, drop = True)
    d = d.where(d.MY != 28, drop = True)
    d = d.mean(dim = "MY")

    d28 = d28.to_array().squeeze()
    d = d.to_array().squeeze()
    diff = d28 - d
    
    axs[0,0].contourf(d28.lat, d28.pfull, d28.transpose('pfull','lat'),
                    levels = [-50]+boundaries+[150],norm=norm,cmap=cmap)
    c0=axs[0,0].contour(d28.lat, d28.pfull, d28.transpose('pfull','lat'),
                    levels = boundaries[slice(None,None,2)], colors='black',
                    linewidths=0.6)
    
    c0.levels = [funcs.nf(val) for val in c0.levels]
    axs[0,0].clabel(c0, c0.levels, inline=1, fmt=fmt, fontsize=14)

    axs[0,1].contourf(d.lat, d.pfull, d.transpose('pfull','lat'),
                    levels = [-50]+boundaries+[150],norm=norm,cmap=cmap)
    c1=axs[0,1].contour(d.lat, d.pfull, d.transpose('pfull','lat'),
                    levels = boundaries[slice(None,None,2)], colors='black',
                    linewidths=0.6)
    
    c1.levels = [funcs.nf(val) for val in c1.levels]
    axs[0,1].clabel(c1, c1.levels, inline=1, fmt=fmt, fontsize=14)

    axs[0,2].contourf(diff.lat, diff.pfull, diff.transpose('pfull','lat'),
                    levels = [-50]+boundaries0+[150],norm=norm0,cmap=cmap0)
    c2=axs[0,2].contour(diff.lat, diff.pfull, diff.transpose('pfull','lat'),
                    levels = boundaries0[slice(None,None,2)], colors='black',
                    linewidths=0.6)
    
    c2.levels = [funcs.nf(val) for val in c2.levels]
    axs[0,2].clabel(c2, c2.levels, inline=1, fmt=fmt, fontsize=14)

    fig.savefig(figpath+'all_psi_cross-section_Ls' + str(Lsmin) + '-' \
            + str(Lsmax) + '.pdf', bbox_inches = 'tight', pad_inches = 0.06)

    


    PATH = '/export/silurian/array-01/xz19136/Isca_data/Streamfn'
        
    infiles = '/topo_MY28_lh_180-360_psi.nc'
    d = xr.open_mfdataset(PATH + infiles, decode_times = False,
                          concat_dim = 'time', combine = 'nested',
                          chunks = {'time' : 'auto'})

    ##### reduce dataset #####
    d = d.astype('float32')

    d = d.where(d.time <= Lsmax, drop=True)
    d = d.where(Lsmin <= d.time, drop=True)
    d = d.mean(dim = "time", skipna = True) / 10 ** 8
    d["pfull"] = d.pfull / 100

    infiles = '/topo_scenario_lh_180-360_psi.nc'
    dclim = xr.open_mfdataset(PATH + infiles, decode_times = False,
                          concat_dim = 'time', combine = 'nested',
                          chunks = {'time' : 'auto'})

    ##### reduce dataset #####
    dclim = dclim.astype('float32')
    dclim = dclim.where(dclim.time <= Lsmax, drop=True)
    dclim = dclim.where(Lsmin <= dclim.time, drop=True)
    dclim = dclim.mean(dim = "time", skipna = True) / 10 ** 8
    dclim["pfull"] = dclim.pfull / 100

    dclim = dclim.to_array().squeeze()
    d = d.to_array().squeeze()
    diff = d - dclim
    
    axs[1,0].contourf(d.lat, d.pfull, d.transpose('pfull','lat'),
                    levels = [-50]+boundaries+[150],norm=norm,cmap=cmap)
    c0=axs[1,0].contour(d.lat, d.pfull, d.transpose('pfull','lat'),
                    levels = boundaries[slice(None,None,2)], colors='black',
                    linewidths=0.6)
    
    c0.levels = [funcs.nf(val) for val in c0.levels]
    axs[1,0].clabel(c0, c0.levels, inline=1, fmt=fmt, fontsize=14)

    axs[1,1].contourf(dclim.lat, dclim.pfull, dclim.transpose('pfull','lat'),
                    levels = [-50]+boundaries+[150],norm=norm,cmap=cmap)
    c1=axs[1,1].contour(dclim.lat, dclim.pfull, dclim.transpose('pfull','lat'),
                    levels = boundaries[slice(None,None,2)], colors='black',
                    linewidths=0.6)
    
    c1.levels = [funcs.nf(val) for val in c1.levels]
    axs[1,1].clabel(c1, c1.levels, inline=1, fmt=fmt, fontsize=14)

    axs[1,2].contourf(diff.lat, diff.pfull, diff.transpose('pfull','lat'),
                    levels = [-50]+boundaries0+[150],norm=norm0,cmap=cmap0)
    c2=axs[1,2].contour(diff.lat, diff.pfull, diff.transpose('pfull','lat'),
                    levels = boundaries0[slice(None,None,2)], colors='black',
                    linewidths=0.6)
    
    c2.levels = [funcs.nf(val) for val in c2.levels]
    axs[1,2].clabel(c2, c2.levels, inline=1, fmt=fmt, fontsize=14)

    fig.savefig(figpath+'all_psi_cross-section_Ls' + str(Lsmin) + '-' \
            + str(Lsmax) + '.pdf', bbox_inches = 'tight', pad_inches = 0.06)
