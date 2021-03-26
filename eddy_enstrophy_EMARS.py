'''
Calculates and plots eddy enstrophy for OpenMARS, Isca - all years and exps.
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

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'


    EMARS = True
    SD = False
    ilev = 350

    latmax = 90
    latmin = 60

    if EMARS == True:
        PATH = '/export/anthropocene/array-01/xz19136/EMARS'
        files = '/*isentropic*'
        reanalysis = 'EMARS'
    else:
        reanalysis = 'OpenMARS'
        PATH = '/export/anthropocene/array-01/xz19136/OpenMARS/Isentropic'
        files = '/*isentropic*'

    if SD == True:
        sd = '_SD'
    else:
        sd = ''

    linestyles = ['solid', 'dotted','dashed', 'dashdot']

    newlabel = ['Northern\nsummer solstice', 'Northern\nwinter solstice']
    newpos = [90,270]
    
    fig, axs = plt.subplots(1, 2, figsize = (18,8))

    for i, ax in enumerate(fig.axes):
        ax.set_xlim([0,360])
        ax.tick_params(length = 6, labelsize = 18)

        ax.set_ylim([-0.1, 149])
        ax.set_yticks([0,25,50,75,100,125])

        ax2 = ax.twiny()
        ax2.set_xticks(newpos)

        ax2.xaxis.set_ticks_position('top')
        ax2.xaxis.set_label_position('top') 
        ax2.tick_params(length = 6)
        ax2.set_xlim(ax.get_xlim())

        ax.text(-0.04, 1.00, string.ascii_lowercase[i], size = 20,
                    transform = ax.transAxes, weight = 'bold')



        
        ax.text(0.03, 0.93, reanalysis, size = 20,
                        transform = ax.transAxes, weight = 'bold')
        ax.set_xlabel('solar longitude (degrees)', fontsize = 20)
        
        if i == 0:
            ax.set_ylabel('eddy enstrophy ($10^{6}$PVU$^2$)', fontsize = 20)
        else:
            ax.set_yticklabels([])

    
        ax2.set_xticklabels(newlabel,fontsize=18)



    plt.subplots_adjust(hspace=.06, wspace = 0.05)



    plt.savefig('Thesis/eddy_enstrophy_'+reanalysis+'_' +str(ilev)+ 'K'+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)


    d = xr.open_mfdataset(PATH+files, decode_times=False, concat_dim='time',
                           combine='nested',chunks={'time':'auto'})

    if EMARS==True:
        d["Ls"] = d.Ls.expand_dims({"lat":d.lat})
        #d = d.rename({"pfull":"plev"})
        #d = d.rename({"t":"tmp"})
        smooth = 200
        yearmax = 32
    else:
        smooth = 250
        d = d.sortby('time', ascending=True)
        yearmax = 33

    d = d.where(d.lat > latmin, drop = True)
    d = d.where(d.lat < latmax, drop = True)
    d = d.sortby('lat', ascending=True)
    d = d.sortby('lon', ascending=True)
    d = d.sel(ilev = ilev, method='nearest').squeeze()
    

    for i in list(np.arange(24,yearmax,1)):
        di = d.where(d.MY == i, drop=True)
        print(i)
        di["Ls"] = di.Ls.sel(lat=di.lat[0]).sel(lon=di.lon[0])
        if EMARS == True:
            di = di.sortby(di.Ls, ascending=True)
        Zi = funcs.calc_eddy_enstr(di.PV) * 10**6

        
        Ls = di.Ls
        
        Zi = Zi.chunk({'time':'auto'})
        Zi = Zi.rolling(time=smooth,center=True)

        Zi = Zi.mean()
        
        if i < 28:
            ax = axs[0]
            color = 'black'
            linestyle = linestyles[i-28]

        elif i == 28:
            ax = axs[1]
            color = 'red'
            linestyle = 'solid'

        else:
            ax = axs[1]
            color = 'black'
            linestyle = linestyles[i-29]


        ci = ax.plot(Ls, Zi, label='MY '+str(i), color=color,
                     linestyle = linestyle)
                     
        plt.savefig('Thesis/eddy_enstrophy_'+reanalysis+'_' +str(ilev)+ 'K'+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)
    
    axs[0].legend(fontsize = 15, loc = 'upper center')
    axs[1].legend(fontsize = 15, loc = 'upper center')

    plt.savefig('Thesis/eddy_enstrophy_'+reanalysis+'_' +str(ilev)+ 'K'+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)


