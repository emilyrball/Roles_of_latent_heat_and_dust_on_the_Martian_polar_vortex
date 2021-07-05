'''
Calculates and plots polar PV for EMARS - all years.
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
from matplotlib.legend_handler import HandlerTuple

import pandas as pd

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

if __name__ == "__main__":

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    theta0 = 200.
    kappa = 1/4.0
    p0 = 610.

    EMARS = False
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
    labels = []

    newlabel = ['Northern\nsummer solstice', 'Northern\nwinter solstice']
    newpos = [90,270]
    
    fig, axs = plt.subplots(1, 2, figsize = (18,8))

    latax = []

    for i, ax in enumerate(fig.axes):
        ax.set_xlim([0,360])
        ax.tick_params(length = 6, labelsize = 18)

        ax.set_ylim([-0.8, 23.4])
        ax.set_yticks([0,5,10,15,20])

        ax2 = ax.twiny()
        ax2.set_xticks(newpos)

        ax2.xaxis.set_ticks_position('top')
        ax2.xaxis.set_label_position('top') 
        ax2.tick_params(length = 6)
        ax2.set_xlim(ax.get_xlim())

        ax3 = ax.twinx()
        
        ax3.tick_params(length = 6, labelsize = 16)
        
        ax3.yaxis.set_label_position('right')
        ax3.yaxis.set_ticks_position('right')
        ax3.tick_params(length = 6, labelsize = 18)
        
        ax3.set_ylim([55,88])
        if i == 0:
            ax3.set_yticklabels([])
        else:
            ax3.set_ylabel('latitude ($^\circ$N)', fontsize = 20)
        ax3.plot(np.linspace(265,310,20),np.ones(20)*56, color = 'k',
                 linewidth = '3.5',)
        latax.append(ax3)

        ax.text(-0.04, 1.00, string.ascii_lowercase[i], size = 20,
                    transform = ax.transAxes, weight = 'bold')



        
        ax.text(0.03, 0.93, reanalysis, size = 20,
                        transform = ax.transAxes, weight = 'bold')
        ax.set_xlabel('solar longitude (degrees)', fontsize = 20)
        
        if i == 0:
            ax.set_ylabel('Lait-scaled PV (MPVU)', fontsize = 20)
        else:
            ax.set_yticklabels([])

    
        ax2.set_xticklabels(newlabel,fontsize=18)



    plt.subplots_adjust(hspace=.06, wspace = 0.05)



    plt.savefig('Thesis/polar_PV/'+reanalysis+'_' +str(ilev)+ 'K'+sd+'.pdf',
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
    d = d.mean(dim="lon")

    ci = []
    cimax = []
    cu = []
    cumax = []

    for i in list(np.arange(24,yearmax,1)):
        di = d.where(d.MY == i, drop=True)
        print(i)
        di["Ls"] = di.Ls.mean(dim="lat")
        #di["Ls"] = di.Ls.sel(lat=di.lat[0]).sel(lon=di.lon[0])
        if EMARS == True:
            di = di.sortby(di.Ls, ascending=True)

        # Lait scale PV
        theta = di.ilev
        print("Scaling PV")
        laitPV = funcs.lait(di.PV, theta, theta0, kappa = kappa)
        di["scaled_PV"] = laitPV * 10**4

        Zi = di.scaled_PV
        q0 = []
        qm = []

        
        Zi.load()

        Ls = di.Ls.load()
        

        for l in range(len(Zi.time)):
            if EMARS == True:
                q = Zi[l,:]
            else:
                q = Zi.sel(time=Zi.time[l],method="nearest")
            qlat, qmax = funcs.calc_jet_lat(q, q.lat)
            q0.append(qlat)
            qm.append(qmax)
        
        #Zi = Zi.chunk({'time':'auto'})
        #Zi = Zi.rolling(time=smooth,center=True)

        #Zi = Zi.mean()

        Ls = funcs.moving_average(Ls, smooth)
        q0 = funcs.moving_average(q0, smooth)
        qm = funcs.moving_average(qm, smooth)
        
        if i < 28:
            ax = axs[0]
            ax2 = latax[0]
            color1 = 'black'
            color2 = 'blue'
            linestyle1 = linestyles[i-28]
            linestyle2 = linestyles[i-28]

        elif i == 28:
            ax = axs[1]
            ax2 = latax[1]
            color1 = 'red'
            color2 = 'red'
            linestyle1 = 'solid'
            linestyle2 = 'dashed'

        else:
            ax = axs[1]
            ax2 = latax[1]
            color1 = 'black'
            color2 = 'blue'
            linestyle1 = linestyles[i-29]
            linestyle2 = linestyles[i-29]

        labels.append('MY '+str(i))


        c1, = ax.plot(Ls, qm, label='MY '+str(i), color=color1,
                     linestyle = linestyle1)

        Ls = np.where(Ls > 180, Ls, None)
        c2, = ax2.plot(Ls, q0, label="", color=color2,
                        linestyle = linestyle2)

        if i < 28:
            ci.append(c1)
            cimax.append(c2)
        else:
            cu.append(c1)
            cumax.append(c2)
                     
        plt.savefig('Thesis/polar_PV/'+reanalysis+'_' +str(ilev)+ 'K'+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

    c0 = [[ci[j],cimax[j]] for j in range(len(ci))]
    c1 = [[cu[j],cumax[j]] for j in range(len(cu))]

    axs[0].legend([tuple(c0[j]) for j in list(np.arange(0,yearmax-24,1))[:4]], [i for i in labels[:4]],
                 fontsize = 12, loc = 'center left',
            handler_map={tuple: HandlerTuple(ndivide=None)})

    axs[1].legend([tuple(c1[j]) for j in list(np.arange(-4,yearmax-28,1))[4:]], [i for i in labels[4:]],
                 fontsize = 12, loc = 'center left',
            handler_map={tuple: HandlerTuple(ndivide=None)})
    
    #axs[0].legend(fontsize = 15, loc = 'upper center')
    #axs[1].legend(fontsize = 15, loc = 'upper center')

    plt.savefig('Thesis/polar_PV/'+reanalysis+'_' +str(ilev)+ 'K'+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)