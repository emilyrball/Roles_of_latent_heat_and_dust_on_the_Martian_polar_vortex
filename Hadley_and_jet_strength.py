'''
Calculates and plots Hadley circulation strength and latitude evolution for OpenMARS, Isca - all years and exps.
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
from matplotlib.legend_handler import HandlerTuple

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

if __name__ == "__main__":

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    theta0 = 200.
    kappa = 1/4.0
    p0 = 610.

    SD = False
    ilev = 50

    latmax = 90
    latmin = 0

    Lsmax = 360
    Lsmin = 180

    figpath = 'OpenMARS_figs/Hadley_lats/'

    reanalysis = 'OpenMARS'
    PATH = '/export/anthropocene/array-01/xz19136/OpenMARS/Isobaric'
    files = '/*isobaric*'
    yearmax = 33
    smooth = 250

    if SD == True:
        sd = '_SD'
    else:
        sd = ''

    linestyles = ['solid', 'dotted','dashed', 'dashdot']
    newlabel = ['Northern\nsummer solstice', 'Northern\nwinter solstice']
    newpos = [90,270]
    cols = ['#5F9ED1', '#C85200','#898989']
    labs = ["MY 24-27", "MY 28", "MY 29-32"]
    
    fig, axs = plt.subplots(2, 2, figsize = (16.6,15))

    for i, ax in enumerate(fig.axes):
        ax.set_xlim([200,360])
        ax.tick_params(length = 6, labelsize = 18)

        ax3 = ax.twinx()
        ax3.tick_params(length = 6, labelsize = 18)
        
        ax3.yaxis.set_label_position('right')
        ax3.yaxis.set_ticks_position('right')

        
        ax3.set_ylim([5,76])
        ax3.set_yticks([])
        ax3.set_yticklabels([])
        

        ax.text(-0.04, 1.01, string.ascii_lowercase[i], size = 20,
                    transform = ax.transAxes, weight = 'bold')
        
        ax3.plot(np.linspace(265,310,20),np.ones(20)*7, color = 'k',
                 linewidth = '3.5',)
        ax3.set_yticklabels([])



        if i == 0:
            ax.text(0.01, 0.95, '$\psi$ strength', size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_ylabel('$\psi$ strength (kg s$^{-1}$)', fontsize = 20)
            ax.set_xticklabels([])
            ax.set_ylim([-1, 51])
            ax.set_yticks([0,10,20,30,40,50])

        elif i == 1:
            ax.text(0.01, 0.95, 'Jet strength', size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_ylabel('jet strength (m s$^{-1}$)', fontsize = 20)
            ax.set_xticklabels([])
            ax.set_ylim([60, 170])
            ax.set_yticks([75,100,125,150])

        elif i == 2:
            ax.text(0.01, 0.95, 'Edge of Hadley cell', size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_ylabel('Hadley cell edge latitude($^{\circ}$N)', fontsize = 20)
            
            ax.set_xlabel('solar longitude (degrees)', fontsize = 20)
            ax.set_xticklabels([200,220,240,260,280,300,320,340])
            ax.set_ylim([46, 81])
        else:
            ax.text(0.01, 0.95, 'Jet latitude', size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_xlabel('solar longitude (degrees)', fontsize = 20)
            ax.set_ylabel('jet latitude ($^{\circ}$N)', fontsize = 20)
            ax.set_xticklabels([200,220,240,260,280,300,320,340])
            ax.set_ylim([46, 81])



    plt.subplots_adjust(hspace=.06, wspace = 0.18)




    plt.savefig(figpath+'all_'+str(ilev)+'Pa_'+reanalysis+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

    
    reanalysis_clim = []
    reanalysis_ls = []
    reanalysis_lat = []
    labels = []

    ci = []
    cimax = []
    cu = []
    cumax = []
    PATH = '/export/anthropocene/array-01/xz19136/Data_Ball_etal_2021/OpenMARS_'
    d = xr.open_mfdataset(PATH + 'Ls200-360_u_50Pa.nc', decode_times=False)
    
    d = d.where(d.lat > latmin, drop = True)
    d = d.where(d.lat < latmax, drop = True)
    d = d.where(d.Ls > 200, drop = True)
    d = d.where(d.Ls <= 360, drop = True)
    latm = d.lat.max().values
    d = d.transpose("lat","lon","time")
    d["Ls"] = d.Ls.sel(lat=d.lat[0]).sel(lon=d.lon[0])
    d = d.sortby("lat", ascending = True)
    d = d.sortby("lon", ascending = True)
    for i in list(np.arange(24,yearmax,1)):
        if i == 27:
            continue
        print(i)
        di = d.where(d.MY == i, drop = True)
        
        di["Ls"] = di.Ls.sel(lat = 5, method = 'nearest', drop = True).squeeze()
        #di = di.sortby("time", ascending = True)
        di = di.mean(dim = "lon")
        Zi = di.uwnd

        q0 = []
        qm = []

        Zi.load()

        Ls = di.Ls.load()

        for l in range(len(Zi.time)):
            q = Zi.sel(time=Zi.time[l],method="nearest")
            
            qlat, qmax = funcs.calc_jet_lat(q, q.lat)
            if qlat > latm:
                qlat = latm
            q0.append(qlat)
            qm.append(qmax)
        
        if i != 28:
            reanalysis_clim.append(qm)
            reanalysis_lat.append(q0)
            reanalysis_ls.append(Ls)

        Ls = funcs.moving_average(Ls, smooth)
        q0 = funcs.moving_average(q0, smooth)
        qm = funcs.moving_average(qm, smooth)

        ax = axs[0,1]
        ax2 = axs[1,1]
        linestyle1='solid'
        
        
        if i < 28:
            color1 = cols[0]
            label = labs[0]
            w = '1'

        elif i == 28:
            color1 = cols[1]
            label = labs[1]
            w = '1.5'

        else:
            color1 = cols[2]
            label = labs[2]
            w = '1'

        labels.append('MY '+str(i))

        c1, = ax.plot(Ls, qm, label=label, color=color1,
                     linestyle = linestyle1, linewidth = w)

        Ls = funcs.moving_average(Ls, 2)
        q0 = funcs.moving_average(q0, 2)
        c2, = ax2.plot(Ls, q0, label=label, color=color1,
                        linestyle = linestyle1, linewidth = w)

        cu.append(c1)
        cumax.append(c2)
                     
        plt.savefig(figpath+'all_' +str(ilev)+ 'Pa_'+reanalysis+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)


        year = 'MY'+str(i)
        di = xr.open_mfdataset(PATH + year + '_Ls200-360_psi.nc',
                                            decode_times = False)
        
                           

        ##### reduce dataset #####
        di = di.astype('float32')
        di = di.sortby('time', ascending =True) / 10 ** 8
        di = di.sortby('lat', ascending = False)

        di = di.where(di.time <= Lsmax, drop=True)

        plev = di.pfull.sel(pfull = ilev, method = "nearest").values

        latm = di.lat.max().values
        Ls = di.time.load()

        Ls.load()
        di.load()

        q0 = []
        qm = []


        for j in range(Ls.shape[0]):
            lsj = Ls[j]
            psi_j = di.where(di.time == lsj, drop = True).squeeze()
            psi_j = psi_j.to_array().squeeze()

            psi_j = psi_j.sel(pfull = plev, method = "nearest").squeeze()
            psi_j.load()
            _, psi_max = funcs.calc_jet_lat(psi_j, psi_j.lat)
            psi0_lat, _ = funcs.calc_Hadley_lat(psi_j, psi_j.lat)
            if psi0_lat > latm:
                psi0_lat = latm

            q0.append(psi0_lat)
            qm.append(psi_max)



        if i != 28:
            reanalysis_clim.append(qm)
            reanalysis_lat.append(q0)
            reanalysis_ls.append(Ls)

        Ls = funcs.moving_average(Ls, smooth)
        q0 = funcs.moving_average(q0, smooth)
        qm = funcs.moving_average(qm, smooth)

        
        ax = axs[0,0]
        ax2 = axs[1,0]
        linestyle1='solid'
        
        
        if i < 28:
            color1 = cols[0]
            label = labs[0]
            w = '1'

        elif i == 28:
            color1 = cols[1]
            label = labs[1]
            w = '1.5'

        else:
            color1 = cols[2]
            label = labs[2]
            w = '1'

        labels.append('MY '+str(i))

        c1, = ax.plot(Ls, qm, label=label, color=color1,
                     linestyle = linestyle1, linewidth = w)

        Ls = funcs.moving_average(Ls, 2)
        q0 = funcs.moving_average(q0, 2)
        c2, = ax2.plot(Ls, q0, label=label, color=color1,
                        linestyle = linestyle1, linewidth = w)


        ci.append(c1)
        cimax.append(c2)
                     
        plt.savefig(figpath+'all_' +str(ilev)+ 'Pa_'+reanalysis+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)
    
    c0 = [[ci[j]] for j in range(len(ci))]
    c1 = [[cimax[j]] for j in range(len(cimax))]

    axs[0,0].legend([tuple(c0[j]) for j in [0,3,4]], [i for i in labs],
                 fontsize = 14, loc = 'upper right', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    axs[1,0].legend([tuple(c1[j]) for j in [0,3,4]], [i for i in labs],
                 fontsize = 14, loc = 'upper right', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    
    c0 = [[cu[j]] for j in range(len(cu))]
    c1 = [[cumax[j]] for j in range(len(cumax))]

    axs[0,1].legend([tuple(c0[j]) for j in [0,3,4]], [i for i in labs],
                 fontsize = 14, loc = 'upper right', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    axs[1,1].legend([tuple(c1[j]) for j in [0,3,4]], [i for i in labs],
                 fontsize = 14, loc = 'upper right', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.savefig(figpath+'all_' +str(ilev)+ 'Pa_'+reanalysis+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)