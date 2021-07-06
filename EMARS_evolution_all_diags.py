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

    latmax = 90
    latmin = 60

    ilev = 350

    PATH = '/export/anthropocene/array-01/xz19136/Data_Ball_etal_2021'
    files = '/*EMARS*'
    reanalysis = 'EMARS'

    figpath = reanalysis+'_figs/'

    linestyles = ['solid', 'dotted','dashed', 'dashdot']
    newlabel = ['Northern\nsummer solstice', 'Northern\nwinter solstice']
    newpos = [90,270]
    cols = ['#5F9ED1', '#C85200','#898989']
    labs = ["MY 24-27", "MY 28", "MY 29-32"]
    
    fig, axs = plt.subplots(1, 3, figsize = (25,7.5))

    for i, ax in enumerate(fig.axes):
        ax.set_xlim([0,360])
        ax.tick_params(length = 6, labelsize = 18)

        ax2 = ax.twiny()
        ax2.set_xticks(newpos)
        ax2.set_xticklabels(newlabel,fontsize=18)

        ax2.xaxis.set_ticks_position('top')
        ax2.xaxis.set_label_position('top') 
        ax2.tick_params(length = 6)
        ax2.set_xlim(ax.get_xlim())

        
        
        ax4 = ax.twinx()
        ax4.tick_params(length = 6, labelsize = 18)

        ax4.yaxis.set_label_position('right')
        ax4.yaxis.set_ticks_position('right')
        ax4.set_ylim([-0.05,1])
        ax4.set_yticks([])
        ax4.set_yticklabels([])
        
        ax4.plot(np.linspace(265,310,20),np.zeros(20), color = 'k',
                 linewidth = '3.5',)

        ax.text(-0.04, 1.01, string.ascii_lowercase[i], size = 20,
                    transform = ax.transAxes, weight = 'bold')

        if i == 0:
            ax.set_ylim([-0.4, 11.9])
            ax.set_yticks([0,2,4,6,8,10])
            ax.text(0.01, 0.95, "Lait-scaled PV", size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_ylabel('Lait-scaled PV (MPVU)', fontsize = 20)
            ax.set_xlabel('solar longitude (degrees)', fontsize = 20)
        elif i == 1:
            ax.set_ylim([68,88])
            ax.set_yticks([70,72.5,75,77.5,80,82.5,85,87.5])
            ax.set_yticklabels([70,None,75,None,80,None,85,None])
            ax.text(0.01, 0.9, "Max PV\nlatitude", size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_ylabel('latitude of maximum PV ($^\circ$N)', fontsize = 20)
            ax.set_xlabel('solar longitude (degrees)', fontsize = 20)
        elif i == 2:
            ax.set_ylim([-1, 55])
            ax.set_yticks([0,10,20,30,40,50])
            ax.text(0.01, 0.95, "Eddy enstrophy", size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_ylabel('eddy enstrophy (MPVU$^2$)', fontsize = 20)
            ax.set_xlabel('solar longitude (degrees)', fontsize = 20)

    plt.subplots_adjust(hspace=.06, wspace = 0.2)

    plt.savefig(figpath+'evo_all_' +str(ilev)+ 'K_'+reanalysis+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

    d = xr.open_mfdataset(PATH+files, decode_times=False, concat_dim='time',
                           combine='nested',chunks={'time':'auto'})

    #d["Ls"] = d.Ls.expand_dims({"lat":d.lat})
    #d = d.rename({"pfull":"plev"})
    #d = d.rename({"t":"tmp"})
    smooth = 200
    yearmax = 32

    #d = d.sel(ilev = ilev, method='nearest').squeeze()

    latm = d.lat.max().values

    # Lait scale PV
    theta = ilev
    print("Scaling PV")
    laitPV = funcs.lait(d.PV, theta, theta0, kappa = kappa)
    d["scaled_PV"] = laitPV
    
    reanalysis_clim = []
    reanalysis_ls = []
    reanalysis_lat = []
    labels = []

    ci = []
    cimax = []
    cu = []
    cumax = []

    ui = []
    uimax = []
    uu = []
    uumax = []

    for i in list(np.arange(24,yearmax,1)):
        di = d.where(d.MY == i, drop=True)
        print(i)
        di["Ls"] = di.Ls.sel(lat=di.lat[0]).sel(lon=di.lon[0])
        di = di.sortby(di.Ls, ascending=True)
        di = di.transpose("lat","lon","time")
        di = di.sortby("lat", ascending = True)
        di = di.sortby("lon", ascending = True)
               
        
        Ei = di.scaled_PV
        Ei = Ei.where(Ei.lat > 60, drop = True)
        Ei = Ei.where(Ei.lat < 90, drop = True)

        Zi = Ei.mean(dim = "lon") * 10**4

        q0 = []
        qm = []
        
        Zi.load()

        Ls = di.Ls.load()
        
        qmax = Zi.mean(dim="lat")
        for l in range(len(Zi.time)):
            qmul = qmax[l]
            q = Zi[:,l]
            
            qlat, _ = funcs.calc_jet_lat(q, q.lat)
            if qlat > latm:
                qlat = latm
            q0.append(qlat)
            qm.append(qmul)
        
        #Zi = Zi.chunk({'time':'auto'})
        #Zi = Zi.rolling(time=smooth,center=True)

        #Zi = Zi.mean()

        Ls = funcs.moving_average(Ls, smooth)
        q0 = funcs.moving_average(q0, smooth)
        qm = funcs.moving_average(qm, smooth)
        
        
        if i < 28:
            color1 = cols[0]
            label = labs[0]
            w = '1.2'
        elif i == 28:
            color1 = cols[1]
            label = labs[1]
            w = '2'
        else:
            color1 = cols[2]
            label = labs[2]
            w = '1.2'

        labels.append('MY '+str(i))

        c1, = axs[0].plot(Ls, qm, label=label, color=color1,
                        linestyle = '-', linewidth = w)

        Ls = np.where(Ls > 170, Ls, None)
        c2, = axs[1].plot(Ls, q0, label=label, color=color1,
                        linestyle = '-', linewidth = w)

        ci.append(c1)
        cimax.append(c2)
                     
        plt.savefig(figpath+'evo_all_' +str(ilev)+ 'K_'+reanalysis+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

        
        

        Ei = funcs.calc_eddy_enstr(Ei) * 10**8
        
        Ls = di.Ls.load()
        
        Ei = Ei.chunk({'time':'auto'})
        Ei = Ei.rolling(time=smooth,center=True)

        Ei = Ei.mean()
        
        if i < 28:
            color1 = cols[0]
            label = labs[0]
            w = '1.2'
        elif i == 28:
            color1 = cols[1]
            label = labs[1]
            w = '2'
        else:
            color1 = cols[2]
            label = labs[2]
            w = '1.2'

        labels.append('MY '+str(i))

        c3, = axs[2].plot(Ls, Ei, label=label, color=color1,
                        linestyle = '-', linewidth = w)

        ui.append(c3)

                     
        plt.savefig(figpath+'evo_all_' +str(ilev)+ 'K_'+reanalysis+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)
    
    c0 = [[ci[j]] for j in range(len(ci))]
    c1 = [[cimax[j]] for j in range(len(cimax))]
    u0 = [[ui[j]] for j in range(len(ui))]

    axs[0].legend([tuple(c0[j]) for j in [0,4,5]], [i for i in labs],
                 fontsize = 14, loc = 'center left', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    axs[1].legend([tuple(c1[j]) for j in [0,4,5]], [i for i in labs],
                 fontsize = 14, loc = 'center left', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    axs[2].legend([tuple(u0[j]) for j in [0,4,5]], [i for i in labs],
                 fontsize = 14, loc = 'center left', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.savefig(figpath+'evo_all_' +str(ilev)+ 'K_'+reanalysis+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)
