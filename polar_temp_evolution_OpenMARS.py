'''
Calculates and plots polar PV evolution for OpenMARS, Isca - all years and exps.
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

    EMARS = False
    SD = True
    ilev = 50

    latmax = 90
    latmin = 60

    figpath = 'Thesis/polar_PV/temp_'

    if EMARS == True:
        PATH = '/export/anthropocene/array-01/xz19136/EMARS'
        files = '/*isobaric*'
        reanalysis = 'EMARS'
    else:
        reanalysis = 'OpenMARS'
        PATH = '/export/anthropocene/array-01/xz19136/OpenMARS/Isobaric'
        files = '/*isobaric*'

    if SD == True:
        sd = '_SD'
    else:
        sd = ''

    cols = ['#5F9ED1', '#C85200','#898989']
    labs = ["MY 24-27", "MY 28", "MY 29-32"]
    newlabel = ['Northern\nsummer solstice', 'Northern\nwinter solstice']
    newpos = [90,270]
    
    fig, axs = plt.subplots(2, 3, figsize = (25,15))

    for i, ax in enumerate(fig.axes):
        ax.set_xlim([0,360])
        ax.tick_params(length = 6, labelsize = 18)

        if i < 3:
            print(i)
            #ax.set_ylim([-0.4, 11.9])
            #ax.set_yticks([0,2,4,6,8,10])
        else:
            ax.set_ylim([68,88])
            ax.set_yticks([70,72.5,75,77.5,80,82.5,85,87.5])
            ax.set_yticklabels([70,None,75,None,80,None,85,None])

        ax2 = ax.twiny()
        ax2.set_xticks(newpos)

        ax2.xaxis.set_ticks_position('top')
        ax2.xaxis.set_label_position('top') 
        ax2.tick_params(length = 6)
        ax2.set_xlim(ax.get_xlim())

        ax3 = ax.twinx()
        ax3.tick_params(length = 6, labelsize = 18)
        
        ax3.yaxis.set_label_position('right')
        ax3.yaxis.set_ticks_position('right')

        
        ax3.set_ylim([-0.05,1])
        
        
        ax3.set_yticks([])
        ax3.set_yticklabels([])

        ax.text(-0.04, 1.01, string.ascii_lowercase[i], size = 20,
                    transform = ax.transAxes, weight = 'bold')


        if i == 0:
            ax.set_ylabel('Lait-scaled PV (MPVU)', fontsize = 20)
            ax.text(0.01, 0.95, reanalysis, size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax3.plot(np.linspace(265,310,20),np.zeros(20), color = 'k',
                 linewidth = '3.5',)
        elif i == 3:
            ax.set_ylabel('latitude of maximum PV ($^\circ$N)', fontsize = 20)
            ax.text(0.01, 0.95, reanalysis, size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax3.plot(np.linspace(265,310,20),np.zeros(20), color = 'k',
                 linewidth = '3.5',)

        elif i == 1 or i == 4:
            ax.text(0.01, 0.9, 'Yearly\nsimulations', size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_yticklabels([])
            ax3.plot(np.linspace(265,310,20),np.zeros(20), color = 'k',
                 linewidth = '3.5',)
            
        elif i == 2:
            ax.text(0.01, 0.9, 'Process-attribution\nsimulations', size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_yticklabels([])
        elif i == 5:
            ax.text(0.01, 0.9, 'Process-attribution\nsimulations', size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_yticklabels([])
        if i > 2:
            ax.set_xlabel('solar longitude (degrees)', fontsize = 20)
            ax2.set_xticklabels([])
        else:
            ax.set_xticklabels([])
            ax2.set_xticklabels(newlabel,fontsize=18)

    plt.subplots_adjust(hspace=.06, wspace = 0.05)




    plt.savefig(figpath+'all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
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
    d = d.sel(plev = ilev/100, method='nearest').squeeze()

    latm = d.lat.max().values

    # Lait scale PV
    ##theta = d.ilev
    #print("Scaling PV")
    #laitPV = funcs.lait(d.PV, theta, theta0, kappa = kappa)
    ##d["scaled_PV"] = laitPV
    
    reanalysis_clim = []
    reanalysis_ls = []
    reanalysis_lat = []
    labels = []

    ci = []
    cimax = []
    cu = []
    cumax = []

    for i in list(np.arange(24,yearmax,1)):
        di = d.where(d.MY == i, drop=True)
        print(i)
        di["Ls"] = di.Ls.sel(lat=di.lat[0]).sel(lon=di.lon[0])
        if EMARS == True:
            di = di.sortby(di.Ls, ascending=True)
        di = di.transpose("lat","lon","time")
        di = di.sortby("lat", ascending = True)
        di = di.sortby("lon", ascending = True)
        di = di.mean(dim = "lon")
        #Zi = di.scaled_PV * 10**4
        Zi = di.tmp
        q0 = []
        qm = []

        Zi.load()

        Ls = di.Ls.load()
        

        for l in range(len(Zi.time)):
            if EMARS == True:
                q = Zi[l,:]
            else:
                q = Zi.sel(time=Zi.time[l],method="nearest")
            
            qmax = q.mean(dim="lat").values
            qlat, _ = funcs.calc_jet_lat(q, q.lat)
            if qlat > latm:
                qlat = latm
            q0.append(qlat)
            qm.append(qmax)
        
        #Zi = Zi.chunk({'time':'auto'})
        #Zi = Zi.rolling(time=smooth,center=True)

        #Zi = Zi.mean()
        if i != 28:
            reanalysis_clim.append(qm)
            reanalysis_lat.append(q0)
            reanalysis_ls.append(Ls)

        Ls = funcs.moving_average(Ls, smooth)
        q0 = funcs.moving_average(q0, smooth)
        qm = funcs.moving_average(qm, smooth)
        
        ax = axs[0,0]
        ax2 = axs[1,0]
        linestyle1 = 'solid'

        if i < 28:
            color1 = cols[0]
            label = labs[0]
            w = '1'
        elif i == 28:
            color1 = cols[1]
            label = labs[1]
            w = '2'
        else:
            color1 = cols[2]
            label = labs[2]
            w = '1'
        labels.append('MY '+str(i))

        c1, = ax.plot(Ls, qm, label=label, color=color1,
                     linestyle = linestyle1, linewidth = w)

        Ls = funcs.moving_average(Ls, 5)
        q0 = funcs.moving_average(q0, 5)
        Ls = np.where(Ls > 170, Ls, None)
        c2, = ax2.plot(Ls, q0, label="", color=color1,
                        linestyle = linestyle1, linewidth = w)

        ci.append(c1)
        cimax.append(c2)
                     
        plt.savefig(figpath+'all_' +str(ilev)+ 'K_'+reanalysis+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)
    
    c0 = [[ci[j]] for j in [0,4,5]]
    c1 = [[cimax[j]] for j in [0,4,5]]


    axs[0,0].legend([tuple(c0[j]) for j in range(len(labs))], [i for i in labs],
                 fontsize = 14, loc = 'center left', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    axs[1,0].legend([tuple(c1[j]) for j in range(len(labs))], [i for i in labs],
                 fontsize = 14, loc = 'center left', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.savefig(figpath+'all_' +str(ilev)+ 'K_'+reanalysis+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

    ls0 = np.arange(0,360,0.05)

    PV_new = []

    for i in range(len(reanalysis_clim)):
        PV = xr.Dataset({
            "lat": (["time"], reanalysis_lat[i]),
            "mag": (["time"], reanalysis_clim[i]),
        },
        coords = {
            "time": (["time"], reanalysis_ls[i]),
        },
        )
        #PV_new[i] = PV_new[i].assign_coords(time = (reanalysis_ls[i].values))
        PV_new.append(PV.assign_coords(my = (i+24)))
        
        x = PV_new[i]
        x.load()
        x = x.interp({"time" : ls0})#,
                            #kwargs={"fill_value":np.nan})
        reanalysis_clim[i] = x
    
    year_open = xr.concat(reanalysis_clim, dim = "my")
    year_open = year_open.mean(dim="my",skipna=True)
    year_open = year_open.chunk({'time':'auto'})
    year_open = year_open.rolling(time=100,center=True)
    year_open = year_open.mean().compute()

    

    ci = []
    cimax = []
    cu = []
    cumax = []

    clim0, = axs[0,2].plot(ls0,year_open.mag, label="Reanalysis",color='k',
                         linestyle='-',linewidth='1.5')
    #clim1, = axs[1,2].plot(ls0,year_open.mag, label="Reanalysis",color='k',
    #                     linestyle='-',linewidth='1.2')
    ls0 = np.where(ls0 > 170, ls0, None)
    clim00, = axs[1,2].plot(ls0,year_open.lat, label="Reanalysis",color='k',
                         linestyle='-',linewidth='1.5')
    #clim11, = latax[5].plot(ls0,year_open.lat, label="",color='k',
    #                     linestyle='dotted',linewidth='1.2')

    
    ci.append(clim0)
    cimax.append(clim00)

    
    plt.savefig(figpath+'all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)