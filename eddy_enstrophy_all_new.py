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

    ilev = 350

    latmax = 90
    latmin = 60

    figpath = 'Thesis/eddy_enstrophy/'

    reanalysis = 'OpenMARS'
    PATH = '/export/anthropocene/array-01/xz19136/Data_Ball_etal_2021/'
    
    NORM = ''
    ymin = - 1
    ymax = 55
    tics = [0,10,20,30,40,50]

    linestyles = ['solid', 'dotted','dashed', 'dashdot']
    cols = ['#5F9ED1', '#C85200','#898989']
    labs = ["MY 24-27", "MY 28", "MY 29-32"]
    newlabel = ['Northern\nsummer solstice', 'Northern\nwinter solstice']
    newpos = [90,270]
    
    fig, axs = plt.subplots(2, 2, figsize = (16.6,15))

    for i, ax in enumerate(fig.axes):
        ax.set_xlim([0,360])
        ax.tick_params(length = 6, labelsize = 18)

        ax.set_ylim([ymin,ymax])
        ax.set_yticks(tics)

        ax2 = ax.twiny()
        ax2.set_xticks(newpos)
        ax2.set_xticklabels(newlabel,fontsize=18)

        ax2.xaxis.set_ticks_position('top')
        ax2.xaxis.set_label_position('top') 
        ax2.tick_params(length = 6)
        ax2.set_xlim(ax.get_xlim())

        ax.text(-0.04, 1.01, string.ascii_lowercase[i], size = 20,
                    transform = ax.transAxes, weight = 'bold')

        
        ax3 = ax.twinx()
        ax3.tick_params(length = 6, labelsize = 18)
        
        ax3.yaxis.set_label_position('right')
        ax3.yaxis.set_ticks_position('right')

        
        ax3.set_ylim([-0.05,1])
        
        
        ax3.set_yticks([])
        ax3.set_yticklabels([])

        if i == 0:# or i == 3:
            ax.text(0.01, 0.95, reanalysis, size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_ylabel('eddy enstrophy (MPVU$^2$)', fontsize = 20)
            ax3.plot(np.linspace(265,310,20),np.zeros(20), color = 'k',
                 linewidth = '3.5',)
            
            ax.set_xticklabels([])
        elif i == 1:
            ax.text(0.01, 0.9, 'Process-attribution\nsimulations', size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        elif i == 2:# or i == 4:
            ax.text(0.01, 0.9, 'Yearly\nsimulations', size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_ylabel('eddy enstrophy (MPVU$^2$)', fontsize = 20)
            ax3.plot(np.linspace(265,310,20),np.zeros(20), color = 'k',
                 linewidth = '3.5',)
            ax.set_xlabel('solar longitude (degrees)', fontsize = 20)
            ax2.set_xticklabels([])
        else:
            ax.text(0.01, 0.9, 'Process-attribution\nsimulations', size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_yticklabels([])
            ax.set_xlabel('solar longitude (degrees)', fontsize = 20)
            ax2.set_xticklabels([])

    plt.subplots_adjust(hspace=.06, wspace = 0.05)


    plt.savefig(figpath+'eddy_enstrophy_new_'+str(ilev)+'K_'+reanalysis+NORM+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

    


    
    d = xr.open_dataset(PATH+reanalysis+'_Ls0-360_PV_350K.nc', decode_times=False)


    smooth = 250
    d = d.sortby('time', ascending=True)
    yearmax = 33

    d = d.where(d.lat > latmin, drop = True)
    d = d.where(d.lat < latmax, drop = True)
    #d = d.sel(ilev = ilev, method='nearest').squeeze()
    
    reanalysis_clim = []
    reanalysis_ls = []

    c = []

    for i in list(np.arange(24,yearmax,1)):
        di = d.where(d.MY == i, drop=True)
        print(i)
        di["Ls"] = di.Ls.sel(lat=di.lat[0]).sel(lon=di.lon[0])
        di = di.transpose("lat","lon","time")
        di = di.sortby("lat", ascending = True)
        di = di.sortby("lon", ascending = True)

        # Lait scale PV
        theta = ilev
        laitPV = funcs.lait(di.PV, theta, theta0, kappa = kappa)
        di["scaled_PV"] = laitPV

        Zi = funcs.calc_eddy_enstr(di.scaled_PV) * 10**8
        
        if i != 28:
            reanalysis_clim.append(Zi)
        
        Ls = di.Ls
        if i != 28:
            reanalysis_ls.append(Ls)
        
        Zi = Zi.chunk({'time':'auto'})
        Zi = Zi.rolling(time=smooth,center=True)

        Zi = Zi.mean()

        Zi = Zi.load()
        
        ax = axs[0,0]
        if i < 28:
            color = cols[0]
            label = labs[0]
            w = '1.2'
        elif i == 28:
            color = cols[1]
            label = labs[1]
            w = '2'
        else:
            color = cols[2]
            label = labs[2]
            w = '1.2'


        ci, = ax.plot(Ls, Zi, label = label, color = color,
                     linestyle = '-', linewidth = w)

        c.append(ci)
                     
        plt.savefig(figpath+'eddy_enstrophy_new_' +str(ilev)+ 'K_'+reanalysis+NORM+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

    c0 = [[c[j]] for j in [0,4,5]]
    
    axs[0,0].legend([tuple(c0[j]) for j in range(len(labs))], [i for i in labs],
                 fontsize = 14, loc = 'upper center', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    ls0 = np.arange(0,360,0.05)
    for i in range(len(reanalysis_clim)):
        reanalysis_clim[i] = reanalysis_clim[i].assign_coords(time = (reanalysis_ls[i].values))
        reanalysis_clim[i] = reanalysis_clim[i].assign_coords(my = (i+24))
        
        x = reanalysis_clim[i]
        x.load()
        x = x.interp({"time" : ls0})#,
                            #kwargs={"fill_value":np.nan})
        reanalysis_clim[i] = x
    
    year_open = xr.concat(reanalysis_clim, dim = "my")
    year_open = year_open.mean(dim="my",skipna=True)
    year_open = year_open.chunk({'time':'auto'})
    year_open = year_open.rolling(time=100,center=True)
    year_open = year_open.mean().compute()

    c = []
    f = []

    clim0, = axs[0,1].plot(ls0,year_open, label="Reanalysis",color='k',
                         linestyle='-',linewidth='2.0')
    clim1, = axs[1,1].plot(ls0,year_open, label="Reanalysis",color='k',
                         linestyle='-',linewidth='2.0')

    c.append(clim0)
    f.append(clim1)
    
    plt.savefig(figpath+'eddy_enstrophy_new_'+str(ilev)+'K_'+reanalysis+NORM+'.pdf',
                bbox_inches='tight',pad_inches=0.1)


    
    exp = [
        'control_',
        'lh_',
        'dust_',
        'dust_lh_',
        'topo_',
        'topo_lh_',
        'topo_dust_',
        'topo_dust_lh_',
    ]
    
    labels = [
        'Reanalysis','Control', 'LH', 'D', 'LH+D',
        'Reanalysis','T', 'LH+T', 'D+T', 'LH+D+T',
        ]

    colors = [
        '#56B4E9',
        '#0072B2',
        #'#F0E442',
        '#E69F00',
        #'#009E73',
        #'#CC79A7',
        '#D55E00',
        #'#000000',
        ]

    filepath = '/export/anthropocene/array-01/xz19136/Data_Ball_etal_2021/'

    for i in range(len(exp)):
        print(exp[i])
        
        d = xr.open_mfdataset(filepath+exp[i]+'Ls0-360_PV_350K.nc',
                            decode_times=False)

        # reduce dataset
        d = d.astype('float32')
        d = d.sortby('new_time', ascending=True)
        d = d.sortby('lat', ascending=True)
        d = d.sortby('lon', ascending=True)
        d = d[["PV", "mars_solar_long", "MY"]]

        d["mars_solar_long"] = d.mars_solar_long.where(d.mars_solar_long != 354.3762, other=359.762)
        d["mars_solar_long"] = d.mars_solar_long.where(d.mars_solar_long != 354.37808, other = 359.78082)

        x = d.where(d.lat > latmin, drop = True)
        x = x.where(d.lat < latmax, drop = True)

        x = x.transpose("lat","lon","new_time", "MY")

        theta = ilev
        laitPV = funcs.lait(x.PV, theta, theta0, kappa = kappa)
        x["scaled_PV"] = laitPV
        ens = funcs.calc_eddy_enstr(x.scaled_PV) * 10**8

        x["ens"] = ens

        x = x.mean(dim = "MY", skipna = True)
        
        Ls = x.mars_solar_long[0,:]
        year_mean = x.ens.chunk({'new_time':'auto'})
        year_mean = year_mean.rolling(new_time=25,center=True)
        year_mean = year_mean.mean()

        linestyle = '-'

        if i<4:
            ax = axs[0,1]
            label = labels[i+1]
            color = colors[i]
            c1, = ax.plot(Ls, year_mean, label = label, color = color,
                    linewidth = '1.5', linestyle = linestyle)
            c.append(c1)
        else:
            ax = axs[1,1]
            label = labels[i+2]
            color = colors[i-4]
            c1, = ax.plot(Ls, year_mean, label = label, color = color,
                    linewidth = '1.5', linestyle = linestyle)
            f.append(c1)      

        plt.savefig(figpath+'eddy_enstrophy_new_'+str(ilev)+'K_'+reanalysis+NORM+'.pdf',
                bbox_inches='tight',pad_inches=0.1)

    c0 = [[c[j]] for j in range(len(c))]
    c1 = [[f[j]] for j in range(len(f))]

    axs[0,1].legend([tuple(c0[j]) for j in range(len(c0))], [i for i in labels[:5]],
                 fontsize = 14, loc = 'center left', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})


    axs[1,1].legend([tuple(c1[j]) for j in range(len(c1))], [i for i in labels[5:]],
                 fontsize = 14, loc = 'center left', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.savefig(figpath+'eddy_enstrophy_new_'+str(ilev)+'K_'+reanalysis+NORM+'.pdf',
                bbox_inches='tight',pad_inches=0.1)

    
    exp = [
        'MY24_',
        'MY25_',
        'MY26_',
        'MY27_',
        'MY28_',
        'MY29_',
        'MY30_',
        'MY31_',
        'MY32_',
    ]

    c = []

    for i in [0,1,2,3,4,5,6,7,8]:
        
        print(exp[i])
        filepath = '/export/anthropocene/array-01/xz19136/Data_Ball_etal_2021/'
        
        i_files = filepath+exp[i]+'Ls0-360_PV_350K.nc'

        d = xr.open_dataset(i_files, decode_times=False)

        # reduce dataset
        d = d.astype('float32')
        d = d.sortby('new_time', ascending=True)
        d = d.sortby('lat', ascending=True)
        d = d.sortby('lon', ascending=True)
        d = d[["PV", "mars_solar_long"]]

        d["mars_solar_long"] = d.mars_solar_long.where(d.mars_solar_long != 354.3762, other=359.762)
        d["mars_solar_long"] = d.mars_solar_long.where(d.mars_solar_long != 354.37808, other = 359.78082)
        
        x = d.where(d.lat > latmin, drop = True)
        x = x.where(d.lat < latmax, drop = True)
               


        x = x.transpose("lat","lon","new_time", "MY")
        theta = ilev
        laitPV = funcs.lait(x.PV, theta, theta0, kappa = kappa)
        x["scaled_PV"] = laitPV
        ens = funcs.calc_eddy_enstr(x.scaled_PV) * 10**8

        x["ens"] = ens
        x = x.mean(dim = "MY", skipna = True)
        Ls = x.mars_solar_long[0,:]
        year_mean = x.ens.chunk({'new_time':'auto'})
        year_mean = year_mean.rolling(new_time=25,center=True)
        year_mean = year_mean.mean()

        ax = axs[1,0]
        linestyle = '-'
        
        if i < 4:
            color = cols[0]
            w = '1.2'
            label = labs[0]
        elif i == 4:
            color = cols[1]
            label = labs[1]
            w = '2'
        else:
            color = cols[2]
            label = labs[2]
            w = '1.2'

        c1, = ax.plot(Ls, year_mean, label = label, color=color,
                    linestyle=linestyle, linewidth = w)

        c.append(c1)

        plt.savefig(figpath+'eddy_enstrophy_new_'+str(ilev)+'K_'+reanalysis+NORM+'.pdf',
                    bbox_inches='tight',pad_inches=0.1)
    
    c1 = [[c[j]] for j in [0,4,5]]

    axs[1,0].legend([tuple(c1[j]) for j in [0,1,2]], [x for x in labs],
                 fontsize = 14, loc = 'center left', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.savefig(figpath+'eddy_enstrophy_new_'+str(ilev)+'K_'+reanalysis+NORM+'.pdf',
                bbox_inches='tight',pad_inches=0.1)
    plt.savefig(figpath+'eddy_enstrophy_new_'+str(ilev)+'K_'+reanalysis+NORM+'.png',
                bbox_inches='tight',pad_inches=0.1)