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

    SD = False
    ilev = 350

    latmax = 90
    latmin = 60

    figpath = 'Thesis/polar_PV/'

    reanalysis = 'OpenMARS'
    PATH = '/export/anthropocene/array-01/xz19136/Data_Ball_etal_2021/'

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
            ax.set_ylim([-0.4, 11.9])
            ax.set_yticks([0,2,4,6,8,10])
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

    


    
    d = xr.open_mfdataset(PATH+'OpenMARS_Ls0-360_PV_350K.nc', decode_times=False)

    smooth = 250
    d = d.sortby('time', ascending=True)
    yearmax = 33

    d = d.where(d.lat > latmin, drop = True)
    d = d.where(d.lat < latmax, drop = True)
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

    for i in list(np.arange(24,yearmax,1)):
        di = d.where(d.MY == i, drop=True)
        print(i)
        di["Ls"] = di.Ls.sel(lat=di.lat[0]).sel(lon=di.lon[0])
        di = di.transpose("lat","lon","time")
        di = di.sortby("lat", ascending = True)
        di = di.sortby("lon", ascending = True)
        di = di.mean(dim = "lon")
        Zi = di.scaled_PV * 10**4

        q0 = []
        qm = []

        Zi.load()

        Ls = di.Ls.load()
        

        for l in range(len(Zi.time)):
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


    
    exp = ['control_',
           'lh_',
           'dust_',
           'dust_lh_',
           'topo_',
           'topo_lh_',
           'topo_dust_',
           'topo_dust_lh_']

    labels = [
        'Reanalysis', 'Control', 'LH', 'D', 'LH+D', 'T','LH+T', 'D+T', 'LH+D+T',
        ]

    colors = [
        '#56B4E9',
        '#0072B2',
        '#F0E442',
        '#E69F00',
        '#009E73',
        '#CC79A7',
        '#D55E00',
        '#000000',
        ]

    for i in range(len(exp)):
        print(exp[i])
        
        d = xr.open_mfdataset(PATH+exp[i]+'Ls0-360_PV_350K.nc',
                                decode_times=False)

        # reduce dataset
        d = d.astype('float32')
        d = d.sortby('new_time', ascending=True)
        d = d.sortby('lat', ascending=True)
        d = d.sortby('lon', ascending=True)
        d = d[["PV", "mars_solar_long"]]

        #d["mars_solar_long"] = d.mars_solar_long.sel(lon=0)
        d["mars_solar_long"] = d.mars_solar_long.where(d.mars_solar_long != 354.3762, other=359.762)
        d["mars_solar_long"] = d.mars_solar_long.where(d.mars_solar_long != 354.37808, other = 359.762)

        x = d.squeeze()

        x = x.where(d.lat > latmin, drop = True)
        x = x.where(d.lat < latmax, drop = True)

        latm = x.lat.max().values

        x = x.transpose("lat","lon","new_time","MY")

        theta = ilev
        laitPV = funcs.lait(x.PV, theta, theta0, kappa = kappa)
        x["scaled_PV"] = laitPV

        ens = x.scaled_PV * 10**4

        x["ens"] = ens
        x = x.mean(dim="lon")

        year_mean = x.mean(dim='MY')
        
        Ls1 = year_mean.mars_solar_long.mean(dim="lat").load()
        year_mean = year_mean.ens.load()

        q0 = []
        qm = []     

        for l in range(len(year_mean.new_time)):
            q = year_mean.sel(new_time=year_mean.new_time[l],method="nearest")
            qmax = q.mean(dim="lat").values
            qlat, _ = funcs.calc_jet_lat(q, q.lat)
            if qlat > latm:
                qlat = latm
            q0.append(qlat)
            qm.append(qmax)

        Ls = funcs.moving_average(Ls1, 15)
        q0 = funcs.moving_average(q0, 15)
        qm = funcs.moving_average(qm, 15)

        ax = axs[0,2]
        ax2 = axs[1,2]
        color = colors[i]
        label = labels[i+1]
        if i < 4:
            linestyle = 'solid'
        else:
            linestyle = '--'

        c1, = ax.plot(Ls, qm, label = label, color = color,
                    linewidth = '1.2', linestyle = linestyle)
        Ls = funcs.moving_average(Ls, 5)
        q0 = funcs.moving_average(q0, 5)
        Ls = np.where(Ls >170, Ls, None)
        c2, = ax2.plot(Ls, q0, label = label, color = color,
                    linewidth = '1.2', linestyle = linestyle)

        
        ci.append(c1)
        cimax.append(c2)

        if SD == True:
            x = x.mean(dim="lat")
            year_max = x.max(dim='MY')
            year_min = x.min(dim='MY')
            year_max = year_max.ens.chunk({'new_time':'auto'})
            year_max = year_max.rolling(new_time=15,center=True)
            year_max = year_max.mean()
            year_min = year_min.ens.chunk({'new_time':'auto'})
            year_min = year_min.rolling(new_time=15,center=True)
            year_min = year_min.mean()
    
            ax.fill_between(Ls1, year_min, year_max, color=color, alpha=.1)
        

        plt.savefig(figpath+'all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)

    c0 = [[ci[j]] for j in range(len(ci))]
    c1 = [[cimax[j]] for j in range(len(cimax))]

    axs[0,2].legend([tuple(c0[j]) for j in range(len(c0))], [i for i in labels],
                 fontsize = 14, loc = 'center left', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})


    axs[1,2].legend([tuple(c1[j]) for j in range(len(c1))], [i for i in labels],
                 fontsize = 14, loc = 'center left', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.savefig(figpath+'all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
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

    
    labels = ["MY 24", "MY 25", "MY 26", "MY 27", "MY 28",
              "MY 29", "MY 30", "MY 31", "MY 32",]

    ci = []
    cimax = []
    cu = []
    cumax = []

    for i in range(len(exp)):
        print(exp[i])
        label = labels[i]

        d = xr.open_mfdataset(PATH+exp[i]+'Ls0-360_PV_350K.nc', decode_times=False)

        # reduce dataset
        d = d.astype('float32')
        d = d.sortby('new_time', ascending=True)
        d = d.sortby('lat', ascending=True)
        d = d.sortby('lon', ascending=True)
        d = d[["PV", "mars_solar_long"]]

        #d["mars_solar_long"] = d.mars_solar_long.sel(lon=0)
        d["mars_solar_long"] = d.mars_solar_long.where(d.mars_solar_long != 354.3762, other=359.762)


        #x = d.sel(ilev=ilev, method='nearest').squeeze()

        x = d.where(d.lat > latmin, drop = True)
        x = x.where(d.lat < latmax, drop = True)

        latm = x.lat.max().values

        x = x.transpose("lat","lon","new_time","MY")

        theta = ilev
        laitPV = funcs.lait(x.PV, theta, theta0, kappa = kappa)
        x["scaled_PV"] = laitPV

        ens = x.scaled_PV * 10**4

        x["ens"] = ens
        
        x = x.mean(dim="lon")

        year_mean = x.mean(dim='MY')

        Ls = year_mean.mars_solar_long.mean(dim="lat").load()
        year_mean = year_mean.ens.load()

        q0 = []
        qm = []

        for l in range(len(year_mean.new_time)):
            q = year_mean.sel(new_time=year_mean.new_time[l],method="nearest")
            qmax = q.mean(dim="lat").values
            qlat, _ = funcs.calc_jet_lat(q, q.lat)
            if qlat > latm:
                qlat = latm
            q0.append(qlat)
            qm.append(qmax)
        

        Ls = funcs.moving_average(Ls, 15)
        q0 = funcs.moving_average(q0, 15)
        qm = funcs.moving_average(qm, 15)

        ax = axs[0,1]
        ax2 = axs[1,1]
        linestyle1 = 'solid'

        if i < 4:
            color1 = cols[0]
            label = labs[0]
            w = '1'

        elif i == 4:
            color1 = cols[1]
            label = labs[1]
            w = '2'
        else:
            color1 = cols[2]
            label = labs[2]
            w = '1'
        c1, = ax.plot(Ls, qm, label = label, color=color1,
                        linestyle=linestyle1, linewidth = w)
        
        Ls = funcs.moving_average(Ls, 5)
        q0 = funcs.moving_average(q0, 5)
        Ls = np.where(Ls >170, Ls, None)
        c2, = ax2.plot(Ls, q0, label = label, color=color1,
                        linestyle=linestyle1, linewidth = w)

        ci.append(c1)
        cimax.append(c2)

        plt.savefig(figpath+'all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
                    bbox_inches='tight',pad_inches=0.1)
    
    c0 = [[ci[j]] for j in [0,4,5]]
    c1 = [[cimax[j]] for j in [0,4,5]]

    axs[0,1].legend([tuple(c0[j]) for j in [0,1,2]], [i for i in labs],
                 fontsize = 14, loc = 'center left', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    axs[1,1].legend([tuple(c1[j]) for j in [0,1,2]], [i for i in labs],
                 fontsize = 14, loc = 'center left', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.savefig(figpath+'all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)