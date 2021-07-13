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

    EMARS = False
    SD = False
    ilev = 50

    latmax = 90
    latmin = 0

    Lsmax = 360
    Lsmin = 180

    figpath = 'Thesis/Hadley_strength/'

    if EMARS == True:
        PATH = '/export/anthropocene/array-01/xz19136/EMARS'
        files = '/*isobaric*'
        reanalysis = 'EMARS'
        yearmax = 32
        smooth = 200
    else:
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
    
    fig, axs = plt.subplots(2, 3, figsize = (25,15))

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



        if i == 0:
            ax.text(0.01, 0.95, reanalysis, size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_ylabel('$\psi$ strength (kg s$^{-1}$)', fontsize = 20)
            ax3.plot(np.linspace(265,310,20),np.ones(20)*7, color = 'k',
                 linewidth = '3.5',)
        elif i == 3:
            ax.text(0.01, 0.95, reanalysis, size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_ylabel('latitude ($^{\circ}$N)', fontsize = 20)
            ax3.set_yticklabels([])
            ax3.plot(np.linspace(265,310,20),np.ones(20)*7, color = 'k',
                 linewidth = '3.5',)
        elif i == 1 or i == 4:
            ax.text(0.01, 0.9, 'Yearly\nsimulations', size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_yticklabels([])
            ax3.set_yticklabels([])
            ax3.plot(np.linspace(265,310,20),np.ones(20)*7, color = 'k',
                 linewidth = '3.5',)
        elif i == 2:
            ax.text(0.01, 0.9, 'Process-attribution\nsimulations', size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_yticklabels([])
            #ax3.set_ylabel('latitude ($^\circ$N)', fontsize = 20)
        elif i == 5:
            ax.text(0.01, 0.9, 'Process-attribution\nsimulations', size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_yticklabels([])
            #ax3.set_ylabel('latitude ($^\circ$N)', fontsize = 20)
        if i > 2:
            ax.set_xlabel('solar longitude (degrees)', fontsize = 20)
            ax.set_xticklabels([200,220,240,260,280,300,320,340])
            ax.set_ylim([26, 81])
        else:
            ax.set_xticklabels([])
            ax.set_ylim([0, 51])
            ax.set_yticks([0,10,20,30,40,50])

        #latax.append(ax3)



    plt.subplots_adjust(hspace=.06, wspace = 0.05)




    plt.savefig(figpath+'all_'+str(ilev)+'Pa_'+reanalysis+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

    


    
    PATH = '/export/anthropocene/array-01/xz19136/OpenMARS/Streamfn'
    
    reanalysis_clim = []
    reanalysis_ls = []
    reanalysis_lat = []
    labels = []

    ci = []
    cimax = []
    cu = []
    cumax = []

    for i in list(np.arange(24,yearmax,1)):
        if i == 27:
            continue

        year = 'MY'+str(i)
        infiles = '/'+year+'_180-360_psi.nc'
        d = xr.open_mfdataset(PATH + infiles, decode_times = False,
                          concat_dim = 'time', combine = 'nested',
                          chunks = {'time' : 'auto'})

        ##### reduce dataset #####
        d = d.astype('float32')
        d = d.sortby('time', ascending =True) / 10 ** 8
        d = d.sortby('lat', ascending = False)

        d = d.where(d.time <= Lsmax, drop=True)
        #d = d.where(Lsmin - 10 <= d.time, drop=True)

        plev = d.pfull.sel(pfull = ilev, method = "nearest").values

        #d = d.sel(time = d.time[slice(None,None,5)], method = "nearest")
        latm = d.lat.max().values
        Ls = d.time.load()

        Ls.load()
        d.load()

        q0 = []
        qm = []


        for j in range(Ls.shape[0]):
            lsj = Ls[j]
            psi_j = d.where(d.time == lsj, drop = True).squeeze()
            psi_j = psi_j.to_array().squeeze()
            #psi_j = psi_j.where(psi_j.pfull < 250, drop = True)
            #psi_max = psi_j.max(skipna = True).values
                     
            
            # edge and strength of Hadley cell
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
            w = '2'

        else:
            color1 = cols[2]
            label = labs[2]
            w = '1'

        labels.append('MY '+str(i))

        c1, = ax.plot(Ls, qm, label=label, color=color1,
                     linestyle = linestyle1, linewidth = w)

        Ls = funcs.moving_average(Ls, 10)
        q0 = funcs.moving_average(q0, 10)
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

    plt.savefig(figpath+'all_' +str(ilev)+ 'Pa_'+reanalysis+sd+'.pdf',
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
    clim00, = axs[1,2].plot(ls0,year_open.lat, label="Reanalysis",color='k',
                         linestyle='-',linewidth='1.5')
    #clim11, = latax[5].plot(ls0,year_open.lat, label="",color='k',
    #                     linestyle='dotted',linewidth='1.2')

    
    ci.append(clim0)
    cimax.append(clim00)

    
    plt.savefig(figpath+'all_'+str(ilev)+'Pa_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)


    
    exp = ['stand',
           'lh',
           'scenario',
           'scenario_lh',
           'topo',
           'topo_lh',
           'topo_scenario',
           'topo_scenario_lh']

    labels = [
        'Reanalysis', 'Control', 'LH', 'D', 'LH+D', 'T','LH+T', 'D+T', 'LH+D+T',
        ]

    location = [
        'silurian',
        'silurian',
        'silurian',
        'silurian',
        'silurian',
        'silurian',
        'silurian',
        'silurian',
    ]

    #filepath = '/export/triassic/array-01/xz19136/Isca_data'
    #start_file = [33, 33, 33, 33, 33, 33, 33, 33]
    #end_file = [99, 99, 99, 99, 99, 99, 99, 139]

    #axes.prop_cycle: cycler('color', ['006BA4', 'FF800E', 'ABABAB', '595959', '5F9ED1', 'C85200', '898989', 'A2C8EC', 'FFBC79', 'CFCFCF'])
    #color = plt.cm.
    #matplotlib.rcParams['axes.prop_cycle'] = cycler('color', color)
    colors = ['#5F9ED1','#006BA4','#FF800E','#C85200']
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

    freq = 'daily'

    interp_file = 'atmos_'+freq+'_interp_new_height_temp_isentropic.nc'

    for i in range(len(exp)):
        print(exp[i])
        

        filepath = '/export/' + location[i] + '/array-01/xz19136/Isca_data/Streamfn/'
        #start = start_file[i]
        #end = end_file[i]

        #_, _, i_files = funcs.filestrings(exp[i], filepath, start, end, interp_file)

        d = xr.open_mfdataset(filepath + exp[i] + '_180-360_psi.nc',
                            decode_times=False, concat_dim='time',
                            combine='nested')

        # reduce dataset
        d = d.astype('float32') / 10**8
        d = d.sortby('time', ascending=True)
        d = d.where(d.time <= Lsmax, drop=True)
        d = d.where(Lsmin <= d.time, drop=True)
        d = d.sortby('lat', ascending=False)

        x = d.sel(pfull=ilev, method='nearest').squeeze()

        x = x.where(d.lat > latmin, drop = True)
        x = x.where(d.lat < latmax, drop = True)

        latm = x.lat.max().values

        x = x.transpose("lat","time")
        
        Ls = x.time.load()
        year_mean = d.load()

        q0 = []
        qm = []     

        for j in range(len(year_mean.time)):
            lsj = Ls[j]
            psi_j = d.where(d.time == lsj, drop = True).squeeze()
            psi_j = psi_j.to_array().squeeze()
            #psi_j = psi_j.where(psi_j.pfull < 250, drop = True)
            #psi_max = psi_j.max(skipna = True).values

            psi_j = psi_j.sel(pfull = plev, method = "nearest").squeeze()
            psi_j.load()
            _, psi_max = funcs.calc_jet_lat(psi_j, psi_j.lat)
            psi0_lat, _ = funcs.calc_Hadley_lat(psi_j, psi_j.lat)
            if psi0_lat > latm:
                psi0_lat = latm

            q0.append(psi0_lat)
            qm.append(psi_max)

        Ls = funcs.moving_average(Ls, 15)
        q0 = funcs.moving_average(q0, 15)
        qm = funcs.moving_average(qm, 15)

        ax = axs[0,2]
        ax2 = axs[1,2]
        color = colors[i]
        label = labels[i+1]
        if i < 4:
            linestyle1 = 'solid'
        else:
            linestyle1 = '--'

        c1, = ax.plot(Ls[:-8], qm[:-8], label = label, color = color,
                    linewidth = '1.2', linestyle = linestyle1)
        Ls = funcs.moving_average(Ls, 3)
        q0 = funcs.moving_average(q0, 3)
        c2, = ax2.plot(Ls[:-8], q0[:-8], label = label, color = color,
                    linewidth = '1.2', linestyle = linestyle1)


        ci.append(c1)
        cimax.append(c2)

        if SD == True:
            year_max = dsr.max(dim='MY')
            year_min = dsr.min(dim='MY')
            year_max = year_max.ens.chunk({'new_time':'auto'})
            year_max = year_max.rolling(new_time=25,center=True)
            year_max = year_max.mean()
            year_min = year_min.ens.chunk({'new_time':'auto'})
            year_min = year_min.rolling(new_time=25,center=True)
            year_min = year_min.mean()
    
            ax.fill_between(Ls, year_min, year_max, color=color, alpha=.1)
        

        plt.savefig(figpath+'all_'+str(ilev)+'Pa_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)

    c0 = [[ci[j]] for j in range(len(ci))]
    c1 = [[cimax[j]] for j in range(len(cimax))]


    axs[0,2].legend([tuple(c0[j]) for j in range(len(c0))], [i for i in labels],
                 fontsize = 14, loc = 'upper right', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    axs[1,2].legend([tuple(c1[j]) for j in range(len(c1))], [i for i in labels],
                 fontsize = 14, loc = 'lower right', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.savefig(figpath+'all_'+str(ilev)+'Pa_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)

    
    exp = ['topo_MY24_lh',
           'topo_MY25_lh',
           'topo_MY26_lh',
           'topo_MY27_lh',
           'topo_MY28_lh',
           'topo_MY29_lh',
           'topo_MY30_lh',
           'topo_MY31_lh',
           'topo_MY32_lh',]

    location = ['silurian','silurian', 'silurian', 'silurian',
                'silurian','silurian', 'silurian', 'silurian', 'silurian']
    
    labels = ["MY 24", "MY 25", "MY 26", "MY 27", "MY 28",
              "MY 29", "MY 30", "MY 31", "MY 32",]

    freq = 'daily'

    interp_file = 'atmos_'+freq+'_interp_new_height_temp_isentropic.nc'

    #filepath = '/export/triassic/array-01/xz19136/Isca_data'
    #start_file=[30, 35, 35, 35, 35, 35, 35, 35, 35]
    #end_file = [80, 99, 88, 99, 99, 96, 99, 99, 88]

    ci = []
    cimax = []
    cu = []
    cumax = []

    for i in range(len(exp)):
        print(exp[i])
        label = labels[i]

        filepath = '/export/' + location[i] + '/array-01/xz19136/Isca_data/Streamfn/'
        
        #_ ,_ , i_files = funcs.filestrings(exp[i], filepath, start,
        #                                end, interp_file)

        d = xr.open_mfdataset(filepath + exp[i] + '_180-360_psi.nc', decode_times=False, concat_dim='time',
                            combine='nested',chunks={'time':'auto'})

        # reduce dataset
        d = d.astype('float32') / 10**8
        d = d.sortby('time', ascending=True)
        d = d.where(d.time <= Lsmax, drop=True)
        d = d.where(Lsmin <= d.time, drop=True)
        d = d.sortby('lat', ascending=False)

        x = d.sel(pfull=ilev, method='nearest').squeeze()

        x = x.where(d.lat > latmin, drop = True)
        x = x.where(d.lat < latmax, drop = True)

        latm = x.lat.max().values

        x = x.transpose("lat","time")
        
        Ls = x.time.load()
        year_mean = d.load()

        q0 = []
        qm = []     

        for j in range(len(year_mean.time)):
            lsj = Ls[j]
            psi_j = d.where(d.time == lsj, drop = True).squeeze()
            psi_j = psi_j.to_array().squeeze()
            #psi_j = psi_j.where(psi_j.pfull < 250, drop = True)
            #psi_max = psi_j.max(skipna = True).values

            psi_j = psi_j.sel(pfull = plev, method = "nearest").squeeze()
            psi_j.load()
            _, psi_max = funcs.calc_jet_lat(psi_j, psi_j.lat)
            psi0_lat, _ = funcs.calc_Hadley_lat(psi_j, psi_j.lat)
            if psi0_lat > latm:
                psi0_lat = latm

            q0.append(psi0_lat)
            qm.append(psi_max)

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
        c1, = ax.plot(Ls[:-8], qm[:-8], label = label, color=color1,
                        linestyle=linestyle1, linewidth = w)
        Ls = funcs.moving_average(Ls, 3)
        q0 = funcs.moving_average(q0, 3)
        c2, = ax2.plot(Ls[:-8], q0[:-8], label = label, color=color1,
                        linestyle=linestyle1, linewidth = w)

        ci.append(c1)
        cimax.append(c2)

        plt.savefig(figpath+'all_'+str(ilev)+'Pa_'+reanalysis+sd+'.pdf',
                    bbox_inches='tight',pad_inches=0.1)
    
    c0 = [[ci[j]] for j in [0,4,5]]
    c1 = [[cimax[j]] for j in [0,4,5]]

    axs[0,1].legend([tuple(c0[j]) for j in [0,1,2]], [i for i in labs],
                 fontsize = 14, loc = 'upper right', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    axs[1,1].legend([tuple(c1[j]) for j in [0,1,2]], [i for i in labs],
                 fontsize = 14, loc = 'upper right', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.savefig(figpath+'all_'+str(ilev)+'Pa_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)