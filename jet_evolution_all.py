'''
Calculates and plots jet strength and latitude evolution for OpenMARS, Isca - all years and exps.
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
    ilev = 350

    latmax = 90
    latmin = 0

    figpath = 'Thesis/jet_strength/'

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
            ax.set_ylabel('jet strength (ms$^{-1}$)', fontsize = 20)
            ax3.plot(np.linspace(265,310,20),np.ones(20)*7, color = 'k',
                 linewidth = '3.5',)
        elif i == 3:
            ax.text(0.01, 0.95, reanalysis, size = 18,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_ylabel('jet latitude ($^{\circ}$N)', fontsize = 20)
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
            ax.set_ylim([45, 79])
        else:
            ax.set_xticklabels([])
            ax.set_ylim([45, 170])
            ax.set_yticks([50,75,100,125,150])
            #ax2.set_xticklabels(newlabel,fontsize=18)

        #latax.append(ax3)



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
    d = d.where(d.Ls > 200, drop = True)
    d = d.sel(ilev = ilev, method='nearest').squeeze()

    latm = d.lat.max().values
    
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
        di = d.where(d.MY == i, drop=True)
        print(i)
        di = di.transpose("lat","lon","time")
        di["Ls"] = di.Ls.sel(lat=di.lat[0]).sel(lon=di.lon[0])
        if EMARS == True:
            di = di.sortby(di.Ls, ascending=True)
        di = di.sortby("lat", ascending = True)
        di = di.sortby("lon", ascending = True)
        di = di.mean(dim = "lon")
        Zi = di.uwnd

        q0 = []
        qm = []

        Zi.load()

        Ls = di.Ls.load()

        for l in range(len(Zi.time)):
            if EMARS == True:
                q = Zi[:,l]
            else:
                q = Zi.sel(time=Zi.time[l],method="nearest")
            
            qlat, qmax = funcs.calc_jet_lat(q, q.lat)
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

        Ls = funcs.moving_average(Ls, 2)
        q0 = funcs.moving_average(q0, 2)
        c2, = ax2.plot(Ls, q0, label=label, color=color1,
                        linestyle = linestyle1, linewidth = w)


        ci.append(c1)
        cimax.append(c2)
                     
        plt.savefig(figpath+'all_' +str(ilev)+ 'K_'+reanalysis+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)
    
    c0 = [[ci[j]] for j in range(len(ci))]
    c1 = [[cimax[j]] for j in range(len(cimax))]


    axs[0,0].legend([tuple(c0[j]) for j in [0,3,4]], [i for i in labs],
                 fontsize = 14, loc = 'upper right', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    axs[1,0].legend([tuple(c1[j]) for j in [0,3,4]], [i for i in labs],
                 fontsize = 14, loc = 'upper right', handlelength = 3,
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
    clim00, = axs[1,2].plot(ls0,year_open.lat, label="Reanalysis",color='k',
                         linestyle='-',linewidth='1.5')
    #clim11, = latax[5].plot(ls0,year_open.lat, label="",color='k',
    #                     linestyle='dotted',linewidth='1.2')

    
    ci.append(clim0)
    cimax.append(clim00)

    
    plt.savefig(figpath+'all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)


    
    exp = ['soc_mars_mk36_per_value70.85_none_mld_2.0',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_cdod_clim_scenario_7.4e-05',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_cdod_clim_scenario_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05_lh_rel']

    labels = [
        'Reanalysis', 'Control', 'LH', 'D', 'LH-D', 'T','LH-T', 'D-T', 'LH-D-T',
        ]

    location = ['triassic','triassic', 'anthropocene', 'anthropocene',
                'triassic','triassic', 'anthropocene', 'silurian']

    #filepath = '/export/triassic/array-01/xz19136/Isca_data'
    start_file = [33, 33, 33, 33, 33, 33, 33, 33]
    end_file = [99, 99, 99, 99, 99, 99, 99, 139]

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

    for i in range(len(start_file)):
        print(exp[i])
        

        filepath = '/export/' + location[i] + '/array-01/xz19136/Isca_data'
        start = start_file[i]
        end = end_file[i]

        _, _, i_files = funcs.filestrings(exp[i], filepath, start, end, interp_file)

        d = xr.open_mfdataset(i_files, decode_times=False, concat_dim='time',
                            combine='nested')

        # reduce dataset
        d = d.astype('float32')
        d = d.sortby('time', ascending=True)
        d = d.sortby('lat', ascending=True)
        d = d.sortby('lon', ascending=True)
        d = d[["uwnd", "mars_solar_long"]]

        d["mars_solar_long"] = d.mars_solar_long.sel(lon=0)
        d = d.where(d.mars_solar_long != 354.3762, other=359.762)
        d = d.where(d.mars_solar_long != 354.37808, other = 359.7808)

        d, index = funcs.assign_MY(d)

        x = d.sel(ilev=ilev, method='nearest').squeeze()

        x = x.where(d.lat > latmin, drop = True)
        x = x.where(d.lat < latmax, drop = True)

        latm = x.lat.max().values

        print('Averaged over '+str(np.max(x.MY.values))+' MY')

        x = x.transpose("lat","lon","time")
        x = x.mean(dim="lon")

        dsr, N, n = funcs.make_coord_MY(x, index)

        year_mean = dsr.mean(dim='MY')
        
        Ls = year_mean.mars_solar_long.mean(dim="lat").load()
        year_mean = year_mean.uwnd.load()

        q0 = []
        qm = []     

        for l in range(len(year_mean.new_time)):
            q = year_mean.sel(new_time=year_mean.new_time[l],method="nearest")
            qlat, qmax = funcs.calc_jet_lat(q, q.lat)
            if qlat > latm:
                qlat = latm
            q0.append(qlat)
            qm.append(qmax)

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
        Ls = funcs.moving_average(Ls, 2)
        q0 = funcs.moving_average(q0, 2)
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
        

        plt.savefig(figpath+'all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)

    c0 = [[ci[j]] for j in range(len(ci))]
    c1 = [[cimax[j]] for j in range(len(cimax))]


    axs[0,2].legend([tuple(c0[j]) for j in range(len(c0))], [i for i in labels],
                 fontsize = 14, loc = 'center left', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    axs[1,2].legend([tuple(c1[j]) for j in range(len(c1))], [i for i in labels],
                 fontsize = 14, loc = 'upper right', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.savefig(figpath+'all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)

    
    exp = ['soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY24_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY25_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY26_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY27_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY28_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY29_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY30_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY31_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY32_7.4e-05_lh_rel',]

    location = ['silurian','silurian', 'silurian', 'silurian',
                'silurian','silurian', 'silurian', 'silurian', 'silurian']
    
    labels = ["MY 24", "MY 25", "MY 26", "MY 27", "MY 28",
              "MY 29", "MY 30", "MY 31", "MY 32",]

    freq = 'daily'

    interp_file = 'atmos_'+freq+'_interp_new_height_temp_isentropic.nc'

    #filepath = '/export/triassic/array-01/xz19136/Isca_data'
    start_file=[30, 35, 35, 35, 35, 35, 35, 35, 35]
    end_file = [80, 99, 88, 99, 99, 96, 99, 99, 88]

    ci = []
    cimax = []
    cu = []
    cumax = []

    for i in range(len(exp)):
        print(exp[i])
        label = labels[i]

        filepath = '/export/' + location[i] + '/array-01/xz19136/Isca_data'
        start = start_file[i]
        end = end_file[i]

        _ ,_ , i_files = funcs.filestrings(exp[i], filepath, start,
                                        end, interp_file)

        d = xr.open_mfdataset(i_files, decode_times=False, concat_dim='time',
                            combine='nested',chunks={'time':'auto'})

        # reduce dataset
        d = d.astype('float32')
        d = d.sortby('time', ascending=True)
        d = d.sortby('lat', ascending=True)
        d = d.sortby('lon', ascending=True)
        d = d[["uwnd", "mars_solar_long"]]

        d["mars_solar_long"] = d.mars_solar_long.sel(lon=0)
        d = d.where(d.mars_solar_long != 354.3762, other=359.762)
        print(d.mars_solar_long[0].values)

        d, index = funcs.assign_MY(d)

        x = d.sel(ilev=ilev, method='nearest').squeeze()

        x = x.where(d.lat > latmin, drop = True)
        x = x.where(d.lat < latmax, drop = True)

        latm = x.lat.max().values

        print('Averaged over '+str(np.max(x.MY.values))+' MY')

        x = x.transpose("lat","lon","time")        
        x = x.mean(dim="lon")

        dsr, N, n = funcs.make_coord_MY(x, index)
        year_mean = dsr.mean(dim='MY')

        Ls = year_mean.mars_solar_long.mean(dim="lat").load()
        year_mean = year_mean.uwnd.load()

        q0 = []
        qm = []

        for l in range(len(year_mean.new_time)):
            q = year_mean.sel(new_time=year_mean.new_time[l],method="nearest")
            qlat, qmax = funcs.calc_jet_lat(q, q.lat)
            if qlat > latm:
                qlat = latm
            q0.append(qlat)
            qm.append(qmax)
        
        #Zi = Zi.chunk({'time':'auto'})
        #Zi = Zi.rolling(time=smooth,center=True)

        #Zi = Zi.mean()

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
        Ls = funcs.moving_average(Ls, 2)
        q0 = funcs.moving_average(q0, 2)
        c2, = ax2.plot(Ls[:-8], q0[:-8], label = label, color=color1,
                        linestyle=linestyle1, linewidth = w)

        ci.append(c1)
        cimax.append(c2)

        plt.savefig(figpath+'all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
                    bbox_inches='tight',pad_inches=0.1)
    
    c0 = [[ci[j]] for j in [0,4,5]]
    c1 = [[cimax[j]] for j in [0,4,5]]

    axs[0,1].legend([tuple(c0[j]) for j in [0,1,2]], [i for i in labs],
                 fontsize = 14, loc = 'upper right', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    axs[1,1].legend([tuple(c1[j]) for j in [0,1,2]], [i for i in labs],
                 fontsize = 14, loc = 'upper right', handlelength = 3,
            handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.savefig(figpath+'all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)