'''
Calculates and plots eddy enstrophy for OpenMARS, Isca - all years and exps.
'''

import numpy as np
import xarray as xr
import os, sys

import calculate_PV as cPV
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

from calculate_PV_Isca_anthro import filestrings

from eddy_enstrophy_Isca_all_years import (assign_MY, make_coord_MY)


if __name__ == "__main__":

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'


    EMARS = False
    SD = False
    ilev = 350

    latmax = 90
    latmin = 60

    if EMARS == True:
        PATH = 'link-to-anthro/EMARS'
        files = '/*isentropic*'
        reanalysis = 'EMARS'
    else:
        reanalysis = 'OpenMARS'
        PATH = 'link-to-anthro/OpenMARS/Isentropic'
        files = '/*isentropic*'

    if SD == True:
        sd = '_SD'
    else:
        sd = ''

    linestyles = ['solid', 'dotted','dashed', 'dashdot']

    newlabel = ['Northern\nsummer solstice', 'Northern\nwinter solstice']
    newpos = [90,270]
    
    fig, axs = plt.subplots(2, 3, figsize = (25,15))

    for i, ax in enumerate(fig.axes):
        ax.set_xlim([0,360])
        ax.tick_params(length = 6, labelsize = 18)

        ax.set_ylim([-0.01, 1.7])
        #ax.set_yticks([0,10,20,30,40,50])

        ax2 = ax.twiny()
        ax2.set_xticks(newpos)

        ax2.xaxis.set_ticks_position('top')
        ax2.xaxis.set_label_position('top') 
        ax2.tick_params(length = 6)
        ax2.set_xlim(ax.get_xlim())

        ax.text(-0.04, 1.01, string.ascii_lowercase[i], size = 20,
                    transform = ax.transAxes, weight = 'bold')



        if i == 0 or i == 3:
            ax.text(0.03, 0.93, reanalysis, size = 20,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_ylabel('eddy enstrophy ($10^{-6}$PVU$^2$)', fontsize = 20)
        elif i == 1 or i == 4:
            ax.text(0.03, 0.93, 'Model', size = 20,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_yticklabels([])
        elif i == 2:
            ax.text(0.03, 0.88, 'Model without\ntopography', size = 20,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_yticklabels([])
        elif i == 5:
            ax.text(0.03, 0.88, 'Model with\ntopography', size = 20,
                        transform = ax.transAxes, weight = 'bold')
            ax.set_yticklabels([])
    
        if i > 2:
            ax.set_xlabel('solar longitude (degrees)', fontsize = 20)
            ax2.set_xticklabels([])
        else:
            ax2.set_xticklabels(newlabel,fontsize=18)
            ax.set_xticklabels([])



    plt.subplots_adjust(hspace=.06, wspace = 0.05)




    plt.savefig('Thesis/eddy_enstrophy_all_' +str(ilev)+ 'K_'+reanalysis+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

    


    exp = ['soc_mars_mk36_per_value70.85_none_mld_2.0',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_cdod_clim_scenario_7.4e-05',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_cdod_clim_scenario_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05_lh_rel']

    labels = ['Standard', 'Latent Heat', 'Dust', 'Latent Heat and Dust',
              #None, None, None, None]
              'Standard', 'Latent Heat', 'Dust', 'Latent Heat and Dust']

    location = ['triassic','triassic', 'anthropocene', 'anthropocene',
                'triassic','triassic', 'anthropocene', 'silurian']

    #filepath = '/export/triassic/array-01/xz19136/Isca_data'
    start_file = [33, 33, 33, 33, 33, 33, 33, 33]
    end_file = [99, 99, 99, 99, 99, 99, 99, 139]

    #axes.prop_cycle: cycler('color', ['006BA4', 'FF800E', 'ABABAB', '595959', '5F9ED1', 'C85200', '898989', 'A2C8EC', 'FFBC79', 'CFCFCF'])
    #color = plt.cm.
    #matplotlib.rcParams['axes.prop_cycle'] = cycler('color', color)
    colors = ['#5F9ED1','#006BA4','#FF800E','#C85200']

    freq = 'daily'

    interp_file = 'atmos_'+freq+'_interp_new_height_temp_isentropic.nc'

    for i in range(len(start_file)):
        print(exp[i])
        label = labels[i]

        filepath = '/export/' + location[i] + '/array-01/xz19136/Isca_data'
        start = start_file[i]
        end = end_file[i]

        _, _, i_files = filestrings(exp[i], filepath, start, end, interp_file)

        d = xr.open_mfdataset(i_files, decode_times=False, concat_dim='time',
                            combine='nested')

        # reduce dataset
        d = d.astype('float32')
        d = d.sortby('time', ascending=True)
        d = d.sortby('lat', ascending=True)
        d = d.sortby('lon', ascending=True)
        d = d[["PV", "mars_solar_long"]]

        d["mars_solar_long"] = d.mars_solar_long.sel(lon=0)
        d = d.where(d.mars_solar_long != 354.3762, other=359.762)
        d = d.where(d.mars_solar_long != 354.37808, other = 359.7808)

        d, index = assign_MY(d)

        x = d.sel(ilev=ilev, method='nearest').squeeze()

        x = x.where(d.lat > latmin, drop = True)
        x = x.where(d.lat < latmax, drop = True)

        print('Averaged over '+str(np.max(x.MY.values))+' MY')

        x = x.transpose("lat","lon","time")
        ens = cPV.calc_eddy_enstr(x.PV) * 10**6 * 2


        x["ens"] = ens

        dsr, N, n = make_coord_MY(x, index)

        year_mean = dsr.mean(dim='MY')
        
        Ls = year_mean.mars_solar_long[0,:]
        year_mean = year_mean.ens.chunk({'new_time':'auto'})
        year_mean = year_mean.rolling(new_time=25,center=True)
        year_mean = year_mean.mean()


        if i<4:
            ax = axs[0,2]
            color = colors[i]
            print(color)
        else:
            ax = axs[1,2]
            color = colors[i-4]


        c = ax.plot(Ls, year_mean, label = label, color = color,
                    linewidth = '0.8')
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
        

        plt.savefig('Thesis/eddy_enstrophy_all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)

    

    plt.savefig('Thesis/eddy_enstrophy_all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)

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
    d = d.sel(ilev = ilev, method='nearest').squeeze()
    
    reanalysis_clim = []
    reanalysis_ls = []

    for i in list(np.arange(24,yearmax,1)):
        di = d.where(d.MY == i, drop=True)
        print(i)
        di["Ls"] = di.Ls.sel(lat=di.lat[0]).sel(lon=di.lon[0])
        if EMARS == True:
            di = di.sortby(di.Ls, ascending=True)
        di = di.transpose("lat","lon","time")
        di = di.sortby("lat", ascending = True)
        di = di.sortby("lon", ascending = True)
        Zi = cPV.calc_eddy_enstr(di.PV) * 10**6
        reanalysis_clim.append(Zi)
        
        Ls = di.Ls
        reanalysis_ls.append(Ls)
        
        Zi = Zi.chunk({'time':'auto'})
        Zi = Zi.rolling(time=smooth,center=True)

        Zi = Zi.mean()
        
        
        if i < 28:
            ax = axs[0,0]
            color = 'black'
            linestyle = linestyles[i-28]

        elif i == 28:
            ax = axs[1,0]
            color = 'red'
            linestyle = 'solid'

        else:
            ax = axs[1,0]
            color = 'black'
            linestyle = linestyles[i-29]


        ci = ax.plot(Ls, Zi, label='MY '+str(i), color=color,
                     linestyle = linestyle)
                     
        plt.savefig('Thesis/eddy_enstrophy_all_' +str(ilev)+ 'K_'+reanalysis+sd+'.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)
    
    axs[0,0].legend(fontsize = 15, loc = 'upper center')
    axs[1,0].legend(fontsize = 15, loc = 'upper center')

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
    year_open = year_open.mean()

    clim0 = axs[0,2].plot(ls0,year_open, label="Reanalysis",color='k',
                         linestyle='-',linewidth='1.2')
    clim1 = axs[1,2].plot(ls0,year_open, label="Reanalysis",color='k',
                         linestyle='-',linewidth='1.2')

    axs[0,2].legend(fontsize = 15, loc = 'upper right')
    axs[1,2].legend(fontsize = 15, loc = 'upper right')
    
    plt.savefig('Thesis/eddy_enstrophy_all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
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

    for i in range(len(exp)):
        print(exp[i])
        label = labels[i]

        filepath = '/export/' + location[i] + '/array-01/xz19136/Isca_data'
        start = start_file[i]
        end = end_file[i]

        _ ,_ , i_files = filestrings(exp[i], filepath, start,
                                        end, interp_file)

        d = xr.open_mfdataset(i_files, decode_times=False, concat_dim='time',
                            combine='nested',chunks={'time':'auto'})

        # reduce dataset
        d = d.astype('float32')
        d = d.sortby('time', ascending=True)
        d = d.sortby('lat', ascending=True)
        d = d.sortby('lon', ascending=True)
        d = d[["PV", "mars_solar_long"]]

        d["mars_solar_long"] = d.mars_solar_long.sel(lon=0)
        d = d.where(d.mars_solar_long != 354.3762, other=359.762)
        print(d.mars_solar_long[0].values)

        d, index = assign_MY(d)

        x = d.sel(ilev=ilev, method='nearest').squeeze()

        x = x.where(d.lat > latmin, drop = True)
        x = x.where(d.lat < latmax, drop = True)

        print('Averaged over '+str(np.max(x.MY.values))+' MY')

        x = x.transpose("lat","lon","time")
        ens = cPV.calc_eddy_enstr(x.PV) * 10**6 * 2

        x["ens"] = ens

        dsr, N, n = make_coord_MY(x, index)
        year_mean = dsr.mean(dim='MY')
        Ls = year_mean.mars_solar_long[0,:]
        year_mean = year_mean.ens.chunk({'new_time':'auto'})
        year_mean = year_mean.rolling(new_time=25,center=True)
        year_mean = year_mean.mean()

        if i < 4:
            ax = axs[0,1]
            color = 'black'
            linestyle = linestyles[i]

        elif i == 4:
            ax = axs[1,1]
            color = 'red'
            linestyle = 'solid'

        else:
            ax = axs[1,1]
            color = 'black'
            linestyle = linestyles[i-5]

        c = ax.plot(Ls, year_mean, label = label, color=color, linestyle=linestyle)

        plt.savefig('Thesis/eddy_enstrophy_all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
                    bbox_inches='tight',pad_inches=0.1)
    
    axs[0,1].legend(fontsize = 15, loc = 'upper center')
    axs[1,1].legend(fontsize = 15, loc = 'upper center')



    plt.savefig('Thesis/eddy_enstrophy_all_'+str(ilev)+'K_'+reanalysis+sd+'.pdf',
                bbox_inches='tight',pad_inches=0.1)


