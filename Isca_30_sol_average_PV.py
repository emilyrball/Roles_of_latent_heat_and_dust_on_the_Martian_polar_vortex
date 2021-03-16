# 8 panels of 30-sol average PV from each year of OpenMARS data, on 350K surface.

import numpy as np
import xarray as xr
import os, sys

import analysis_functions as funcs
import colorcet as cc
import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import (cm, colors)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

if __name__== "__main__":

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    Lsmin = 255
    Lsmax = 285

    theta0 = 200.
    kappa = 1/4.0
    p0 = 610.

    ### choose your files
    exp = ['soc_mars_mk36_per_value70.85_none_mld_2.0',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_cdod_clim_scenario_7.4e-05',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_cdod_clim_scenario_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05_lh_rel']

    labels = ['Standard', 'Latent Heat', 'Dust', 'Latent Heat\nand Dust',
              #None, None, None, None]
              'Standard', 'Latent Heat', 'Dust', 'Latent Heat\nand Dust']

    location = ['triassic','silurian', 'silurian', 'silurian',
                'triassic','silurian', 'silurian', 'silurian']

    #filepath = '/export/triassic/array-01/xz19136/Isca_data'
    start_file = [100, 35, 35, 35, 165, 35, 35, 35]
    end_file = [175, 99, 99, 99, 225, 99, 99, 92]

    figpath = 'Isca_figs/'

    freq = 'daily'

    interp_file = 'atmos_'+freq+'_interp_new_height_temp_isentropic.nc'
    ilev = 350.


    theta, center, radius, verts, circle = funcs.stereo_plot()

    vmin = 10
    vmax = 101
    step = 10

    fig, axs = plt.subplots(nrows=2,ncols=4, figsize = (14,9),
                            subplot_kw = {'projection':ccrs.NorthPolarStereo()})

    for j, ax in enumerate(fig.axes):
        ax.text(0.05, 0.95, string.ascii_lowercase[j], transform=ax.transAxes, 
                size=20, weight='bold')
        
        ax.set_title(labels[j],weight='bold',fontsize=20)

    boundaries, _, _, cmap, norm = funcs.make_colourmap(vmin, vmax, step,
                                                col = 'viridis', extend = 'both')



    for i in range(len(start_file)):

        filepath = '/export/' + location[i] + '/array-01/xz19136/Isca_data'
        start = start_file[i]
        end = end_file[i]

        _, _, i_files = funcs.filestrings(exp[i], filepath, start, end, interp_file)

        d = xr.open_mfdataset(i_files, decode_times=False, concat_dim='time',
                            combine='nested')

        # reduce dataset
        d = d.astype('float32')
        d = d.sortby('time', ascending=True)

        d["mars_solar_long"] = d.mars_solar_long.sel(lon=0)
        d = d.where(d.mars_solar_long != 354.37808, other=359.762)
        print(d.mars_solar_long[-1].values)

        
        d = d.where(d.mars_solar_long <= Lsmax, drop=True)
        d = d.where(Lsmin <= d.mars_solar_long, drop=True)

        d = d.sel(lat=d.lat[50<d.lat])

        latm = d.lat.max().values

        x = d.sel(ilev=ilev, method='nearest').squeeze()

        # Lait scale PV
        theta = x.ilev
        print("Scaling PV")
        laitPV = funcs.lait(x.PV,theta,theta0,kappa=kappa)
        x["scaled_PV"]=laitPV

        # individual plots
        for j, ax in enumerate(fig.axes):
            if j==i:
                funcs.make_stereo_plot(ax, [latm, 80, 60, 50],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)
                a = x.scaled_PV.mean(dim='time').squeeze() * 10**5
                u = x.uwnd.mean(dim='time').squeeze()

                c0 = ax.contourf(a.lon,a.lat,a,cmap=cmap,transform=ccrs.PlateCarree(),
                                norm=norm,levels=[-50]+boundaries+[150])
                c = ax.contour(x.lon, x.lat, u, colors='0.8', levels=[0,40,80,120],
                                transform=ccrs.PlateCarree(),linewidths = 1)

                c.levels = [funcs.nf(val) for val in c.levels]
                ax.clabel(c, c.levels, inline=1, fmt=fmt, fontsize=15)
                print(i)
                plt.savefig(figpath+'/Isca_average_winter_PV_'+str(ilev)+'K_'+str(Lsmin)+'-'+str(Lsmax)+'_all_exp.png',
                    bbox_inches='tight', pad_inches = 0.02)
        #####

    
    plt.subplots_adjust(hspace=.2,wspace=.02)

    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=axs,
                      orientation='horizontal', extend='both', aspect=30,shrink=0.9,
                      pad=.03,ticks=boundaries[slice(None,None,1)])

    cb.set_label(label='Lait-scaled PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)',
                 fontsize=20)
    cb.ax.tick_params(labelsize=15)


    plt.savefig(figpath+'/Isca_average_winter_PV_'+str(ilev)+'K_'+str(Lsmin)+'-'+str(Lsmax)+'_all_exp.png',
                bbox_inches='tight', pad_inches = 0)


