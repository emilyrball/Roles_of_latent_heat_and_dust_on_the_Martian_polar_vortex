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

    Lsmin = 270
    Lsmax = 300

    theta0 = 200.
    kappa = 1/4.0
    p0 = 610.

    ### choose your files
    exp = [
        'control_',
        'topo_',
        'lh_',
        'topo_lh_',
        'dust_',
        'topo_dust_',
        'dust_lh_',
        'topo_dust_lh_',        
    ]

    labels = [
        'Control',
        'T', 
        'LH',
        'LH+T',
        'D',
        'D+T',
        'LH+D',
        'LH+D+T',
    ]

    filepath = '/export/anthropocene/array-01/xz19136/Data_Ball_etal_2021/'
    

    figpath = 'Isca_figs/PV_maps/'

    ilev = 300.

    theta, center, radius, verts, circle = funcs.stereo_plot()

    vmin = [
        2,
        2,
        2,
        2,
        ]
    vmax = [
        9.1,
        5.1,
        8.1,
        6.6,
        ]
    step = [
        0.5,
        0.5,
        0.5,
        0.5,
        ]
    
    y = [
        0.35,
        0.49,
        0.48,
        0.4,
        0.49,
        0.47,
        0.4,
        0.3,
    ]

    fig, axs = plt.subplots(nrows=4,ncols=2, figsize = (9,14),
                            subplot_kw = {'projection':ccrs.NorthPolarStereo()})

    for j, ax in enumerate(fig.axes):
        ax.text(0.05, 0.95, string.ascii_lowercase[j], transform=ax.transAxes, 
            size=20, weight='bold')
        ax.text(-0.06, y[j], labels[j], transform = ax.transAxes, 
                        size = 20, weight = 'bold',
                        rotation = 'vertical', rotation_mode = 'anchor')
        
    axs[0,0].set_title('No topography',weight='bold',fontsize=20,y=1.04)
    axs[0,1].set_title('MOLA topography',weight='bold',fontsize=20,y=1.04)

    

    for i in range(len(exp)):
        if i % 2 == 0:
            boundaries, _, _, cmap, norm = funcs.make_colourmap(
                                                    vmin[int(i/2)], vmax[int(i/2)], step[int(i/2)],
                                                    col = 'viridis', extend = 'both')

        i_files = filepath+exp[i]+'Ls270-300_300K.nc'

        d = xr.open_dataset(i_files, decode_times=False)

        # reduce dataset
        d = d.astype('float32')
        d = d.sortby('new_time', ascending=True)
        
        d = d.where(d.mars_solar_long <= Lsmax, drop=True)
        d = d.where(Lsmin <= d.mars_solar_long, drop=True)

        d = d.sel(lat=d.lat[55<d.lat])

        latm = d.lat.max().values

        x = d.sel(ilev=ilev, method='nearest').squeeze()

        # Lait scale PV
        theta = ilev
        print("Scaling PV")
        laitPV = funcs.lait(x.PV,theta,theta0,kappa=kappa)
        x["scaled_PV"]=laitPV
        x = x.mean(dim="MY")

        # individual plots
        for j, ax in enumerate(fig.axes):
            if j==i:
                funcs.make_stereo_plot(ax, [latm, 80, 70, 60, 55],
                          [-180, -120, -60, 0, 60, 120, 180],
                          circle, alpha = 0.3, linestyle = '--',)
                a = x.scaled_PV.mean(dim='new_time').squeeze() * 10**4
                u = x.uwnd.mean(dim='new_time').squeeze()
                q_max = []
                a = a.load()
                for l in range(len(a.lon)):
                    q = a.sel(lon = a.lon[l],method="nearest")
                    q0, _ = funcs.calc_jet_lat(q, a.lat)
                    q_max.append(q0)
                c0 = ax.contourf(a.lon,a.lat,a,cmap=cmap,transform=ccrs.PlateCarree(),
                                norm=norm,levels=[-50]+boundaries+[150])
                c0 = ax.plot(a.lon, q_max,transform=ccrs.PlateCarree(),
                     color='blue', linewidth=1)
                c = ax.contour(x.lon, x.lat, u, colors='1', levels=[0,50,100],
                                transform=ccrs.PlateCarree(),linewidths = 1)

                c.levels = [funcs.nf(val) for val in c.levels]
                ax.clabel(c, c.levels, inline=1, fmt=fmt, fontsize=15)
                print(i)
                plt.savefig(figpath+'/Isca_average_winter_PV_'+str(ilev)+'K_'+str(Lsmin)+'-'+str(Lsmax)+'_all_exp.pdf',
                    bbox_inches='tight', pad_inches = 0.02)
                plt.savefig(figpath+'/Isca_average_winter_PV_'+str(ilev)+'K_'+str(Lsmin)+'-'+str(Lsmax)+'_all_exp.png',
                    bbox_inches='tight', pad_inches = 0.02)
        #####

    
    plt.subplots_adjust(hspace=.15,wspace=.04)
    slicei = [2,2,2,2]
    for i in [0,2,4,6]:
        boundaries, _, _, cmap, norm = funcs.make_colourmap(
                                                    vmin[int(i/2)], vmax[int(i/2)], step[int(i/2)],
                                                    col = 'viridis', extend = 'both')
        cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=axs[int(i/2),:],
                      extend='both', aspect = 13,
                      pad=.03,ticks=boundaries[slice(None,None,slicei[int(i/2)])])
        cb.set_label(label='Lait-scaled PV (MPVU)',
                 fontsize=13)
        cb.ax.tick_params(labelsize=11)


    plt.savefig(figpath+'/Isca_average_winter_PV_'+str(ilev)+'K_'+str(Lsmin)+'-'+str(Lsmax)+'_all_exp.pdf',
                bbox_inches='tight', pad_inches = 0.02)
    plt.savefig(figpath+'/Isca_average_winter_PV_'+str(ilev)+'K_'+str(Lsmin)+'-'+str(Lsmax)+'_all_exp.png',
                bbox_inches='tight', pad_inches = 0.02)

