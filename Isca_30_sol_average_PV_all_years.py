import numpy as np
import xarray as xr
import os, sys
import pandas as pd

import analysis_functions as funcs
import colorcet as cc
import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import (cm, colors)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def make_coord_MY(x, index):
    x = x.where(x.time > index[0], drop=True)
    #x = x.where(x.time <= index[-1], drop=True)

    N=int(np.max(x.MY))
    n = range(N)

    y = x.time[:len(x.time)//N]

    ind = pd.MultiIndex.from_product((n,y),names=('MY','new_time'))
    dsr = x.assign_coords({'time':ind}).unstack('time')
    dsr = dsr.squeeze()

    return dsr, N, n

if __name__ == "__main__":
    Lsmin = 270
    Lsmax = 300

    theta0 = 200.
    kappa = 1/4.0
    p0 = 610.

    ### choose your files
    exp = ['soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY24_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY25_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY26_7.4e-05_lh_rel',
#           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY27_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY28_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY29_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY30_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY31_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY32_7.4e-05_lh_rel',]


    location = ['silurian']

    #filepath = '/export/triassic/array-01/xz19136/Isca_data'
    start_file=[30, 30, 30, 30, 30, 30, 30, 30]
    end_file = [80, 99, 88, 99, 96, 99, 99, 88]


    figpath = 'Isca_figs/PV_maps'

    freq = 'daily'

    interp_file = 'atmos_'+freq+'_interp_new_height_temp_isentropic.nc'
    ilev = 300.


    theta, center, radius, verts, circle = funcs.stereo_plot()

    vmin = 1
    vmax = 6.6
    step = 0.5

    fig, axs = plt.subplots(nrows=2,ncols=4, figsize = (14,8),
                            subplot_kw = {'projection':ccrs.NorthPolarStereo()})

    boundaries, _, _, cmap, norm = funcs.make_colourmap(vmin, vmax, step,
                                                col = 'viridis', extend = 'both')

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'


    for i in range(len(exp)):

        filepath = '/export/' + location[0] + '/array-01/xz19136/Isca_data'
        start = start_file[i]
        end = end_file[i]

        _, _, i_files = funcs.filestrings(exp[i], filepath, start, end, interp_file)

        d = xr.open_mfdataset(i_files, decode_times=False, concat_dim='time',
                            combine='nested')

        # reduce dataset
        d = d.astype('float32')
        d = d.sortby('time', ascending=True)

        latm = d.lat.max().values

        d["mars_solar_long"] = d.mars_solar_long.sel(lon=0)
        d = d.where(d.mars_solar_long != 354.37808, other=359.762)
        print(d.mars_solar_long[-1].values)


        d = d.where(d.mars_solar_long <= Lsmax, drop=True)
        d = d.where(Lsmin <= d.mars_solar_long, drop=True)

        d = d.sel(lat=d.lat[55<d.lat])
        d = d.sel(ilev=ilev, method='nearest').squeeze()

        #d, index = assign_MY(d)
        x = d.squeeze()



        #x = x.where(x.time <= index[-1], drop=True)

        # Lait scale PV
        theta = x.ilev
        print("Scaling PV")
        laitPV = funcs.lait(x.PV,theta,theta0,kappa=kappa)
        x["scaled_PV"]=laitPV


        #dsr, N, n = make_coord_MY(x, index)

        plt.subplots_adjust(hspace=.17,wspace=.02, bottom=0.1)



        # individual plots
        for j, ax in enumerate(fig.axes):
            if j == i:
                if j < 3:
                    my = j + 24
                else:
                    my = j + 25

                funcs.make_stereo_plot(ax, [latm, 80, 70, 60,55],
                                      [-180, -120, -60, 0, 60, 120, 180],
                                      circle, alpha = 0.3, linestyle = '--',)

                #a = dsr.where(dsr.MY==my-24,drop=True)
                a = x

                u = a.uwnd.mean(dim='time').squeeze()
                a = a.scaled_PV.mean(dim='time').squeeze() * 10**4

                q_max = []
                a = a.load()
                for l in range(len(a.lon)):
                    q = a.sel(lon = a.lon[l],method="nearest")
                    q0, _ = funcs.calc_jet_lat(q, a.lat)
                    q_max.append(q0)

                cf = ax.contourf(a.lon,a.lat,a,cmap=cmap,transform=ccrs.PlateCarree(),
                                norm=norm,levels=[-50]+boundaries+[150])
                c0 = ax.plot(a.lon, q_max,transform=ccrs.PlateCarree(),
                     color='blue', linewidth=1)
    
                #c0.levels = [funcs.nf(val) for val in c0.levels]
                #axs[0,0].clabel(c0, c0.levels, inline=1, fmt=fmt, fontsize=14)

                c = ax.contour(x.lon, x.lat, u, colors='0.8', levels=[0,50,100],
                                transform=ccrs.PlateCarree(),linewidths = 1)
                ax.text(0.05, 0.95, string.ascii_lowercase[j], transform=ax.transAxes, 
                        size=20, weight='bold')

                c.levels = [funcs.nf(val) for val in c.levels]
                ax.set_title('MY '+str(my),weight='bold',fontsize=20)
                ax.clabel(c, c.levels, inline=1, fmt=fmt, fontsize=15)
                print(i)
                plt.savefig(figpath+'/Isca_average_winter_PV_'+str(ilev)+'K_'+str(Lsmin)+'-'+str(Lsmax)+'_all_years_7.4e-05.pdf',
                    bbox_inches='tight', pad_inches = 0.02)
                plt.savefig(figpath+'/Isca_average_winter_PV_'+str(ilev)+'K_'+str(Lsmin)+'-'+str(Lsmax)+'_all_years_7.4e-05.png',
                    bbox_inches='tight', pad_inches = 0.02)
        #####

    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=axs,
                      orientation='horizontal', extend='both', aspect=30,shrink=0.9,
                      pad=.03,ticks=boundaries[slice(None,None,1)])

    cb.set_label(label='Lait-scaled PV (MPVU)',
                 fontsize=20)
    cb.ax.tick_params(labelsize=15)


    plt.savefig(figpath+'/Isca_average_winter_PV_'+str(ilev)+'K_'+str(Lsmin)+'-'+str(Lsmax)+'_all_years_7.4e-05.pdf',
                bbox_inches='tight', pad_inches = 0.02)
    plt.savefig(figpath+'/Isca_average_winter_PV_'+str(ilev)+'K_'+str(Lsmin)+'-'+str(Lsmax)+'_all_years_7.4e-05.png',
                bbox_inches='tight', pad_inches = 0.02)


