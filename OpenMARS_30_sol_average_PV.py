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

def calc_jet_lat(u, lats, plot=False):
    """Function to calculate location and strenth of maximum given zonal wind
    u(lat) field """
    # Restict to 10 points around maximum
    u_max = np.where(u == np.ma.max(u))[0][0]
    u_near = u[u_max-1:u_max+2]
    lats_near = lats[u_max-1:u_max+2]
    # Quartic fit, with smaller lat spacing
    coefs = np.ma.polyfit(lats_near,u_near,2)
    fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
    quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
    # Find jet lat and max
    jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]
    jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2
    # Plot fit?
    if plot:
        print (jet_max)
        print (jet_lat)
        plt.plot(lats_near, u_near)
        plt.plot(fine_lats, quad)
        plt.show()

    return jet_lat, jet_max

if __name__ == "__main__":


    Lsmin = 270
    Lsmax = 300

    theta0 = 200.
    kappa = 1/4.0
    p0 = 610.

    ilev = 300

    sh = False

    if sh == True:
        latmin = -50
        hem = 'sh'
        proj = ccrs.SouthPolarStereo()
    else:
        hem = 'nh'
        latmin = 50
        proj = ccrs.NorthPolarStereo()


    PATH = '/export/anthropocene/array-01/xz19136/OpenMARS/Isentropic'
    infiles = '/isentropic*'

    figpath = 'OpenMARS_figs/PV_maps'

    d = xr.open_mfdataset(PATH+infiles, decode_times=False, concat_dim='time',
                           combine='nested',chunks={'time':'auto'})


    # reduce dataset
    d = d.astype('float32')
    d = d.sortby('time', ascending=True)
    d = d.where(d.Ls <= Lsmax, drop=True)
    d = d.where(Lsmin <= d.Ls, drop=True)
    d = d.sel(ilev=ilev, method='nearest')
    
    if sh == True:
        latm = d.lat.min().values
        x = d.sel(lat=d.lat[latmin>d.lat])
    else:
        latm = d.lat.max().values
        x = d.sel(lat=d.lat[latmin<d.lat])

    theta, center, radius, verts, circle = funcs.stereo_plot()


    fig, axs = plt.subplots(nrows=2,ncols=4, figsize = (14,8),
                            subplot_kw = {'projection':proj})



    vmin = 1
    vmax = 6.1
    step = 0.5

    boundaries0, _, _, cmap0, norm0 = funcs.make_colourmap(vmin, vmax, step,
                                                col = 'viridis', extend = 'both')



    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'


    # Lait scale PV
    theta = x.ilev
    laitPV = funcs.lait(x.PV,theta,theta0,kappa=kappa)
    x["scaled_PV"]=laitPV

    for i, ax in enumerate(fig.axes):
        print(i)
        if i < 3:
            my = i + 24
        else:
            my = i + 25

        if sh == True:
            funcs.make_stereo_plot(ax, [-55, -60, -70, -80, latm],
                              [-180, -120, -60, 0, 60, 120, 180],
                                      circle, alpha = 0.3, linestyle = '--',)
            #x["scaled_PV"] = - x.scaled_PV
        else:
            funcs.make_stereo_plot(ax, [latm, 80, 70, 60, 55],
                              [-180, -120, -60, 0, 60, 120, 180],
                              circle, alpha = 0.3, linestyle = '--',)

        x0 = x.where(d.MY==my,drop=True).mean(dim='time')
        a0 = x0.scaled_PV*10**4
        if sh == True:
            a0 = -a0
        ax.set_title('MY '+str(my),weight='bold',fontsize=20)
        ax.text(0.05, 0.95, string.ascii_lowercase[i], transform=ax.transAxes, 
                        size=20, weight='bold')
        if my == 28:
            if sh == True:
                continue
        
        q_max = []
        a0 = a0.load()
        for l in range(len(a0.lon)):
            q = a0.sel(lon = a0.lon[l],method="nearest")
            q0, _ = funcs.calc_jet_lat(q, a0.lat)
            if q0 > latm:
                q0 = latm
            q_max.append(q0)

        ax.contourf(a0.lon,a0.lat,a0,transform=ccrs.PlateCarree(),
                    cmap=cmap0,levels=[-100]+boundaries0+[500], norm=norm0)
        c0 = ax.plot(a0.lon, q_max,transform=ccrs.PlateCarree(),
                     color='blue', linewidth=1)
        c0 = ax.contour(x0.lon, x0.lat, x0.uwnd,colors='0.8',levels=[0,50,100],
                        transform=ccrs.PlateCarree(),linewidths = 1)

        c0.levels = [funcs.nf(val) for val in c0.levels]

        ax.clabel(c0, c0.levels, inline=1, fmt=fmt, fontsize=15)
        plt.savefig(figpath+'/OpenMARS_average_PV_yearly_'+str(ilev)+'K_Ls'+str(Lsmin)+'-'+str(Lsmax)+'_'+hem+'.pdf',
                bbox_inches='tight', pad_inches = 0.02)
        plt.savefig(figpath+'/OpenMARS_average_PV_yearly_'+str(ilev)+'K_Ls'+str(Lsmin)+'-'+str(Lsmax)+'_'+hem+'.png',
                bbox_inches='tight', pad_inches = 0.02)

    plt.subplots_adjust(hspace=.17,wspace=.02, bottom=0.1)



    cb = fig.colorbar(cm.ScalarMappable(norm=norm0,cmap=cmap0),ax=axs,
                      orientation='horizontal',extend='both',aspect=30,shrink=0.9,
                      ticks=boundaries0[slice(None,None,2)],pad=.03)

    cb.set_label(label='Lait-scaled PV (MPVU)',
                 fontsize=20)

    cb.ax.tick_params(labelsize=15)


    plt.savefig(figpath+'/OpenMARS_average_PV_yearly_'+str(ilev)+'K_Ls'+str(Lsmin)+'-'+str(Lsmax)+'_'+hem+'.pdf',
                bbox_inches='tight', pad_inches = 0.02)
    plt.savefig(figpath+'/OpenMARS_average_PV_yearly_'+str(ilev)+'K_Ls'+str(Lsmin)+'-'+str(Lsmax)+'_'+hem+'.png',
                bbox_inches='tight', pad_inches = 0.02)
