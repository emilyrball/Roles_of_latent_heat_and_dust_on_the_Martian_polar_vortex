### Plot vertical/latitudinal dust distributions, as well as size distribution ###

import numpy as np
import xarray as xr
import os, sys

import analysis_functions as funcs
import colorcet as cc

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import (cm, colors,cycler)
import matplotlib.path as mpath
import matplotlib
import matplotlib.gridspec as gridspec

from scipy.interpolate import interp2d

import pandas as pd
import string
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def newfmt(x):
    _, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return b

def duplicate_axis(ax, newpos):
    ax2 = ax.twiny()
    ax2.set_xticks(newpos)
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top') 
    ax2.tick_params(length = 6)
    ax2.set_xlim(ax.get_xlim())
    
    return ax2

def create_dust(pfull, phi, ls, tau):
    '''
    Creates surface dust mmr and vertical dust mmr from column dust optical
    depth using the Conrath-nu profile.

    Parameters
    ----------

    pfull     : array-like, pressure levels for vertical profile, in Pa.
    phi       : array-like, latitudes, in degrees.
    ls        : array-like, solar longitudes, in degrees.
    tau       : array-like, column dust optical depth product, for now must be from MCD.

    Returns
    -------

    q0_clim   : array-like, surface dust mmr
    dust_clim : array-like, vertical dust mmr
    '''
    sin_lat = np.sin(phi * np.pi/180)
    sinls = np.sin((ls-158.) * np.pi/180)
    

    dust_clim = np.zeros((len(phi),len(ls),len(pfull)))

    q0_clim = np.zeros((len(phi),len(ls)))

    for i in range(len(phi)):
        for j in range(len(ls)):
            zmax = 60 + 18*sinls[j] - (32 + 18*sinls[j]*sin_lat[i]**4) - 8*sinls[j]*sin_lat[i]**5
            tau_ij = tau.cdod.sel(latitude=tau.latitude[i]).sel(Time=tau.Time[j])

            q0_clim[i,j] = 7.4e-5*tau_ij/(16.4 - 7.4e-5*tau_ij)

            for k in range(len(pfull)):
                dust_clim[i,j,k] = q0_clim[i,j]*np.exp(0.007*(1-(700./pfull[k])**(70./zmax)))

    return q0_clim, dust_clim


if __name__ == "__main__":

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    newlabel = ['Northern\nsummer solstice', 'Northern\nwinter solstice']
    newpos = [90,270]

    fig = plt.figure(figsize = (22,12))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[5, 2, 2])

    axs1 = fig.add_subplot(gs[0, 0])
    axs2 = fig.add_subplot(gs[0, 1])
    axs3 = fig.add_subplot(gs[1, 0])
    axs4 = fig.add_subplot(gs[1:,1])
    axs5 = fig.add_subplot(gs[2, 0])

    axs1.set_xlim([0,360])
    axs2.set_xlim([0,360])
    axs3.set_xlim([0,360])
    axs4.set_xlim([0,360])
    axs5.set_xlim([0,360])

    axs1.set_xticklabels([])
    axs2.set_xticklabels([])

    axs1.tick_params(length = 6, labelsize = 18)
    axs2.tick_params(length = 6, labelsize = 18)
    axs3.tick_params(length = 6, labelsize = 18)
    axs4.tick_params(length = 6, labelsize = 18)
    axs5.tick_params(length = 6, labelsize = 18)

    axs1.text(-0.03, 1.02, string.ascii_lowercase[0], size = 20,
                    transform = axs1.transAxes, weight = 'bold')
    axs2.text(-0.03, 1.02, string.ascii_lowercase[1], size = 20,
                    transform = axs2.transAxes, weight = 'bold')
    axs3.text(-0.03, 1.02, string.ascii_lowercase[2], size = 20,
                    transform = axs3.transAxes, weight = 'bold')
    axs4.text(-0.03, 1.02, string.ascii_lowercase[3], size = 20,
                    transform = axs4.transAxes, weight = 'bold')

    axs1.set_ylabel('latitude ($^{\circ}$N)', fontsize = 20)
    axs5.set_ylabel('latitude ($^{\circ}$N)', fontsize = 20)
    axs2.set_ylabel('pressure (hPa)', fontsize = 20)
    axs4.set_ylabel('pressure (hPa)', fontsize = 20)

    axs5.yaxis.set_label_coords(x = -0.12, y = 0.5)
    axs5.yaxis.set_label_coords(x = -0.11, y = 1.15)

    axs1.set_ylim([-90, 90])
    axs2.set_ylim([6.5,0.005])
    axs3.set_ylim([55, 90])
    axs4.set_ylim([6.5,0.05])
    axs5.set_ylim([-90, -55])

    axs2.set_yscale('log')
    axs4.set_yscale('log')

    axs1.set_yticks([-75,-50,-25,0,25,50,75])
    axs2.set_yticks([6,1,0.1,0.01])
    axs2.set_yticklabels([6,1,0.1,])
    axs3.set_yticks([80,70,60])
    axs4.set_yticks([6,1,0.1,0.05])
    axs4.set_yticklabels([6,1,0.1,])
    axs5.set_yticks([-80,-70,-60])

    axs4.set_xlabel('solar longitude (degrees)', fontsize = 20)
    axs5.set_xlabel('solar longitude (degrees)', fontsize = 20)

    ax1 = duplicate_axis(axs1, newpos)
    ax2 = duplicate_axis(axs2, newpos)
    ax3 = duplicate_axis(axs3, newpos)
    ax4 = duplicate_axis(axs4, newpos)

    ax1.set_xticklabels(newlabel,fontsize=18)
    ax2.set_xticklabels(newlabel,fontsize=18)
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])

    axs3.set_xticklabels([])
    axs3.set_xticks([])
    axs3.tick_params(labelbottom='off')
    ax3.tick_params(labelbottom='off')
    axs3.spines['bottom'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)

    axs5.spines['top'].set_visible(False)

    d = 0.012
    
    kwargs = dict(transform=axs3.transAxes, color='k', clip_on=False)
    
    axs3.plot((-d, +d), (-d, +d), **kwargs)  # bottom-left diagonal
    axs3.plot((1-d, 1+d), (-d, +d), **kwargs)

    kwargs = dict(transform=axs5.transAxes, color='k', clip_on=False)
    
    axs5.plot((-d, +d), (1-d, 1+d), **kwargs)  # bottom-left diagonal
    axs5.plot((1-d, 1+d), (1-d, 1+d), **kwargs)


    plt.savefig('Thesis/dust_distributions.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)
        
    plt.subplots_adjust(wspace = 0.2)

    boundaries0 = [2e-7, 5e-7, 8e-7, 1.1e-6, 1.4e-6, 1.7e-6]
    
    cmap_new0 = cm.get_cmap('cet_CET_L18', len(boundaries0)+1)
    colours0 = list(cmap_new0(np.arange(len(boundaries0)+1)))
    cmap0 = colors.ListedColormap(colours0[1:-1],"")
    cmap0.set_over(colours0[-1])
    cmap0.set_under(colours0[0])
    norm0 = colors.BoundaryNorm(boundaries0,ncolors = len(boundaries0) - 1,
                               clip = False)

    cb0 = fig.colorbar(cm.ScalarMappable(norm = norm0, cmap = cmap0),
                      ax = axs1, extend = 'both',
                      ticks = boundaries0)

    cb0.set_label(label='dust mmr (kg/kg)',
                 fontsize=20)
    cb0.ax.set_yticklabels(["{:.1e}".format(i) for i in cb0.get_ticks()])

    cb0.ax.tick_params(labelsize=16)


    boundaries1 = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    
    cmap_new1 = cm.get_cmap('cet_CET_L18', len(boundaries1)+1)
    colours1 = list(cmap_new1(np.arange(len(boundaries1)+1)))
    cmap1 = colors.ListedColormap(colours1[1:-1],"")
    cmap1.set_over(colours1[-1])
    cmap1.set_under(colours1[0])
    norm1 = colors.BoundaryNorm(boundaries1,ncolors = len(boundaries1) - 1,
                               clip = False)

    cb1 = fig.colorbar(cm.ScalarMappable(norm = norm1, cmap = cmap1),
                      ax = axs2, extend = 'both',
                      ticks = boundaries1)

    cb1.set_label(label='dust mmr (kg/kg)',
                 fontsize=20)
    cb1.ax.set_yticklabels(["{:.1e}".format(i) for i in cb1.get_ticks()])

    cb1.ax.tick_params(labelsize=16)

    boundaries2 = [0.0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
                   0.01,]
    boundaries2 = [20 * i for i in boundaries2]
    
    cmap_new2 = cm.get_cmap('cet_kbc', len(boundaries2))
    colours2 = list(cmap_new2(np.arange(len(boundaries2))))
    cmap2 = colors.ListedColormap(colours2[:-1],"")
    cmap2.set_over(colours2[-1])
    cmap2.set_under('white')
    norm2 = colors.BoundaryNorm(boundaries2,ncolors = len(boundaries2) - 1,
                               clip = False)

    cb2 = fig.colorbar(cm.ScalarMappable(norm = norm2, cmap = cmap2),
                      ax = [axs3,axs5], extend = 'both',
                      ticks = boundaries2[slice(None,None,2)])

    cb2.set_label(label='$T_c - T^*$ (K)',
                 fontsize=20)
    #cb2.ax.set_yticklabels(["{:.1e}".format(i) for i in cb2.get_ticks()])

    cb2.ax.tick_params(labelsize=16)

    boundaries3 = [0.000,0.0004,0.0008,0.0012,0.0016,0.002,0.0024,0.0028,
                  0.0032,0.0036,0.004,0.0044,0.0048,0.0052,0.0056,0.006,0.0064,
                  0.0068, 0.0072]
    boundaries3 = [20 * i for i in boundaries3]
    cmap_new3 = cm.get_cmap('cet_kbc', len(boundaries3))
    colours3 = list(cmap_new3(np.arange(len(boundaries3))))
    cmap3 = colors.ListedColormap(colours3[:-1],"")
    cmap3.set_over(colours3[-1])
    cmap3.set_under('white')
    norm3 = colors.BoundaryNorm(boundaries3, ncolors = len(boundaries3) - 1,
                               clip = False)

    cb3 = fig.colorbar(cm.ScalarMappable(norm = norm3, cmap = cmap3),
                      ax = [axs4], extend = 'both',
                      ticks = boundaries3[slice(None,None,2)])

    cb3.set_label(label='$T_c - T^*$ (K)', fontsize=20)
    #cb3.ax.set_yticklabels(["{:.1e}".format(i) for i in cb3.get_ticks()])

    cb3.ax.tick_params(labelsize=16)



    plt.savefig('Thesis/dust_distributions.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)

    isca_years = []
    isca_ls = []


    filepath = '/export/anthropocene/array-01/xz19136/Data_Ball_etal_2021/'
    d_isca = xr.open_dataset(filepath+'topo_dust_lh_Ls0-360_dt_tg_zonal.nc', decode_times=False)
    # reduce dataset
    d = d_isca.astype('float32')
    d = d.sortby('new_time', ascending=True)
    #d = d[["dt_tg_lh_condensation", "mars_solar_long"]]
    d = d.rename({'pfull':'plev'})

    d["mars_solar_long"] = d.mars_solar_long.squeeze()
    
    #d["dt_tg_lh_condensation"] = d.dt_tg_lh_condensation.where(d.dt_tg_lh_condensation != np.nan, other = 0.0)
    
    
    dsr =d.mean(dim="MY",skipna=True)
    x = dsr.dt_tg_lh_condensation.mean(dim="lon",skipna=True) * 8
    
    print(x.min(skipna=True).values)
    print(x.max(skipna=True).values)

    Ls = dsr.mars_solar_long
    xp = x.sel(plev=2.0, method='nearest').squeeze()
    x80 = x.sel(lat=85, method='nearest').squeeze()
    #dsr = d.chunk({"new_time":"auto"})
    #dsr = dsr.rolling(new_time=5,center=True)
    #dsr = dsr.mean(skipna=True)
    cf3 = axs3.contourf(Ls, xp.lat, xp.transpose('lat','new_time'),
                        norm = norm2, cmap=cmap2, levels = [-50] \
                                                 + boundaries2 + [150])
    cf5 = axs5.contourf(Ls, xp.lat, xp.transpose('lat','new_time'),
                        norm = norm2, cmap=cmap2, levels = [-50] \
                                                 + boundaries2 + [150])
    cf4 = axs4.contourf(Ls, x80.plev, x80.transpose('plev','new_time'),
                        norm = norm3, cmap=cmap3, levels = [-50] \
                                                 + boundaries3 + [150])
    plt.savefig('Thesis/dust_distributions.pdf',
        bbox_inches = 'tight', pad_inches = 0.1)
    plt.savefig('Thesis/dust_distributions.png',
        bbox_inches = 'tight', pad_inches = 0.1)

    pfull = [625, 610, 600, 590, 580, 570, 560, 550, 530, 510, 490, 
             470, 450, 425, 400, 375, 350, 325, 300, 275, 250, 225, 
             200, 175, 150, 125, 100, 75, 50, 40, 30, 20, 10, 5, 1]


    tau = xr.open_dataset('/export/anthropocene/array-01/xz19136/dust_dists/dust_clim.nc',
                                                decode_times=False)
    tau = tau.mean(dim='longitude')
    tau = tau.squeeze()

    ls = tau.Ls
    phi = tau.latitude

    q0_clim, dust_clim = create_dust(pfull, phi, ls, tau)

    
    pfull = [pfull[i]/100 for i in range(len(pfull))]
    
    c0 = axs1.contourf(ls, phi, dust_clim[:,:,0], levels = [1e-10] + boundaries0 + [1e-4],
                      cmap=cmap0, norm=norm0)
    c1 = axs2.contourf(ls, pfull, np.transpose(dust_clim[int(len(phi)/2),:,:]),
                        levels = [1e-14] + boundaries1 + [1e-3], cmap=cmap1, norm=norm1)
    

    plt.savefig('Thesis/dust_distributions.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)


    
