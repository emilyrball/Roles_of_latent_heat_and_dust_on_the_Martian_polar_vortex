# plot Fig. 5 of Darryn Waugh's 2016 paper 'Comparison of reanalyses' for
# MACDA, EMARS and OpenMars

import numpy as np
import xarray as xr
import os, sys
import glob
import calculate_PV as cPV
import PVmodule_Emily as cpv

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import (cm, colors)
import matplotlib.path as mpath

import colorcet as cc
import string
from Isca_instantaneous_PV_all import make_colourmap
from Isca_OpenMARS_profiles import calc_PV_max
from eddy_enstrophy_Isca_all_years import (assign_MY, make_coord_MY)

Lsmin = 255
Lsmax = 285
my = 24

Waugh_plot=False
difference_Waugh=False
open_mars=True
emars=False
isca_years=False
isca_years_old=False

thetalevs=[200., 250., 300., 350., 400., 450., 500., 550., 600., 650., 700., 750., 800., 850., 900., 950.]

theta_0 = 200.
kappa = 1/4.0
p0 = 610.

inpath_E = 'link-to-anthro/EMARS/'
inpath_M = 'link-to-anthro/MACDA/'
inpath_O = 'link-to-anthro/OpenMARS/Isobaric/'

figpath = 'Figs/'


d_E1 = xr.open_mfdataset(inpath_E + '*baric*_MY*', decode_times=False, concat_dim='time',
                       combine='nested',chunks={'time':'auto'})
d_O1 = xr.open_mfdataset(inpath_O + '*mars_my*', decode_times=False, concat_dim='time',
                       combine='nested',chunks={'time':'auto'})

d_E = d_E1.where(d_E1.MY == my , drop = True)
d_E = d_E.where(Lsmin <= d_E.Ls, drop = True)
d_E = d_E.where(d_E.Ls <= Lsmax, drop = True)


d_O = d_O1.where(d_O1.MY == my, drop = True)
d_O = d_O.where(Lsmin <= d_O.Ls, drop = True)
d_O = d_O.where(d_O.Ls <= Lsmax, drop = True)

lait_E = cPV.lait(d_E.PV, d_E.theta, theta_0, kappa=kappa)
lait_O = cPV.lait(d_O.PV, d_O.theta, theta_0, kappa=kappa)

pv_e = lait_E.mean(dim='time').mean(dim='lon') *10**5
pv_o = lait_O.mean(dim='time').mean(dim='lon') *10**5

t_e = d_E.theta.mean(dim='time').mean(dim='lon')
t_o = d_O.theta.mean(dim='time').mean(dim='lon')

u_e = d_E.u.mean(dim='time').mean(dim='lon')
u_o = d_O.uwnd.mean(dim='time').mean(dim='lon')

p_e = d_E.pfull/100
lat_e = d_E.lat

p_o = d_O.plev/100
lat_o = d_O.lat

#d_M = xr.open_mfdataset(inpath_M + '*baric*mars_MY*', decode_times=False, concat_dim='time',
#                        combine='nested',chunks={'time':'auto'})

#d_M = d_M.where(d_M.MY == my, drop = True)
#d_M = d_M.where(Lsmin <= d_M.Ls, drop = True)
#d_M = d_M.where(d_M.Ls <= Lsmax, drop = True)

#lait_M = cPV.lait(d_M.PV, d_M.theta, theta_0, kappa=kappa)
#pv_m = lait_M.mean(dim='time').mean(dim='lon') *10**5
#t_m = d_M.theta.mean(dim='time').mean(dim='lon')
#u_m = d_M.uwnd.mean(dim='time').mean(dim='lon')

#p_m = d_M.plev/100
#lat_m = d_M.lat


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)



class nf(float):
    def __repr__(self):
        s = f'{self:.1f}'
        return f'{self:.0f}' if s[-1] == '0' else s

if Waugh_plot==True:
    fig, axs = plt.subplots(nrows=1,ncols=3, sharey=True,sharex=True,
                            figsize=(21, 5))
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')
    #ticker.LogFormatter(base=10.0, labelOnlyBase=False,
    #                               minor_thresholds=None, linthresh=None)
    axs[0].set_xlim([0,90])
    axs[0].set_ylim([10,0.005])
    axs[1].set_xlim([0,90])
    axs[1].set_ylim([10,0.005])
    axs[2].set_xlim([0,90])
    axs[2].set_ylim([10,0.005])

    axs[0].yaxis.set_major_formatter(ticker.ScalarFormatter())
    axs[1].yaxis.set_major_formatter(ticker.ScalarFormatter())
    axs[2].yaxis.set_major_formatter(ticker.ScalarFormatter())

    axs[0].set_xlabel('latitude (degrees)')
    axs[1].set_xlabel('latitude (degrees)')
    axs[2].set_xlabel('latitude (degrees)')
    axs[0].set_ylabel('pressure (hPa)')

    vmax = 65.
    vmin = -5.

    cmap = cm.OrRd
    boundaries = [-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65]
    new_bds = [-30,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,200]
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='OrRd'),ax=axs,extend='max',
                label='Scaled PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)')


    axs[1].contourf(lat_e, p_e, pv_e.transpose('pfull','lat'),
                levels = new_bds,
                cmap='OrRd', vmin=vmin, vmax=vmax)
    axs[1].contour(lat_e, p_e, t_e.transpose('pfull','lat'),
                levels=[200,300,400,500,600,700,800,900,1000,1100],
                linestyles = '--', colors='black', linewidths=1)
    cs1 = axs[1].contour(lat_e, p_e, u_e.transpose('pfull','lat'),
                    levels=[0,50,100,150], colors='black',linewidths=1)

    axs[0].contourf(lat_m, p_m, pv_m.transpose('plev','lat'),
                levels = new_bds,
                cmap='OrRd', vmin=vmin, vmax=vmax)
    axs[0].contour(lat_m, p_m, t_m.transpose('plev','lat'),
                levels=[200,300,400,500,600,700,800,900,1000,1100],
                linestyles = '--', colors='black', linewidths=1)
    cs0 = axs[0].contour(lat_m, p_m, u_m.transpose('plev','lat'),
                    levels=[0,50,100,150], colors='black',linewidths=1)

    axs[2].contourf(lat_o, p_o, pv_o.transpose('plev','lat'),
                levels = new_bds,
                cmap='OrRd', vmin=vmin, vmax=vmax)
    axs[2].contour(lat_o, p_o, t_o.transpose('plev','lat'),
                levels=[200,300,400,500,600,700,800,900,1000,1100],
                linestyles = '--', colors='black', linewidths=1)
    cs2 = axs[2].contour(lat_o, p_o, u_o.transpose('plev','lat'),
                    levels=[0,50,100,150], colors='black',linewidths=1)

    # Recast levels to new class
    cs0.levels = [nf(val) for val in cs0.levels]
    cs1.levels = [nf(val) for val in cs1.levels]
    cs2.levels = [nf(val) for val in cs2.levels]


    # Label levels with specially formatted floats
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    axs[0].clabel(cs0, cs0.levels, inline=1, fmt=fmt, fontsize=10)
    axs[1].clabel(cs1, cs1.levels, inline=1, fmt=fmt, fontsize=10)
    axs[2].clabel(cs2, cs2.levels, inline=1, fmt=fmt, fontsize=10)

    axs[0].set_title('MACDA')
    axs[1].set_title('EMARS')
    axs[2].set_title('OpenMARS')

    plt.subplots_adjust(hspace=.2,wspace=.1)

    plt.savefig('Figs/Waugh2016_MACDA_EMARS_OpenMARS'+str(my)+'.pdf')
    plt.clf()

if difference_Waugh==True:
    if my <27:
        fig, axs = plt.subplots(nrows=1,ncols=3, sharey=True,sharex=True,
                                figsize=(21, 5))
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        axs[2].set_yscale('log')
        #ticker.LogFormatter(base=10.0, labelOnlyBase=False,
        #                               minor_thresholds=None, linthresh=None)
        axs[0].set_xlim([0,90])
        axs[0].set_ylim([10,0.005])
        axs[1].set_xlim([0,90])
        axs[1].set_ylim([10,0.005])
        axs[2].set_xlim([0,90])
        axs[2].set_ylim([10,0.005])

        axs[0].yaxis.set_major_formatter(ticker.ScalarFormatter())
        axs[1].yaxis.set_major_formatter(ticker.ScalarFormatter())
        axs[2].yaxis.set_major_formatter(ticker.ScalarFormatter())

        axs[0].set_xlabel('latitude (degrees)')
        axs[1].set_xlabel('latitude (degrees)')
        axs[2].set_xlabel('latitude (degrees)')
        axs[0].set_ylabel('pressure (hPa)')

        ax2 = axs[2].twinx()
        ax2.set_yscale('log')
        ax2.set_ylabel('altitude (km)')
        newlabel = [0, 9.4, 20.5, 31.6]
        newpos = [10, 1, 0.1, 0.01]
        labels = ["","","",""]
        ax2.set_yticks(newpos)
        ax2.set_yticklabels(newlabel)
        ax2.yaxis.set_ticks_position('right') # set the position of the second x-axis to bottom
        ax2.yaxis.set_label_position('right') # set the position of the second x-axis to bottom
        #ax2.spines['right'].set_position(('outward'))
        ax2.set_ylim(axs[2].get_ylim())

        ax1 = axs[1].twinx()
        ax1.set_yscale('log')
        #ax1.spines['right'].set_position(('outward'))
        ax1.set_ylim(axs[1].get_ylim())
        #ax1.set_yticks(newpos)
        ax1.yaxis.set_ticks_position('right')
        ax1.set_yticklabels(labels)

        ax0 = axs[0].twinx()
        ax0.set_yscale('log')
        #ax0.spines['right'].set_position(('outward'))
        ax0.set_ylim(axs[0].get_ylim())
        #ax0.set_yticks(newpos)
        ax0.yaxis.set_ticks_position('right')
        ax0.set_yticklabels(labels)

        
        plt.subplots_adjust(hspace=.1,wspace=.1)

        vmax = 65
        vmin = -5

        cmap = cm.OrRd
        boundaries = [-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65]
        new_bds = [-30,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,200]
        norm = colors.BoundaryNorm(boundaries, cmap.N)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='OrRd'),ax=axs,extend='both',
                    label='Scaled PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)',pad=0.05)

        diff_m = pv_m.transpose('plev','lat') - pv_o.transpose('plev','lat')

        pv_e = pv_e.rename({'pfull':'plev'})
        pv_e = pv_e.interp_like(pv_o.plev)
        t_e = t_e.rename({'pfull':'plev'})
        t_e = t_e.interp_like(pv_o.plev)
        u_e = u_e.rename({'pfull':'plev'})
        u_e = u_e.interp_like(pv_o.plev)
        diff_e = pv_e.transpose('plev','lat') - pv_o.transpose('plev','lat')
        p_e = pv_e.plev/100

        axs[0].contourf(lat_o, p_o, pv_o.transpose('plev','lat'),
                    levels = new_bds,
                    cmap='OrRd', vmin=vmin, vmax=vmax)
        axs[0].contour(lat_o, p_o, t_o.transpose('plev','lat'),
                    levels=[200,300,400,500,600,700,800,900,1000,1100],
                    linestyles = '--', colors='black', linewidths=1)
        cs0 = axs[0].contour(lat_o, p_o, u_o.transpose('plev','lat'),
                        levels=[0,50,100,150], colors='black',linewidths=1)
                        
        axs[1].contourf(lat_m, p_m, diff_m,
                    levels = new_bds,
                    cmap='OrRd', vmin=vmin, vmax=vmax)
        axs[1].contour(lat_m, p_m, t_m.transpose('plev','lat'),
                    levels=[200,300,400,500,600,700,800,900,1000,1100],
                    linestyles = '--', colors='black', linewidths=1)
        cs1 = axs[1].contour(lat_m, p_m, u_m.transpose('plev','lat'),
                        levels=[0,50,100,150], colors='black',linewidths=1)

        axs[2].contourf(lat_e, p_e, diff_e,
                    levels = new_bds,
                    cmap='OrRd', vmin=vmin, vmax=vmax)
        axs[2].contour(lat_e, p_e, t_e.transpose('plev','lat'),
                    levels=[200,300,400,500,600,700,800,900,1000,1100],
                    linestyles = '--', colors='black', linewidths=1)
        cs2 = axs[2].contour(lat_e, p_e, u_e.transpose('plev','lat'),
                        levels=[0,50,100,150], colors='black',linewidths=1)

        

        # Recast levels to new class
        cs0.levels = [nf(val) for val in cs0.levels]
        cs1.levels = [nf(val) for val in cs1.levels]
        cs2.levels = [nf(val) for val in cs2.levels]

        # Label levels with specially formatted floats
        if plt.rcParams["text.usetex"]:
            fmt = r'%r \%'
        else:
            fmt = '%r'

        axs[0].clabel(cs0, cs0.levels, inline=1, fmt=fmt, fontsize=10)
        axs[1].clabel(cs1, cs1.levels, inline=1, fmt=fmt, fontsize=10)
        axs[2].clabel(cs2, cs2.levels, inline=1, fmt=fmt, fontsize=10)

        axs[0].set_title('OpenMARS')
        axs[1].set_title('MACDA (difference)')
        axs[2].set_title('EMARS (difference)')

    elif my > 27:
        fig, axs = plt.subplots(nrows=1,ncols=2, sharey=True,sharex=True,
                                figsize=(14, 5))
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        #ticker.LogFormatter(base=10.0, labelOnlyBase=False,
        #                               minor_thresholds=None, linthresh=None)
        axs[0].set_xlim([0,90])
        axs[0].set_ylim([10,0.005])
        axs[1].set_xlim([0,90])
        axs[1].set_ylim([10,0.005])

        axs[0].yaxis.set_major_formatter(ticker.ScalarFormatter())
        axs[1].yaxis.set_major_formatter(ticker.ScalarFormatter())

        axs[0].set_xlabel('latitude (degrees)')
        axs[1].set_xlabel('latitude (degrees)')
        axs[0].set_ylabel('pressure (hPa)')

        ax2 = axs[1].twinx()
        ax2.set_yscale('log')
        ax2.set_ylabel('altitude (km)')
        newlabel = [0, 9.4, 20.5, 31.6]
        newpos = [10, 1, 0.1, 0.01]
        labels = ["","","",""]
        ax2.set_yticks(newpos)
        ax2.set_yticklabels(newlabel)
        ax2.yaxis.set_ticks_position('right') # set the position of the second x-axis to bottom
        ax2.yaxis.set_label_position('right') # set the position of the second x-axis to bottom
        #ax2.spines['right'].set_position(('outward'))
        ax2.set_ylim(axs[1].get_ylim())

        ax0 = axs[0].twinx()
        ax0.set_yscale('log')
        #ax0.spines['right'].set_position(('outward'))
        ax0.set_ylim(axs[0].get_ylim())
        #ax0.set_yticks(newpos)
        ax0.yaxis.set_ticks_position('right')
        ax0.set_yticklabels(labels)

        plt.subplots_adjust(hspace=.1,wspace=.1)

        vmax = 65
        vmin = -15

        cmap = cm.OrRd
        boundaries = [-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65]
        new_bds = [-50,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,200]
        norm = colors.BoundaryNorm(boundaries, cmap.N)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='OrRd'),ax=axs,extend='both',
                    label='Scaled PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)',pad=0.09)

        pv_e = pv_e.rename({'pfull':'plev'})
        pv_e = pv_e.interp_like(pv_o.plev)
        t_e = t_e.rename({'pfull':'plev'})
        t_e = t_e.interp_like(pv_o.plev)
        u_e = u_e.rename({'pfull':'plev'})
        u_e = u_e.interp_like(pv_o.plev)
        diff_e = pv_e.transpose('plev','lat') - pv_o.transpose('plev','lat')
        p_e = pv_e.plev/100

        axs[0].contourf(lat_o, p_o, pv_o.transpose('plev','lat'),
                    levels = new_bds,
                    cmap='OrRd', vmin=vmin, vmax=vmax)
        axs[0].contour(lat_o, p_o, t_o.transpose('plev','lat'),
                    levels=[200,300,400,500,600,700,800,900,1000,1100],
                    linestyles = '--', colors='black', linewidths=1)
        cs0 = axs[0].contour(lat_o, p_o, u_o.transpose('plev','lat'),
                        levels=[0,50,100,150], colors='black',linewidths=1)

        axs[1].contourf(lat_e, p_e, diff_e,
                    levels = new_bds,
                    cmap='OrRd', vmin=vmin, vmax=vmax)
        axs[1].contour(lat_e, p_e, t_e.transpose('plev','lat'),
                    levels=[200,300,400,500,600,700,800,900,1000,1100],
                    linestyles = '--', colors='black', linewidths=1)
        cs1 = axs[1].contour(lat_e, p_e, u_e.transpose('plev','lat'),
                        levels=[0,50,100,150], colors='black',linewidths=1)

        

        # Recast levels to new class
        cs0.levels = [nf(val) for val in cs0.levels]
        cs1.levels = [nf(val) for val in cs1.levels]

        # Label levels with specially formatted floats
        if plt.rcParams["text.usetex"]:
            fmt = r'%r \%'
        else:
            fmt = '%r'

        axs[0].clabel(cs0, cs0.levels, inline=1, fmt=fmt, fontsize=10)
        axs[1].clabel(cs1, cs1.levels, inline=1, fmt=fmt, fontsize=10)

        axs[0].set_title('OpenMARS')
        axs[1].set_title('EMARS (difference)')

    
    plt.savefig('Figs/Waugh2016_difference_'+str(my)+'.pdf')
    plt.clf()

if open_mars==True:

    nrow = 2
    ncol = 4

    vmin = -5
    vmax = 65
    step = 5

    fig, axs = plt.subplots(nrow, ncol, sharey=True,sharex=True,
                            figsize=(25, 8))

    
    plt.subplots_adjust(hspace=.2,wspace=.1)

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'


    boundaries, _, _, cmap, norm = make_colourmap(vmin, vmax, step,
                                            col = 'OrRd', extend = 'max')

    for i, ax in enumerate(fig.axes):
        
        ax.set_yscale('log')
        ax.set_xlim([0,90])
        ax.set_ylim([10,0.005])
        ax.text(-0.05, 1.05, string.ascii_lowercase[i], transform=ax.transAxes,
                size=22, weight='bold')

        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        if i < 3:
            my = i+24
        else:
            my = i+25
        print(my)
        d_O = d_O1.where(d_O1.MY==my, drop=True)
        d_O = d_O.where(Lsmin <= d_O.Ls, drop = True)
        d_O = d_O.where(d_O.Ls <= Lsmax, drop = True)
        d_O = d_O.where(d_O.lat >= 0, drop = True)
        
        lait_O = cPV.lait(d_O.PV, d_O.theta, theta_0, kappa=kappa)
        pv_o = lait_O.mean(dim='time').mean(dim='lon') *10**5
        t_o = d_O.theta.mean(dim='time').mean(dim='lon')
        u_o = d_O.uwnd.mean(dim='time').mean(dim='lon')

        lats_max = []
        arr = pv_o.load()
        for jlev in range(len(arr.plev)):
            marr = arr.sel(plev=arr.plev[jlev])
            marr_max = marr.max().values
            marr = marr.where(marr >= marr_max,drop=True)
            lat_max = marr.lat.values

            #lat_max, max_mag = calc_PV_max(marr[:,ilev], lait1.lat)
            lats_max.append(lat_max)

        ax.plot(lats_max, arr.plev/100, linestyle='-', color='blue',linewidth=2)

        ax.contourf(lat_o.where(lat_o.lat >= 0, drop=True), p_o, pv_o.transpose('plev','lat'),
                    levels = [-50]+boundaries+[150],norm=norm,cmap=cmap)
        ax.contour(lat_o.where(lat_o.lat >= 0, drop=True), p_o, t_o.transpose('plev','lat'),
                    levels=[200,300,400,500,600,700,800,900,1000,1100],
                    linestyles = '--', colors='black', linewidths=1)
        csi = ax.contour(lat_o.where(lat_o.lat >= 0, drop=True), p_o, u_o.transpose('plev','lat'),
                        levels=[0,50,100,150], colors='black',linewidths=1)

        csi.levels = [nf(val) for val in csi.levels]
        ax.clabel(csi, csi.levels, inline=1, fmt=fmt, fontsize=14)
        ax.tick_params(labelsize=18, length=8)
        ax.tick_params(length=4, which='minor')

        ax.set_title('MY '+str(my), fontsize = 22, weight = 'bold', y = 1.02)
        plt.savefig('OpenMARS_figs/Waugh_yearly.pdf', bbox_inches='tight',
                pad_inches=0.04)

    
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs,
                extend='max', ticks=boundaries[slice(None,None,1)],pad=0.01)

    cb.set_label(label='Lait-scaled PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)',
                 fontsize=20)
    cb.ax.tick_params(labelsize=18)


    axs[1,0].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[1,1].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[1,2].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[1,3].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[0,0].set_ylabel('pressure (hPa)',fontsize=20)
    axs[1,0].set_ylabel('pressure (hPa)',fontsize=20)

    plt.savefig('OpenMARS_figs/Waugh_yearly.pdf', bbox_inches='tight',
                pad_inches=0.04)





if emars==True:
    d_O2 = d_O1.where(Lsmin <= d_O1.Ls, drop = True)
    d_O2 = d_O2.where(d_O2.Ls <= Lsmax, drop = True)

    d_E2 = d_E1.where(Lsmin <= d_E1.Ls, drop = True)
    d_E2 = d_E2.where(d_E2.Ls <= Lsmax, drop = True)

    fig, axs = plt.subplots(nrows=5,ncols=2, sharey=True,sharex=True,
                            figsize=(6, 12))
    #plt.subplots_adjust(hspace=.1,wspace=.1)
    vmax = 65
    vmin = -15

    cmap = cm.OrRd
    boundaries = [-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65]
    new_bds = [-50,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,200]
    norm = colors.BoundaryNorm(boundaries, cmap.N)

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    for i in [0,1,2,3,4]:
        my = i + 28

        if i==4:
            axs[i,j].set_xlabel('latitude (degrees)')

        d_O = d_O2.where(d_O2.MY==my, drop=True)
        d_E = d_E2.where(d_E2.MY==my, drop=True)

        lait_O = cPV.lait(d_O.PV, d_O.theta, theta_0, kappa=kappa)
        pv_o = lait_O.mean(dim='time').mean(dim='lon') *10**5
        t_o = d_O.theta.mean(dim='time').mean(dim='lon')
        u_o = d_O.uwnd.mean(dim='time').mean(dim='lon')

        lait_E = cPV.lait(d_E.PV, d_E.theta, theta_0, kappa=kappa)
        pv_e = lait_E.mean(dim='time').mean(dim='lon') *10**5
        t_e = d_E.theta.mean(dim='time').mean(dim='lon')
        u_e = d_E.u.mean(dim='time').mean(dim='lon')

        pv_e = pv_e.rename({'pfull':'plev'})
        pv_e = pv_e.interp_like(pv_o.plev)
        t_e = t_e.rename({'pfull':'plev'})
        t_e = t_e.interp_like(pv_o.plev)
        u_e = u_e.rename({'pfull':'plev'})
        u_e = u_e.interp_like(pv_o.plev)
        diff_pve = pv_e.transpose('plev','lat') - pv_o.transpose('plev','lat')
        diff_ue = u_e.transpose('plev','lat') - u_o.transpose('plev','lat')
        diff_te = t_e.transpose('plev','lat') - t_o.transpose('plev','lat')
        p_e = pv_e.plev/100

        for j in [0,1]:
            axs[i,j].set_yscale('log')
            axs[i,j].set_xlim([0,90])
            axs[i,j].set_ylim([10,0.005])
            axs[i,j].yaxis.set_major_formatter(ticker.ScalarFormatter())

            if j==1:
                ax = axs[i,j].twinx()
                ax.set_yscale('log')
                ax.set_ylabel('altitude (km)')
                newlabel = [0, 9.4, 20.5, 31.6]
                newpos = [10, 1, 0.1, 0.01]
                labels = ["","","",""]
                ax.set_yticks(newpos)
                ax.set_yticklabels(newlabel)
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_label_position('right')
                ax.set_ylim(axs[i,j].get_ylim())


                axs[i,j].contourf(lat_e, p_e, diff_pve,
                                levels = new_bds,
                                cmap='OrRd', vmin=vmin, vmax=vmax)
                csj = axs[i,j].contour(lat_e, p_e, diff_te,
                                levels=[-50,-25,0,25,50],
                                linestyles = '--', colors='black', linewidths=1)
                csi = axs[i,j].contour(lat_e, p_e, diff_ue,
                                    levels=[-50,-25,0,25,50],
                                    colors='black',linewidths=1,linestyles='-')


            elif j==0:
                axs[i,j].set_ylabel('pressure (hPa)')
                axs[i,j].contourf(lat_o, p_o, pv_o.transpose('plev','lat'),
                            levels = new_bds,
                            cmap='OrRd', vmin=vmin, vmax=vmax)
                axs[i,j].contour(lat_o, p_o, t_o.transpose('plev','lat'),
                            levels=[200,300,400,500,600,700,800,900,1000,1100],
                            linestyles = '--', colors='black', linewidths=1)
                csj = axs[i,j].contour(lat_o, p_o, t_o.transpose('plev','lat'),
                            levels=[5000])
                csi = axs[i,j].contour(lat_o, p_o, u_o.transpose('plev','lat'),
                            levels=[0,50,100,150], colors='black',linewidths=1)

            csi.levels = [nf(val) for val in csi.levels]

            axs[i,j].clabel(csi, csi.levels, inline=1, fmt=fmt, fontsize=10)
            csj.levels = [nf(val) for val in csj.levels]

            axs[i,j].clabel(csj, csj.levels, inline=1, fmt=fmt, fontsize=10)
        
    axs[0,0].set_title('OpenMARS')
    axs[0,1].set_title('EMARS (difference)')
    
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='OrRd'),ax=axs,extend='both',
                label='Scaled PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)',pad=0.15)
    
    plt.savefig('Figs/Waugh2016_yearly_emars_difference_MCS.pdf')
    plt.clf()

from calculate_PV_Isca_anthro import filestrings

if isca_years==True:

    exp = ['soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY24_4.2e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY25_4.2e-05_lh_rel',
#           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY26_4.2e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY27_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY28_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY29_4.2e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY30_4.2e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY31_4.2e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY32_4.2e-05_lh_rel',]


    location = ['silurian']

    #filepath = '/export/triassic/array-01/xz19136/Isca_data'
    start_file = [23]
    end_file = [28]



    freq = 'daily'

    interp_file = 'atmos_'+freq+'_interp_new_height_temp_PV.nc'

    theta0 = 200.
    kappa = 1/4.0
    p0 = 610.

    

    figpath = 'Thesis/new_sims/'

    Lsmin = 255.
    Lsmax = 285.


    nrow = 2
    ncol = 4

    vmin = -5
    vmax = 65
    step = 2.5

    fig, axs = plt.subplots(nrow, ncol, sharey=True,sharex=True,
                            figsize=(25, 8))

    
    plt.subplots_adjust(hspace=.2,wspace=.1)

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'


    boundaries, _, _, cmap, norm = make_colourmap(vmin, vmax, step,
                                            col = 'OrRd', extend = 'max')

    for i in range(len(exp)):

        filepath = '/export/' + location[0] + '/array-01/xz19136/Isca_data'
        start = start_file[0]
        end = end_file[0]

        _, _, i_files = filestrings(exp[i], filepath, start, end, interp_file)

        d = xr.open_mfdataset(i_files, decode_times=False, concat_dim='time',
                            combine='nested')

        # reduce dataset
        d = d.astype('float32')
        d = d.sortby('time', ascending=True)


        d["mars_solar_long"] = d.mars_solar_long.sel(lon=0)
        d = d.where(d.mars_solar_long != 354.37808, other=359.762)
        print(d.mars_solar_long[-1].values)
        print(d.mars_solar_long[0].values)


        d = d.where(d.mars_solar_long <= Lsmax, drop=True)
        d = d.where(Lsmin <= d.mars_solar_long, drop=True)

        #d, index = assign_MY(d)


        #x = x.where(x.time <= index[-1], drop=True)

        # Lait scale PV
        theta = cpv.potential_temperature(d.pfull*100, d.temp, p0=p0, kappa=kappa)

        lait = cpv.laitscale(d.PV, theta, theta0, kappa=kappa)*10**5

        lait = lait.mean(dim='time').mean(dim='lon').squeeze()

        d = d.mean(dim="time").mean(dim="lon").squeeze()
        theta = theta.mean(dim="time").mean(dim="lon").squeeze()

        temp = d.temp
        u = d.ucomp


        temp = temp.rename({'pfull':'plev'})

        u = u.rename({'pfull':'plev'})

        theta = theta.rename({'pfull':'plev'})

        lait = lait.rename({'pfull':'plev'})


        tmpi = temp.transpose('plev','lat')
        ui = u.transpose('plev','lat')
        laiti = lait.transpose('plev','lat')
        thetai = theta.transpose('plev','lat')


        for j, ax in enumerate(fig.axes):
            if j == i:
                ax.set_yscale('log')
                ax.set_xlim([0,90])
                ax.set_ylim([10,0.005])
                ax.text(-0.05, 1.05, string.ascii_lowercase[j], transform=ax.transAxes,
                        size=22, weight='bold')

                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

                if j < 3:
                    my = i+24
                else:
                    my = j+25
                print(my)

                lats_max = []
                arr = lait.load()
                for jlev in range(len(arr.plev)):
                    marr = arr.sel(plev=arr.plev[jlev])
                    marr_max = marr.max().values
                    marr = marr.where(marr >= marr_max,drop=True)
                    lat_max = marr.lat.values

                    #lat_max, max_mag = calc_PV_max(marr[:,ilev], lait1.lat)
                    lats_max.append(lat_max)

                ax.plot(lats_max, arr.plev, linestyle='-', color='blue',linewidth=2)

                ax.contourf(laiti.lat, laiti.plev, laiti,
                        levels=[-150]+boundaries+[350],norm=norm,cmap=cmap)
                ax.contour(thetai.lat, thetai.plev, thetai,
                            levels=[200,300,400,500,600,700,800,900,1000,1100],
                            linestyles = '--', colors='black', linewidths=1)
                csi = ax.contour(tmpi.lat, tmpi.plev, ui,
                        levels=[-50,0,50,100,150],colors='black',linewidths=1)

                csi.levels = [nf(val) for val in csi.levels]
                ax.clabel(csi, csi.levels, inline=1, fmt=fmt, fontsize=14)
                ax.tick_params(labelsize=18)

                ax.set_title('MY '+str(my), fontsize = 22, weight = 'bold', y = 1.02)
                plt.savefig('Isca_figs/Waugh_yearly_new.pdf', bbox_inches='tight',
                        pad_inches=0.02)

                #####
                ax.plot(lats_max, arr.plev, linestyle='-', color='blue')
                #####        
            

    
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs,
                extend='max', ticks=boundaries[slice(None,None,2)],pad=0.01)

    cb.set_label(label='Lait-scaled PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)',
                 fontsize=18)
    cb.ax.tick_params(labelsize=18)


    axs[1,0].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[1,1].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[1,2].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[1,3].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[0,0].set_ylabel('pressure (hPa)',fontsize=20)
    axs[1,0].set_ylabel('pressure (hPa)',fontsize=20)

    plt.savefig('Isca_figs/Waugh_yearly_new.pdf', bbox_inches='tight',
                pad_inches=0.02)



if isca_years_old==True:
    theta0 = 200.
    kappa = 1/4.0
    p0 = 610.

    

    figpath = 'Thesis/new_sims/'

    Lsmin = 255.
    Lsmax = 285.


    nrow = 2
    ncol = 4

    vmin = -5
    vmax = 65
    step = 2.5

    fig, axs = plt.subplots(nrow, ncol, sharey=True,sharex=True,
                            figsize=(25, 8))

    
    plt.subplots_adjust(hspace=.2,wspace=.1)

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'


    boundaries, _, _, cmap, norm = make_colourmap(vmin, vmax, step,
                                            col = 'OrRd', extend = 'max')

    exp = 'soc_mars_mk36_per_value70.85_none_mld_2.0_all_years'
    filepath = '/export/silurian/array-01/xz19136/Isca_data/link-to-silurian/Isca_data'
    start_file = 21
    end_file = 120
    freq = 'daily'
    interp_file = 'atmos_'+freq+'_interp_new_height_temp_PV.nc'
        
    _ ,_ , i_files = filestrings(exp, filepath, start_file,
                                    end_file, interp_file)
    
    d = xr.open_mfdataset(i_files, decode_times=False, concat_dim='time',
                        combine='nested',chunks={'time':'auto'})

    # reduce dataset
    d = d.astype('float32')
    d = d.sortby('time', ascending=True)
    
    d["mars_solar_long"] = d.mars_solar_long.sel(lon=0)
    d = d.where(d.mars_solar_long != 354.37808, other=359.762)
    print(d.mars_solar_long[0].values)

    
    d, index = assign_MY(d)
    x = d.squeeze()
    
    
    dsr, N, n = make_coord_MY(x, index)

    dsr = dsr.where(dsr.mars_solar_long <= Lsmax, drop=True)
    dsr = dsr.where(Lsmin <= dsr.mars_solar_long, drop=True)

    #d, index = assign_MY(d)


    #x = x.where(x.time <= index[-1], drop=True)

    for i in list(np.arange(0,9,1)):
        
        if i < 2:
            my = i + 25
        else:
            my = i + 26

        di = dsr.where(dsr.MY == my-25, drop=True).squeeze()

        # Lait scale PV
        theta = cpv.potential_temperature(di.pfull*100, di.temp, p0=p0, kappa=kappa)

        lait = cpv.laitscale(di.PV, theta, theta0, kappa=kappa)*10**5

        lait = lait.mean(dim='new_time').mean(dim='lon').squeeze()

        theta = theta.mean(dim="new_time").mean(dim="lon").squeeze()

        temp = di.temp.mean(dim='new_time').mean(dim='lon').squeeze()
        u = di.ucomp.mean(dim='new_time').mean(dim='lon').squeeze()


        temp = temp.rename({'pfull':'plev'})

        u = u.rename({'pfull':'plev'})

        theta = theta.rename({'pfull':'plev'})

        lait = lait.rename({'pfull':'plev'})


        tmpi = temp.transpose('plev','lat')
        ui = u.transpose('plev','lat')
        laiti = lait.transpose('plev','lat')
        thetai = theta.transpose('plev','lat')
        

        for j, ax in enumerate(fig.axes):
            if j == i:
                ax.set_yscale('log')
                ax.set_xlim([0,90])
                ax.set_ylim([10,0.005])
                ax.text(-0.05, 1.05, string.ascii_lowercase[j], transform=ax.transAxes,
                        size=22, weight='bold')

                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

                
                lats_max = []
                arr = lait.load()
                for jlev in range(len(arr.plev)):
                    marr = arr.sel(plev=arr.plev[jlev])
                    marr_max = marr.max().values
                    marr = marr.where(marr >= marr_max,drop=True)
                    lat_max = marr.lat.values

                    #lat_max, max_mag = calc_PV_max(marr[:,ilev], lait1.lat)
                    lats_max.append(lat_max)

                ax.plot(lats_max, arr.plev, linestyle='-', color='blue',linewidth=2)

                ax.contourf(laiti.lat, laiti.plev, laiti,
                        levels=[-150]+boundaries+[350],norm=norm,cmap=cmap)
                ax.contour(thetai.lat, thetai.plev, thetai,
                            levels=[200,300,400,500,600,700,800,900,1000,1100],
                            linestyles = '--', colors='black', linewidths=1)
                csi = ax.contour(tmpi.lat, tmpi.plev, ui,
                        levels=[-50,0,50,100,150],colors='black',linewidths=1)

                csi.levels = [nf(val) for val in csi.levels]
                ax.clabel(csi, csi.levels, inline=1, fmt=fmt, fontsize=14)
                ax.tick_params(labelsize=18)

                ax.set_title('MY '+str(my), fontsize = 22, weight = 'bold', y = 1.02)
                plt.savefig('Isca_figs/Waugh_yearly_old.pdf', bbox_inches='tight',
                        pad_inches=0.02)

                #####
                ax.plot(lats_max, arr.plev, linestyle='-', color='blue')
                #####        
        

    
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs,
                extend='max', ticks=boundaries[slice(None,None,2)],pad=0.01)

    cb.set_label(label='Lait-scaled PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)',
                 fontsize=18)
    cb.ax.tick_params(labelsize=18)


    axs[1,0].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[1,1].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[1,2].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[1,3].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)
    axs[0,0].set_ylabel('pressure (hPa)',fontsize=20)
    axs[1,0].set_ylabel('pressure (hPa)',fontsize=20)

    plt.savefig('Isca_figs/Waugh_yearly_old.pdf', bbox_inches='tight',
                pad_inches=0.02)