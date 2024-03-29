'''
Calculates and plots a latent heat release proxy for OpenMARS, Isca - all years
and climatology.
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
if __name__ == "__main__":

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    colors = cc.cm.bmy(np.linspace(0,1,4))
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color', colors)

    pfull = 2.0
    latmin = 60
    latmax = 90
    
    nh = True
    if nh:
        nh = 'nh_'
    else:
        nh = ''

    title = "OpenMARS"

    linestyles = ['solid', 'dotted','dashed', 'dashdot']

    newlabel = ['Northern\nsummer solstice', 'Northern\nwinter solstice']
    newpos = [90,270]
    
    fig, axs = plt.subplots(3, 2, figsize = (17,12),
                            gridspec_kw={'height_ratios':[7,3,3]})

    #axs[0,0].set_title('OpenMARS', y=1.04, size=20,weight='bold')
    #axs[0,1].set_title('Model', y=1.04, size=20,weight='bold')

    d = .012  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    

    for i, ax in enumerate(fig.axes):
        ax.set_xlim([0,360])
        ax.tick_params(length = 6, labelsize = 18)

        
        if i < 4:
            ax2 = ax.twiny()
            ax2.set_xticks(newpos)

            ax2.xaxis.set_ticks_position('top')
            ax2.xaxis.set_label_position('top') 
            ax2.tick_params(length = 6)
            ax2.set_xlim(ax.get_xlim())

            ax.text(-0.04, 1.00, string.ascii_lowercase[i], size = 20,
                    transform = ax.transAxes, weight = 'bold')



        if i == 0:
            ax.set_ylabel('$T_c - T^*$ (K)', fontsize = 20)
            ax2.set_title(title, weight = "bold", fontsize = 20)
            ax2.set_xticklabels(newlabel,fontsize=18)
            ax.set_xticklabels([])
            ax.set_ylim([-0.005, 0.45])
            ax.set_yticks([0,0.1,0.2,0.3,0.4])

        elif i == 1:
            ax.set_yticklabels([])
            ax2.set_title("Yearly Simulations", weight = "bold", fontsize = 20)
            ax2.set_xticklabels(newlabel,fontsize=18)
            ax.set_xticklabels([])
            ax.set_ylim([-0.005, 0.45])
            ax.set_yticks([0,0.1,0.2,0.3,0.4])

        elif 1 < i < 4:
            ax2.set_xticklabels([])
            ax.tick_params(labelbottom='off')
            ax2.tick_params(labelbottom='off')
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_ylim([50,90])
            ax.set_yticks([60,70,80])
            ax.spines['bottom'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)

            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    
            ax.plot((-d, +d), (-d, +d), **kwargs)  # bottom-left diagonal
            ax.plot((1-d, 1+d), (-d, +d), **kwargs)

        elif i == 4:
            ax.set_ylabel('latitude ($^\circ$N)', fontsize = 20)
            ax.yaxis.set_label_coords(x = -0.15, y = 1)
            ax.set_ylim([-90,-50])
            ax.set_yticks([-80,-70,-60])
            ax.set_xlabel('solar longitude (degrees)', fontsize = 20)
            ax.spines['top'].set_visible(False)
            #ax.tick_params(labeltop='off')

            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    
            ax.plot((-d, +d), (1-d, 1+d), **kwargs)  # bottom-left diagonal
            ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

        elif i == 5:
            ax.set_ylim([-90,-50])
            ax.set_yticks([-80,-70,-60])
            ax.set_yticklabels([])
            ax.set_xlabel('solar longitude (degrees)', fontsize = 20)
            ax.spines['top'].set_visible(False)

            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    
            ax.plot((-d, +d), (1-d, 1+d), **kwargs)  # bottom-left diagonal
            ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
        
        if i == 3:
            ax.set_yticklabels([])
            
            
            

    boundaries, _, _, cmap, norm = funcs.make_colourmap(0, 0.41, 0.025,
                                        col = 'cet_kbc', extend = 'max')

    plt.subplots_adjust(hspace=.11, wspace = 0.05)

    cb = fig.colorbar(cm.ScalarMappable(norm = norm, cmap = cmap),
                      ax = axs, orientation = 'horizontal', extend='max',
                      aspect = 40, pad = 0.09,
                      ticks = boundaries[slice(None, None, 2)])

    cb.set_label(label = '$T_c - T^*$ (K)', fontsize = 20)
    cb.ax.tick_params(labelsize=18)



    
    roo = '/export/anthropocene/array-01/xz19136/Data_Ball_etal_2021/'




    
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


    isca_years = []
    isca_ls = []

    for i in range(len(exp)):
        print(exp[i])
        label = labels[i]

        d_isca = xr.open_mfdataset(roo+exp[i]+'Ls0-360_dt_tg_zonal.nc', decode_times=False)

        # reduce dataset
        d = d_isca.astype('float32')
        d = d.sortby('new_time', ascending=True)
        d = d[["dt_tg_lh_condensation", "mars_solar_long"]]

        d["mars_solar_long"] = d.mars_solar_long.squeeze()
        d["mars_solar_long"] = d.mars_solar_long.where(d.mars_solar_long != 354.3762, other=359.762)
        
        x = d.squeeze()
        a = x.dt_tg_lh_condensation.where(x.MY == x.MY[-1],
                            drop=True)
        a = a.squeeze()
        isca_years.append(a)

        ens = x.dt_tg_lh_condensation.where(x.dt_tg_lh_condensation != np.nan, other = 0.0)
        coslat = np.cos(np.pi/180 * ens.lat)
        ens = ens * coslat
        x["ens"] = ens
        
        
        if i == 4:
            x = x.where(x.MY < 4, drop = True)
        year_mean = x.mean(dim='MY', skipna=True)

        year_mean = year_mean.where(year_mean.lat > latmin, drop = True)
        
        ens = year_mean.ens.mean(dim="lon",skipna=True)
        ens = ens.where(ens > 0.0, other = np.nan)
        tlat = np.tan(np.pi/180 * abs(ens.lat[2] - ens.lat[1])/2)
        
        ens = ens.sum(dim='lat',skipna=True) / (2*tlat)
        Ls = year_mean.mars_solar_long.sel(lat=0,method="nearest")
        isca_ls.append(Ls)
        year_mean = ens.chunk({'new_time':'auto'})
        year_mean = year_mean.rolling(new_time=30,center=True)
        year_mean = year_mean.mean(skipna=True) * 8
        

        ax = axs[0,1]

        if i < 4:
            color = '#5F9ED1'
            linestyle = linestyles[i]
            lw=1.0

        elif i == 4:
            color = '#C85200'
            linestyle = 'solid'
            lw=1.2

        else:
            color =  '#ABABAB'
            linestyle = linestyles[i-5]
            lw=1.0

        c = ax.plot(Ls, year_mean, label = label, color=color,
                    linestyle=linestyle, linewidth=lw)

        plt.savefig('Thesis/lh_evolution_'+nh+str(pfull)+'hPa.pdf',
                    bbox_inches='tight',pad_inches=0.1)

    axs[0,1].legend(fontsize = 14, loc = 'upper left')



    for i in range(len(isca_years)):
        isca_years[i] = isca_years[i].assign_coords(MY = (i+24))
        isca_years[i] = isca_years[i].assign_coords(new_time = (isca_ls[i].values))
        

        x = isca_years[i]
        x = x.load()
        if i != 0:
            x = x.interp({"new_time":isca_ls[0].values})#,
                            #kwargs={"fill_value":"np.nan"})
    
    
    isca_yearly = xr.concat(isca_years, dim="MY")
    isca_yearly=isca_yearly.mean(dim="lon",skipna=True)

    isca_yearly = isca_yearly.where(isca_yearly != 0, other = np.nan) * 8
    year_isca = isca_yearly
    isca_std = year_isca.std(dim="MY",skipna=True)
    isca_std = isca_std.chunk({'new_time':'auto'})
    isca_std = isca_std.rolling(new_time=10,center=True)
    isca_std = isca_std.mean(skipna=True)
    print(isca_std.mean(skipna=True).values)

    year_isca = year_isca.mean(dim='MY',skipna = True)
    print(year_isca)
    print(isca_ls[0])

    cf1 = axs[1,1].contourf(isca_ls[0], year_isca.lat,
                    year_isca.transpose('lat','new_time'), norm = norm,
                    cmap=cmap, levels = [-50] + boundaries + [150])

    c1 = axs[1,1].contour(isca_ls[0], year_isca.lat,
                    isca_std.transpose('lat','new_time'), colors='0.8',
                    levels = [0.05,0.1], linewidths=0.8)
    c1.levels = [funcs.nf(val) for val in c1.levels]
    axs[1,1].clabel(c1, c1.levels, inline=1, fontsize=10)

    cf1_ = axs[2,1].contourf(isca_ls[0], year_isca.lat,
                    year_isca.transpose('lat','new_time'), norm = norm,
                    cmap=cmap, levels = [-50] + boundaries + [150])

    c1_ = axs[2,1].contour(isca_ls[0], year_isca.lat,
                    isca_std.transpose('lat','new_time'), colors='0.8',
                    levels = [0.05,0.1], linewidths=0.8)
    c1_.levels = [funcs.nf(val) for val in c1_.levels]
    axs[2,1].clabel(c1_, c1_.levels, inline=1, fontsize=10)



    plt.savefig('Thesis/lh_evolution_'+nh+str(pfull)+'hPa.pdf',
                bbox_inches='tight',pad_inches=0.1)
    





    PATH = roo + 'OpenMARS_Ls0-360_t_2hPa.nc'

    
    d_open = xr.open_mfdataset(PATH, decode_times=False)
    
    smooth = 200
    d_open = d_open.sortby('time', ascending=True)
        

    plt.savefig('Thesis/lh_evolution_'+nh+str(pfull)+'hPa.pdf',
                bbox_inches='tight',pad_inches=0.1)

    
    d = d_open.astype('float32')
    
    tc = 149.2 + 6.48*np.log(0.135*pfull)

    t = d.tmp.where(d.tmp < tc, other = tc)
    t = tc - t
    #t = t.where(t > 0, other = np.nan)
    d["tstar"] = t

    op_years = []
    op_ls = []


    for i in list(np.arange(24,33,1)):
        print('Beginning MY '+str(i))
        di = d.where(d.MY == i, drop=True)
        op_years.append(di.tstar)
        
        di = di.where(di.lat > latmin, drop = True)
        di["Ls"] = di.Ls.sel(lat=di.lat[0])
        
        Ls = di.Ls.sel(lon=di.lon[0]).squeeze()

        coslat = np.cos(np.pi/180 * di.lat)
        tot = sum(np.cos(np.pi/180 * di.lat))
        tlat = np.tan(np.pi/180 * abs(di.lat[2] - di.lat[1])/2)
        

        Zi = di.tmp.where(di.tmp < tc, other=tc)

        Zi = tc - Zi
        
        Zi = Zi * coslat
        Zi = Zi.sum(dim='lat',skipna = True) / (2*tlat)
        Zi = Zi.mean(dim="lon",skipna=True)
        Zi = Zi.chunk({'time':'auto'})
        Zi = Zi.rolling(time=smooth,center=True)

        Zi = Zi.mean()
        

        op_ls.append(Ls)

        ax = axs[0,0]
        
        if i < 28:
            color = '#5F9ED1'
            linestyle = linestyles[i-28]
            lw=1.0

        elif i == 28:
            color = '#C85200'
            linestyle = 'solid'
            lw=1.5

        else:
            color =  '#ABABAB'
            linestyle = linestyles[i-29]
            lw=1.0


        ci = ax.plot(Ls, Zi, label='MY '+str(i), color=color,
                     linestyle = linestyle, linewidth=lw)
                     
        plt.savefig('Thesis/lh_evolution_'+nh+str(pfull)+'hPa.pdf',
            bbox_inches = 'tight', pad_inches = 0.1)
    
    axs[0,0].legend(fontsize = 14, loc = 'upper left')

    
    plt.savefig('Thesis/lh_evolution_'+nh+str(pfull)+'hPa.pdf',
                bbox_inches='tight',pad_inches=0.1)

    ls0 = np.arange(0,360,0.05)

    for i in range(len(op_years)):
        op_years[i] = op_years[i].assign_coords(time = (op_ls[i].values))
        op_years[i] = op_years[i].assign_coords(my = (i+24))
        
        x = op_years[i]
        x.load()
        x = x.interp({"time" : ls0})#,
                            #kwargs={"fill_value":np.nan})
        op_years[i] = x
    
    open_mars_yearly = xr.concat(op_years, dim="my").mean(dim="lon",skipna=True)
    open_mars_yearly = open_mars_yearly.where(open_mars_yearly != 0, other = np.nan)
    year_open = open_mars_yearly
    open_std = year_open.std(dim='my', skipna=True)
    open_std = open_std.chunk({'time':'auto'})
    open_std = open_std.rolling(time=150,center=True)
    open_std = open_std.mean(skipna=True)

    year_open = year_open.mean(dim="my",skipna=True)
    year_open = year_open.chunk({'time':'auto'})
    year_open = year_open.rolling(time=10,center=True)
    year_open = year_open.mean(skipna=True).load()

    cf0 = axs[1,0].contourf(ls0, year_open.lat,
                    year_open.transpose('lat','time'), norm = norm, cmap=cmap,
                    levels = [-50] + boundaries + [150])

    c0 = axs[1,0].contour(ls0, year_open.lat,
                    open_std.transpose('lat','time'), colors='0.8',
                    levels = [0.05,0.1],linewidths=0.8)
    
    c0.levels = [funcs.nf(val) for val in c0.levels]
    axs[1,0].clabel(c0, c0.levels, inline=1, #fmt=fmt,
                    fontsize=10)


    cf0_ = axs[2,0].contourf(ls0, year_open.lat,
                    year_open.transpose('lat','time'), norm = norm, cmap=cmap,
                    levels = [-50] + boundaries + [150])

    c0_ = axs[2,0].contour(ls0, year_open.lat,
                    open_std.transpose('lat','time'), colors='0.8',
                    levels = [0.05,0.1],linewidths=0.8)
    
    c0_.levels = [funcs.nf(val) for val in c0_.levels]
    axs[2,0].clabel(c0_, c0_.levels, inline=1, #fmt=fmt,
                    fontsize=10)


    plt.savefig('Thesis/lh_evolution_'+nh+str(pfull)+'hPa.pdf',
                bbox_inches='tight',pad_inches=0.1)

