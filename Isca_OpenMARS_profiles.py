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

#def fmt(x, pos):
#    a, b = '{:.2e}'.format(x).split('e')
#    b = int(b)
#    return r'${} \times 10^{{{}}}$'.format(a, b)



class nf(float):
    def __repr__(self):
        s = f'{self:.1f}'
        return f'{self:.0f}' if s[-1] == '0' else s

def calc_PV_max(PV, lats, plot=False):
    '''
    Function to calculate location and strenth of maximum given zonal-mean PV
    PV(height) field
    '''
    # Restict to 10 points around maximum
    PV_max = np.where(PV == np.ma.max(PV))[0][0]
    PV_near = PV[PV_max-1:PV_max+2]
    lats_near = lats[PV_max-1:PV_max+2]
    # Quartic fit, with smaller lat spacing
    coefs = np.ma.polyfit(lats_near,PV_near,2)
    fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
    quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2
    # Find jet lat and max
    jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]
    jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2
    # Plot fit?
    if plot:
        print (jet_max)
        print (jet_lat)
        plt.plot(lats_near, PV_near)
        plt.plot(fine_lats, quad)
        plt.show()

    return jet_lat, jet_max

if __name__ == "__main__":

    theta0 = 200.
    kappa = 1/4.0
    p0 = 610.

    ### choose your files
    exp = [
        #'soc_mars_mk36_per_value70.85_none_mld_2.0',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_lh_rel',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_cdod_clim_scenario_7.4e-05',
        #'soc_mars_mk36_per_value70.85_none_mld_2.0_cdod_clim_scenario_7.4e-05_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_lh_rel',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05',
        'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05_lh_rel',
    ]


    location = [
        #'triassic',
        #'triassic',
        #'anthropocene',
        #'anthropocene',
        'triassic',
        'triassic',
        'anthropocene',
        'silurian',
    ]

    #filepath = '/export/triassic/array-01/xz19136/Isca_data'
    start_file = [
        #33,
        #33,
        #33,
        #33,
        33,
        33,
        33,
        33,
        ]
    end_file = [
        #99,
        #99, 
        #99, 
        #99, 
        99, 
        99, 
        99, 
        139,
        ]

    freq = 'daily'

    interp_file = 'atmos_'+freq+'_interp_new_height_temp_PV.nc'
    pfull = 1.

    figpath = 'Thesis/vertical_profiles/'

    Lsmin = 270.
    Lsmax = 300.


    inpath = 'link-to-anthro/OpenMARS/Isobaric/'


    fig0, axs0 = plt.subplots(nrows=1,ncols=5, figsize = (20,6))
    #fig0.subplots_adjust(wspace = 0.05)#,hspace=0.01)
    plt.tight_layout()
    fig1, axs1 = plt.subplots(nrows=1,ncols=5, figsize = (20,6))
    #fig1.subplots_adjust(wspace = 0.09)#,hspace=0.01)
    plt.tight_layout()
    fig2, axs2 = plt.subplots(nrows=1,ncols=5, figsize = (20,6))
    #fig2.subplots_adjust(wspace = 0.02)#,hspace=0.01)
    plt.tight_layout()



    boundaries0, _, _, cmap0, norm0 = funcs.make_colourmap(130, 221, 5,
                                        col = 'cet_coolwarm', extend = 'both')
    boundaries1, _, _, cmap1, norm1 = funcs.make_colourmap(-150, 151, 25,
                                        col = 'cet_coolwarm', extend = 'both')
    boundaries, _, _, cmap, norm = funcs.make_colourmap(-0.5, 7.1, 0.5,
                                        col = 'OrRd', extend = 'max')

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'
        


    # individual plots
    for j, ax in enumerate(fig0.axes):
        ax.set_yscale('log')
        ax.set_xlabel('latitude ($^{\circ}$N)', fontsize=22)
        ax.set_xlim([0,90])
        ax.set_ylim([10,0.005])
        ax.set_yticks([10,1,0.1,0.01])
        ax.tick_params(labelsize=18, length=8)
        ax.tick_params(length=4, which='minor')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.text(-0.02, 1.06, string.ascii_lowercase[j], transform=ax.transAxes, 
                size=20, weight='bold')
        if j == 0:
            ax.set_ylabel('pressure (hPa)', fontsize=22)
            ax.tick_params(width=1.5, which = 'both')
            ax.tick_params(length=5, which='minor')
            ax.tick_params(length=9,)
        else:
            ax.set_yticklabels([])

    for j, ax in enumerate(fig1.axes):
        ax.set_yscale('log')
        ax.set_xlabel('latitude ($^{\circ}$N)', fontsize=22)
        ax.set_xlim([0,90])
        ax.set_ylim([10,0.005])
        ax.set_yticks([10,1,0.1,0.01])
        ax.tick_params(labelsize=18, length=8)
        ax.tick_params(length=4, which='minor')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.text(-0.02, 1.06, string.ascii_lowercase[j], transform=ax.transAxes, 
                size=20, weight='bold')
        if j == 0:
            ax.set_ylabel('pressure (hPa)', fontsize=22)
            ax.tick_params(width=1.5, which = 'both')
            ax.tick_params(length=5, which='minor')
            ax.tick_params(length=9,)
        else:
            ax.set_yticklabels([])

    for j, ax in enumerate(fig2.axes):
        ax.set_yscale('log')
        ax.set_xlabel('latitude ($^{\circ}$N)', fontsize=22)
        ax.set_xlim([0,90])
        ax.set_ylim([10,0.005])
        ax.set_yticks([10,1,0.1,0.01])
        ax.tick_params(labelsize=18, length=8)
        ax.tick_params(length=4, which='minor')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.text(-0.02, 1.06, string.ascii_lowercase[j], transform=ax.transAxes, 
                size=20, weight='bold')
        if j == 0:
            ax.set_ylabel('pressure (hPa)', fontsize=22)
            ax.tick_params(width=1.5, which = 'both')
            ax.tick_params(length=5, which='minor')
            ax.tick_params(length=9,)
        else:
            ax.set_yticklabels([])



    cb0 = fig0.colorbar(cm.ScalarMappable(norm0, cmap0), ax = axs0, extend = 'both',
                        aspect = 50, pad = 0.2, orientation = 'horizontal',
                        ticks = boundaries0[slice(None, None, 2)])
    cb1 = fig1.colorbar(cm.ScalarMappable(norm1, cmap1), ax = axs1, extend = 'both',
                        aspect = 50, pad = 0.2, orientation = 'horizontal',
                        ticks = boundaries1[slice(None, None, 2)])
    cb2 = fig2.colorbar(cm.ScalarMappable(norm, cmap), ax = axs2, extend = 'max',
                        aspect = 50, pad = 0.2, orientation = 'horizontal',
                        ticks = boundaries[slice(1, None, 2)])


    cb0.set_label(label='Temperature (K)',
                 fontsize=22)
    cb1.set_label(label='u-wind (ms$^{-1}$)',
                 fontsize=22)
    cb2.set_label(label='Lait-scaled PV (MPVU)',
                 fontsize=22)

    cb0.ax.tick_params(labelsize=18)
    cb1.ax.tick_params(labelsize=18)
    cb2.ax.tick_params(labelsize=18)


    axs0[0].set_title('OpenMARS', fontsize=20, weight='bold', y = 1.06)
    axs0[1].set_title('T', fontsize=20, weight='bold', y = 1.06)
    axs0[2].set_title('LH+T', fontsize=20, weight='bold', y = 1.06)
    axs0[3].set_title('D+T', fontsize=20, weight='bold', y = 1.06)
    axs0[4].set_title('LH+D+T', fontsize=20, weight='bold', y = 1.06)

    axs1[0].set_title('OpenMARS', fontsize=20, weight='bold', y = 1.06)
    axs1[1].set_title('T', fontsize=20, weight='bold', y = 1.06)
    axs1[2].set_title('LH+T', fontsize=20, weight='bold', y = 1.06)
    axs1[3].set_title('D+T', fontsize=20, weight='bold', y = 1.06)
    axs1[4].set_title('LH+D+T', fontsize=20, weight='bold', y = 1.06)

    axs2[0].set_title('OpenMARS', fontsize=20, weight='bold', y = 1.06)
    axs2[1].set_title('T', fontsize=20, weight='bold', y = 1.06)
    axs2[2].set_title('LH+T', fontsize=20, weight='bold', y = 1.06)
    axs2[3].set_title('D+T', fontsize=20, weight='bold', y = 1.06)
    axs2[4].set_title('LH+D+T', fontsize=20, weight='bold', y = 1.06)




    axs0[0].spines['left'].set_linewidth(3)
    axs0[0].spines['right'].set_linewidth(3)
    axs0[0].spines['top'].set_linewidth(3)
    axs0[0].spines['bottom'].set_linewidth(3)

    axs1[0].spines['left'].set_linewidth(3)
    axs1[0].spines['right'].set_linewidth(3)
    axs1[0].spines['top'].set_linewidth(3)
    axs1[0].spines['bottom'].set_linewidth(3)

    axs2[0].spines['left'].set_linewidth(3)
    axs2[0].spines['right'].set_linewidth(3)
    axs2[0].spines['top'].set_linewidth(3)
    axs2[0].spines['bottom'].set_linewidth(3)

    ds = xr.open_mfdataset(inpath + '*mars_my*', decode_times=False, concat_dim='time',
                           combine='nested')

    ds = ds.where(Lsmin <= ds.Ls, drop = True)
    ds = ds.where(ds.Ls <= Lsmax, drop = True)

    ds["plev"] = ds.plev/100

    ds = ds.where(ds.lat >= 0, drop=True)

    t1 = ds.tmp.mean(dim='time').mean(dim='lon')
    u1 = ds.uwnd.mean(dim='time').mean(dim='lon')
    theta1 = ds.theta.mean(dim='time').mean(dim='lon')

    lait1 = funcs.lait(ds.PV, ds.theta, theta0, kappa=kappa)
    lait1 = lait1.mean(dim='time').mean(dim='lon')*10**4

    p = ds.plev
    lat = ds.lat


    #marr = np.ma.masked_array(lait1, np.isnan(lait1),fill_value=-999)
    #arr = lait1.where(lait1.plev < 5.5, drop=True)
    arr = lait1.load()

    lats_max = []
    for ilev in range(len(arr.plev)):
        marr = arr.sel(plev=arr.plev[ilev])
        marr_max = marr.max().values
        marr = marr.where(marr >= marr_max,drop=True)
        lat_max = marr.lat.values
        #lat_max, max_mag = calc_PV_max(marr[:,ilev], lait1.lat)
        lats_max.append(lat_max)

    axs0[0].contourf(lat, p, t1.transpose('plev','lat'),
        levels=[90]+boundaries0+[350],norm=norm0,cmap=cmap0)
    c0 = axs0[0].contour(lat, p, u1.transpose('plev','lat'),
        levels=[-50,0,50,100,150], colors='black',linewidths=1)
    c0.levels = [nf(val) for val in c0.levels]
    axs0[0].clabel(c0, c0.levels, inline=1, fmt = fmt, fontsize =18)
    axs1[0].contourf(lat, p, u1.transpose('plev','lat'),
        levels=[-200]+boundaries1+[350],norm=norm1,cmap=cmap1)
    axs2[0].contourf(lat, p, lait1.transpose('plev','lat'),
        levels=boundaries+[350],norm=norm,cmap=cmap)

    axs2[0].contour(lat, p, theta1.transpose('plev','lat'),
                    levels=[200,300,400,500,600,700,800,900,1000,1100],
                    linestyles = '--', colors='black', linewidths=1)


    axs2[0].plot(lats_max, arr.plev, linestyle='-', color='blue')




    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    data = []

    for i in range(len(start_file)):

        filepath = '/export/'+location[i]+'/array-01/xz19136/Isca_data'
        start = start_file[i]
        end = end_file[i]
    
        _, _, i_files = funcs.filestrings(exp[i], filepath, start, end, interp_file)
        

        d = xr.open_mfdataset(i_files, decode_times=False, concat_dim='time',
                            combine='nested')


        # reduce dataset
        d = d.astype('float32')
        d = d.sortby('time', ascending=True)
        d = d[["temp", "ucomp", "PV", "mars_solar_long"]]



        d["mars_solar_long"] = d.mars_solar_long.sel(lon=0)
        #d = d.where(d.mars_solar_long != 354.3762, other=359.762)
        d = d.where(d.mars_solar_long != 354.3762, drop=True)

        d = d.where(d.mars_solar_long <= 285., drop = True)
        d = d.where(d.mars_solar_long >= 255., drop = True)


        theta = funcs.calculate_theta(d.temp, d.pfull*100, p0=p0, kappa=kappa)

        lait = funcs.lait(d.PV, theta, theta0, kappa=kappa)*10**4

        lait = lait.mean(dim='time').mean(dim='lon').squeeze()

        d = d.mean(dim="time").mean(dim="lon").squeeze()
        theta = theta.mean(dim="time").mean(dim="lon").squeeze()


        temp = d.temp.squeeze()
        u = d.ucomp.squeeze()


        temp = temp.rename({'pfull':'plev'})

        u = u.rename({'pfull':'plev'})

        theta = theta.rename({'pfull':'plev'})

        lait = lait.rename({'pfull':'plev'})

        #temp = temp.mean(dim='scalar_axis')
        #u = u.mean(dim='scalar_axis')
        
        tmpi = temp.transpose('plev','lat')
        ui = u.transpose('plev','lat')
        laiti = lait.transpose('plev','lat')
        thetai = theta.transpose('plev','lat')


        lats_max = []
        arr = laiti.where(laiti.lat>0,drop=True).load()
        for ilev in range(len(arr.plev)):
            marr = arr.sel(plev=arr.plev[ilev])
            marr_max = marr.max().values
            marr = marr.where(marr >= marr_max,drop=True)
            lat_max = marr.lat.values

            #lat_max, max_mag = calc_PV_max(marr[:,ilev], lait1.lat)
            lats_max.append(lat_max)


        axs0[i+1].contourf(tmpi.lat, tmpi.plev, tmpi,
                levels=[90]+boundaries0+[350],norm=norm0,cmap=cmap0)
        csi = axs0[i+1].contour(tmpi.lat, tmpi.plev, ui,
                levels=[-50,0,50,100,150],colors='black',linewidths=1)

        csi.levels = [nf(val) for val in csi.levels]
        axs0[i+1].clabel(csi, csi.levels, inline=1, fmt = fmt, fontsize =18)
        axs1[i+1].contourf(ui.lat, ui.plev, ui,
                levels=[-250]+boundaries1+[350],norm=norm1,cmap=cmap1)
        axs2[i+1].contourf(laiti.lat, laiti.plev, laiti,
                levels=[-150]+boundaries+[350],norm=norm,cmap=cmap)
        #####
        axs2[i+1].plot(lats_max, arr.plev, linestyle='-', color='blue')
        #####        
        axs2[i+1].contour(thetai.lat, thetai.plev, thetai,
                    levels=[200,300,400,500,600,700,800,900,1000,1100],
                    linestyles = '--', colors='black', linewidths=1)


        fig0.savefig(figpath+'vertical_temp_Isca_OpenMARS_winter_dusts.pdf',bbox_inches='tight', pad_inches = 0.04)
        fig1.savefig(figpath+'vertical_uwnd_Isca_OpenMARS_winter_dusts.pdf',bbox_inches='tight', pad_inches = 0.04)
        fig2.savefig(figpath+'vertical_PV_Isca_OpenMARS_winter_dusts.pdf',bbox_inches='tight', pad_inches = 0.04)



    fig0.savefig(figpath+'vertical_temp_Isca_OpenMARS_winter_dusts.pdf',bbox_inches='tight', pad_inches = 0.04)
    fig1.savefig(figpath+'vertical_uwnd_Isca_OpenMARS_winter_dusts.pdf',bbox_inches='tight', pad_inches = 0.04)
    fig2.savefig(figpath+'vertical_PV_Isca_OpenMARS_winter_dusts.pdf',bbox_inches='tight', pad_inches = 0.04)

