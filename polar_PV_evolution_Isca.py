# polar temperature evolution

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

from dust_distribution import create_dust

def data_reduct(d, Lsmin, Lsmax, latmin, latmax):
    d = d.astype('float32')
    d = d.sortby('time', ascending=True)
    
    d = d.where(d.mars_solar_long != 354.3762, other=359.762)
    d = d.where(d.mars_solar_long != 354.37808, other = 359.762)
    d = d.where(d.lat > latmin, drop = True)
    d = d.where(d.lat < latmax, drop = True)
    d = d.mean(dim='lon')
    
    d = d.where(d.mars_solar_long > Lsmin, drop = True)
    d = d.where(d.mars_solar_long < Lsmax, drop = True)

    return d

def calc_jet_lat(u, lats, plot = False):
    '''
    Function to calculate location and strength of maximum given zonal wind
    u(lat) field

    Parameters
    ----------

    u    : array-like
    lats : array-like. Default use will be to calculate jet on a given pressure
           level, but this array may also represent pressure level.

    Returns
    -------

    jet_lat : latitude (pressure level) of maximum zonal wind
    jet_max : strength of maximum zonal wind
    '''

    # Restrict to 10 points around maximum
    u_max = np.where(u == np.ma.max(u.values))[0][0]
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
    
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    theta0 = 200.
    kappa = 1/4.0
    p0 = 610.

    year = "MY 28"

    sh = False
    ilev = 350
    pfull = 30
    jetmax = False

    latmin = 60
    latmax = 90

    Lsmin = 200
    Lsmax = 360

    red  = '#C85200'
    blue = '#006BA4'
    grey = '#595959'

    colors = [blue, red, grey]

    if sh == True:
        latmin0 = -latmax
        latmax = -latmin
        latmin = latmin0
        hem = 'sh'
    else:
        hem = 'nh'

    if jetmax == True:
        jet = "jet_"
    else:
        jet = ""

    ### choose your files
    exp = ['soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_MY28_7.4e-05_lh_rel',
           'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05_lh_rel',
           #'soc_mars_mk36_per_value70.85_none_mld_2.0_all_years',
    ]
    location = ['silurian',
                'silurian',
                #'anthropocene',
    ]
    start_file = [30,
                  30,
                  #21,
    ]
    end_file = [222,
                139,
                #222,
    ]

    freq = 'daily'

    interp_file = 'atmos_'+freq+'_interp_new_height_temp_isentropic.nc'
    p_file = 'atmos_'+freq+'_interp_new_height_temp_PV.nc'

    labels = [
        'Temperature (K)',
        'Zonal wind (ms $^{-1}$)',
        'Lait-scaled PV\n(10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)',
    ]


    #pfulls = [625, 610, 600, 590, 580, 570, 560, 550, 530, 510, 490, 
    #         470, 450, 425, 400, 375, 350, 325, 300, 275, 250, 225, 
    #         200, 175, 150, 125, 100, 75, 50, 40, 30, 20, 10, 5, 1]


    #tau_clim = xr.open_dataset('/export/anthropocene/array-01/xz19136/' \
    #                      + 'dust_dists/dust_clim.nc', decode_times = False)
    #tau_clim = tau_clim.mean(dim='longitude')
    #tau_clim = tau_clim.squeeze()

    #ls_clim = tau_clim.Ls
    #phi_clim = tau_clim.latitude

    #q0_clim, _ = create_dust(pfulls, phi_clim, ls_clim, tau_clim)

    #q0_clim = xr.DataArray(data = q0_clim, dims = ["latitude", "Time"],
    #                coords = dict(latitude = (["latitude"], phi_clim),
    #                              Time = (["Time"], tau_clim.Time)),
    #                attrs = dict(description="Surface dust mmr",
    #                                     units="kg/kg"))

    #tau_MY28 = xr.open_dataset('/export/anthropocene/array-01/xz19136/dust_' \
    #                 + 'dists/dustscenario_MY28_v2-0.nc', decode_times = False)
    #tau_MY28 = tau_MY28.mean(dim='longitude')
    #tau_MY28["cdod"] = tau_MY28.cdod610
    #tau_MY28 = tau_MY28[["cdod", "Ls"]]
    #tau_MY28 = tau_MY28.squeeze()
    #print(tau_MY28)


    #ls_MY28 = tau_MY28.Ls
    #phi_MY28 = tau_MY28.latitude

    #q0_MY28, _ = create_dust(pfulls, phi_MY28, ls_MY28, tau_MY28)

    #q0_MY28 = xr.DataArray(data = q0_MY28, dims = ["latitude", "Time"],
    #                coords = dict(latitude = (["latitude"], phi_MY28),
    #                              Time = (["Time"], tau_MY28.Time)),
    #                attrs = dict(description="Surface dust mmr",
    #                                     units="kg/kg"))

    #if sh:
    #    q0_clim = q0_clim.where(q0_clim.latitude <= 0, drop = True)
    #    q0_MY28 = q0_MY28.where(q0_MY28.latitude <= 0, drop = True)
    #else: 
    #    q0_clim = q0_clim.where(q0_clim.latitude >= 0, drop = True)
    #    q0_MY28 = q0_MY28.where(q0_MY28.latitude >= 0, drop = True)

    #q0_clim = q0_clim.mean(dim = "latitude", skipna = True)
    #q0_MY28 = q0_MY28.mean(dim = "latitude", skipna = True)

    fig, axs = plt.subplots(3,1,figsize=(10,14))

    for i, ax in enumerate(fig.axes):
        ax.set_xlim([Lsmin, Lsmax])
        ax.set_xticklabels([])
        ax.tick_params(length = 6, labelsize = 16)
        ax.set_ylabel(labels[i], fontsize = 18)

        ax3 = ax.twinx()
        ax3.tick_params(length = 6, labelsize = 16)
        
        ax3.yaxis.set_label_position('right')
        ax3.yaxis.set_ticks_position('right')
        #ax3.set_ylabel('surface dust mmr (kg/kg)', fontsize = 18,
        #                color = red)

        #ax3.plot(ls_clim, q0_clim, color = red,
        #         linewidth = '1.0', linestyle = '--')
        #ax3.plot(ls_MY28, q0_MY28, color = red,
        #         linewidth = '1.0', linestyle = '-')
        #ax3.set_yticklabels(["{:.1e}".format(i) for i in ax3.get_yticks()])
        ax3.set_ylim([-0.05,1])
        ax3.set_yticks([])
        ax3.set_yticklabels([])
        ax3.plot(np.linspace(265,310,200),np.zeros(200), color = 'k',
                 linewidth = '3.5',)
        
    axs[2].set_xticklabels(axs[2].get_xticks())
    axs[2].set_xlabel('solar longitude (degrees)', fontsize = 18)
    
    fig.savefig('Isca_figs/variability_all_'+hem+'_'+str(ilev)+'K.pdf',
                    bbox_inches='tight', pad_inches = 0.1)
    

    for i in range(len(start_file)):
        print(exp[i])

        filepath = '/export/' + location[i] + '/array-01/xz19136/Isca_data'
        figpath = 'Isca_figs/'+exp[i]+'/'
        start = start_file[i]
        end = end_file[i]

        
        fig.savefig(figpath+'variability_all_'+hem+'_'+str(ilev)+'K.pdf',
                    bbox_inches='tight', pad_inches = 0.1)

        _, _, i_files = filestrings(exp[i], filepath, start, end, interp_file)

        d = xr.open_mfdataset(i_files, decode_times = False,
                              concat_dim = 'time', combine='nested')

        _, _, p_files = filestrings(exp[i], filepath, start, end, p_file)

        dp = xr.open_mfdataset(p_files, decode_times = False,
                              concat_dim = 'time', combine='nested')


        # reduce dataset
        d = data_reduct(d, Lsmin, Lsmax, latmin, latmax)
        dp = data_reduct(dp, Lsmin, Lsmax, latmin, latmax)

        if jetmax == False:
            d = d.mean(dim = "lat")
        
        dp = dp.mean(dim = "lat")
        
        d = d.sel(ilev = ilev, method = 'nearest').squeeze()
        dp = dp.sel(pfull = pfull/100, method = 'nearest').squeeze()

        # Lait scale PV
        theta = d.ilev
        print("Scaling PV")
        laitPV = cPV.lait(d.PV, theta, theta0, kappa = kappa)
        d["scaled_PV"] = laitPV


        x, index = assign_MY(d)
        dsr, N, n = make_coord_MY(x, index)

        Ls = dsr.mars_solar_long.mean(dim = 'MY')
        year_mean = dsr.mean(dim = 'MY')
        year_mean = year_mean.chunk({'new_time' : 'auto'})
        year_mean = year_mean.rolling(new_time = 4, center = True)
            
        year_mean = year_mean.mean()
        mean_PV = year_mean.scaled_PV * 10**5
        mean_wind = year_mean.uwnd

        if jetmax == True:
            mean_PV = mean_PV.mean(dim = "lat")
            mean_wind = mean_wind.max(dim = "lat")
        
        linestyle = '-'
        labc = year
        col = colors[i]

        if exp[i] == 'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_' \
                        + 'topo_cdod_clim_scenario_7.4e-05_lh_rel':
            linestyle = '--'
            labc = "climatology"

        ct=axs[2].plot(Ls[:-5], mean_PV[:-5], color = col, label = labc,
                       linewidth = '1.1', linestyle = linestyle)
        cw=axs[1].plot(Ls[:-5],mean_wind[:-5],color = col, label = labc,
                       linewidth='1.1', linestyle = linestyle)

        fig.savefig(figpath + 'variability_all_' + hem + '_' + str(ilev) \
                     + 'K.pdf', bbox_inches = 'tight', pad_inches = 0.1)

        for j in n:
            di = dsr.where(dsr.MY == j, drop = True).squeeze()

            Zi = di.chunk({'new_time' : 'auto'})
            Zi = Zi.rolling(new_time = 4, center = True)

            Zi = Zi.mean()

            Ls = di.mars_solar_long
            labj = ""
            if exp[i] == 'soc_mars_mk36_per_value70.85_none_mld_2.0_all_years':
                if j == 3:
                    labj = "MY 28"
                    col = grey

            cti = axs[2].plot(Ls[:-5], Zi.scaled_PV[:-5]*10**5,
                           label = labj, color = col,
                           linewidth = '0.6', linestyle = '-',
                           alpha = 0.4)
            cwi = axs[1].plot(Ls[:-5], Zi.uwnd[:-5], label = labj,
                           color = col, linewidth = '0.6',
                           linestyle = '-', alpha = 0.4)
            fig.savefig(figpath + 'variability_all_' + hem + '_' \
                        + str(ilev) + 'K.pdf',
                        bbox_inches = 'tight', pad_inches = 0.1)


        
        
        x, index = assign_MY(dp)
        dsr, N, n = make_coord_MY(x, index)

        Ls = dsr.mars_solar_long.mean(dim = 'MY')
        year_mean = dsr.mean(dim = 'MY')
        year_mean = year_mean.chunk({'new_time' : 'auto'})
        year_mean = year_mean.rolling(new_time = 4, center = True)
            
        year_mean = year_mean.mean()
        mean_temp = year_mean.temp
        
        linestyle = '-'


        ct=axs[0].plot(Ls[:-5],mean_temp[:-5],color = col, label = labc,
                       linewidth = '1.1', linestyle = linestyle)
        
        fig.savefig(figpath + 'variability_all_' + hem + '_' + str(ilev) \
                     + 'K.pdf', bbox_inches = 'tight', pad_inches = 0.1)

        for j in n:
            di = dsr.where(dsr.MY == j, drop = True).squeeze()

            Zi = di.chunk({'new_time' : 'auto'})
            Zi = Zi.rolling(new_time = 4, center = True)

            Zi = Zi.mean()

            Ls = di.mars_solar_long
            labj = ""
            if exp[i] == 'soc_mars_mk36_per_value70.85_none_mld_2.0_all_years':
                if j == 3:
                    labj = "MY 28"
                    col = grey

            cti = axs[0].plot(Ls[:-5], Zi.temp[:-5], label = labj,
                           color = col, linewidth = '0.6',
                           linestyle = '-', alpha = 0.4)
            fig.savefig(figpath + 'variability_all_' + hem + '_' \
                        + str(ilev) + 'K.pdf',
                        bbox_inches = 'tight', pad_inches = 0.1)

        axs[0].legend(fontsize = 14, loc = 'upper left')
        axs[1].legend(fontsize = 14, loc = 'upper left')
        axs[2].legend(fontsize = 14, loc = 'upper left')

        fig.savefig(figpath + 'variability_all_' + hem + '_' + str(ilev) \
                     + 'K.pdf', bbox_inches = 'tight', pad_inches = 0.1)
