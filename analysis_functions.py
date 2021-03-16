'''
A selection of functions used in the analysis of OpenMARS and Isca data, for the given paper.
'''
# Written by Emily Ball, final version 16/03/2021
# Tested on Anthropocene

# NOTES and known problems
# Flexible to planetary parameters such as radius/gravity/rotation rate etc.

# Updates
# 16/03/2021 Upload to GitHub and commenting EB

# Begin script ========================================================================================================

import numpy as np
import xarray as xr
import dask
import os, sys

import metpy.interpolate
from metpy.units import units
import windspharm.xarray as windx
#import time
import scipy.optimize as so
from metpy.calc.tools import (broadcast_indices, find_bounding_indices,
                              _less_or_close)

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import (cm, colors)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#### planetary parameters, set for Mars. These will be the default values used
#### in any subsequent function

g       = 3.72076
p0      = 610. *units.Pa
kappa   = 1/4.4
omega   = 7.08822e-05
rsphere = 3.3962e6

def calculate_pfull(psurf, siglev):
    '''
    Calculates full pressures using surface pressures and sigma coordinates

    psurf  : array-like
            Surface pressures
    siglev : array-like
            Sigma-levels
    '''

    return psurf*siglev

def calculate_pfull_EMARS(ps,bk,ak):
    '''
    Calculates full pressures using surface pressures and sigma coordinates

    psurf  : array-like
            Surface pressures
    siglev : array-like
            Sigma-levels
    '''
    p_i = ps*bk + ak

    p = xr.zeros_like()
    p[k] = (p_i[k+1]-p_i[k])/np.log(p_i[k+1]/p_i[k])
    return 

def calculate_theta(tmp, plevs, **kwargs):
    '''
    Calculates potential temperature theta

    Input
    -----
    tmp   : temperature, array-like
    plevs : pressure levels, array-like
    p0    : reference pressure, optional. Default: 610. Pa
    kappa : optional. Default: 0.25
    '''
    p0 = kwargs.pop('p0', 610.)
    kappa = kwargs.pop('kappa', 1/4.4)

    ret = tmp * (p0/plevs)**kappa
    return ret
 
def wrapped_gradient(da, coord):
    '''
    Finds the gradient along a given dimension of a dataarray.
    '''

    dims_of_coord = da.coords[coord].dims
    if len(dims_of_coord) == 1:
        dim = dims_of_coord[0]
    else:
        raise ValueError('Coordinate ' + coord + ' has multiple dimensions: ' + str(dims_of_coord))

    coord_vals = da.coords[coord].values

    return xr.apply_ufunc(np.gradient, da, coord_vals, kwargs={'axis': -1},
                          input_core_dims=[[dim], [dim]],
                          output_core_dims=[[dim]],
                          dask='parallelized',
                          output_dtypes=[da.dtype])


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


class nf(float):
    def __repr__(self):
        s = f'{self:.1f}'
        return f'{self:.0f}' if s[-1] == '0' else s


def lait(PV,theta,theta0, **kwargs):
    '''
    Perform Lait scaling of PV
    kwargs
    ------
    kappa: R/c_p, optional, defaults to 0.25.
    '''
    kappa = kwargs.pop('kappa', 0.25)
    ret = PV*(theta/theta0)**(-(1+1/kappa))

    return ret



def calc_eddy_enstr(q):
    '''
    Calculate eddy enstrophy
    -------------
    Input:
    q : xarray DataArray with dimensions "lat","lon","time"
    Output:
    Z : xarray DataArray with dimensions "time"
    '''
    q = q.where(q.lon < 359.5, drop = True)
    qbar = q.mean(dim='lon')
    qbar = qbar.expand_dims({'lon':q.lon})

    qprime = q - qbar
    
    cosi = np.cos(np.pi/180 * (q.lat))


    qp2 = qprime ** 2 * cosi
    qpi = qp2.sum(dim = "lat")
    qp = qpi.sum(dim = "lon")
    
    tlat = np.tan(np.pi/180 * (q.lat[2] - q.lat[1])/2)
    tlon = np.tan(np.pi/180 * (q.lon[2] - q.lon[1])/2)
    
    Z = 1/(np.pi)*tlon*tlat*qp

    return Z
  
def calc_streamfn(lats, pfull, vz, **kwargs):
    '''
    Calculate meridional streamfunction from zonal mean meridional wind.
    
    Parameters
    ----------

    lats   : array-like, latitudes, units (degrees)
    pfull  : array-like, pressure levels, units (Pa)
    vz     : array-like, zonal mean meridional wind, dimensions (lat, pfull)
    radius : float, planetary radius, optional, default 3.39e6 m
    g      : float, gravity, optional, default 3.72 m s**-2

    Returns
    -------

    psi   : array-like, meridional streamfunction, dimensions (lat, pfull),
            units (kg/s)
    '''

    radius = kwargs.pop('radius', 3.39e6)
    g      = kwargs.pop('g', 3.72)

    coeff = 2 * np.pi * radius / g

    psi = np.empty_like(vz.values)
    for ilat in range(lats.shape[0]):
        psi[0, ilat] = coeff * np.cos(np.deg2rad(lats[ilat]))*vz[0, ilat] * pfull[0]
        for ilev in range(pfull.shape[0])[1:]:
            psi[ilev, ilat] = psi[ilev - 1, ilat] + coeff*np.cos(np.deg2rad(lats[ilat])) \
                              * vz[ilev, ilat] * (pfull[ilev] - pfull[ilev - 1])
    
    #psi = xr.DataArray(psi, coords = {"pfull" : pfull.values,
    #                                  "lat"   : lats.values})
    #psi.attrs['units'] = 'kg/s'

    return psi
  
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

def calc_Hadley_lat(u, lats, plot = False):
    '''
    Function to calculate location of 0 streamfunction.

    Parameters
    ----------

    u    : array-like
    lats : array-like. Default use will be to calculate jet on a given pressure
           level, but this array may also represent pressure level.

    Returns
    -------

    jet_lat : latitude (pressure level) of 0 streamfunction
    '''

    asign = np.sign(u)#.values)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    signchange[0] = 0

    

    for i in range(len(signchange)):
        if u[i] > 0 and i < len(signchange) - 4:
            continue
        signchange[i] = 0
    
    for i in range(len(signchange)):
        if signchange[i] == 0:
            continue
        u_0 = i
    
    if all(signchange[i] == 0 for i in range(len(signchange))):
        if u[0] > 0:
            u_0 = 0
        else:
            u_0 = 1

        #u_0 = np.where(u == np.ma.min(np.absolute(u)))[0][0]

    # Restrict to 10 points around maximum
    #u_0 = np.where(u == np.ma.min(np.absolute(u.values)))[0][0]
    if u_0 > 1:
        u_near = u[u_0-2:u_0+2]
        lats_near = lats[u_0-2:u_0+2]

        # Quartic fit, with smaller lat spacing
        coefs = np.ma.polyfit(lats_near,u_near,3)
        fine_lats = np.linspace(lats_near[0], lats_near[-1],300)
        quad = coefs[3]+coefs[2]*fine_lats+coefs[1]*fine_lats**2 \
                    +coefs[0]*fine_lats**3
        # Find jet lat and max
        #jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]

        minq = min(np.absolute(quad))
        jet_lat = fine_lats[np.where(np.absolute(quad) == minq)[0][0]]
        jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2
        # Plot fit?
        if plot:
            print (jet_max)
            print (jet_lat)
            plt.plot(lats_near, u_near)
            plt.plot(fine_lats, quad)
            plt.show()
    elif u_0 == 0:
        jet_lat = 90
        jet_max = u[-1]
    else:
        jet_lat = np.nan
        jet_max = np.nan

    return jet_lat, jet_max
