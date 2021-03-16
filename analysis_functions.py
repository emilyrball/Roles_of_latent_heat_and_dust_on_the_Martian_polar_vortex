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
