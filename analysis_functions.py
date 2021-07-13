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
import pandas as pd

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

    sumc = sum(cosi)
    
    Z = 1/(4*np.pi)* qp/sumc

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

    if u_max == 0:
        jet_lat = np.max(lats)
        jet_max = u[-1]
    else:
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
            u_0 = -1

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
        jet_max = coefs[3]+coefs[2]*jet_lat+coefs[1]*jet_lat**2 \
                    +coefs[0]*jet_lat**3

    elif u_0 == 0:
        jet_lat = 90
        jet_max = u[-1]

    elif u_0 == 1:
        u_near = u[u_0-1:u_0+2]
        lats_near = lats[u_0-1:u_0+2]

        # Quartic fit, with smaller lat spacing
        coefs = np.ma.polyfit(lats_near,u_near,2)
        fine_lats = np.linspace(lats_near[0], lats_near[-1],200)
        quad = coefs[2]+coefs[1]*fine_lats+coefs[0]*fine_lats**2 
        # Find jet lat and max
        #jet_lat = fine_lats[np.where(quad == max(quad))[0][0]]

        minq = min(np.absolute(quad))
        jet_lat = fine_lats[np.where(np.absolute(quad) == minq)[0][0]]
        jet_max = coefs[2]+coefs[1]*jet_lat+coefs[0]*jet_lat**2

    else:
        jet_lat = np.nan
        jet_max = np.nan

    return jet_lat, jet_max
  


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
  
def filestrings(exp, filepath, start, end, filename, **kwargs):
    '''
    Generates lists of strings, for Isca runs.
    '''
    outpath = kwargs.pop('outpath', filepath)

    if start<10:
        st='000'+str(start)
    elif start<100:
        st='00'+str(start)
    elif start<1000:
        st='0'+str(start)
    else:
        st=str(start)

    if end<10:
        en='000'+str(end)
    elif end<100:
        en='00'+str(end)
    elif end<1000:
        en='0'+str(end)
    else:
        en=str(end)


    nfiles = end - start + 1
    infiles = []
    runs = []
    out = []

    for i in range(nfiles):
        run_no = start+i
        if run_no<10:
            run='run000'+str(run_no)
        elif run_no<100:
            run='run00'+str(run_no)
        elif run_no<1000:
            run='run0'+str(run_no)
        else:
            run='run'+str(run_no)

        runs.append(run)
        out.append(outpath +'/'+exp+'/'+run+'/'+filename)
        infiles.append(filepath+'/'+exp+'/'+run+'/'+filename)

    return runs, out, infiles
  
def stereo_plot():
    '''
    Returns variables to define a circular plot in matplotlib.
    '''
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    return theta, center, radius, verts, circle

def make_stereo_plot(ax, lats, lons, circle, **kwargs):
    '''
    Makes the polar stereographic plot and plots gridlines at choice of lats
    and lons.
    '''
    linewidth = kwargs.pop('linewidth', 1)
    linestyle = kwargs.pop('linestyle', '-')
    color = kwargs.pop('color', 'black')
    alpha = kwargs.pop('alpha', 1)

    gl = ax.gridlines(crs = ccrs.PlateCarree(), linewidth = linewidth,
                      linestyle = linestyle, color = color, alpha = alpha)

    ax.set_boundary(circle, transform=ax.transAxes)

    gl.ylocator = ticker.FixedLocator(lats)
    gl.xlocator = ticker.FixedLocator(lons)

def make_colourmap(vmin, vmax, step, **kwargs):
    '''
    Makes a colormap from ``vmin`` (inclusive) to ``vmax`` (exclusive) with
    boundaries incremented by ``step``. Optionally includes choice of color and
    to extend the colormap.
    '''
    col = kwargs.pop('col', 'viridis')
    extend = kwargs.pop('extend', 'both')

    boundaries = list(np.arange(vmin, vmax, step))

    if extend == 'both':
        cmap_new = cm.get_cmap(col, len(boundaries) + 1)
        colours = list(cmap_new(np.arange(len(boundaries) + 1)))
        cmap = colors.ListedColormap(colours[1:-1],"")
        cmap.set_over(colours[-1])
        cmap.set_under(colours[0])

    elif extend == 'max':
        cmap_new = cm.get_cmap(col, len(boundaries))
        colours = list(cmap_new(np.arange(len(boundaries))))
        cmap = colors.ListedColormap(colours[:-1],"")
        cmap.set_over(colours[-1])

    elif extend == 'min':
        cmap_new = cm.get_cmap(col, len(boundaries))
        colours = list(cmap_new(np.arange(len(boundaries))))
        cmap = colors.ListedColormap(colours[1:],"")
        cmap.set_under(colours[0])

    norm = colors.BoundaryNorm(boundaries, ncolors = len(boundaries) - 1,
                               clip = False)

    return boundaries, cmap_new, colours, cmap, norm
  
def assign_MY(d):
    '''
    Calculates new MY for Isca simulations and adds this to input dataset.
    Also returns the indices of the time axis that correspond to a new MY.
    '''
    t = np.zeros_like(d.time)
    index=[]
    for i in range(len(t)-1):
        if d.mars_solar_long[i+1]<d.mars_solar_long[i]:
            print(d.mars_solar_long[i].values)
            print(d.mars_solar_long[i+1].values)
            t[i+1] = t[i]+1
            index.append(d.time[i])
        else:
            t[i+1] = t[i]
    t1 = xr.Dataset({"MY" : (("time"), t)},
                    coords = {"time":d.time})
    d = d.assign(MY=t1["MY"])
    return d, index

def make_coord_MY(x, index):
    x = x.where(x.time > index[0], drop=True)
    x = x.where(x.time <= index[-1], drop=True)

    N=int(np.max(x.MY))
    n = range(N)

    y = x.time[:len(x.time)//N]

    ind = pd.MultiIndex.from_product((n,y),names=('MY','new_time'))
    dsr = x.assign_coords({'time':ind}).unstack('time')
    #dsr = dsr.squeeze()

    return dsr, N, n
  
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

