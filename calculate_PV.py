# Python 'package' to calculate potential vorticity and other metereological quantities
# Extra functionality is to interpolate to isentropic levels
# No output. Package must be loaded into another script to call functions
# Written by Emily Ball, final version 13/05/2020
# Tested on Anthropocene

# NOTES and known problems
# Flexible to planetary parameters such as radius/gravity/rotation rate etc.
# Heavily based on the Python package MetPy (https://github.com/Unidata/MetPy)
# Requires global data for the dependency on Windspharm package (https://ajdawson.github.io/windspharm/latest/)

# Updates
# 02/11/2020 Upload to GitHub EB

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
    r"""Calculates full pressures using surface pressures and sigma coordinates

    psurf  : array-like
            Surface pressures
    siglev : array-like
            Sigma-levels
    """

    return psurf*siglev

def calculate_pfull_EMARS(ps,bk,ak):
    r"""Calculates full pressures using surface pressures and sigma coordinates

    psurf  : array-like
            Surface pressures
    siglev : array-like
            Sigma-levels
    """
    p_i = ps*bk + ak

    p = xr.zeros_like()
    p[k] = (p_i[k+1]-p_i[k])/np.log(p_i[k+1]/p_i[k])
    return 



def interpolate_1d(x, xp, *args, **kwargs):
    r"""Interpolates data with any shape over a specified axis.
    Interpolation over a specified axis for arrays of any shape.
    Parameters
    ----------
    x : array-like
        1-D array of desired interpolated values.
    xp : array-like
        The x-coordinates of the data points.
    args : array-like
        The data to be interpolated. Can be multiple arguments, all must
        be the same shape as xp.
    axis : int, optional
        The axis to interpolate over. Defaults to 0.
    fill_value: float, optional
        Specify handling of interpolation points out of data bounds. If
        None, will return ValueError if points are out of bounds. Defaults
        to nan.
    return_list_always: bool, optional
        Whether to always return a list of interpolated arrays, even when
        only a single array is passed to `args`. Defaults to ``False``.

    Returns
    -------
    array-like
        Interpolated values for each point with coordinates sorted in
        ascending order.

    Notes
    -----
    xp and args must be the same shape.

    """
    # Pull out keyword args
    fill_value = kwargs.pop('fill_value', np.nan)
    axis = kwargs.pop('axis', 0)
    return_list_always = kwargs.pop('return_list_always', False)

    # Handle units
    x, xp = _strip_matching_units(x, xp)

    # Make x an array
    x = np.asanyarray(x).reshape(-1)

    # Save number of dimensions in xp
    ndim = xp.ndim

    # Sort input data
    sort_args = np.argsort(xp, axis=axis)
    sort_x = np.argsort(x)

    # indices for sorting
    sorter = broadcast_indices(xp, sort_args, ndim, axis)

    # sort xp
    xp = xp.values[sorter]
    # Ensure pressure in increasing order
    variables = [arr.values[sorter] for arr in args]

    # Make x broadcast with xp
    x_array = x[sort_x]
    expand = [np.newaxis] * ndim
    expand[axis] = slice(None)
    x_array = x_array[tuple(expand)]

    # Calculate value above interpolated value
    minv = np.apply_along_axis(np.searchsorted, axis, xp, x[sort_x])
    minv2 = np.copy(minv)

    # If fill_value is none and data is out of bounds, raise value error
    if ((np.max(minv) == xp.shape[axis]) or (np.min(minv) == 0)) and fill_value is None:
        raise ValueError(
            'Interpolation point out of data bounds encountered')

    # Warn if interpolated values are outside data bounds, will make these
    #the values at end of data range.
    if np.max(minv) == xp.shape[axis]:
        warnings.warn(
            'Interpolation point out of data bounds encountered')
        minv2[minv == xp.shape[axis]] = xp.shape[axis] - 1
    if np.min(minv) == 0:
        minv2[minv == 0] = 1

    # Get indices for broadcasting arrays
    above = broadcast_indices(xp, minv2, ndim, axis)
    below = broadcast_indices(xp, minv2 - 1, ndim, axis)

    if np.any(x_array < xp[below]):
        warnings.warn(
            'Interpolation point out of data bounds encountered')

    # Create empty output list
    ret = []

    # Calculate interpolation for each variable
    for var in variables:
        # Var needs to be on the *left* of the multiply to ensure that if
        # it's a pint Quantity, it gets to control the operation--at least
        # until we make sure masked arrays and pint play together better.
        # See https://github.com/hgrecco/pint#633
        var_interp=var[below]+(var[above]-var[below])*((x_array-xp[below])/(xp[above]-xp[below]))

        # Set points out of bounds to fill value.
        var_interp[minv == xp.shape[axis]] = fill_value
        var_interp[x_array < xp[below]] = fill_value

        # Check for input points in decreasing order and return output to
        # match.
        if x[0] > x[-1]:
            var_interp = np.swapaxes(
                np.swapaxes(var_interp, 0, axis)[::-1], 0, axis)
        # Output to list
        ret.append(var_interp)

    if return_list_always or len(ret) > 1:
        return ret
    else:
        return ret[0]



def log_interpolate_1d(x, xp, *args, **kwargs):
    r"""Interpolates data with logarithmic x-scale over a specified axis.
    Interpolation on a logarithmic x-scale for interpolation values in
    pressure coordintates.
    Parameters
    ----------
    x : array-like
        1-D array of desired interpolated values.
    xp : array-like
        The x-coordinates of the data points.
    args : array-like
        The data to be interpolated. Can be multiple arguments, all must
        be the same shape as xp.
    axis : int, optional
        The axis to interpolate over. Defaults to 0.
    fill_value: float, optional
        Specify handling of interpolation points out of data bounds. If
        None, will return ValueError if points are out of bounds. Defaults
        to nan.

    Returns
    -------
    array-like
        Interpolated values for each point with coordinates sorted in
        ascending order.

    Notes
    -----
    xp and args must be the same shape.
    """
    # Pull out kwargs
    fill_value = kwargs.pop('fill_value', np.nan)
    axis = kwargs.pop('axis', 0)

    # Handle units
    x, xp = _strip_matching_units(x, xp)

    # Log x and xp
    log_x = np.log(x)
    log_xp = np.log(xp)
    return interpolate_1d(log_x, log_xp,
                          *args, axis=axis, fill_value=fill_value)



def _strip_matching_units(*args):
    """Ensure arguments have same units and return with units stripped.

    Replaces `@units.wraps(None, ('=A', '=A'))`, which breaks with `*args`
    handling for pint>=0.9.
    """
    if all(hasattr(arr, 'units') for arr in args):
        return [arr.values for arr in args]
    else:
        return args



def calculate_theta(tmp, plevs, **kwargs):
    r"""Calculates potential temperature theta

    Input
    -----
    tmp   : temperature, array-like
    plevs : pressure levels, array-like
    p0    : reference pressure, optional. Default: 610. Pa
    kappa : optional. Default: 0.25
    """
    p0 = kwargs.pop('p0', 610.)
    kappa = kwargs.pop('kappa', 1/4.4)

    ret = tmp * (p0/plevs)**kappa
    return ret
 


def wrapped_gradient(da, coord):
    """Finds the gradient along a given dimension of a dataarray."""

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



def pot_vort(uwnd, vwnd, theta, axis, **kwargs):
    r"""Calculate potential vorticity on isobaric levels. Requires input uwnd,
    vwnd and tmp arrays to have (lat, lon, ...) format for Windspharm.

    Input
    -----
    uwnd    : zonal winds, array-like
    vwnd    : meridional winds, array-like
    tmp     : temperature, array-like
    theta   : potential temperature, array-like
    axis    : name of pressure axis eg 'plev' or 'pfull' or 'level'
    omega   : planetary rotation rate, optional
    g       : planetary gravitational acceleration, optional
    rsphere : planetary radius, in metres, optional
    """
    omega = kwargs.pop('omega', 7.08822e-05)
    g = kwargs.pop('g', 3.72076)
    rsphere = kwargs.pop('rsphere', 3.3962e6)
    w = windx.VectorWind(uwnd.fillna(0), vwnd.fillna(0), rsphere = rsphere)

    relvort = w.vorticity()
    relvort = relvort.where(relvort!=0, other=np.nan)
    planvort = w.planetaryvorticity(omega=omega)
    absvort = relvort + planvort

    dthtady, dthtadx = w.gradient(theta.fillna(0))
    dthtadx = dthtadx.where(dthtadx!=0, other=np.nan)
    dthtady = dthtady.where(dthtady!=0, other=np.nan)

    dthtadp = wrapped_gradient(theta, axis)
    dudp = wrapped_gradient(uwnd, axis)
    dvdp = wrapped_gradient(vwnd, axis)


    s = - dthtadp

    PV1 = dvdp * dthtadx - dudp * dthtady
    PV2 = PV1 / s
    PV3 = absvort + PV2
    ret  = g * PV3 * s

    return ret



def isent_interp(isentlevs, plev, tmp_iso, *args, **kwargs):
    r"""Interpolate data in isobaric coordinates to isentropic coordinates.
    
    Parameters
    ----------
    isentlevs : array
        1D array of desired theta surfaces
    plev : array
        1D array of pressure levels
    tmp_iso : array
        array of temperatures on isobaric levels
    args : array, optional
        any additional variables to interpolate to each isentropic level
    
    Returns
    -------
    list
        List with pressure at each isentropic level, followed by each additional
        argument interpolated to isentropic coordinates.
    
    Other Parameters
    ----------------
    axis : int, optional
        axis corresponding to vertical in temperature array, Default=0
    max_iters : int, optional
        max number of iterations to use in calculation, Default=50
    eps : float, optional
        desired absolute error in calculated value, Default=1e-6
    p0 : float, optional
        constant reference pressure, Default=6.hPa (Mars)
    kappa : float, optional
        constant Poisson exponent, Rd/Cp_d. Default=0.222 (Mars)
    bottom_up_search : bool, optional
        Controls whether to search for theta levels bottom-up, or top-down. Defaults to
        True, which is bottom-up search.
    temperature_out : bool, optional
        If true, calculates temperature and output as last item in output list. Defaults
        to False.
    
    """
    def _isen_iter(iter_log_p, isentlevs_nd, kappa, a, b, pok):
        exner = pok * np.exp(-kappa * iter_log_p)
        t = a * iter_log_p + b
        # Newton-Raphson iteration
        f = isentlevs_nd - t * exner
        fp = exner * (kappa * t - a)
        return iter_log_p - (f / fp)
    
    def _potential_temperature(levs,tmpk, p0, kappa):
        return tmpk/_exner_function(levs, p0, kappa)
    
    def _exner_function(levs, p0, kappa):
        return (levs/p0.magnitude)**kappa
    
    def interpolate_1d(x, xp, *args, **kwargs):
        r"""Interpolates data with any shape over a specified axis.
        Interpolation over a specified axis for arrays of any shape.
        Parameters
        ----------
        x : array-like
            1-D array of desired interpolated values.
        xp : array-like
            The x-coordinates of the data points.
        args : array-like
            The data to be interpolated. Can be multiple arguments, all must
            be the same shape as xp.
        axis : int, optional
            The axis to interpolate over. Defaults to 0.
        fill_value: float, optional
            Specify handling of interpolation points out of data bounds. If
            None, will return ValueError if points are out of bounds. Defaults
            to nan.
        return_list_always: bool, optional
            Whether to always return a list of interpolated arrays, even when
            only a single array is passed to `args`. Defaults to ``False``.
    
        Returns
        -------
        array-like
            Interpolated values for each point with coordinates sorted in
            ascending order.
    
        Notes
        -----
        xp and args must be the same shape.
    
        """
        # Pull out keyword args
        fill_value = kwargs.pop('fill_value', np.nan)
        axis = kwargs.pop('axis', 0)
        return_list_always = kwargs.pop('return_list_always', False)
    
        # Handle units
        x, xp = _strip_matching_units(x, xp)
    
        # Make x an array
        x = np.asanyarray(x).reshape(-1)
    
        # Save number of dimensions in xp
        ndim = xp.ndim
    
        # Sort input data
        sort_args = np.argsort(xp, axis=axis)
        sort_x = np.argsort(x)
    
        # indices for sorting
        sorter = broadcast_indices(xp, sort_args, ndim, axis)
    
        # sort xp
        xp = xp[sorter]
        # Ensure pressure in increasing order
        variables = [arr[sorter] for arr in args]
    
        # Make x broadcast with xp
        x_array = x[sort_x]
        expand = [np.newaxis] * ndim
        expand[axis] = slice(None)
        x_array = x_array[tuple(expand)]
    
        # Calculate value above interpolated value
        minv = np.apply_along_axis(np.searchsorted, axis, xp, x[sort_x])
        minv2 = np.copy(minv)
    
        # If fill_value is none and data is out of bounds, raise value error
        if ((np.max(minv) == xp.shape[axis]) or (np.min(minv) == 0)) and fill_value is None:
            raise ValueError(
                'Interpolation point out of data bounds encountered')
    
        # Warn if interpolated values are outside data bounds, will make these
        #the values at end of data range.
        if np.max(minv) == xp.shape[axis]:
            warnings.warn(
                'Interpolation point out of data bounds encountered')
            minv2[minv == xp.shape[axis]] = xp.shape[axis] - 1
        if np.min(minv) == 0:
            minv2[minv == 0] = 1
    
        # Get indices for broadcasting arrays
        above = broadcast_indices(xp, minv2, ndim, axis)
        below = broadcast_indices(xp, minv2 - 1, ndim, axis)
    
        if np.any(x_array < xp[below]):
            warnings.warn(
                'Interpolation point out of data bounds encountered')
    
        # Create empty output list
        ret = []
    
        # Calculate interpolation for each variable
        for var in variables:
            # Var needs to be on the *left* of the multiply to ensure that if
            # it's a pint Quantity, it gets to control the operation--at least
            # until we make sure masked arrays and pint play together better.
            # See https://github.com/hgrecco/pint#633
            var_interp=var[below]+(var[above]-var[below])*((x_array-xp[below])/(xp[above]-xp[below]))
    
            # Set points out of bounds to fill value.
            var_interp[minv == xp.shape[axis]] = fill_value
            var_interp[x_array < xp[below]] = fill_value
    
            # Check for input points in decreasing order and return output to
            # match.
            if x[0] > x[-1]:
                var_interp = np.swapaxes(
                    np.swapaxes(var_interp, 0, axis)[::-1], 0, axis)
            # Output to list
            ret.append(var_interp)
    
        if return_list_always or len(ret) > 1:
            return ret
        else:
            return ret[0]

    #def log_interpolate_1d(x, xp, *args, **kwargs):
    #    r"""Interpolates data with logarithmic x-scale over a specified axis.
    #    Interpolation on a logarithmic x-scale for interpolation values in
    #    pressure coordintates.
    #    Parameters
    #    ----------
    #    x : array-like
    #        1-D array of desired interpolated values.
    #    xp : array-like
     ##       The x-coordinates of the data points.
    #    args : array-like
    #        The data to be interpolated. Can be multiple arguments, all must
    ##        be the same shape as xp.
    #    axis : int, optional
    #        The axis to interpolate over. Defaults to 0.
    #    fill_value: float, optional
     #       Specify handling of interpolation points out of data bounds. If
    #        None, will return ValueError if points are out of bounds. Defaults
    #        to nan.
    
    #    Returns
    #    -------
    #    array-like
    #        Interpolated values for each point with coordinates sorted in
    #        ascending order.
    
    #    Notes
    #    -----
    #    xp and args must be the same shape.
    #    """
    #    # Pull out kwargs
    #    fill_value = kwargs.pop('fill_value', np.nan)
    #    axis = kwargs.pop('axis', 0)
    
        # Handle units
    #    x, xp = _strip_matching_units(x, xp)
    
        # Log x and xp
    #    log_x = np.log(x)
    #    log_xp = np.log(xp)
    #    return interpolate_1d(log_x, log_xp,
    #                          *args, axis=axis, fill_value=fill_value)
    
    def _strip_matching_units(*args):
        """Ensure arguments have same units and return with units stripped.
    
        Replaces `@units.wraps(None, ('=A', '=A'))`, which breaks with `*args`
        handling for pint>=0.9.
        """
        if all(hasattr(arr, 'units') for arr in args):
            return [arr.values for arr in args]
        else:
            return args
    
    
    max_iters = kwargs.pop('max_iters', 50)
    eps = kwargs.pop('eps', 1e-6)
    axis = kwargs.pop('axis', 0)
    p0 = kwargs.pop('p0', 610.*units.Pa)
    kappa = kwargs.pop('kappa', 0.25)
    bottom_up_search = kwargs.pop('bottom_up_search', True)
    temperature_out = kwargs.pop('temperature_out', False)
    
    #if tmp_iso.units != 'K':
    #    raise ValueError('Temperature must be in K')
        
    #if plev.units != 'Pa':
    #    raise ValueError('Pressure must be in Pa')
    #else:
    pres = plev
    
    ndim = tmp_iso.ndim
    slices = [np.newaxis] * ndim
    slices[axis] = slice(None)
    slices = tuple(slices)
    pres = np.broadcast_to(pres.values[slices], tmp_iso.shape)               # is this doing the same thing as MetPy?
    
    ## sort input data    
    sort_pres = np.argsort(pres, axis=axis)
    sort_pres = np.swapaxes(np.swapaxes(sort_pres, 0, axis)[::-1], 0, axis)
    sorter = broadcast_indices(pres, sort_pres, ndim, axis)
    levs = pres[sorter]
    tmpk = tmp_iso.values[sorter]
    
    isentlevs = np.asanyarray(isentlevs).reshape(-1)
    isentlevels = isentlevs[np.argsort(isentlevs)]
    
    # Make the desired isentropic levels the same shape as temperature
    shape = list(tmp_iso.shape)
    shape[axis] = isentlevels.size
    isentlevs_nd = np.broadcast_to(isentlevels[slices], shape)
    
    pres_theta = _potential_temperature(levs,tmpk, p0, kappa)
    
    if pres_theta.max() < np.max(isentlevs):
        raise ValueError('Input theta level out of data bounds')
        
    # Find log of pressure to implement assumption of linear temperature dependence on
    # ln(p)
    log_p = np.log(levs)
    
    pok=p0**kappa
    
    # index values for each point for the pressure level nearest to the desired theta level
    above, below, good = find_bounding_indices(pres_theta, isentlevs, axis,
                                                           from_below=bottom_up_search)
    
    # calculate constants for the interpolation
    a = (tmpk[above] - tmpk[below]) / (log_p[above] - log_p[below])
    b = tmpk[above] - a * log_p[above]
    
    isentprs = 0.5 * (log_p[above] + log_p[below])
    good &= ~np.isnan(a)
    
    log_p_solved = np.empty_like(a[good])
    
    #for i in range(len(a[good])):
    #    log_p_solved[i] = so.fixed_point(_isen_iter, isentprs[good][i],
    #                              args=(isentlevs_nd[good][i], kappa, a[good][i], b[good][i], pok),
    #                              xtol=eps, maxiter=max_iters)
    
    log_p_solved = so.fixed_point(_isen_iter, isentprs[good],
                                  args=(isentlevs_nd[good], kappa, a[good], b[good], pok.m),
                                  xtol=eps, maxiter=max_iters)

    isentprs[good]=np.exp(log_p_solved)
    isentprs[~(good & _less_or_close(isentprs,np.max(pres)))] = np.nan
    # _less_or_close returns (isentprs < np.max(pres) | np.isclose(isentprs, np.max(pres))
    
    ret = [isentprs]
    
    if temperature_out:
        ret.append(isentlevs_nd / ((p0.magnitude / isentprs) ** kappa))

    # do an interpolation for each additional argument
    if args:
        others = interpolate_1d(isentlevels, pres_theta, *(arr.values[sorter] for arr in args),
                                axis=axis, return_list_always=True)
        ret.extend(others)
    
    return ret



def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)



class nf(float):
    def __repr__(self):
        s = f'{self:.1f}'
        return f'{self:.0f}' if s[-1] == '0' else s



def map_inst_PV_MACDA(lvl, d_isent, Ls, MY, directory, **kwargs):
    r""" map of instantaneous PV on isentropic level from MACDA dataset,
    North polar stereographic projection.
    
    Input
    -----
    lvl     : integer to index isentropic levels
    time0   : integer to index time
    d_isent : dataset containing PV, uwnds on isentropic surface(s)
    Ls      : solar longitude, array
    MY      : Martian year, array
    """
    time0 = kwargs.pop('time0', False)
    d_isent = d_isent.sel(lat=d_isent.lat[50<d_isent.lat])
    d_isent = d_isent.sel(lat=d_isent.lat[d_isent.lat<88])

    fig, ax = plt.subplots(1, figsize = (10,10),
                           subplot_kw = {'projection':ccrs.NorthPolarStereo()})

    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)

    if time0:
        newtime = d_isent.time[time0]

        strLs = str(int(10*Ls.sel(time=newtime))/10)
        strMY = str(int(10*MY.sel(time=newtime))/10)
        strlev = str(int(d_isent.ilev[lvl]))
    
        vmax = d_isent.PV[time0,lvl,:,:].max().values * 10**4
        vmin = d_isent.PV[time0,lvl,:,:].min().values * 10**4

        norm = colors.Normalize(vmin,vmax)
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap='plasma'),ax=ax,
                 label='PV (10$^{-4}$ K m$^2$ kg$^{-1}$ s$^{-1}$)',
                 orientation='horizontal', extend='both', shrink=0.5)
        
        ax.contourf(d_isent.lon,d_isent.lat,d_isent.PV[time0,lvl,:,:]*10**4,
                          40, cmap = 'plasma', transform=ccrs.PlateCarree(),
                          vmin=vmin, vmax=vmax)

        ax.contour(d_isent.lon, d_isent.lat, d_isent.uwnd[time0,lvl,:,:],
                          colors = 'black', transform=ccrs.PlateCarree(),
                          linewidths = 1)
    
    else:
        d_isent=d_isent.squeeze()
        Ls=Ls.squeeze()
        MY=MY.squeeze()
        strLs = str(int(10*Ls.values)/10)
        strMY = str(int(10*MY.values)/10)
        strlev = str(int(d_isent.ilev[lvl]))

        vmax = d_isent.PV[lvl,:,:].max().values * 10**4
        vmin = d_isent.PV[lvl,:,:].min().values * 10**4

        norm = colors.Normalize(vmin,vmax)
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap='plasma'),ax=ax,
                 label='PV (10$^{-4}$ K m$^2$ kg$^{-1}$ s$^{-1}$)',
                 orientation='horizontal', extend='both', shrink=0.5)
        
        ax.contourf(d_isent.lon,d_isent.lat,d_isent.PV[lvl,:,:]*10**4,
                          40, cmap = 'plasma', transform=ccrs.PlateCarree(),
                          vmin=vmin, vmax=vmax)

        c0 = ax.contour(d_isent.lon, d_isent.lat, d_isent.uwnd[lvl,:,:],
                          colors = 'black', transform=ccrs.PlateCarree(),
                          linewidths = 1)
    

    gl0 = ax.gridlines(crs=ccrs.PlateCarree(),linewidth=1,
                 linestyle='--',color='black',alpha=0.3)
    gl0.xlocator = ticker.FixedLocator([-180,-120,-60,0,60,120,180])
    gl0.ylocator = ticker.FixedLocator([90,80,60,50])

    # Recast levels to new class
    c0.levels = [nf(val) for val in c0.levels]
    # Label levels with specially formatted floats
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    ax.clabel(c0, c0.levels, inline=1, fmt=fmt, fontsize=10)

    plt.title('MACDA, PV on '+strlev+'K surface at MY'+strMY+', Ls'+strLs)
    plt.savefig(directory+'PV_'+strlev+'K_MY'+strMY+'Ls'+strLs+'.png')



def map_inst_PV_Isca(lvl, time0, d_isent, axis, Ls, directory, exp):
    r""" map of instantaneous PV on isentropic level from Isca simulations,
    North polar stereographic projection.
    
    Input
    -----
    lvl     : integer to index isentropic levels
    time0   : integer to index time
    d_isent : dataset containing PV, uwnds on isentropic surface(s)
    axis    : 0 = PV
              1 = laitPV
    Ls      : solar longitude, array
    """
    strLs = str(int(Ls[time0]*10)/10)
    strlev = str(int(d_isent.ilev[lvl]))

    fig, ax = plt.subplots(1, figsize = (10,10),
                           subplot_kw = {'projection':ccrs.SouthPolarStereo()})
    if axis == 0:

        ax.set_extent([-180,180,-90,-50], crs=ccrs.PlateCarree())
        vmax = d_isent.PV[lvl,time0,:,:].max() * 10**6
        #vmin = d_isent.PV[lvl,time0,33:64,:].min() * 10**5
        vmin = 0
        norm = colors.Normalize(vmin,vmax)
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap='coolwarm'),ax=ax,
                 label='PV (10$^{-6}$ K m$^2$ kg$^{-1}$ s$^{-1}$)',
                 extend='max', shrink=0.82)
        
        ax.contourf(d_isent.lon,d_isent.lat,d_isent.PV[lvl,time0,:,:]*10**6,
                          40, cmap = 'coolwarm', transform=ccrs.PlateCarree(),
                          vmin=vmin, vmax=vmax)

        ax.contour(d_isent.lon, d_isent.lat, d_isent.uwnd[lvl,time0,:,:],
                          colors = 'black', transform=ccrs.PlateCarree(),
                          linewidths = 1)

        plt.title('Isca, PV on '+strlev+'K surface at Ls '+strLs)
        plt.savefig(directory+'/'+exp+'/PV_'+strlev+'K_Ls_'+strLs+'.png')

    else:
        vmax = d_isent.scaledPV[lvl,time0,:,:].max() * 10**5
        #vmin = d_isent.PV[lvl,time0,33:64,:].min() * 10**5
        vmin = 0
        norm = colors.Normalize(vmin,vmax)
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap='plasma'),ax=ax,
                 label='Scaled PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1})$',
                 extend='max', shrink=0.82)
        
        ax.contourf(d_isent.lon,d_isent.lat,
                    d_isent.scaledPV[lvl,time0,:,:]*10**5,40, cmap = 'plasma',
                    transform=ccrs.PlateCarree(),vmin=vmin, vmax=vmax)

        ax.contour(d_isent.lon, d_isent.lat, d_isent.uwnd[lvl,time0,:,:],
                          colors = 'black', transform=ccrs.PlateCarree(),
                          linewidths = 1)

        plt.title('Isca, PV on '+strlev+'K surface at Ls '+strLs)
        plt.savefig(directory+'/PV_'+strlev+'K_Ls_'+strLs+'.png')



def lait(PV,theta,theta0, **kwargs):
    r"""Perform Lait scaling of PV
    kwargs
    ------
    kappa: R/c_p, optional, defaults to 0.25.
    """
    kappa = kwargs.pop('kappa', 0.25)
    ret = PV*(theta/theta0)**(-(1+1/kappa))

    return ret



def waugh_plot_macda(d_iso, Ls):
    p = d_iso.PVlait.mean(dim='time').mean(dim='lon') *10**5
    t = d_iso.theta.mean(dim='time').mean(dim='lon')
    u = d_iso.uwnd.mean(dim='time').mean(dim='lon')
    plev = d_iso.plev
    lat = d_iso.lat
    
    #extract dates
    strLs1 = str(int(10*Ls[0])/10)
    strLs2 = str(int(10*Ls[-1])/10)
    #strMY = str(int(10*MY.sel(time=newtime))/10)

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.set_yscale('log')
    #ticker.LogFormatter(base=10.0, labelOnlyBase=False,
    #                               minor_thresholds=None, linthresh=None)
    ax.set_xlim([0,90])
    ax.set_ylim([10,0.005])
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.xlabel('latitude (degrees)')
    plt.ylabel('pressure (hPa)')

    #vmax = p[:,0:18].max().values
    #vmin = p[:,0:18].min().values
    vmax = 65.
    vmin = -5.

    cmap = cm.OrRd
    boundaries = [-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65]
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='OrRd'),ax=ax,extend='max',
                 label='Scaled PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)')


    ax.contourf(lat, plev/100, p.transpose('plev','lat'),
                levels = [-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,100],
                cmap='OrRd', vmin=vmin, vmax=vmax)
    ax.contour(lat, plev/100, t.transpose('plev','lat'),
                levels=[200,300,400,500,600,700,800,900,1000,1100],
                linestyles = '--', colors='black', linewidths=1)
    cs3 = ax.contour(lat, plev/100, u.transpose('plev','lat'),
                     levels=[0,50,100,150], colors='black',linewidths=1)

    # Recast levels to new class
    cs3.levels = [nf(val) for val in cs3.levels]

    # Label levels with specially formatted floats
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    ax.clabel(cs3, cs3.levels, inline=1, fmt=fmt, fontsize=10)
    
    #plt.title('PV on '+strlev+'K surface')
    #plt.savefig(home+'/figs/19790220_PV_'+strlev+'Ksurface.png')
    plt.savefig('Figs/MACDA_Waugh2016_Fig5_Ls'+strLs1+'-'+strLs2+'.png')


def waugh_plot_emars(d, Ls):
    p = d.PVlait.mean(dim='time').mean(dim='lon') *10**5
    t = d.thta.mean(dim='time').mean(dim='lon')
    u = d.u.mean(dim='time').mean(dim='lon')
    plev = d.pfull
    lat = d.lat/87.5*90
    
    #extract dates
    strLs1 = str(int(10*Ls[0])/10)
    strLs2 = str(int(10*Ls[-1])/10)
    #strMY = str(int(10*MY.sel(time=newtime))/10)

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.set_yscale('log')
    #ticker.LogFormatter(base=10.0, labelOnlyBase=False,
    #                               minor_thresholds=None, linthresh=None)
    ax.set_xlim([0,87])
    ax.set_ylim([10,0.005])
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.xlabel('latitude (degrees)')
    plt.ylabel('pressure (hPa)')

    #vmax = p[:,0:18].max().values
    #vmin = p[:,0:18].min().values
    vmax = 65.
    vmin = -5.

    cmap = cm.OrRd
    boundaries = [-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65]
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='OrRd'),ax=ax,extend='max',
                 label='Scaled PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)')


    ax.contourf(lat, plev/100, p.transpose('pfull','lat'),
                levels = [-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,150],
                cmap='OrRd', vmin=vmin, vmax=vmax)
    ax.contour(lat, plev/100, t.transpose('pfull','lat'),
                levels=[200,300,400,500,600,700,800,900,1000,1100],
                linestyles = '--', colors='black', linewidths=1)
    cs3 = ax.contour(lat, plev/100, u.transpose('pfull','lat'),
                     levels=[0,50,100,150], colors='black',linewidths=1)

    # Recast levels to new class
    cs3.levels = [nf(val) for val in cs3.levels]

    # Label levels with specially formatted floats
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    ax.clabel(cs3, cs3.levels, inline=1, fmt=fmt, fontsize=10)
    
    #plt.title('PV on '+strlev+'K surface')
    #plt.savefig(home+'/figs/19790220_PV_'+strlev+'Ksurface.png')
    plt.savefig('Figs/EMARS_Waugh2016_Fig5_Ls'+strLs1+'-'+strLs2+'.png')




def waugh_plot_isca(d, Ls, directory, exp):
    p = d.PVlait.mean(dim='time').mean(dim='lon') *10**5
    t = d.theta.mean(dim='time').mean(dim='lon')
    u = d.ucomp.mean(dim='time').mean(dim='lon')
    plev = d.pfull
    lat = d.lat
    
    #extract dates
    strLs1 = str(int(10*Ls[0])/10)
    strLs2 = str(int(10*Ls[-1])/10)
    #strMY = str(int(10*MY.sel(time=newtime))/10)

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.set_yscale('log')
    #ticker.LogFormatter(base=10.0, labelOnlyBase=False,
    #                               minor_thresholds=None, linthresh=None)
    ax.set_xlim([0,90])
    ax.set_ylim([10,0.005])
    
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xlabel('latitude (degrees)')
    plt.ylabel('pressure (hPa)')

    #vmax = p.max().values/2
    #vmin = p.min().values
    #vmax = 65
    vmax = 90
    vmin = -5

    cmap = cm.OrRd
    boundaries = [-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65]
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='OrRd'),ax=ax,extend='max',
                 label='Scaled PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)')


    ax.contourf(lat, plev/100, p, 20,
                cmap='OrRd', vmin=vmin, vmax=vmax)
    ax.contour(lat, plev/100, t,
                levels=[250,300,350,400,450,500,550,600,650,700],
                linestyles = '--', colors='black', linewidths=1)
    cs3 = ax.contour(lat, plev/100, u,
                     levels=[0,50,100,150], colors='black',linewidths=1)

    # Recast levels to new class
    cs3.levels = [nf(val) for val in cs3.levels]

    # Label levels with specially formatted floats
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    ax.clabel(cs3, cs3.levels, inline=1, fmt=fmt, fontsize=10)
    plt.savefig(directory+'/'+exp+'/Waugh2016_Fig5_Ls'+strLs1+'-'+strLs2+'.png')



def theta_plot_era5(d):
    p = d.PV.mean(dim='time').mean(dim='longitude') *10**6
    t = d.theta.mean(dim='time').mean(dim='longitude')
    #u = d.u.mean(dim='time').mean(dim='longitude')
    plev = d.level
    lat = d.latitude
    
    #extract dates
    #strLs1 = str(int(10*Ls[0])/10)
    #strLs2 = str(int(10*Ls[-1])/10)
    #strMY = str(int(10*MY.sel(time=newtime))/10)

    fig, ax = plt.subplots(1, figsize=(8, 6))
    #ax.set_yscale('log')
    #ticker.LogFormatter(base=10.0, labelOnlyBase=False,
    #                               minor_thresholds=None, linthresh=None)
    ax.set_xlim([0,90])
    ax.set_ylim([1000,100])
    plt.xlabel('latitude (degrees)')
    plt.ylabel('pressure (hPa)')

    #vmax = p.max().values/2
    #vmin = p.min().values/2
    vmax = 5
    vmin = 0
    
    cmap = cm.OrRd
    boundaries = [0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,
                  2.75,3,3.25,3.5,3.75,4.,4.25,4.5,4.75,5.]
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='OrRd'),ax=ax,extend='max',
                 label='PV (10$^{-6}$ K m$^2$ kg$^{-1}$ s$^{-1}$)')


    ax.contourf(lat, plev/100, p, levels = boundaries,
                cmap='OrRd', vmin=vmin, vmax=vmax)
    cs2 = ax.contour(lat, plev/100, t,
                     levels=[260,270,280,290,300,310,320,330,340,350,360,
                             370,380,390,400,410,420,430,440],
                     linestyles = '--', colors='black', linewidths=1)
    #cs3 = ax.contour(lat, plev/100, u,
    #                 levels=[0,50,100,150], colors='black',linewidths=1)

    # Recast levels to new class
    cs2.levels = [nf(val) for val in cs2.levels]

    # Label levels with specially formatted floats
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    ax.clabel(cs2, cs2.levels, inline=1, fmt=fmt, fontsize=10)
    
    plt.title('Zonal PV and potential temps from Era 5 data, 10/02/1979')
    #plt.savefig(home+'/figs/19790220_PV_'+strlev+'Ksurface.png')
    plt.savefig('Figs/era5_theta_levels_test.png')



def era5_pv_compare(d):
    pv_calc = d.PV.mean(dim='time').mean(dim='longitude') *10**6
    pv = d.pv.mean(dim='time').mean(dim='longitude') * 10**6
    t = d.theta.mean(dim='time').mean(dim='longitude')
    #u = d.u.mean(dim='time').mean(dim='longitude')
    plev = d.level
    lat = d.latitude

    fig, axs = plt.subplots(nrows=1,ncols=3,sharey=True,sharex=True,
                           figsize=(21, 6))

    axs[0].set_xlim([0,90])
    axs[0].set_ylim([1000,100])
    axs[1].set_xlim([0,90])
    axs[1].set_ylim([1000,100])
    axs[2].set_xlim([0,90])
    axs[2].set_ylim([1000,100])

    plt.xlabel('latitude (degrees)')
    plt.ylabel('pressure (hPa)')

    vmax = 5
    vmin = 0
    
    cmap = cm.OrRd
    boundaries = [0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,
                  2.75,3,3.25,3.5,3.75,4.,4.25,4.5,4.75,5.]
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='OrRd'),ax=axs,extend='max',
                 label='PV (10$^{-6}$ K m$^2$ kg$^{-1}$ s$^{-1}$)')


    axs[0].contourf(lat, plev/100, pv_calc, levels = boundaries,
                    cmap='OrRd', vmin=vmin, vmax=vmax)
    cs2 = axs[0].contour(lat, plev/100, t,
                     levels=[260,270,280,290,300,310,320,330,340,350,360,
                             370,380,390,400,410,420,430,440],
                     linestyles = '--', colors='black', linewidths=1)

    axs[1].contourf(lat, plev/100, pv, levels = boundaries,
                    cmap='OrRd', vmin=vmin, vmax=vmax)
    cs4 = axs[1].contour(lat, plev/100, t,
                     levels=[260,270,280,290,300,310,320,330,340,350,360,
                             370,380,390,400,410,420,430,440],
                     linestyles = '--', colors='black', linewidths=1)
    axs[2].contourf(lat, plev/100, pv-pv_calc, levels = boundaries,
                    cmap='OrRd', vmin=vmin, vmax=vmax)
    

    # Recast levels to new class
    cs2.levels = [nf(val) for val in cs2.levels]
    cs4.levels = [nf(val) for val in cs4.levels]

    # Label levels with specially formatted floats
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    axs[0].clabel(cs2, cs2.levels, inline=1, fmt=fmt, fontsize=10)
    axs[1].clabel(cs4, cs4.levels, inline=1, fmt=fmt, fontsize=10)  

    axs[0].set_title('calculated PV')
    axs[1].set_title('Era5 PV')
    axs[2].set_title('difference in PV')
    
    fig.suptitle('Zonal PV and potential temps from Era 5 data, 10/02/1979')

    plt.savefig('Figs/era5_pv_compare.png')



def relvort_compare(d):
    vo = d.vo.mean(dim='time').mean(dim='longitude')
    relvo = d.relvort.mean(dim='time').mean(dim='longitude')
    plev = d.level
    lat = d.latitude

    fig, axs = plt.subplots(nrows=1,ncols=2,sharey=True,
                           figsize=(18, 6))

    axs[0].set_xlim([0,90])
    axs[0].set_ylim([1000,100])
    axs[1].set_xlim([0,90])
    axs[1].set_ylim([1000,100])
    plt.xlabel('latitude (degrees)')
    plt.ylabel('pressure (hPa)')

    vmax = relvo.max()
    vmin = relvo.min()

    norm = colors.Normalize(vmin,vmax)
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap='plasma'),ax=axs,
                 label='relative vorticity', extend='both')


    axs[0].contourf(lat, plev/100, vo, 20,
                    cmap='OrRd', vmin=vmin, vmax=vmax)
    axs[1].contourf(lat, plev/100, relvo, 20,
                    cmap='OrRd', vmin=vmin, vmax=vmax) 

    axs[0].set_title('ERA5 relative vorticity')
    axs[1].set_title('calculated relative vorticity')
    
    fig.suptitle('Zonal vorticity from Era 5 data, 10/02/1979')
    plt.savefig('Figs/era5_vort_compare.png')



def PV_isobaric_era5(d):

    fig, axs = plt.subplots(nrows=1,ncols=3, figsize = (15,5),
                           subplot_kw = {'projection':ccrs.NorthPolarStereo()})
    
    vmax = 15
    vmin = -15

    strlev = str(int(d.level[10]/100))

    norm = colors.Normalize(vmin,vmax)
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap='plasma'),ax=axs,
             label='PV (10$^{-6}$ K m$^2$ kg$^{-1}$ s$^{-1})$',
             extend='max', shrink=0.8)
        
    axs[0].contourf(d.longitude,d.latitude,d.PV[0,10,:,:]*10**6,
                    40, cmap = 'plasma', transform=ccrs.PlateCarree(),
                    vmin=vmin, vmax=vmax)

    axs[1].contourf(d.longitude,d.latitude,d.pv[0,10,:,:]*10**6,
                    40, cmap = 'plasma', transform=ccrs.PlateCarree(),
                    vmin=vmin, vmax=vmax)
    axs[2].contourf(d.longitude,d.latitude,d.PV[0,10,:,:]*10**6-d.pv[0,10,:,:]*10**6,
                    40, cmap = 'plasma', transform=ccrs.PlateCarree(),
                    vmin=vmin, vmax=vmax)

    axs[0].set_title('calculated PV')
    axs[1].set_title('Era5 PV')
    axs[2].set_title('difference in PV')

    fig.suptitle('Era5, PV on '+strlev+'hPa surface')
    plt.savefig('Figs/PV_'+strlev+'hPa.png')



def isca_temp_lat_plot(d1,ds):
    r"""Plot temperature vs latitude at the first time point for an Isca run,
    for both pfull and interpolated pressure.
    """
    fig, axs = plt.subplots(nrows=1,ncols=2,sharex=True,figsize = (10,5))
    
    t1 = d1.temp.mean(dim='time').mean(dim='lon')
    ts = ds.temp.mean(dim='time').mean(dim='lon')

    axs[0].set_xlim([-90,90])
    axs[0].set_ylim([6.1,0])
    axs[1].set_xlim([-90,90])
    axs[1].set_ylim([6.1,0])

    plt.xlabel('latitude (degrees)')
    axs[0].set_ylabel('Sigma level')
    axs[1].set_ylabel('Pressure (hPa)')

    vmax = np.max(t1.values)
    vmin = np.min(t1.values)

    norm = colors.Normalize(vmin,vmax)
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap='OrRd'),ax=axs,
             label='Temperature (K)',
             extend='max', shrink=0.8)

    axs[0].contourf(d1.lat, d1.pfull, t1, 20,
                    cmap='OrRd', vmin=vmin, vmax=vmax)
    #cs2 = axs[0].contour(lat, plev/100, t,
    #                 levels=[260,270,280,290,300,310,320,330,340,350,360,
    #                         370,380,390,400,410,420,430,440],
    #                 linestyles = '--', colors='black', linewidths=1)

    axs[1].contourf(ds.lat, ds.pfull, ts, 20,
                    cmap='OrRd', vmin=vmin, vmax=vmax)
    #cs4 = axs[1].contour(lat, plev/100, t,
    #                 levels=[260,270,280,290,300,310,320,330,340,350,360,
    #                         370,380,390,400,410,420,430,440],
    #                 linestyles = '--', colors='black', linewidths=1)
    
    axs[0].set_title('Sigma levels')
    axs[1].set_title('Full pressure levels')

    fig.suptitle('Comparison of temperature values from Isca simulations on sigma and pressure levels')
    plt.savefig('Isca_figs/temp_compare.png')


def calc_eddy_enstr(q):
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
