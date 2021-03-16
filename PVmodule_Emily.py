# Python 'package' to calculate potential vorticity and other metereological quantities
# Extra functionality is to interpolate to isentropic levels
# No output. Package must be loaded into another script to call functions
# Written by Emily Ball, final version 13/05/2020
# Tested on Anthropocene

# NOTES and known problems
# Flexible to planetary parameters such as radius/gravity/rotation rate etc.
# Default parameters are Martian
# Functions listed below are from or have been adapted from the Metpy module:
# broadcast_indices, find_bounding_indices, _less_or_close,
# strip_matching_units, interpolate_1d, log_interpolate (https://github.com/Unidata/MetPy)
# Requires global data for the dependency on Windspharm package
# (https://ajdawson.github.io/windspharm/latest/)

# Updates
# 02/11/2020 Upload to GitHub EB
# 10/11/2020 Documentation EB

# Begin script ========================================================================================================

import numpy as np
import xarray as xr
import dask
import os, sys
from windspharm.xarray import VectorWind
import scipy.optimize as so

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def broadcast_indices(x, minv, ndim, axis):
    '''
    Calculate index values to properly broadcast index array within data array.
    '''
    ret = []
    for dim in range(ndim):
        if dim == axis:
            ret.append(minv)
        else:
            broadcast_slice = [np.newaxis] * ndim
            broadcast_slice[dim] = slice(None)
            dim_inds = np.arange(x.shape[dim])
            ret.append(dim_inds[tuple(broadcast_slice)])
    return tuple(ret)

def find_bounding_indices(arr, values, axis, from_below=True):
    '''Find the indices surrounding the values within arr along axis.

    Returns a set of above, below, good. Above and below are lists of arrays of indices.
    These lists are formulated such that they can be used directly to index into a numpy
    array and get the expected results (no extra slices or ellipsis necessary). `good` is
    a boolean array indicating the "columns" that actually had values to bound the desired
    value(s).

    Parameters
    ----------
    arr : array-like
        Array to search for values

    values: array-like
        One or more values to search for in `arr`

    axis : int
        The dimension of `arr` along which to search.

    from_below : bool, optional
        Whether to search from "below" (i.e. low indices to high indices). If `False`,
        the search will instead proceed from high indices to low indices. Defaults to `True`.

    Returns
    -------
    above : list of arrays
        List of broadcasted indices to the location above the desired value

    below : list of arrays
        List of broadcasted indices to the location below the desired value

    good : array
        Boolean array indicating where the search found proper bounds for the desired value

    '''
    # The shape of generated indices is the same as the input, but with the axis of interest
    # replaced by the number of values to search for.
    indices_shape = list(arr.shape)
    indices_shape[axis] = len(values)

    # Storage for the found indices and the mask for good locations
    indices = np.empty(indices_shape, dtype=np.int)
    good = np.empty(indices_shape, dtype=np.bool)

    # Used to put the output in the proper location
    store_slice = [slice(None)] * arr.ndim

    # Loop over all of the values and for each, see where the value would be found from a
    # linear search
    for level_index, value in enumerate(values):
        # Look for changes in the value of the test for <= value in consecutive points
        # Taking abs() because we only care if there is a flip, not which direction.
        switches = np.abs(np.diff((arr <= value).astype(np.int), axis=axis))

        # Good points are those where it's not just 0's along the whole axis
        good_search = np.any(switches, axis=axis)

        if from_below:
            # Look for the first switch; need to add 1 to the index since argmax is giving the
            # index within the difference array, which is one smaller.
            index = switches.argmax(axis=axis) + 1
        else:
            # Generate a list of slices to reverse the axis of interest so that searching from
            # 0 to N is starting at the "top" of the axis.
            arr_slice = [slice(None)] * arr.ndim
            arr_slice[axis] = slice(None, None, -1)

            # Same as above, but we use the slice to come from the end; then adjust those
            # indices to measure from the front.
            index = arr.shape[axis] - 1 - switches[tuple(arr_slice)].argmax(axis=axis)

        # Set all indices where the results are not good to 0
        index[~good_search] = 0

        # Put the results in the proper slice
        store_slice[axis] = level_index
        indices[tuple(store_slice)] = index
        good[tuple(store_slice)] = good_search

    # Create index values for broadcasting arrays
    above = broadcast_indices(arr, indices, arr.ndim, axis)
    below = broadcast_indices(arr, indices - 1, arr.ndim, axis)

    return above, below, good


def _less_or_close(a, value, **kwargs):
    '''Compare values for less or close to boolean masks.

    Returns a boolean mask for values less than or equal to a target within a
    specified absolute or relative tolerance (as in :func:`numpy.isclose`).

    Parameters
    ----------
    a : array-like
        Array of values to be compared
    value : float
        Comparison value

    Returns
    -------
    array-like
        Boolean array where values are less than or nearly equal to value.

    '''
    return (a < value) | np.isclose(a, value, **kwargs)


def _strip_matching_units(*args):
    '''
    Ensure arguments have same units and return with units stripped.

    Replaces `@units.wraps(None, ('=A', '=A'))`, which breaks with `*args`
    handling for pint>=0.9.
    '''
    if all(hasattr(arr, 'units') for arr in args):
        return [arr.values for arr in args]
    else:
        return args

def interpolate_1d(x, xp, *args, **kwargs):
    '''
    Interpolates data with any shape over a specified axis.
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
        Axis to interpolate over. Defaults to 0.
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

    '''
    # Pull out keyword args
    axis = kwargs.pop('axis', 0)
    fill_value = kwargs.pop('fill_value', np.nan)
    return_list_always = kwargs.pop('return_list_always', False)

    # Handle units
    x, xp = _strip_matching_units(x, xp)

    # Make x, xp, args an array
    x = np.asanyarray(x).reshape(-1)
    xp = np.asanyarray(xp)
    args = [np.asanyarray(arr) for arr in args]


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

def get_axis(da, coord):
    '''
    Return axis of a dataset corresponding to a given coordinate name.
    '''
    return list(da.dims).index(coord)


def log_interpolate_1d(x, xp, *args, **kwargs):
    '''
    Interpolates data with logarithmic x-scale over a specified axis.
    Interpolation on a logarithmic x-scale for interpolation values in
    pressure coordinates.
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
        Axis to interpolate over. Defaults to 0.
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
    '''
    # Pull out kwargs
    axis = kwargs.pop('axis', 0)
    fill_value = kwargs.pop('fill_value', np.nan)

    # Handle units
    x, xp = _strip_matching_units(x, xp)

    # Log x and xp
    log_x = np.log(x)
    log_xp = np.log(xp)
    return interpolate_1d(log_x, log_xp,
                          *args, axis=axis, fill_value=fill_value)


def wrapped_gradient(da, coord):
    '''
    Finds the gradient along a given dimension of a dataarray.
    Input
    -----
    da    : array-like
    coord : dimension name for axis along which to take gradient
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

def wrapped_argsort(da, coord):
    '''
    Applies numpy.argsort along a given dimension of a dataarray.
    Input
    -----
    da    : array-like
    coord : dimension name for axis along which to take gradient
    '''

    dims_of_coord = da.coords[coord].dims
    if len(dims_of_coord) == 1:
        dim = dims_of_coord[0]
    else:
        raise ValueError('Coordinate ' + coord + ' has multiple dimensions: ' + str(dims_of_coord))

    return xr.apply_ufunc(np.argsort, da, kwargs={'axis': -1},
                          input_core_dims=[[dim]],
                          output_core_dims=[[dim]],
                          dask='parallelized',
                          output_dtypes=[da.dtype])

def _isen_iter(iter_log_p, isentlevs_nd, kappa, a, b, pok):
    exner = pok * np.exp(-kappa * iter_log_p)
    t = a * iter_log_p + b
    # Newton-Raphson iteration
    f = isentlevs_nd - t * exner
    fp = exner * (kappa * t - a)
    return iter_log_p - (f / fp)

def _exner_function(levs, **kwargs):
    '''
    Calculates the Exner function.

    Input
    -----
    levs  : pressure levels, array-like
    p0    : reference pressure in Pascals, optional. Default: 610. Pa
    kappa : optional. Default: 0.25
    '''
    p0 = kwargs.pop('p0', 610.)
    kappa = kwargs.pop('kappa', 0.25)

    return (levs/p0)**kappa   

def potential_temperature(levs, tmp, **kwargs):
    '''
    Calculates potential temperature theta

    Input
    -----
    tmp   : temperature, array-like
    levs  : pressure levels, array-like
    p0    : reference pressure in Pascals, optional. Default: 610. Pa
    kappa : optional. Default: 0.25
    '''
    p0 = kwargs.pop('p0', 610.)
    kappa = kwargs.pop('kappa', 0.25)

    ret = tmp / _exner_function(levs, p0=p0, kappa=kappa)
    return ret

def laitscale(PV, theta, theta0, **kwargs):
    '''
    Perform Lait scaling of PV
    kwargs
    ------
    kappa: R/c_p, optional. Default = 0.25.
    '''
    kappa = kwargs.pop('kappa', 0.25)
    ret = PV*(theta/theta0)**(-(1+1/kappa))

    return ret


def potential_vorticity_baroclinic(uwnd, vwnd, theta, coord, **kwargs):
    '''
    Calculate potential vorticity on isobaric levels. Requires input uwnd,
    vwnd and tmp arrays to have (lat, lon, ...) format for Windspharm.

    Input
    -----
    uwnd    : zonal winds, array-like
    vwnd    : meridional winds, array-like
    tmp     : temperature, array-like
    theta   : potential temperature, array-like
    coord   : dimension name for pressure axis (eg. 'pfull')
    omega   : planetary rotation rate, optional
    g       : planetary gravitational acceleration, optional
    rsphere : planetary radius, in metres, optional
    '''
    omega = kwargs.pop('omega', 7.08822e-05)
    g = kwargs.pop('g', 3.72076)
    rsphere = kwargs.pop('rsphere', 3.3962e6)
    w = VectorWind(uwnd.fillna(0), vwnd.fillna(0), rsphere = rsphere)

    relvort = w.vorticity()
    relvort = relvort.where(relvort!=0, other=np.nan)
    planvort = w.planetaryvorticity(omega=omega)
    absvort = relvort + planvort

    dthtady, dthtadx = w.gradient(theta.fillna(0))
    dthtadx = dthtadx.where(dthtadx!=0, other=np.nan)
    dthtady = dthtady.where(dthtady!=0, other=np.nan)

    dthtadp = wrapped_gradient(theta, coord)
    dudp = wrapped_gradient(uwnd, coord)
    dvdp = wrapped_gradient(vwnd, coord)

    s = -dthtadp
    f = dvdp * dthtadx - dudp * dthtady
    ret  = g * (absvort + f/s) * s

    return ret


def isent_interp(isentlevs, pres, tmp, *args, **kwargs):
    '''
    Interpolate data in isobaric coordinates to isentropic coordinates.
    
    Parameters
    ----------
    isentlevs : array
        1D array of desired theta surfaces
    pres : array
        1D array of pressure levels
    tmp  : array
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
        constant reference pressure in Pascals, Default = 610 Pa (Mars)
    kappa : float, optional
        constant Poisson exponent, Rd/Cp_d. Default = 0.25 (Mars)
    bottom_up_search : bool, optional
        Controls whether to search for theta levels bottom-up, or top-down. Defaults to
        True, which is bottom-up search.
    temperature_out : bool, optional
        If true, calculates temperature and output as last item in output list. Defaults
        to False.
    
    '''

    max_iters = kwargs.pop('max_iters', 50)
    eps = kwargs.pop('eps', 1e-6)
    axis = kwargs.pop('axis', 0)
    p0 = kwargs.pop('p0', 610.)
    kappa = kwargs.pop('kappa', 0.25)
    bottom_up_search = kwargs.pop('bottom_up_search', True)
    temperature_out = kwargs.pop('temperature_out', False)
    
    
    if hasattr(tmp, 'units'):
        if not (tmp.units == 'K') and not (tmp.units == 'Kelvins') and not (tmp.units == 'Kelvin'):
            raise ValueError('Temperature must be in Kelvins.')

    if hasattr(pres, 'units'):
        if not (pres.units == 'Pa') and not (pres.units == 'Pascals') and not (pres.units == 'Pascal'):
            raise ValueError('Pressure must be in Pascals.')

    pres = np.asanyarray(pres)
    
    ndim = tmp.ndim
    slices = [np.newaxis] * ndim
    slices[axis] = slice(None)
    slices = tuple(slices)

    pres = np.broadcast_to(pres[slices], tmp.shape)
    
    ## sort input data    
    sort_pres = np.argsort(pres, axis=axis)
    sort_pres = np.swapaxes(np.swapaxes(sort_pres, 0, axis)[::-1], 0, axis)
    sorter = broadcast_indices(pres, sort_pres, ndim, axis)
    levs = pres[sorter]
    tmpk = tmp.values[sorter]
    
    isentlevs = np.asanyarray(isentlevs).reshape(-1)
    isentlevels = isentlevs[np.argsort(isentlevs)]
    
    # Make the desired isentropic levels the same shape as temperature
    shape = list(tmp.shape)
    shape[axis] = isentlevels.size
    isentlevs_nd = np.broadcast_to(isentlevels[slices], shape)
    
    pres_theta = potential_temperature(levs,tmpk, p0=p0, kappa=kappa)
    
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
    
    log_p_solved = so.fixed_point(_isen_iter, isentprs[good],
                                  args=(isentlevs_nd[good], kappa, a[good], b[good], pok),
                                  xtol=eps, maxiter=max_iters)

    isentprs[good]=np.exp(log_p_solved)
    isentprs[~(good & _less_or_close(isentprs,np.max(pres)))] = np.nan
    # _less_or_close returns (isentprs < np.max(pres) | np.isclose(isentprs, np.max(pres))
    
    ret = [isentprs]
    
    if temperature_out:
        ret.append(isentlevs_nd / ((p0 / isentprs) ** kappa))

    # do an interpolation for each additional argument
    if args:
        args = [np.asanyarray(arr) for arr in args]
        others = interpolate_1d(isentlevels, pres_theta, *(arr[sorter] for arr in args),
                                axis=axis, return_list_always=True)
        ret.extend(others)
    
    return ret