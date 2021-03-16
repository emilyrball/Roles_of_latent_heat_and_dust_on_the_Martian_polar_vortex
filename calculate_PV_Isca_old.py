'''
Calculates potential vorticity on isobaric levels from Isca data. Optionally
interpolates the data to isentropic coordinates.

This is an older version of the PV calculation script calculate_PV_Isca.py
that has not been parallelized, but it is slightly easier to follow.
'''

import numpy as np
import xarray as xr
import os, sys
import PVmodule_Emily as PV

def netcdf_prep(ds):
    '''
    Appends longitude 360 to file and reduces file to only variables necessary
    for PV calculation. Also converts pressure to Pa.
    '''
    ens_list = []
    tmp1 = ds.sel(lon=0.)
    tmp1 = tmp1.assign_coords({'lon':359.9999})
    ens_list.append(ds)
    ens_list.append(tmp1)

    d = xr.concat(ens_list, dim='lon')
    d = d.astype('float32')
    d = d[["ucomp", "vcomp", "temp", "mars_solar_long"]]
    # pressure is in hPa, must be in Pa for calculations
    d["pfull"] = d.pfull*100
    return d

def get_ls(d):
    '''
    Extracts solar longitude from dataset.
    '''
    Ls = d.mars_solar_long.sel(lon=0).squeeze()
    return Ls

def wind_prep(d):
    '''
    Reformats uwind, vwind and temperature to be in correct shape for
    Windspharm calculations.
    '''
    uwnd = d.ucomp.transpose('lat','lon','pfull','time')
    vwnd = d.vcomp.transpose('lat','lon','pfull','time')
    tmp = d.temp.transpose('lat','lon','pfull','time')

    return uwnd, vwnd, tmp

def save_PV_isobaric(d, outpath):
    '''
    Saves potential vorticity on isobaric levels to ``outpath``.
    '''

    print('Saving PV on isobaric levels to '+outpath)
    d["pfull"] = d.pfull/100  # data back to hPa
    d.to_netcdf(outpath+'_PV.nc')
    d["pfull"] = d.pfull*100  # back to Pa for isentropic interpolation

    return d

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



if __name__ == "__main__":

    # Mars-specific!
    theta0 = 200. # reference temperature
    kappa = 0.25 # ratio of specific heats
    p0 = 610. # reference pressure
    omega = 7.08822e-05 # planetary rotation rate
    g = 3.72076 # gravitational acceleration
    rsphere = 3.3962e6 # mean planetary radius

    # Earth-specific!
    #theta0 = 223.
    #kappa = 287.04/1004.6
    #p0 = 101325.
    #omega=7.2921e-5
    #rsphere=6.371e6
    #g=9.81

    # choose your experiment
    exp = 'soc_mars_mk36_per_value70.85_none_mld_2.0_with_mola_topo_cdod_clim_scenario_7.4e-05_lh_rel'
    filepath = '/export/silurian/array-01/xz19136/Isca_data'
    start_file = 30
    end_file = 140
    freq = 'daily'
    interp_file = 'atmos_'+freq+'_interp_new_height_temp'
    outpath1 = '/export/silurian/array-01/xz19136/Isca_data'
    runs, out, i_files = filestrings(exp, filepath, start_file,
                                     end_file, interp_file, outpath=outpath1)


    for i in range(len(i_files)):
        f = i_files[i]
        run = runs[i]
        outpath = out[i]

        exists = os.path.isfile(outpath+'_isentropic.nc')
        if not exists:        
            ds = xr.open_mfdataset(f+'.nc', decode_times=False,
                                   concat_dim='time', combine='nested',
                                   chunks={'time':'auto'})

            d = netcdf_prep(ds)

            Ls = get_ls(d)

            print('Calculating potential temperature...')
            theta = PV.potential_temperature(d.pfull, d.temp,
                                             kappa = kappa, p0 = p0)

            uwnd_trans,vwnd_trans,tmp_trans = wind_prep(d)

            print('Calculating potential vorticity...')
            PV_iso = PV.potential_vorticity_baroclinic(uwnd_trans, vwnd_trans,
                      theta, 'pfull', omega = omega, g = g, rsphere = rsphere)

            PV_iso = PV_iso.transpose('time','pfull','lat','lon')
            d["PV"] = PV_iso

            d = save_PV_isobaric(d, outpath)

            d = d.squeeze()
            d = d.transpose('time','pfull','lat','lon')

            # define isentropic levels to interpolate data to
            thetalevs = [200., 225., 250., 275., 300., 310., 320., 330., 340.,
                         350., 360., 370., 380., 390., 400., 425., 450., 475.,
                         500., 525., 550., 575., 600., 625., 650., 675., 700.,
                         725., 750., 775., 800., 850., 900., 950.]
            isentlevs = np.array(thetalevs)


            isent_prs, isent_PV, isent_u = PV.isent_interp(isentlevs, d.pfull,
                                               d.temp, d.PV, d.ucomp, axis = 1)

            d_isent = xr.Dataset({
                "prs" : (("time","ilev","lat","lon"), isent_prs),
                "PV"    : (("time","ilev","lat","lon"), isent_PV),
                "uwnd"  : (("time","ilev","lat","lon"), isent_u)
                },
                coords = {
                    "ilev": isentlevs,
                    "time": d.time,
                    "lat" : d.lat,
                    "lon" : d.lon
                    })

            d_isent["mars_solar_long"] = d.mars_solar_long
            d_isent.to_netcdf(outpath+'_isentropic.nc')

        else:
             print("Skipping "+run)
