__version__ = "2025-03-29"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

"""
To plot outputs from the RNS SY_1km experiment
"""

import xarray as xr
import iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os
import importlib

oshome=os.getenv('HOME')
sys.path.append(f'{oshome}/git/RNS_Sydney_1km/plotting_code')
import common_functions as cf
importlib.reload(cf)

################## set up ##################

oshome=os.getenv('HOME')

# stationpath = '/g/data/gb02/mf9078/AWS_5mindata_38stations'
stationpath = '/g/data/ce10/users/mjl561/observations/AWS/au-2000_2024_5min'

cylc_id = 'rns_MC_202002'
project = 'fy29'

datapath = f'/g/data/ce10/users/mjl561/cylc-run/{cylc_id}/netcdf'
plotpath = f'/g/data/ce10/users/mjl561/cylc-run/{cylc_id}/figures'
cycle_path = f'/scratch/{project}/mjl561/cylc-run/{cylc_id}/share/cycle'

local_time_offset = 10
local_time = False

variables_done = [
    'land_sea_mask','air_temperature','surface_temperature','relative_humidity',
    'latent_heat_flux','sensible_heat_flux','air_pressure_at_sea_level',
    'dew_point_temperature', 'surface_net_downward_longwave_flux','wind_u','wind_v',
    'specific_humidity','specific_humidity_lowest_atmos_level','wind_speed_of_gust',
    'soil_moisture_l1','soil_moisture_l2','soil_moisture_l3','soil_moisture_l4',
    'soil_temperature_l1','soil_temperature_l2','soil_temperature_l3','soil_temperature_l4',
    'surface_runoff_flux','subsurface_runoff_flux','surface_total_moisture_flux',
    'boundary_layer_thickness','surface_air_pressure',
    'fog_area_fraction','visibility','cloud_area_fraction',
    'stratiform_rainfall_amount','stratiform_rainfall_flux',
    'toa_outgoing_shortwave_flux','toa_outgoing_shortwave_flux_corrected','toa_outgoing_longwave_flux',
    'surface_net_longwave_flux',
    ]

variables = ['soil_moisture_l1','soil_moisture_l2','soil_moisture_l3','soil_moisture_l4']
variables = ['air_temperature_10min']
variables = ['upward_air_velocity_at_300m']
variables = ['air_temperature']
variables = ['wind_speed_of_gust']
variables = ['air_pressure_at_sea_level']
variables = ['soil_moisture_l1']
variables = ['latent_heat_flux']
variables = ['specific_humidity']
variables = ['dew_point_temperature']
variables = ['toa_outgoing_longwave_flux']
variables = ['surface_temperature']
variables = ['cloud_area_fraction']
variables = ['stratiform_rainfall_amount']
variables = ['stratiform_rainfall_flux']
variables = ['toa_outgoing_shortwave_flux_corrected']
variables = ['toa_outgoing_longwave_flux']

exps = [
        '5',
        # '11p1',
        ### BARRA operational reanalysis ###
        # 'BARRA-R2',
        # 'BARRA-C2',
        ]

################## functions ##################

def main_plotting():


    return

def create_spatial_timeseries_animation(exps, ds, vopts, sids, stations, obs, times, 
                                        suffix='', plot_vs_obs=False, masked=True, distance=100):

    fnamein = f"{plotpath}/{opts['plot_fname']}_spatial*.png"
    fnameout = f"{plotpath}/{opts['plot_fname']}_spatial{suffix}"

    # remove all spatial png files
    for file in glob.glob(fnamein):
        os.remove(file)

    for i,time in enumerate(times):
        print(f'{i+1} of {len(times)}')
        if plot_vs_obs:
            fig, fname = cf.create_spatial_timeseries_plot_vs_obs(exps, ds, vopts, sids, stations, obs, datapath,
                                                      itime=i, masked=masked, distance=distance, suffix=suffix)
        else:
            fig, fname = cf.create_spatial_timeseries_plot(exps, ds, vopts, sids, stations, obs, datapath,
                                   itime=i, masked=masked, distance=distance, suffix=suffix)
        fig.savefig(f'{plotpath}/{fname}', bbox_inches='tight', dpi=200)
        plt.close('all')

    cf.make_mp4(fnamein,fnameout,fps=24,quality=26)

    # # remove all spatial png files
    # for file in glob.glob(fnamein):
    #     os.remove(file)

    return

def create_sst_animation(suffix='_sst'):

    lsm_opts = cf.get_variable_opts('land_sea_mask')
    lsm_ds = cf.open_output_netcdf(exps, lsm_opts, 'land_sea_mask')
    
    masked = xr.Dataset()
    for exp in exps:
        masked[exp] = ds[exp].where(lsm_ds[exp].isel(time=0)==0)


    vopts = cf.update_opts(opts,
        vmin=16,
        vmax=28,
        # cmap='turbo',
    )

    # # test
    # dss = masked.isel(time=48)
    # fig, fname = cf.plot_spatial(exps, dss, vopts, sids, stations, obs, distance=250)


    # select only 1 masked data per day
    dss = masked.sel(time=masked.time.dt.hour==12)

    cf.plot_spatial_anim(exps,dss,vopts,sids,stations,obs,plotpath,slabels=False,fill_obs=False,
                         distance=250,suffix=suffix)
    fnamein = f"{plotpath}/{opts['plot_fname']}_spatial*.png"
    fnameout = f"{plotpath}/{opts['plot_fname']}_spatial{suffix}"
    cf.make_mp4(fnamein,fnameout,fps=12,quality=26)

    # # make gif
    # command = f'convert -delay 5 -loop 0 {fnamein} {fnameout}.gif'
    # os.system(command)
    
    # remove all spatial png files
    for file in glob.glob(fnamein):
        os.remove(file)

    return

def set_up_plot_attrs(exps, plotpath):

    # predefine colours for experiments
    exp_colours = {
        '11p1'      : 'tab:blue',
        'd0198'     : 'tab:orange',
        'BARRA-C2'  : 'tab:green',
        'BARRA-R2'  : 'green',
        'ACCESS-G'  : 'grey',
        }

    exp_plot_titles = {
        'ACCESS-G'       : 'ACCESS-G3 Global',
        'BARRA-R2'       : 'BARRA-R2 12.2km (reanalysis product)',
        'BARRA-C2'       : 'BARRA-C2 4.4km (reanalysis product)',
        }

    # drop keys not in exps
    exp_colours = {key: exp_colours[key] for key in exp_colours if key in exps}
    exp_plot_titles = {key: exp_plot_titles[key] for key in exp_plot_titles if key in exps}
    
    # create entry in exp_plot_titles if key not in exps
    extra_colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'][::-1]
    for exp in exps:
        if exp not in exp_colours:
            colour = extra_colours.pop()
            while colour in list(exp_colours.values()):
                colour = extra_colours.pop()
            exp_colours[exp] = colour

        if exp not in exp_plot_titles:
            exp_plot_titles[exp] = exp
    
    cf.local_time = local_time
    cf.plotpath = plotpath
    cf.exp_colours = exp_colours
    cf.exp_plot_titles = exp_plot_titles
    cf.cycle_path = cycle_path

    return exp_colours, exp_plot_titles

def open_output_netcdf(exps,opts,variable,datapath,pp_to_netcdf=True):
    """
    Open the netcdf files for the experiments and variable

    Args:
    exps: list of experiments
    variable: variable to open
    
    """

    print(f'attempting to open {len(exps)} experiments:')
    print(exps)
   
    ds = xr.Dataset()
    for i_exp,exp in enumerate(exps):

        fname = f'{datapath}/{opts["plot_fname"]}/{exp}_{opts["plot_fname"]}.nc'
        print(f'{i_exp+1}: checking {fname}')
        if os.path.exists(fname):
            print(f'  opening {variable} {exp}')
            try:
                da = xr.open_dataset(fname)
                da_var = list(da.keys())[0]
                da = da[da_var]
                if da_var != variable:
                    print(f'WARNING: {da_var} != {variable}, updating name')
                    da.name = variable
            except Exception as e:
                print(f'failed to open {fname} {e}')
                # print(f'removing {exp} from exps')
                # exps.remove(exp)

        elif 'BARRA' in exp:
            da = get_barra_data(ds,opts,exp)

        else:
            print('trying to open pp file using iris')

            # check if data directory exists
            if not os.path.exists(cycle_path):
                print(f'{cycle_path} does not exist')
                # return None

            cycle_list = sorted([x.split('/')[-2] for x in glob.glob(f'{cycle_path}/*/')])

            da_list = []
            for i,cycle in enumerate(cycle_list):
                
                print('========================')
                print(f'getting {exp} {i}: {cycle}\n')

                # set paths to experiment outputs
                exp_paths = {
                    '11p1': f'{cycle_path}/{cycle}/MC/11p1/GAL9/um',
                    '5': f'{cycle_path}/{cycle}/MC2/5/RAL3P2/um',
                    'd0198': f'{cycle_path}/{cycle}/MC/d0198/RAL3P2/um',
                }

                # check if any of the variables files are in the directory
                if len(glob.glob(f"{exp_paths[exp]}/{opts['fname']}*")) == 0:
                    print(f'no files in {exp_paths[exp]}')
                    cycle_list.remove(cycle)
                    continue

                ppfname = f"{exp_paths[exp]}/{opts['fname']}*"
                ppfnames = sorted(glob.glob(ppfname))
                
                try:
                    if ppfnames[0][-3:] == '.nc':
                        print('Opening as netcdf')
                        # open all ppfnames as netcdf and concatenate
                        nclist = []
                        for ncfname in ppfnames:
                            print(f'opening {fname}')
                            tmp = xr.open_dataset(ncfname)['STASH_'+opts['stash']]
                            nclist.append(tmp)

                        # check coordinates for datetime type and get coordinate name
                        for coord in tmp.coords:
                            if np.issubdtype(tmp[coord].dtype, np.datetime64):
                                time_coord = coord
                                print(f'found time coordinate: {time_coord}')

                        dal = xr.concat(nclist, dim=time_coord)
                        print(f'renaming {time_coord} to time')
                        dal = dal.rename({time_coord: 'time'})

                        # if coordinates are grid_longitude_t, rename to longitude
                        if 'grid_longitude_t' in dal.coords:
                            print('renaming grid_longitude_t to longitude')
                            dal = dal.rename({'grid_longitude_t': 'longitude'})
                        if 'grid_latitude_t' in dal.coords:
                            print('renaming grid_latitude_t to latitude')
                            dal = dal.rename({'grid_latitude_t': 'latitude'})

                    else:
                        cb = iris.load_cube(ppfnames, constraint=opts['constraint'])
                        # fix timestamp/bounds error in accumulations
                        if cb.coord('time').bounds is not None:
                            print('WARNING: updating time point to right bound')
                            cb.coord('time').points = cb.coord('time').bounds[:,1]
                        dal = xr.DataArray().from_iris(cb)
                except Exception as e:
                    print(f'trouble opening {ppfname}')
                    print(e)
                    # return None

                da_list.append(dal)

            if len(da_list) == 0:
                print(f'no data found for {exp} in {cycle_path}')
            else:
                da = xr.concat(da_list, dim='time')
                ##### save pp to netcdf ######
                if pp_to_netcdf:
                    # set decimal precision to reduce filesize (definded fmt precision +1)
                    precision = int(opts['fmt'].split('.')[1][0]) + 1
                    da = da.round(precision)
                    
                    # drop unessasary dimensions
                    if 'forecast_period' in da.coords:
                        da = da.drop_vars('forecast_period')
                    if 'forecast_reference_time' in da.coords:
                        da = da.drop_vars('forecast_reference_time')
                    
                    # chunk to optimise save
                    if len(da.dims)==3:
                        itime, ilon, ilat = da.shape
                        da = da.chunk({'time':24,'longitude':ilon,'latitude':ilat})
                    elif len(da.dims)==2:
                        ilon, ilat = da.shape
                        da = da.chunk({'longitude':ilon,'latitude':ilat})
                    
                    # encoding
                    # da.time.encoding.update({'dtype':'int32'})
                    da.longitude.encoding.update({'dtype':'float32', '_FillValue': -999})
                    da.latitude.encoding.update({'dtype':'float32', '_FillValue': -999})
                    da.encoding.update({'zlib':'true', 'shuffle': True, 'dtype':opts['dtype'], '_FillValue': -999})

                    # create directory if it doesn't exist
                    os.makedirs(f'{datapath}/{opts["plot_fname"]}', exist_ok=True)

                    try:
                        print(f'saving to netcdf: {fname}')
                        da.to_netcdf(fname, unlimited_dims='time')
                    except Exception as e:
                        print(f'failed to save {fname} {e}')
                        # remove fname if it exists
                        if os.path.exists(fname):
                            os.remove(fname)

            # test if substring 'temperature' is part of the variable name
            if 'temperature' in variable:
                print(f'converting {variable} from K to C')
                da = da - 273.15
                da.attrs['units'] = 'C'

        # interpolate to first exp grid
        if i_exp==0:
            template_exp = exp
            ds[exp] = da
        else:
            # check if the coordinates match with the template_exp
            if (da.longitude.equals(ds[template_exp].longitude) and da.latitude.equals(ds[template_exp].latitude)):
                print(f'  {exp} at same resolution as {template_exp}, no regridding necessary')
                ds[exp] = da
            else:
                print(f'  regridding to {template_exp}')
                ds[exp] = da.interp_like(ds[template_exp], method='nearest')

    # update the ds timezone attribute as "UTC"
    ds.time.attrs.update({'timezone':'UTC'})

    return ds

################## main ##################

if __name__ == "__main__":

    from dask.distributed import Client

    n_workers = int(os.environ['PBS_NCPUS'])
    local_directory = os.path.join(os.environ['PBS_JOBFS'], 'dask-worker-space')
    try:
        print(client)
    except Exception:
        client = Client(
            n_workers=n_workers,
            threads_per_worker=1, 
            local_directory = local_directory)

    ################## get model data ##################

    exp_colours, exp_plot_titles = set_up_plot_attrs(exps, plotpath)

    # check if plotpath exists, make if necessary
    if not os.path.exists(plotpath):
        os.makedirs(plotpath)

    for variable in variables:
        print(f'processing {variable}')

        opts = cf.get_variable_opts(variable)
        ds = open_output_netcdf(exps,opts,variable,datapath)

        # convert to local time and update timezone
        if local_time and ds.time.attrs['timezone'] == 'UTC': 
            ds = ds.assign_coords(time=ds.time + pd.Timedelta(f'{local_time_offset}h'))
            ds.time.attrs.update({'timezone': 'AEST'})

        ds = ds.compute()

        #### get observations ####
        if variable in ['not_currently_available']:
        # if variable in ['sensible_heat_flux','latent_heat_flux','soil_moisture_l1','soil_moisture_l2','soil_moisture_l3']:
            # print('getting flux obs')
            obs, stations = cf.get_flux_obs(variable, local_time_offset=None)
        elif opts['constraint'] in ['air_temperature','dew_point_temperature','wind_speed_of_gust']:
            sdate = ds.time[0]
            obs, stations = cf.process_station_netcdf(opts['constraint'], stationpath, local_time_offset=local_time_offset)
        else:
            print('no obs available')
            # set up dummy obs and stations dataframes
            obs, stations = pd.DataFrame(), pd.DataFrame()
            sids, sufix = [], ''

        # convert to local time if obs is not None
        if local_time and not obs.empty:
            obs.index = obs.index + pd.Timedelta(f'{local_time_offset}h')

        # trim obs dataframe to ds time period
        obs = obs.loc[ds.time.min().values:ds.time.max().values]

        # select only obs that align with ds model times
        if not obs.empty:
            obs = obs.loc[ds.time.values]

        # select all stations
        sids, suffix = stations.index.tolist(), '_all'

        # select inner part of ds


        # trim sids to those in ds
        sids, suffix = cf.trim_sids(ds, obs, sids, stations), '_trimmed'

        lsm_opts = cf.get_variable_opts('land_sea_mask')
        lsm_ds = open_output_netcdf([exps[0]], lsm_opts, 'land_sea_mask', datapath)
        
        # only pass obs if their lat/lon is within the lsm mask
        sids_to_pass = []
        for sid in sids:
            lat = stations.loc[sid,'lat']
            lon = stations.loc[sid,'lon']
            if lsm_ds[exps[0]].isel(time=0).sel(latitude=lat,longitude=lon, method = 'nearest')==1:
                sids_to_pass.append(sid)
        
        stations.loc[sids_to_pass]

        # urban sids
        urban_sids = ['67105','66212', '66194', '66037']

        # exlude sids to pass if they have less than 95% data in obs
        for sid in sids_to_pass:
            if obs[sid].count() < 0.95*len(obs):
                sids_to_pass.remove(sid)

        vopts = cf.update_opts(opts,
            # vmax=1000,
            vmin = 20,
            vmax = 340,
            )

        # fig, fname  = cf.plot_spatial(['5'], ds.isel(time=2), vopts, sids, stations, obs, cbar_loc='right', slabels=False,
        #          fill_obs=False, distance=500, fill_size=15, fill_diff=False,
        #          diff_vals=2, show_mean=False, suffix='')

        # fig.savefig(f'{plotpath}/{fname}', bbox_inches='tight', dpi=300)



        cf.plot_spatial_anim(['5'],ds,vopts,sids,stations,obs,plotpath,
                      cbar_loc='right',slabels=False,
                      fill_obs=False,distance=500,fill_size=15,fill_diff=False,
                      show_mean=False,suffix='_5',remove_files=True,fps=12)


    
    # print('done!')

