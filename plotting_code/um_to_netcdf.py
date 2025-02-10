__version__ = "2024-12-27"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

'''
Create netcdf from um files

GADI ENVIRONMENT
----------------
module use /g/data/hh5/public/modules; module load conda/analysis3
'''

import time
import os
import xarray as xr
import iris
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import warnings
import importlib
warnings.simplefilter(action='ignore', category=FutureWarning)

oshome=os.getenv('HOME')
sys.path.append(f'{oshome}/git/RNS_Sydney_1km/plotting_code')
import common_functions as cf
importlib.reload(cf)

tic = time.perf_counter()

######## set up ########
datapath = '/g/data/ce10/users/mjl561/cylc-run/rns_ostia_SY_1km/netcdf'

cylc_id = 'rns_ostia'

variables_done = [
    'land_sea_mask','air_temperature','surface_temperature','relative_humidity',
    'latent_heat_flux','sensible_heat_flux','air_pressure_at_sea_level',
    'dew_point_temperature', 'surface_net_downward_longwave_flux','wind_u','wind_v',
    'specific_humidity','specific_humidity_lowest_atmos_level','wind_speed_of_gust',
    'soil_moisture_l1','soil_moisture_l2','soil_moisture_l3','soil_moisture_l4',
    'soil_temperature_l1','soil_temperature_l2','soil_temperature_l3','soil_temperature_l4',
    'surface_runoff_flux','subsurface_runoff_flux','surface_total_moisture_flux',
    'surface_temperature','boundary_layer_thickness','surface_air_pressure',
    'fog_area_fraction','visibility','cloud_area_fraction',
    'stratiform_rainfall_amount','stratiform_rainfall_flux',
    'toa_outgoing_shortwave_flux','toa_outgoing_shortwave_flux_corrected','toa_outgoing_longwave_flux',
    'surface_net_longwave_flux',
    ]

problematic =['total_precipitation_rate']
variables = ['surface_net_shortwave_flux','surface_downwelling_shortwave_flux','surface_downwelling_longwave_flux']

###############################################################################
# dictionary of experiments

exps = [
        ### Parent models ###
        'E5L_11p1_CCI',
        'BR2_12p2_CCI',
        # ## ERA5-Land CCI ###
        'E5L_5_CCI',
        'E5L_1_CCI',
        'E5L_1_L_CCI',
        # ### ERA5-Land CCI WordCover ###
        'E5L_5_CCI_WC',
        'E5L_1_CCI_WC',
        'E5L_1_L_CCI_WC',
        # ### BARRA CCI ###
        'BR2_5_CCI',
        'BR2_1_CCI',
        'BR2_1_L_CCI',
        # ### BARRA CCI WorldCover ###
        'BR2_5_CCI_WC',
        'BR2_1_CCI_WC',
        'BR2_1_L_CCI_WC',
        # ### BARRA IGBP ###
        'BR2_5_IGBP',
        'BR2_1_IGBP',
        'BR2_1_L_IGBP',
        ### BARRA CCI no urban ###
        'BR2_5_CCI_no_urban',
        'BR2_1_CCI_no_urban',
        'BR2_1_L_CCI_no_urban',
        ]

###############################################################################

def get_um_data(exp,opts):
    '''gets UM data, converts to xarray and local time'''

    print(f'processing {exp} (constraint: {opts["constraint"]})')

    # Operational model data
    if exp in ['BARRA-R2', 'BARRA-C2']:
        print('BARRA not yet implemented')
        # da = get_barra_data(ds,opts,exp)
    else:
        fpath = f"{exp_paths[exp]}/{opts['fname']}*"
        try:
            cb = iris.load_cube(fpath, constraint=opts['constraint'])
            # fix timestamp/bounds error in accumulations
            if cb.coord('time').bounds is not None:
                print('WARNING: updating time point to right bound')
                cb.coord('time').points = cb.coord('time').bounds[:,1]
            da = xr.DataArray().from_iris(cb)
        except Exception as e:
            print(f'trouble opening {fpath}')
            print(e)
            return None

        # fix time dimension name if needed
        if 'time' not in da.dims:
            print('WARNING: updating time dimension name from dim_0')
            da = da.swap_dims({'dim_0': 'time'})

        da = filter_odd_times(da)

        if opts['constraint'] in [
            'air_temperature', 
            'soil_temperature', 
            'dew_point_temperature', 
            'surface_temperature'
            ]:

            print('converting from K to °C')
            da = da - 273.15
            da.attrs['units'] = '°C'

        if opts['constraint'] in ['stratiform_rainfall_flux_mean']:
            print('converting from mm/s to mm/h')
            da = da * 3600.
            da.attrs['units'] = 'mm/h'

        if opts['constraint'] in ['moisture_content_of_soil_layer']:
            da = da.isel(depth=opts['level'])

        # print(da.head())

    return da

def filter_odd_times(da):

    if da.time.size == 1:
        return da

    minutes = da.time.dt.minute.values
    most_common_bins = np.bincount(minutes)
    most_common_minutes = np.flatnonzero(most_common_bins == np.max(most_common_bins))
    filtered = np.isin(da.time.dt.minute,most_common_minutes)
    filtered_da = da.sel(time=filtered)

    return filtered_da

if __name__ == "__main__":

    print('running variables:',variables)

    print('load dask')
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

    # folder in cylc-run name
    cycle_path = f'/scratch/ce10/mjl561/cylc-run/{cylc_id}/share/cycle'

    for variable in variables:
        print(f'processing {variable}')

        opts = cf.get_variable_opts(variable)

        cycle_list = sorted([x.split('/')[-2] for x in glob.glob(f'{cycle_path}/*/')])
        assert len(cycle_list) > 0, f"no cycles found in {cycle_path}"

        for exp in exps:
            out_dir = f'{datapath}/{opts["plot_fname"]}'

            # make directory if it doesn't exist
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            da_list = []
            for i,cycle in enumerate(cycle_list):
                print('========================')
                print(f'getting {exp} {i}: {cycle}\n')

                # set paths to experiment outputs
                reses = ['5','1','1_L']
                exp_paths = {
                    f'E5L_11p1_CCI': f'{cycle_path}/{cycle}/ERA5LAND_CCI/SY_11p1/GAL9/um',
                    f'E5L_11p1_CCI_WC': f'{cycle_path}/{cycle}/ERA5LAND_CCI_WC/SY_11p1/GAL9/um',
                    f'BR2_12p2_CCI':  f'{cycle_path}/{cycle}/BARRA_CCI/SY_12p2/GAL9/um',
                    f'BR2_12p2_CCI_WC':  f'{cycle_path}/{cycle}/BARRA_CCI_WC/SY_12p2/GAL9/um',
                    f'BR2_12p2_IGBP' : f'{cycle_path}/{cycle}/BARRA_IGBP/SY_12p2/GAL9/um',
                    f'BR2_12p2_CCI_no_urban': f'{cycle_path}/{cycle}/BARRA_CCI_no_urban/SY_12p2/GAL9/um',
                }|{
                    f'E5L_{res}_CCI': f'{cycle_path}/{cycle}/ERA5LAND_CCI/SY_{res}/RAL3P2/um' for res in reses
                }|{
                    f'E5L_{res}_CCI_WC': f'{cycle_path}/{cycle}/ERA5LAND_CCI_WC/SY_{res}/RAL3P2/um' for res in reses
                }|{
                    f'BR2_{res}_CCI':  f'{cycle_path}/{cycle}/BARRA_CCI/SY_{res}/RAL3P2/um' for res in reses
                }|{
                    f'BR2_{res}_CCI_WC':  f'{cycle_path}/{cycle}/BARRA_CCI_WC/SY_{res}/RAL3P2/um' for res in reses
                }|{
                    f'BR2_{res}_IGBP' : f'{cycle_path}/{cycle}/BARRA_IGBP/SY_{res}/RAL3P2/um' for res in reses
                }|{
                    f'BR2_{res}_CCI_no_urban': f'{cycle_path}/{cycle}/BARRA_CCI_no_urban/SY_{res}/RAL3P2/um' for res in reses
                }

                # check if first exp in exp_path directory exists, if not drop the cycle from cyclSY_e_list
                if not os.path.exists(exp_paths[exp]):
                    print(f'path {exp_paths[exp]} does not exist')
                    cycle_list.remove(cycle)
                    continue

                # check if any of the variables files are in the directory
                if len(glob.glob(f"{exp_paths[exp]}/{opts['fname']}*")) == 0:
                    print(f'no files in {exp_paths[exp]}')
                    cycle_list.remove(cycle)
                    continue

                da = get_um_data(exp,opts)
                if da is None:
                    print(f'WARNING: no data found at {cycle}')
                else:
                    da_list.append(da)

                # for time invarient variables (land_sea_mask) only get the first cycle
                if variable == 'land_sea_mask':
                    print('land_sea_mask only needs one cycle')
                    break

            print('concatenating, adjusting, computing data')
            ds = xr.concat(da_list, dim='time')
            # da = da.compute()

            # set decimal precision to reduce filesize (definded fmt precision +1)
            precision = int(opts['fmt'].split('.')[1][0]) + 1
            ds = ds.round(precision)

            # drop unessasary dimensions
            if 'forecast_period' in ds.coords:
                ds = ds.drop_vars('forecast_period')
            if 'forecast_reference_time' in ds.coords:
                ds = ds.drop_vars('forecast_reference_time')

            # chunk to optimise save
            if len(ds.dims)==3:
                itime, ilon, ilat = ds.shape
                ds = ds.chunk({'time':24,'longitude':ilon,'latitude':ilat})
            elif len(ds.dims)==2:
                ilon, ilat = ds.shape
                ds = ds.chunk({'longitude':ilon,'latitude':ilat})

            # encoding
            ds.time.encoding.update({'dtype':'int32'})
            ds.longitude.encoding.update({'dtype':'float32', '_FillValue': -999})
            ds.latitude.encoding.update({'dtype':'float32', '_FillValue': -999})
            ds.encoding.update({'zlib':'true', 'shuffle': True, 'dtype':opts['dtype'], '_FillValue': -999})

            fname = f'{out_dir}/{exp}_{opts["plot_fname"]}.nc'
            print(f'saving to netcdf: {fname}')
            ds.to_netcdf(fname, unlimited_dims='time')

    toc = time.perf_counter() - tic

    print(f"Timer {toc:0.4f} seconds")
