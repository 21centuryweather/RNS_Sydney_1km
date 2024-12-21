__version__ = "2024-12-21"
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
warnings.simplefilter(action='ignore', category=FutureWarning)

tic = time.perf_counter()

variables = ['surface_temperature','soil_moisture_l1','soil_moisture_l2','soil_moisture_l3','soil_moisture_l4',]

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
            return xr.DataArray()

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

        print(da.head())

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

def get_variable_opts(variable):
    '''standard variable options for saving. to be updated within master script as needed'''

    # standard ops
    opts = {
        'constraint': variable,
        'plot_title': variable.capitalize().replace('_',' '),
        'plot_fname': variable.replace(' ','_'),
        'fname'     : 'umnsaa_pvera',
        'dtype'     : 'float32'
        }
    
    if variable == 'air_temperature':
        opts.update({
            'constraint': 'air_temperature',
            'plot_title': 'Air temperature (1.5 m)',
            'plot_fname': 'air_temperature_1p5m',
            'units'     : '°C',
            'fname'     : 'umnsaa_pvera',
            })
        
    if variable == 'upward_air_velocity':
        opts.update({
            'constraint': 'upward_air_velocity',
            'units'     : 'm s-1',
            'fname'     : 'umnsaa_pverb',
            })
        
    elif variable == 'updraft_helicity_max':
        opts.update({
            'constraint': 'm01s20i080',
            'plot_title': 'Maximum updraft helicity (2000-5000m)',
            'plot_fname': 'updraft_helicity_2000_5000m_max',
            'units'     : 'm2 s-2',
            })
        
    if variable == 'surface_altitude':
        opts.update({
            'constraint': 'surface_altitude',
            'units'     : 'm',
            'dtype'     : 'int16',
            })
        
    elif variable == 'Tdp':
        opts.update({
            'constraint': 'dew_point_temperature',
            'plot_title': 'Dew point temperature (1.5 m)',
            'plot_fname': 'dew_point_temperature_1p5m',
            'units'     : '°C',
            'fname'     : 'umnsaa_pvera',
            })

    elif variable == 'RH':
        opts.update({
            'constraint': 'relative_humidity',
            'plot_title': 'Relative humidity (1.5 m)',
            'plot_fname': 'relative_humidity_1p5m',
            'units'     : '%',
            'fname'     : 'umnsaa_pvera',
            'dtype'     : 'int16',
            })

    elif variable == 'Qair':
        opts.update({
            'constraint': 'm01s03i237',
            'plot_title': 'Specific humidity (1.5 m)',
            'plot_fname': 'specific_humidity_1p5m',
            'units'     : 'kg/kg',
            'fname'     : 'umnsaa_psurfc',
            })

    elif variable == 'Evap_soil':
        opts.update({
            'constraint': 'Evaporation from soil surface',
            'units'     : 'kg/m2/s',
            'fname'     : 'umnsaa_psurfc',
            })

    elif variable == 'Qle':
        opts.update({
            'constraint': 'surface_upward_latent_heat_flux',
            'plot_title': 'Latent heat flux',
            'plot_fname': 'latent_heat_flux',
            'units'     : 'W/m2',
            'fname'     : 'umnsaa_psurfa',
            })
        
    elif variable == 'Qh':
        opts.update({
            'constraint': 'surface_upward_sensible_heat_flux',
            'plot_title': 'Sensible heat flux',
            'plot_fname': 'sensible_heat_flux',
            'units'     : 'W/m2',
            'fname'     : 'umnsaa_psurfa',
            })

    elif variable == 'soil_moisture_l1':
        opts.update({
            'constraint': 'moisture_content_of_soil_layer',
            'plot_title': 'Soil moisture (layer 1)',
            'plot_fname': 'soil_moisture_l1',
            'units'     : 'kg/m2',
            'level'     : 0,
            'fname'     : 'umnsaa_pverb',
            })

    elif variable == 'soil_moisture_l2':
        opts.update({
            'constraint': 'moisture_content_of_soil_layer',
            'plot_title': 'Soil moisture (layer 2)',
            'plot_fname': 'soil_moisture_l2',
            'units'     : 'kg/m2',
            'level'     : 1,
            'fname'     : 'umnsaa_pverb',
            })

    elif variable == 'soil_moisture_l3':
        opts.update({
            'constraint': 'moisture_content_of_soil_layer',
            'plot_title': 'Soil moisture (layer 3)',
            'plot_fname': 'soil_moisture_l3',
            'units'     : 'kg/m2',
            'level'     : 2,
            'fname'     : 'umnsaa_pverb',
            })
    
    elif variable == 'soil_moisture_l4':
        opts.update({
            'constraint': 'moisture_content_of_soil_layer',
            'plot_title': 'Soil moisture (layer 4)',
            'plot_fname': 'soil_moisture_l4',
            'units'     : 'kg/m2',
            'level'     : 3,
            'fname'     : 'umnsaa_pverb',
            })

    elif variable == 'surface_temperature':
        opts.update({
            'constraint': 'surface_temperature',
            'units'     : '°C',
            'fname'     : 'umnsaa_pvera',
            })

    elif variable == 'soil_temperature_l1':
        opts.update({
            'constraint': 'soil_temperature',
            'plot_title': 'soil temperature (5cm)',
            'plot_fname': 'soil_temperature_5cm',
            'units'     : '°C',
            'level'     : 0.05,
            'fname'     : 'umnsaa_pverb',
            })

    elif variable == 'wind_speed_of_gust':
        opts.update({
            'constraint': 'wind_speed_of_gust',
            'units'     : 'm/s',
            'fname'     : 'umnsaa_pvera',
            })

    elif variable == 'wind':
        opts.update({
            'constraint': 'wind',
            'units'     : 'm/s',
            'fname'     : 'umnsaa_pvera',
            })
        
    elif variable == 'surface_runoff_amount':
        opts.update({
            'constraint': 'surface_runoff_amount',
            'units'     : 'kg m-2',
            'fname'     : 'umnsaa_psurfb',
            })

    elif variable == 'fog_area_fraction':
        opts.update({
            'constraint': 'fog_area_fraction',
            'units'     : '-',
            'fname'     : 'umnsaa_pvera',
            })

    elif variable == 'total_precipitation_rate':
        opts.update({
            'constraint': iris.Constraint(
                name='precipitation_flux', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'precipitation rate',
            'units'     : 'kg m-2',
            'fname'     : 'umnsaa_pverb',
            })
                
    elif variable == 'precipitation_amount_accumulation':
        opts.update({
            'constraint': iris.Constraint(
                name='precipitation_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'precipitation accumulation',
            'units'     : 'kg m-2',
            'fname'     : 'umnsaa_pverb',
            })
        
    elif variable == 'convective_rainfall_amount_accumulation':
        opts.update({
            'constraint': iris.Constraint(
                name='convective_rainfall_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'units'     : 'kg m-2',
            'fname'     : 'umnsaa_pverb',
            })
        
    elif variable == 'stratiform_rainfall_amount_accumulation':
        opts.update({
            'constraint': iris.Constraint(
                name='stratiform_rainfall_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'units'     : 'kg m-2',
            'fname'     : 'umnsaa_pverb',
            })

    elif variable == 'daily_precipitation_amount':
        opts.update({
            'constraint': iris.Constraint(
                name='precipitation_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'units'     : 'mm per day',
            'fname'     : 'umnsaa_pverb',
            })

    elif variable == 'subsurface_runoff_amount':
        opts.update({
            'constraint': 'subsurface_runoff_amount',
            'fname'     : 'umnsaa_psurfb',
            })

    elif variable == 'land_mask':
        opts.update({
            'constraint': 'land_binary_mask',
            'units'     : 'm',
            'fname'     : 'umnsaa_pa000',
            'dtype'     : 'int16',
            })

    # add variable to opts
    opts.update({'variable':variable})

    return opts

if __name__ == "__main__":

    from dask.distributed import Client

    n_workers = int(os.environ['PBS_NCPUS'])
    worker_memory = (int(os.environ['PBS_VMEM']) / n_workers)
    local_directory = os.path.join(os.environ['PBS_JOBFS'], 'dask-worker-space')
    try:
        print(client)
    except Exception:
        client = Client(
            n_workers=n_workers,
            threads_per_worker=1, 
            memory_limit = worker_memory, 
            local_directory = local_directory)

    ################## get model data ##################

    datapath = '/g/data/ce10/users/mjl561/cylc-run/rns_ostia_SY_1km/netcdf'
    
    # folder in cylc-run name
    cylc_id = 'rns_ostia'
    cycle_path = f'/scratch/ce10/mjl561/cylc-run/{cylc_id}/share/cycle'
    exps = ['ERA5-LAND_11p1',
            'ERA5-LAND_5',
            'ERA5-LAND_1',
            'ERA5-LAND_1_L',
            'BARRA-R2_12p2',
            'BARRA-R2_5',
            'BARRA-R2_1',
            'BARRA-R2_1_L'
            ]
    
    for variable in variables:
        print(f'processing {variable}')

        opts = get_variable_opts(variable)

        cycle_list = sorted([x.split('/')[-2] for x in glob.glob(f'{cycle_path}/*/')])
        assert len(cycle_list) > 0, f"no cycles found in {cycle_path}"

        for exp in exps:
            fname = f'{datapath}/{exp}_{opts["plot_fname"]}.nc'
            # check if file has already been processed
            print(f'looking for {variable} {exp}')
            if not os.path.exists(fname):
                da_list = []
                for i,cycle in enumerate(cycle_list):
                    print('========================')
                    print(f'getting {exp} {i}: {cycle}\n')

                    exp_paths = {
                        'ERA5-LAND_11p1': f'{cycle_path}/{cycle}/ERA5LAND_CCI/SY_11p1/GAL9/um',
                        'ERA5-LAND_5':    f'{cycle_path}/{cycle}/ERA5LAND_CCI/SY_5/RAL3P2/um',
                        'ERA5-LAND_1':    f'{cycle_path}/{cycle}/ERA5LAND_CCI/SY_1/RAL3P2/um',
                        'ERA5-LAND_1_L':  f'{cycle_path}/{cycle}/ERA5LAND_CCI/SY_1_L/RAL3P2/um',
                        'BARRA-R2_12p2':  f'{cycle_path}/{cycle}/BARRA_CCI/SY_12p2/GAL9/um',
                        'BARRA-R2_5':     f'{cycle_path}/{cycle}/BARRA_CCI/SY_5/RAL3P2/um',
                        'BARRA-R2_1':     f'{cycle_path}/{cycle}/BARRA_CCI/SY_1/RAL3P2/um',
                        'BARRA-R2_1_L':   f'{cycle_path}/{cycle}/BARRA_CCI/SY_1_L/RAL3P2/um',
                    }

                    da_list.append(get_um_data(exp,opts))

                print('concatenating, adjusting, computing data')
                da = xr.concat(da_list, dim='time')
                # da = da.compute()

                print(f'saving to netcdf: {fname}')
                da.encoding.update({'zlib':'true', 'dtype':opts['dtype']})
                da.to_netcdf(fname)

    toc = time.perf_counter() - tic
    
    print(f"Timer {toc:0.4f} seconds")