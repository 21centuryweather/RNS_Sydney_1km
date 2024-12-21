__version__ = "2024-12-21"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

"""
To plot the um outputs
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

sys.path.append(f'{oshome}/git/RNS_Sydney_1km/plotting_code')
import common_functions as cf
importlib.reload(cf)

################## set up ##################

regrid_to_template = True

variables = ['surface_temperature','soil_moisture_l1','soil_moisture_l2','soil_moisture_l3','soil_moisture_l4']
variables = ['soil_moisture_l2']

exps = ['ERA5-LAND_11p1',
        'ERA5-LAND_5',
        'ERA5-LAND_1',
        'BARRA-R2_12p2',
        'BARRA-R2_5',
        'BARRA-R2_1',
        'BARRA-R2_1_L',
        ]

exps = ['ERA5-LAND_5','BARRA-R2_5']
template_exp = exps[0]

################## functions ##################

def main_plotting():

    dss = ds.isel(time=0).compute()
    fig, fname = cf.plot_spatial(exps, dss, opts, sids=[], stations=None, obs=None)

    fig.savefig(f'{plotpath}/{fname}', dpi=300, bbox_inches='tight')

    return

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

    oshome=os.getenv('HOME')
    datapath = '/g/data/ce10/users/mjl561/cylc-run/rns_ostia_SY_1km/netcdf'

    # folder in cylc-run name
    cylc_id = 'rns_ostia'
    cycle_path = f'/scratch/ce10/mjl561/cylc-run/{cylc_id}/share/cycle'
    plotpath = f'{oshome}/postdoc/02-Projects/P58_Sydney_1km/figures'
    
    # check if plotpath exists, make if necessary
    if not os.path.exists(plotpath):
        os.makedirs(plotpath)

    for variable in variables:
        print(f'processing {variable}')

        opts = get_variable_opts(variable)

        ds = xr.Dataset()
        for i,exp in enumerate(exps):
            fname = f'{datapath}/{exp}_{opts["plot_fname"]}.nc'
            print(f'{i+1}: checking {fname}')
            if os.path.exists(fname):
                print(f'  opening {variable} {exp}')
                try:
                    da = xr.open_dataset(fname)[opts['constraint']]
                    if regrid_to_template:
                        if i==0:
                            # set template to interpolate
                            template_fpath = f'{datapath}/{template_exp}_{opts["plot_fname"]}.nc'
                            template = xr.open_dataset(template_fpath)[opts['constraint']]
                    
                        ds[exp] = da.interp_like(template, method='nearest')
                except Exception as e:
                    print(f'failed to open {fname} {e}')
                    print(f'removing {exp} from exps')
                    exps.remove(exp)




