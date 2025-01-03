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

oshome=os.getenv('HOME')
sys.path.append(f'{oshome}/git/RNS_Sydney_1km/plotting_code')
import common_functions as cf
importlib.reload(cf)

################## set up ##################

oshome=os.getenv('HOME')
datapath = '/g/data/ce10/users/mjl561/cylc-run/rns_ostia_SY_1km/netcdf'
# plotpath = f'{oshome}/postdoc/02-Projects/P58_Sydney_1km/figures'
plotpath = '/g/data/ce10/users/mjl561/cylc-run/rns_ostia_SY_1km/figures'

regrid_to_template = False

variables = ['surface_temperature','soil_moisture_l1','soil_moisture_l2','soil_moisture_l3','soil_moisture_l4']
variables = ['air_temperature']
variables = ['surface_temperature']

exps = [
        ### ERA5-Land CCI ###
        'E5L_5_CCI',
        # 'E5L_1_CCI',
        # 'E5L_1_L_CCI',
        ### ERA5-Land CCI WordCover ###
        # 'E5L_5_CCI_WC',
        # 'E5L_1_CCI_WC',
        # 'E5L_1_L_CCI_WC',
        ### BARRA CCI ###
        'BR2_5_CCI',
        # 'BR2_1_CCI',
        # 'BR2_1_L_CCI',
        ### BARRA CCI WorldCover ###
        # 'BR2_5_CCI_WC',
        # 'BR2_1_CCI_WC',
        # 'BR2_1_L_CCI_WC',
        ### BARRA IGBP ###
        # 'BR2_5_IGBP',
        # 'BR2_1_IGBP',
        # 'BR2_1_L_IGBP',
        ### BARRA CCI no urban ###
        # 'BR2_5_CCI_no_urban',
        # 'BR2_1_CCI_no_urban',
        # 'BR2_1_L_CCI_no_urban',
        ]

template_exp = exps[0]

################## functions ##################

def main_plotting():

    # select UTC = 1 or midday AEST
    # dss = ds.sel(time=ds.time.dt.hour==3)
    dss = ds.isel(time=1)

    vopts = cf.update_opts(opts,
                vmin=15,
                vmax=30,
                cmap='turbo',
            )

    fig, fname = cf.plot_spatial(exps, dss, vopts, sids=[], stations=None, obs=None)
    fig, fname = cf.plot_spatial_difference(exps[0],exps[2], dss, vopts, sids=[], stations=None, obs=None)

    fig.savefig(f'{plotpath}/{fname}', dpi=300, bbox_inches='tight')

    return

def main_animation(suffix):

    vopts = cf.update_opts(opts,
                vmin=15,
                vmax=30,
                cmap='turbo',
            )

    # lsm_ds = open_netcdf(exps,'land_sea_mask')
    # # mask ds with lsm
    # masked = xr.Dataset()
    # for exp in exps:
    #     masked[exp] = ds[exp].where(lsm_ds[exp].isel(time=0)==0)

    cf.plot_spatial_anim(exps,ds,vopts,[],stations=None,obs=None,plotpath=plotpath,distance=250, slabels=False,
                      fill_obs=False)
    fnamein = f"{plotpath}/{opts['plot_fname']}_spatial*.png"
    fnameout = f"{plotpath}/{opts['plot_fname']}_spatial{suffix}"
    cf.make_mp4(fnamein,fnameout,fps=24,quality=26)

    # # make gif
    # command = f'convert -delay 5 -loop 0 {fnamein} {fnameout}.gif'
    # os.system(command)
    
    # remove all spatial png files
    for file in glob.glob(fnamein):
        os.remove(file)

    return

def open_netcdf(exps,opts,variable):
    """
    Open the netcdf files for the experiments and variable

    Args:
    exps: list of experiments
    variable: variable to open
    
    """
   
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
                else:
                    ds[exp] = da
            except Exception as e:
                print(f'failed to open {fname} {e}')
                print(f'removing {exp} from exps')
                exps.remove(exp)

    return ds

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

    # folder in cylc-run name
    cylc_id = 'rns_ostia'
    cycle_path = f'/scratch/ce10/mjl561/cylc-run/{cylc_id}/share/cycle'

    # check if plotpath exists, make if necessary
    if not os.path.exists(plotpath):
        os.makedirs(plotpath)

    for variable in variables:
        print(f'processing {variable}')

        opts = cf.get_variable_opts(variable)
        ds = open_netcdf(exps,variable)

