__version__ = "2025-01-21"
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
stationpath = '/g/data/gb02/mf9078/AWS_5mindata_38stations'


regrid_to_template = True
template_exp = 'E5L_1_L_CCI_WC'
template_exp = 'BR2_1_L_CCI_WC'

variables = ['surface_temperature','soil_moisture_l1','soil_moisture_l2','soil_moisture_l3','soil_moisture_l4']
variables = ['surface_temperature']
variables = ['latent_heat_flux']
variables = ['air_temperature']

exps = [
        ### Parent models ###
        # 'E5L_11p1_CCI',
        # 'BR2_12p2_CCI',
        # ## ERA5-Land CCI ###
        # 'E5L_5_CCI',
        # 'E5L_1_CCI',
        # 'E5L_1_L_CCI',
        # ### ERA5-Land CCI WordCover ###
        # 'E5L_5_CCI_WC',
        # 'E5L_1_CCI_WC',
        # 'E5L_1_L_CCI_WC',
        # ### BARRA CCI ###
        # 'BR2_5_CCI',
        # 'BR2_1_CCI',
        # 'BR2_1_L_CCI',
        # ### BARRA CCI WorldCover ###
        # 'BR2_5_CCI_WC',
        # 'BR2_1_CCI_WC',
        'BR2_1_L_CCI_WC',
        # # ### BARRA IGBP ###
        # 'BR2_5_IGBP',
        # 'BR2_1_IGBP',
        # 'BR2_1_L_IGBP',
        # ### BARRA CCI no urban ###
        # 'BR2_5_CCI_no_urban',
        # 'BR2_1_CCI_no_urban',
        # 'BR2_1_L_CCI_no_urban',
        ]

################## functions ##################

def main_plotting():

    # select UTC = 1 or midday AEST
    # dss = ds.sel(time=ds.time.dt.hour==3)
    dss = ds.isel(time=1)

    # vopts = cf.update_opts(opts,
    #             vmin=15,
    #             vmax=30,
    #             cmap='turbo',
    #         )

    fig, fname = cf.plot_spatial(exps, dss, opts, sids, stations, obs, slabels=False,
                 fill_obs=True)
    fig, fname = cf.plot_spatial_difference(exps[0],exps[2], dss, vopts, sids=[], stations=None, obs=None)

    fig.savefig(f'{plotpath}/{fname}', dpi=300, bbox_inches='tight')

    return

def main_animation(suffix):

    vopts = cf.update_opts(opts,
                vmin=10,
                vmax=45,
                cmap='Spectral_r',
            )

    # lsm_ds = open_output_netcdf(exps,'land_sea_mask')
    # # mask ds with lsm
    # masked = xr.Dataset()
    # for exp in exps:
    #     masked[exp] = ds[exp].where(lsm_ds[exp].isel(time=0)==0)

    cf.plot_spatial_anim(exps,ds,vopts,sids,stations,obs,plotpath,slabels=False,fill_obs=True,distance=200)
    fnamein = f"{plotpath}/{opts['plot_fname']}_spatial*.png"
    fnameout = f"{plotpath}/{opts['plot_fname']}_spatial{suffix}"
    cf.make_mp4(fnamein,fnameout,fps=48,quality=26)

    # # make gif
    # command = f'convert -delay 5 -loop 0 {fnamein} {fnameout}.gif'
    # os.system(command)
    
    # remove all spatial png files
    for file in glob.glob(fnamein):
        os.remove(file)

    return

def _plot_stations(ds, obs, sids, stations, opts, suffix):

    # all stations
    fig, fname, all_stats = cf.plot_all_station_timeseries(ds, obs, sids, exps, stations, opts, 3, suffix)
    fig.savefig(fname, bbox_inches='tight', dpi=200)
    # # avg stations
    # fig,fname, _ = cf.plot_station_data_func(ds, obs, sids, exps ,stations, opts, 'mean', suffix)
    # fig.savefig(fname,bbox_inches='tight',dpi=200)
    # fig,fname, _ = cf.plot_station_data_avg(ds, obs, sids, exps, stations, opts, suffix)
    # fig.savefig(fname,bbox_inches='tight',dpi=200)
    # # bias
    # fig,fname, _ = cf.plot_station_data_bias(ds, obs, sids, exps, stations, opts, suffix)
    # fig.savefig(fname,bbox_inches='tight',dpi=200)
    # save stats to csv
    # all_stats['mean'] = all_stats.mean(axis=1)
    # all_stats.to_csv(f"{plotpath}/{opts['case']}_{opts['constraint']}_allstats{suffix}.csv")

def open_output_netcdf(exps,opts,variable):
    """
    Open the netcdf files for the experiments and variable

    Args:
    exps: list of experiments
    variable: variable to open
    
    """

    print(f'attempting to open {len(exps)} experiments:')
    print(exps)
   
    ds = xr.Dataset()
    for i,exp in enumerate(exps):
        fname = f'{datapath}/{opts["plot_fname"]}/{exp}_{opts["plot_fname"]}.nc'
        print(f'{i+1}: checking {fname}')
        if os.path.exists(fname):
            print(f'  opening {variable} {exp}')
            try:
                da = xr.open_dataset(fname)[opts['constraint']]
                if regrid_to_template:
                    if i==0:
                        # set template to interpolate
                        template_fpath = f'{datapath}/{opts["plot_fname"]}/{template_exp}_{opts["plot_fname"]}.nc'
                        template = xr.open_dataset(template_fpath)[opts['constraint']]
                    if exp != template_exp:
                        print(f'  regridding to {template_exp}')
                        ds[exp] = da.interp_like(template, method='nearest')
                    else:
                        ds[exp] = da
                else:
                    ds[exp] = da
            except Exception as e:
                print(f'failed to open {fname} {e}')
                print(f'removing {exp} from exps')
                exps.remove(exp)

    return ds

def trim_sids(sids):
    
    # remove sids outside ds extent
    xdsmin,ydsmin,xdsmax,ydsmax =cf.get_bounds(ds)
    sids = [sid for sid in sids if (stations.loc[sid,'lon']>xdsmin) and (stations.loc[sid,'lon']<xdsmax)
                and (stations.loc[sid,'lat']>ydsmin) and (stations.loc[sid,'lat']<ydsmax)]
    
    # remove those without obs data
    sdate,edate = pd.Timestamp(ds.time.min().values),pd.Timestamp(ds.time.max().values)
    # remove any column that is all nan between sdate and edate
    sids = [sid for sid in sids if not (obs.loc[sdate:edate, sid].isna().all())]

    return sids

def set_up_plot_attrs(exps, plotpath):

    # predefine colours for experiments
    exp_colours = {
        'E5L_11p1_CCI'    : 'gold',
        'BR2_12p2_CCI'    : 'cyan',
        'BARRA-R2'        : 'grey',
        'ACCESS-G'        : 'grey',
        }

    exp_plot_titles = {
        'ACCESS-G'       : 'ACCESS-G3 Global'
        }

    # drop keys not in exps
    exp_colours = {key: exp_colours[key] for key in exp_colours if key in exps}
    exp_plot_titles = {key: exp_plot_titles[key] for key in exp_plot_titles if key in exps}
    
    # create entry in exp_plot_titles if key not in exps
    extra_colours = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'grey','cyan'][::-1]
    for exp in exps:
        if exp not in exp_colours:
            colour = extra_colours.pop()
            while colour in list(exp_colours.values()):
                colour = extra_colours.pop()
            exp_colours[exp] = colour

        if exp not in exp_plot_titles:
            exp_plot_titles[exp] = exp
    
    cf.plotpath = plotpath
    cf.exp_colours = exp_colours
    cf.exp_plot_titles = exp_plot_titles

    return exp_colours, exp_plot_titles

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

    exp_colours, exp_plot_titles = set_up_plot_attrs(exps, plotpath)

    # check if plotpath exists, make if necessary
    if not os.path.exists(plotpath):
        os.makedirs(plotpath)

    for variable in variables:
        print(f'processing {variable}')

        opts = cf.get_variable_opts(variable)
        ds = open_output_netcdf(exps,opts,variable)

        #### get observations ####
        if variable in ['sensible_heat_flux','latent_heat_flux','soil_moisture_l1','soil_moisture_l2']:
            print('getting flux obs')
            obs, stations = cf.get_flux_obs(variable, local_time_offset=None)
        else:
            print('getting station obs')
            obs, stations = cf.process_station_netcdf(variable, stationpath, local_time_offset=11)
        ###     obs, stations = cf.get_station_obs(stationpath, opts, local_time_offset=None, resample=opts['obs_period'], method='instant')

        # trim obs dataframe to ds time period
        obs = obs.loc[ds.time.min().values:ds.time.max().values]

        # select only obs that align with ds model times
        obs = obs.loc[ds.time.values]

        # select all stations
        sids, suffix = stations.index.tolist(), '_all'

        # trim sids to those in ds
        sids, suffix = trim_sids(sids), '_trimmed'

        # # special sids
        # sids, suffix = ['066062','066137','070351'], '_special'

        # # plotting
        # vopts = cf.update_opts(opts,
        #     vmin=10,
        #     vmax=45,
        #     )
        # _plot_stations(ds, obs, sids, stations, vopts, suffix)

        sids = ['067105','066212','070330','061260','066059','068239','063291','066043','061078','070351','066194','066161','061425','063303','067113']

        main_animation(suffix=exps[0])
    
    print('done!')

