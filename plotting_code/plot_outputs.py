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
# stationpath = '/g/data/gb02/mf9078/AWS_5mindata_38stations'
stationpath = '/g/data/ce10/users/mjl561/observations/AWS/au-2000_2024_5min'

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
variables = ['dew_point_temperature']
variables = ['latent_heat_flux']
variables = ['specific_humidity']
variables = ['soil_moisture_l1']
variables = ['toa_outgoing_longwave_flux']
variables = ['air_temperature']

exps = [
        ### Parent models ###
        # 'E5L_11p1_CCI',
        # 'BR2_12p2_CCI',
        # ## ERA5-Land CCI ###
        # 'E5L_5_CCI',
        'E5L_1_CCI',
        # 'E5L_1_L_CCI',
        # ### ERA5-Land CCI WordCover ###
        # 'E5L_5_CCI_WC',
        # 'E5L_1_CCI_WC',
        # 'E5L_1_L_CCI_WC',
        # ### BARRA CCI ###
        # 'BR2_5_CCI',
        'BR2_1_CCI',
        # 'BR2_1_L_CCI',
        # ### BARRA CCI WorldCover ###
        # 'BR2_5_CCI_WC',
        # 'BR2_1_CCI_WC',
        # 'BR2_1_L_CCI_WC',
        # # ### BARRA IGBP ###
        # 'BR2_5_IGBP',
        # 'BR2_1_IGBP',
        # 'BR2_1_L_IGBP',
        # ### BARRA CCI no urban ###
        # 'BR2_5_CCI_no_urban',
        # 'BR2_1_CCI_no_urban',
        # 'BR2_1_L_CCI_no_urban',
        ### BARRA operational reanalysis ###
        # 'BARRA-R2',
        'BARRA-C2',
        ]

################## functions ##################

def main_plotting():

    # select UTC = 1 or midday AEST
    # dss = ds.sel(time=ds.time.dt.hour==3)
    dss = ds.isel(time=0)
    fig, fname = cf.plot_spatial(exps, dss, opts, sids, stations, obs, slabels=True,
        fill_obs=True, distance=250)
    # dss = ds.sel(latitude=slice(-35,-33), longitude=slice(150,152)).sel(time=slice(None,'2017-01-02'))

    # sids, suffix = ['068192', '066212', '066059', '067119', '066194', '068257', '066037', '066161', '066062', '066137', '061425', '067108', '067113', '063292'], '_sydney_select'

    for hour in range(0,24):

        dss = ds.copy()
        dss = dss.sel(time=slice('2017-01-11','2017-01-18'))
        dss = dss.sel(time=dss.time.dt.hour==hour)
        dss = dss.sel(latitude=slice(-34.4,-33.3), longitude=slice(150.2,151.6))
        dss = dss.compute()

        vopts = cf.update_opts(opts,
                    vmin=20,
                    vmax=35,
                    # cmap='turbo',
                )

        fig, fname = cf.plot_spatial(exps, dss, vopts, sids, stations, obs, slabels=True,
                    fill_obs=True, ncols=3, distance=25)
        # fig, fname = cf.plot_spatial_difference(exps[0],exps[2], dss, vopts, sids=[], stations=None, obs=None)

        fig.savefig(f'{plotpath}/{fname}', dpi=300, bbox_inches='tight')

    return

def _plot_stations(ds, obs, sids, stations, opts, suffix):

    # # all stations
    # fig, fname, all_stats = cf.plot_all_station_timeseries(ds, obs, sids, exps, stations, opts, 3, suffix)
    # fig.savefig(fname, bbox_inches='tight', dpi=200)
    # # avg stations
    fig,fname, _ = cf.plot_station_data_func_timeseries(ds, obs, sids, exps ,stations, opts, 'mean', suffix)
    fig.savefig(fname,bbox_inches='tight',dpi=200)
    fig,fname, _ = cf.plot_station_data_avg_timeseries(ds, obs, sids, exps, stations, opts, suffix)
    fig.savefig(fname,bbox_inches='tight',dpi=200)
    # # bias
    fig,fname, _ = cf.plot_station_data_bias_timeseries(ds, obs, sids, exps, stations, opts, suffix)
    fig.savefig(fname,bbox_inches='tight',dpi=200)
    # save stats to csv
    # all_stats['mean'] = all_stats.mean(axis=1)
    # all_stats.to_csv(f"{plotpath}/{opts['case']}_{opts['constraint']}_allstats{suffix}.csv")

def main_animation(suffix):

    vopts = cf.update_opts(opts,
                vmin=15,
                vmax=45,
                cmap='Spectral_r',
            )

    cf.plot_spatial_anim(exps,ds,opts,sids,stations,obs,plotpath,slabels=False,fill_obs=True,distance=200)
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

def create_soil_moisture_plots(suffix=''):

    for i,time in enumerate(ds.time.values):
        print(f'{i+1} of {len(ds.time)}')
        fig, fname = create_soil_moisture_plot(suffix,i)
        fig.savefig(f'{plotpath}/{fname}', bbox_inches='tight', dpi=200)
        plt.close('all')

    fnamein = f"{plotpath}/{opts['plot_fname']}_spatial*.png"
    fnameout = f"{plotpath}/{opts['plot_fname']}_spatial{suffix}"
    cf.make_mp4(fnamein,fnameout,fps=48,quality=26)

    # remove all spatial png files
    for file in glob.glob(fnamein):
        os.remove(file)

    return

def create_spatial_timeseries_plots(times, suffix='', plot_vs_obs=False):

    fnamein = f"{plotpath}/{opts['plot_fname']}_spatial*.png"
    fnameout = f"{plotpath}/{opts['plot_fname']}_spatial{suffix}"

    # remove all spatial png files
    for file in glob.glob(fnamein):
        os.remove(file)

    for i,time in enumerate(times):
        print(f'{i+1} of {len(times)}')
        if plot_vs_obs:
            fig, fname = create_spatial_timeseries_plot_vs_obs(suffix,i)
        else:
            fig, fname = create_spatial_timeseries_plot(suffix,i)
        fig.savefig(f'{plotpath}/{fname}', bbox_inches='tight', dpi=200)
        plt.close('all')

    cf.make_mp4(fnamein,fnameout,fps=48,quality=26)

    # remove all spatial png files
    for file in glob.glob(fnamein):
        os.remove(file)

    return

# def create_soil_moisture_plot(suffix='',itime=0):

#     # convert layer soil moisture mass to volumentric soil moisture
#     depths = {'soil_moisture_l1':0.05,'soil_moisture_l2':0.225,'soil_moisture_l3':0.675,'soil_moisture_l4':2}

#     # convert to volumetric soil moisture
#     dss = ds.isel(time=itime)/(depths[variable]*1000)
#     dss.attrs['units'] = 'm3/m3'

#     vopts = cf.update_opts(opts,
#         units='m3/m3',
#         vmin=0,
#         vmax=0.8,
#         cmap='Spectral',
#     )

#     fig, fname = cf.plot_spatial(exps, dss, vopts, sids, stations, obs, slabels=True,
#         fill_obs=True, ncols=2, distance=100, suffix=suffix)

#     # create new ax on the bottom of the current axes
#     axins = fig.add_axes([0.08, -0.4, 0.83, 0.3])
#     axins.set_title(f'Domain averaged {opts["plot_title"]}')
#     # calculate the spatially averaged timeseries for each experiment
#     ts = ds.mean(dim=['latitude','longitude'])/(depths[variable]*1000)
#     ts = ts.to_dataframe().drop(columns=['depth'])
#     # plot the timeseries
#     ts_plot = ts.plot(ax=axins)
#     # update axins legend
#     axins.legend([exp_plot_titles[exp] for exp in exps]+['site observations'], loc='upper right', fontsize=8)

#     for i,exp in enumerate(exps):
#         # get colour from ts_plot
#         exp_col = ts_plot.get_lines()[i].get_color()
#         # add point for current time on timeseries for each experiment
#         axins.scatter(ts.index[itime], ts.iloc[itime][exp], color=exp_col, s=20, clip_on=False)

#     return fig, fname

def create_spatial_timeseries_plot(suffix='', itime=0,  masked=False):

    lsm_opts = cf.get_variable_opts('land_sea_mask')
    lsm_ds = open_output_netcdf([exps[0]], lsm_opts, 'land_sea_mask')

    if masked:
         ds_masked = ds.where(lsm_ds[exps[0]].isel(time=0)==1)
    else:
        ds_masked = ds.copy()
    if itime is not None:
        dss_masked = ds_masked.isel(time=itime)
    else:
        dss_masked = ds_masked.copy()

    if itime is None:
        if variable in ['latent_heat_flux','sensible_heat_flux']:
            vopts = cf.update_opts(opts,
                vmin=0,
                vmax=400,
                # cmap='inferno'
            )
        elif variable in ['air_temperature']:
            vopts = cf.update_opts(opts,
                vmin=15,
                vmax=35,
                cmap='Spectral_r',
            )
        elif variable in ['dew_point_temperature']:
            vopts = cf.update_opts(opts,
                vmin=5,
                vmax=25,
                cmap='viridis',
            )
        elif variable in ['specific_humidity']:
            vopts = cf.update_opts(opts,
                vmin=0.006,
                vmax=0.016,
                # cmap='viridis',
            )
        else:
            vopts = opts.copy()
    else:
        if variable in ['latent_heat_flux','sensible_heat_flux']:
            vopts = cf.update_opts(opts,
                vmin=0,
                vmax=400,
                # cmap='inferno'
            )
        elif variable in ['air_temperature']:
            vopts = cf.update_opts(opts,
                vmin=10,
                vmax=45,
                cmap='Spectral_r',
            )
        elif variable in ['dew_point_temperature']:
            vopts = cf.update_opts(opts,
                vmin=5,
                vmax=25,
                cmap='viridis',
            )
        else:
            vopts = opts.copy()

    fig, fname = cf.plot_spatial(exps, dss_masked, vopts, [], stations, obs, slabels=False, fill_size=10,
        fill_obs=False, ncols=len(exps), distance=100, suffix=suffix)

    # create new ax on the bottom of the current axes
    axins = fig.add_axes([0.08, -0.4, 0.83, 0.3])
    axins.set_title(f'Domain averaged {opts["plot_title"]}')
    # calculate the spatially averaged timeseries for each experiment
    ts = ds_masked.mean(dim=['latitude','longitude'])
    ts = ts.to_dataframe()[exps]
    # if ts has multiindex, drop the second index
    if isinstance(ts.index, pd.MultiIndex):
        ts.index = ts.index.droplevel(1)

    # plot the timeseries
    ts_plot = ts.plot(ax=axins)
    # update axins legend
    axins.legend([exp_plot_titles[exp] for exp in exps], loc='upper right', fontsize=8, bbox_to_anchor=(1.0,-0.2))

    # get xticks and labels
    xticks = axins.get_xticks()

    if itime is not None:
        for i,exp in enumerate(exps):
            # get colour from ts_plot
            exp_col = ts_plot.get_lines()[i].get_color()
            # add point for current time on timeseries for each experiment
            axins.scatter(ts.index[itime], ts.iloc[itime][exp], color=exp_col, s=20, clip_on=False)

    return fig, fname

def create_spatial_timeseries_plot_vs_obs(suffix='', itime=0, masked=True):

    lsm_opts = cf.get_variable_opts('land_sea_mask')
    lsm_ds = open_output_netcdf([exps[0]], lsm_opts, 'land_sea_mask')

    # only pass obs if their lat/lon is within the lsm mask
    sids_to_pass = []
    for sid in sids:
        lat = stations.loc[sid,'lat']
        lon = stations.loc[sid,'lon']
        if lsm_ds[exps[0]].isel(time=0).sel(latitude=lat,longitude=lon, method = 'nearest')==1:
            sids_to_pass.append(sid)

    # get masked data
    if masked:
        ds_masked = ds.where(lsm_ds[exps[0]].isel(time=0)==1)
    else:
        ds_masked = ds.copy()
    if itime is not None:
        dss_masked = ds_masked.isel(time=itime)
    else:
        dss_masked = ds_masked.copy()
    # dss_masked['time'] = time

    if itime is None:
        if variable in ['latent_heat_flux','sensible_heat_flux']:
            vopts = cf.update_opts(opts,
                vmin=0,
                vmax=400,
                # cmap='inferno'
            )
        elif variable in ['air_temperature']:
            vopts = cf.update_opts(opts,
                vmin=15,
                vmax=35,
                cmap='Spectral_r',
            )
        elif variable in ['dew_point_temperature']:
            vopts = cf.update_opts(opts,
                vmin=5,
                vmax=25,
                cmap='viridis',
            )
        else:
            vopts = opts.copy()
    else:
        if variable in ['latent_heat_flux','sensible_heat_flux']:
            vopts = cf.update_opts(opts,
                vmin=0,
                vmax=400,
                # cmap='inferno'
            )
        elif variable in ['air_temperature']:
            vopts = cf.update_opts(opts,
                vmin=10,
                vmax=45,
                cmap='Spectral_r',
            )
        elif variable in ['dew_point_temperature']:
            vopts = cf.update_opts(opts,
                vmin=5,
                vmax=25,
                cmap='viridis',
            )
        else:
            vopts = opts.copy()


    ########## plot ##########

    fig, fname = cf.plot_spatial(exps, dss_masked, vopts, sids_to_pass, stations, obs, slabels=False, fill_size=10,
        fill_obs=True, ncols=len(exps), distance=100, suffix=suffix)

    fname = fname.split('.png')[0] + '_vs_obs.png'

    # create new ax on the bottom of the current axes
    axins = fig.add_axes([0.08, -0.4, 0.83, 0.3])
    axins.set_title(f'Site averaged {opts["plot_title"]}: {len(sids_to_pass)} sites')
    # calculate the site location averaged timeseries for each experiment

    sim_site_list = []
    for sid in sids_to_pass:
        lat = stations.loc[sid,'lat']
        lon = stations.loc[sid,'lon']
        sim_site = ds_masked.sel(latitude=lat,longitude=lon, method='nearest').to_dataframe()[exps]
        sim_site_list.append(sim_site)
    # combine sim_site_list and calculate average for each experiment
    sim_ts = pd.concat(sim_site_list).groupby(level=0).mean()

    # plot the timeseries
    ts_plot = sim_ts.plot(ax=axins)
    axins.set_ylabel(opts['units'])

    # now plot the site observation average
    obs_ts = obs[sids_to_pass].mean(axis=1)
    obs_ts.name = 'observations'
    obs_ts.plot(ax=axins, color='black', linestyle='--', label='observations')

    if itime is not None:
        for i,exp in enumerate(exps):
            # get colour from ts_plot
            exp_col = ts_plot.get_lines()[i].get_color()
            # add point for current time on timeseries for each experiment
            axins.scatter(sim_ts.index[itime], sim_ts.iloc[itime][exp], color=exp_col, s=20, clip_on=False)
            # add point for observations
            axins.scatter(sim_ts.index[itime], obs_ts.iloc[itime], color='black', s=20, clip_on=False)

    # get legend handles and labels
    handles, labels = axins.get_legend_handles_labels()
    # add MAE and BIAS to labels in exp
    for exp in exps:
        mae = cf.calc_MAE(sim_ts[exp], obs_ts)
        mbe = cf.calc_MBE(sim_ts[exp], obs_ts)
        labels[exps.index(exp)] = f'{exp_plot_titles[exp]}    MAE: {mae:.2f}, MBE: {mbe:.2f}]'
    axins.legend(handles, labels, loc='upper right', fontsize=8, bbox_to_anchor=(1.0,-0.2))

    #### diurnal plot ####
    if itime is None:
        df = pd.concat([sim_ts, obs_ts], axis=1)
        df_diurnal = df.groupby(df.index.hour).mean()
        # plot with observations column in black
        df_diurnal.plot(color=[exp_colours[exp] for exp in exps]+['black'])
        plt.title(f'Diurnally averaged {opts["plot_title"]}: {len(sids_to_pass)} sites')

    return fig, fname

def create_spatial_timeofday_plot(suffix='', hour=3):

    # # calculate diurnal averages
    # diurnal_obs = cf.calc_diurnal_obs(obs,ds)
    # if hour is not None:
    #     diurnal_sim = cf.calc_diurnal_sim(ds.sel(time=ds.time.dt.hour==hour))
    # else: 
    #     diurnal_sim = cf.calc_diurnal_sim(ds)

    lsm_opts = cf.get_variable_opts('land_sea_mask')
    lsm_ds = open_output_netcdf(exps, lsm_opts, 'land_sea_mask')

    # get masked data
    ds_masked = ds.where(lsm_ds.isel(time=0)==1)
    dss_masked = ds_masked.sel(time=ds_masked.time.dt.hour==hour)

    if variable in ['latent_heat_flux','sensible_heat_flux']:
        vopts = cf.update_opts(opts,
            vmin=0,
            vmax=400,
            # cmap='inferno'
        )
    elif variable in ['air_temperature']:
        vopts = cf.update_opts(opts,
            vmin=15,
            vmax=35,
            # cmap='turbo',
        )

    fig, fname = cf.plot_spatial(exps, dss_masked, vopts, sids, stations, obs, slabels=True,
        fill_obs=True, ncols=2, distance=100, suffix=suffix)

    # create new ax on the bottom of the current axes
    axins = fig.add_axes([0.08, -0.4, 0.83, 0.3])

    tz = 'AEST' if local_time else 'UTC'
    axins.set_title(f'Domain averaged {opts["plot_title"]} at {hour}:00 [{tz}]')

    # calculate the spatially averaged timeseries for each experiment
    ts = dss_masked.mean(dim=['latitude','longitude'])
    ts = ts.to_dataframe()[exps]
    # plot the timeseries
    ts_plot = ts.plot(ax=axins)
    # ts_plot.set_ylim(vopts['vmin'],vopts['vmax'])
    # update axins legend
    axins.legend([exp_plot_titles[exp] for exp in exps], loc='upper right', fontsize=8, bbox_to_anchor=(1.0,-0.2), framealpha=1)

    # if xlabel is at an angle, centre and rotate horizontal
    axins.set_xticklabels(axins.get_xticklabels(), rotation=0, ha='center')

    # for i,exp in enumerate(exps):
        # get colour from ts_plot
    #     exp_col = ts_plot.get_lines()[i].get_color()
    #     # add point for current time on timeseries for each experiment
    #     axins.scatter(ts.index[itime], ts.iloc[itime][exp], color=exp_col, s=20, clip_on=False)

    return fig, fname


def create_sst_animation(suffix='_sst'):

    lsm_opts = cf.get_variable_opts('land_sea_mask')
    lsm_ds = open_output_netcdf(exps, lsm_opts, 'land_sea_mask')
    
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
                try:
                    da = xr.open_dataset(fname)[opts['constraint']]
                except:
                    da = xr.open_dataset(fname)[variable]
                if i==0:
                    template_exp = exp
                    ds[exp] = da
                else:
                    # check if the coordinates match with the template_exp
                    if (da.longitude.equals(ds[template_exp].longitude) and da.latitude.equals(ds[template_exp].latitude)):
                        ds[exp] = da
                    else:
                        print(f'  regridding to {template_exp}')
                        ds[exp] = da.interp_like(ds[template_exp], method='nearest')

            except Exception as e:
                print(f'failed to open {fname} {e}')
                print(f'removing {exp} from exps')
                exps.remove(exp)
        elif 'BARRA' in exp:
            da = cf.get_barra_data(ds,opts,exp)
            ds[exp] = da.interp_like(ds[template_exp], method='nearest')

        else:
            print('no file found')

    return ds

def set_up_plot_attrs(exps, plotpath):

    # predefine colours for experiments
    exp_colours = {
        'E5L_11p1_CCI'    : 'tab:blue',
        'BR2_12p2_CCI'    : 'tab:orange',
        'BARRA-C2'        : 'tab:green',
        'BARRA-R2'        : 'green',
        'ACCESS-G'        : 'grey',
        }

    exp_plot_titles = {
        'ACCESS-G'       : 'ACCESS-G3 Global',
        'E5L_11p1_CCI'   : 'ERA5-Land 11.1km (CCI)',
        'E5L_5_CCI'      : 'ERA5-Land 5km (CCI)',
        'E5L_1_CCI'      : 'ERA5-Land 1km (CCI)',
        'E5L_1_L_CCI'    : 'ERA5-Land 1km (CCI large domain)',

        'BR2_12p2_CCI'   : 'BARRA-R2 12.2km (CCI)',
        'BR2_5_CCI'      : 'BARRA-R2 5km (CCI)',
        'BR2_1_CCI'      : 'BARRA-R2 1km (CCI)',
        'BR2_1_L_CCI'    : 'BARRA-R2 1km (CCI large domain)',
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

    return exp_colours, exp_plot_titles

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
        ds = ds.compute()

        #### get observations ####
        if variable in ['sensible_heat_flux','latent_heat_flux','soil_moisture_l1','soil_moisture_l2','soil_moisture_l3']:
            # print('getting flux obs')
            # obs, stations = cf.get_flux_obs(variable, local_time_offset=None)
            print('no obs available')
            # set up dummy obs and stations dataframes
            obs, stations = pd.DataFrame(), pd.DataFrame()
            sids, sufix = [], ''
        elif variable in ['air_temperature','dew_point_temperature']:
            obs, stations = cf.process_station_netcdf(variable, stationpath, local_time_offset=local_time_offset)
        else:
            print('no obs available')
            # set up dummy obs and stations dataframes
            obs, stations = pd.DataFrame(), pd.DataFrame()
            sids, sufix = [], ''

        # trim obs dataframe to ds time period
        obs = obs.loc[ds.time.min().values:ds.time.max().values]

        # select only obs that align with ds model times
        if not obs.empty:
            obs = obs.loc[ds.time.values]

        # select all stations
        sids, suffix = stations.index.tolist(), '_all'

        # trim sids to those in ds
        sids, suffix = cf.trim_sids(ds, obs, sids, stations), '_trimmed'

        if local_time: # convert to local time
            ds = ds.assign_coords(time=ds.time + pd.Timedelta(f'{local_time_offset}h'))
            obs.index = obs.index + pd.Timedelta(f'{local_time_offset}h')
            cf.local_time = local_time

        # # special sids
        # sids, suffix = ['066062','066137','070351'], '_special'

        # # plotting
        # vopts = cf.update_opts(opts,
        #     vmin=10,
        #     vmax=45,
        #     )
        # _plot_stations(ds, obs, sids, stations, vopts, suffix)

        # sids = ['067105','066212','070330','061260','066059','068239','063291','066043','061078','070351','066194','066161','061425','063303','067113']

        # main_animation(suffix=exps[0])

        # create_soil_moisture_plots(suffix='_volumetric_1km')

        # fig, fname = create_spatial_timeseries_plots(ds.time.values, suffix='_5km')

        # fig, fname = create_spatial_timeofday_plot(suffix='_5km', hour=3)
        # fig.savefig(f'{plotpath}/{fname}', bbox_inches='tight', dpi=200)

        # fig, fname = create_spatial_timeseries_plot(suffix='_1km_domain', itime=None)
        # fig.savefig(f'{plotpath}/{fname}', bbox_inches='tight', dpi=200)

        # fig, fname = create_spatial_timeseries_plot(suffix='_5km', itime=None)
        # fig.savefig(f'{plotpath}/{fname}', bbox_inches='tight', dpi=200)

        fig, fname = create_spatial_timeseries_plot_vs_obs(suffix='_1km', itime=None)
        fig.savefig(f'{plotpath}/{fname}', bbox_inches='tight', dpi=200)
        # plt.savefig(f'{plotpath}/diurnal_{variable}_12km.png', bbox_inches='tight', dpi=200)

        # timeseries_plot_vs_obs animation
        # create_spatial_timeseries_plots(ds.time.values[:48], suffix='_1km', plot_vs_obs=True)

        # _plot_stations(ds, obs, sids, stations, opts, suffix='_12km')
    
    print('done!')

