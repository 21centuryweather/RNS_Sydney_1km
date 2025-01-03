import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colorbar import ColorbarBase
import matplotlib.patheffects as path_effects
import cartopy.crs as ccrs
import cartopy.geodesic as cgeo

def plot_spatial_anim(exps,ds,opts,sids,stations,obs,plotpath,
                      cbar_loc='right',slabels=False,
                      fill_obs=False,distance=100,fill_size=15,ncols=2,fill_diff=False,
                      show_mean=False,suffix=''):

    for i,time in enumerate(ds.time.values):
        print(f'{i+1} of {len(ds.time)}')
        dss = ds.isel(time=i)
        fig,fname = plot_spatial(exps,dss,opts,sids,stations,obs,cbar_loc,
                                 slabels,fill_obs,distance,fill_size,ncols,fill_diff,
                                 show_mean,suffix)
        fig.savefig(f'{plotpath}/{fname}', bbox_inches='tight', dpi=200)
        plt.close('all')

    return

def plot_spatial(exps, dss, opts, sids, stations, obs, cbar_loc='right', slabels=False,
                 fill_obs=False, distance=100, fill_size=15, ncols=2, fill_diff=False,
                 diff_vals=2, show_mean=True, suffix=''):

    if dss.time.size>1:
        print('dss includes time period')
        stime = pd.to_datetime(dss.time[0].values).strftime('%Y-%m-%d %H:%M')
        etime = pd.to_datetime(dss.time[-1].values).strftime('%Y-%m-%d %H:%M')
        print(f'calculating mean between {stime} and {etime}')
        dss = dss.mean(dim='time')
        time = f'{stime} - {etime}'
        timestamp = f'{stime} to {etime}'
    else:
        time = dss.time.values
        stime, etime = time, time
        if isinstance(time,np.datetime64):
            timestamp  = pd.to_datetime(time).strftime('%Y-%m-%d %H:%M')
        elif isinstance(time,np.ndarray):
            if opts['variable'] == 'landfrac':
                timestamp = suffix.replace('_','')
            else:
                timestamp = pd.to_datetime(time, format='%H').strftime('%H:%M')
        else:
            # from int to HH:MM format
            timestamp = pd.to_datetime(time, format='%H').strftime('%H:%M')

    tz = 'UTC'

    cbar_title = f"{opts['plot_title'].capitalize()} [{opts['units']}]"
    cmap       = opts['cmap']
    vmin = opts['vmin'] if opts['vmin'] is not None else dss.to_array().min().compute()
    vmax = opts['vmax'] if opts['vmax'] is not None else dss.to_array().max().compute()

    ncols = 1 if len(exps)==1 else ncols
    rows = len(exps)//ncols + len(exps)%ncols
    height = rows*4+0.5
    width = 5.5 * ncols

    proj = ccrs.PlateCarree()

    #############################################

    print(f"plotting {opts['variable']} {time}")

    plt.close('all')
    fig,axes = plt.subplots(nrows=rows,ncols=ncols,figsize=(width,height),
                            sharey=True,sharex=True,
                            subplot_kw={'projection': proj},
                            )

    for i,exp in enumerate(exps):
        ax = axes.flatten()[i] if len(exps)>1 else axes
        print(exp)
        im = dss[exp].plot(ax=ax,cmap=cmap,vmin=vmin,vmax=vmax,add_colorbar=False, transform=proj)

        # # for cartopy
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.coastlines(color='0.85',linewidth=0.5,zorder=5)
        left, bottom, right, top = get_bounds(dss)
        ax.set_extent([left, right, bottom, top], crs=proj)

        ax = distance_bar(ax,distance)
        ax.set_title(exp)
        
        # show ticks on all subplots but labels only on first column and last row
        subplotspec = ax.get_subplotspec()
        ax.set_ylabel('latitude [degrees]') if subplotspec.is_first_col() else ax.set_ylabel('')
        ax.set_xlabel('longitude [degrees]') if subplotspec.is_last_row() else ax.set_xlabel('')
        ax.tick_params(axis='y', labelleft=subplotspec.is_first_col(), labelright=False, labelsize=8)
        ax.tick_params(axis='x', labelbottom=subplotspec.is_last_row(), labeltop=False, labelsize=8) 

        # if len(sids) > 0:
        #     # station labels and values
        #     obs_to_pass = obs.copy()

        #     if fill_diff:
        #         # select only exp within obs_to_pass dicts
        #         for sid in sids:
        #             lat,lon = stations.loc[sid,'lat'],stations.loc[sid,'lon']
        #             obs_to_pass[sid] = dss[exp].sel(latitude=lat,longitude=lon,method='nearest').to_pandas() - obs[sid].loc[stime:etime].mean().values
        #     print_station_labels(ax,im,obs_to_pass,sids,stations,time,opts,slabels,fill_obs,fill_size,fill_diff,diff_vals)

        # print mean spatial value on plot
        if show_mean:
            value_str = f"mean: {opts['fmt'].format(dss[exp].mean().values)} [{opts['units']}]"
            ax.text(0.01,0.99, value_str, fontsize=6, ha='left', va='top', color='0.65', transform=ax.transAxes)

        if subplotspec.is_last_col():
            cbar = custom_cbar(ax,im,cbar_loc)  
            cbar.ax.set_ylabel(cbar_title)
            cbar.ax.tick_params(labelsize=8)

    title = f"{opts['plot_title'].capitalize()} [{opts['units']}] {timestamp} {tz}"
    fig.suptitle(title,y=0.99)
    fig.subplots_adjust(left=0.08,right=0.9,bottom=0.06,top=0.90,wspace=0.05,hspace=0.15)

    timestamp = timestamp.replace(' ','_').replace(':','')
    fname = f"{opts['plot_fname']}_spatial_{timestamp}{suffix}.png"

    return fig,fname

def plot_spatial_difference(exp1,exp2,dss,opts,sids,stations,obs,
                            cbar_loc='right',cbar_levels=21,slabels=False,fill_obs=False,
                            distance=100,vmin=None,vmax=None,fill_size=15,cmap='seismic',suffix=''):
    import datetime
    if dss.time.size > 1:
        print('excluding missing periods')
        dss = dss.dropna(dim='time')
        print('determining time mean')
        dss_mean = dss[[exp1,exp2]].mean(dim='time')
        da = dss_mean[exp2] - dss_mean[exp1]
        time_start = dss.time.values[0]
        time_end = dss.time.values[-1]
        timestamp = "%s - %s" %(
            pd.to_datetime(time_start).strftime('%Y-%m-%d'),
            pd.to_datetime(time_end).strftime('%Y-%m-%d'))
        fname_timestamp =  timestamp.replace(' - ','_')

    elif dss.time.size == 1:
        da = dss[exp2] - dss[exp1]    
        time = dss.time.values
        if isinstance(time,np.ndarray):
            time = time.item()
        if isinstance(time,datetime.time):
            timestamp = time.strftime('%H:%M')
        else:
            timestamp  = pd.to_datetime(time).strftime('%Y-%m-%d %H:%M')
        fname_timestamp = timestamp.replace(' ','_').replace(':','')

    # set plot options
    if vmin is None or vmax is None:
        vmin = float(da.quantile(0.001))
        vmax = float(da.quantile(0.999))
        vmin = min(vmin,-vmax)
        vmax = max(-vmin,vmax)

    tz = '[UTC]'
    cbar_title = f"{opts['plot_title'].capitalize()} difference [{opts['units']}]"
    plot_title = f"{opts['plot_title'].capitalize()} [{opts['units']}]\n{exp2} minus {exp1}: {timestamp} {tz}"
    cmap       = opts['cmap']

    proj = ccrs.PlateCarree()

    #############################################

    print(f"plotting {opts['variable']} {timestamp}")
    plt.close('all')
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5.5,5),
                          sharey=True,sharex=True,
                          subplot_kw={'projection': proj})

    im = da.plot(ax=ax,cmap=cmap,levels=cbar_levels,vmin=vmin,vmax=vmax,add_colorbar=False,transform=proj)
    ax = distance_bar(ax, distance=distance)
    ax.set_title(plot_title)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # # for cartopy
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.coastlines(color='0.75',linewidth=0.5,zorder=5)
    left, bottom, right, top = get_bounds(dss)
    ax.set_extent([left, right, bottom, top], crs=proj)

    # station labels and values
    if len(sids) > 0:
        if dss.time.size == 1:
            print('plotting observation timestep')
            print_station_labels(ax,im,obs,sids,stations,timestamp,opts,slabels,fill_obs,fill_size)
        elif dss.time.size > 1:
            print('calculating mean of observations')
            print_mean_station_labels(ax,im,da,obs,sids,stations,time_start,time_end,opts['obs_key'],slabels,fill_obs,fill_size,
                                print_diff=False,print_val=True)

    # set colorbar
    cbar = custom_cbar(ax,im,cbar_loc)

    cbar.ax.set_ylabel(cbar_title)
    cbar.ax.tick_params(labelsize=8)

    ax.set_ylabel('latitude [degrees]')
    ax.set_xlabel('longitude [degrees]')

    # title = f"{opts['plot_title'].capitalize()} [{opts['units']}] \n{timestamp} [local]"
    # fig.suptitle(title,y=0.97)
    # fig.tight_layout()
    fig.subplots_adjust(left=0.08,right=0.9,bottom=0.06,top=0.90,wspace=0.05,hspace=0.15)

    plot_fname = f"{opts['plot_fname']}_spatial_diff_{exp1}_{exp2}_{fname_timestamp}{suffix}"
    fname = f'{plot_fname}.png'

    return fig,fname

def custom_cbar(ax,im,cbar_loc='right',ticks=None):
    """
    Create a custom colorbar
    """

    if cbar_loc == 'right':
        cax = inset_axes(ax,
            width='4%',  # % of parent_bbox width
            height='100%',
            loc='lower left',
            bbox_to_anchor=(1.05, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm = im.norm, ticks = ticks)

    elif cbar_loc == 'far_right':
        cax = inset_axes(ax,
            width='4%',  # % of parent_bbox width
            height='100%',
            loc='lower left',
            bbox_to_anchor=(1.25, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm = im.norm, ticks = ticks)
    
    else:
        cbar_loc == 'bottom'
        cax = inset_axes(ax,
            width='100%',  # % of parent_bbox width
            height='4%',
            loc='lower left',
            bbox_to_anchor=(0, -0.1, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm = im.norm, orientation='horizontal', ticks = ticks)

    return cbar

def distance_bar(ax,distance=100):
    """
    Add a distance bar to the plot with geodesic distance in km
    """

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    xdist = abs(xlims[1]-xlims[0])
    offset = 0.03*xdist

    # plot distance bar
    start = (xlims[0]+offset,ylims[0]+offset)
    end = cgeo.Geodesic().direct(points=start,azimuths=90,distances=distance*1000).flatten()
    ax.plot([start[0],end[0]],[start[1],end[1]], color='0.65', linewidth=1.5)
    ax.text(start[0]+offset/7,start[1]+offset/5, f'{distance} km', 
        fontsize=9, ha='left',va='bottom', color='0.65')

    return ax

def get_bounds(ds):
    """
    Make sure that the bounds are in the correct order
    """

    if 'latitude' in ds.coords:
        y_dim = 'latitude'
    elif 'lat' in ds.coords:
        y_dim = 'lat'
    if 'longitude' in ds.coords:
        x_dim = 'longitude'
    elif 'lon' in ds.coords:
        x_dim = 'lon'

    left = float(ds[x_dim].min())
    right = float(ds[x_dim].max())
    top = float(ds[y_dim].max())
    bottom = float(ds[y_dim].min())

    resolution_y = (top - bottom) / (ds[y_dim].size - 1)
    resolution_x = (right - left) / (ds[x_dim].size - 1)

    top = round(top + resolution_y/2, 6)
    bottom = round(bottom - resolution_y/2, 6)
    right = round(right + resolution_x/2, 6)
    left = round(left - resolution_x/2, 6)

    if resolution_y < 0:
        top, bottom = bottom, top
    if resolution_x < 0:
        left,right = right,left

    return left, bottom, right, top

def make_mp4(fnamein,fnameout,fps=9,quality=26):
    '''
    Uses ffmpeg to create mp4 with custom codec and options for maximum compatability across OS.
        fnamein (string): The image files to create animation from, with glob wildcards (*).
        fnameout (string): The output filename (excluding extension)
        fps (float): The frames per second. Default 6.
        quality (float): quality ranges 0 to 51, 51 being worst.
    '''

    import glob
    import imageio.v2 as imageio

    # collect animation frames
    fnames = sorted(glob.glob(fnamein))
    if len(fnames)==0:
        print('no files found to process, check fnamein')
        return
    img_shp = imageio.imread(fnames[0]).shape
    out_h, out_w = img_shp[0],img_shp[1]

    # resize output to blocksize for maximum capatability between different OS
    macro_block_size=16
    if out_h % macro_block_size > 0:
        out_h += macro_block_size - (out_h % macro_block_size)
    if out_w % macro_block_size > 0:
        out_w += macro_block_size - (out_w % macro_block_size)

    # quality ranges 0 to 51, 51 being worst.
    assert 0 <= quality <= 51, "quality must be between 1 and 51 inclusive"

    # use ffmpeg command to create mp4
    command = f'ffmpeg -framerate {fps} -pattern_type glob -i "{fnamein}" \
        -vcodec libx264 -crf {quality} -s {out_w}x{out_h} -pix_fmt yuv420p -y {fnameout}.mp4'
    os.system(command)

    return f'completed, see: {fnameout}.mp4'

def update_opts(opts,**kwargs):
    '''update opts copy with kwargs: 
    e.g. opts = update_opts(opts, plot_title='new title')'''
    zopts = opts.copy()
    zopts.update(kwargs)
    return zopts

def get_variable_opts(variable):
    '''standard variable options for plotting. to be updated within master script as needed'''

    # standard ops
    opts = {
        'constraint': variable,
        'plot_title': variable.replace('_',' '),
        'plot_fname': variable.replace(' ','_'),
        'units'     : '?',
        'obs_key'   : 'None',
        'obs_period': '1H',
        'fname'     : 'umnsaa_pvera',
        'vmin'      : None, 
        'vmax'      : None,
        'cmap'      : 'viridis',
        'threshold' : None,
        'fmt'       : '{:.2f}',
        'dtype'     : 'float32'
        }
    
    if variable == 'air_temperature':
        opts.update({
            'constraint': 'air_temperature',
            'plot_title': 'air temperature (1.5 m)',
            'plot_fname': 'air_temperature_1p5m',
            'units'     : '°C',
            'obs_key'   : 'Tair',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 50,
            'cmap'      : 'inferno',
            'threshold' : 2,
            'fmt'       : '{:.2f}',
            })
        
    if variable == 'Qf':
        opts.update({
            'constraint': 'm01s03i721',
            'plot_title': 'anthropogenic heat flux',
            'plot_fname': 'anthrop_heat',
            'units'     : 'W m-2',
            'fname'     : 'umnsaa_psurfb',
            'vmin'      : 0,
            'vmax'      : 80,
            'cmap'      : 'inferno',
            'fmt'       : '{:.1f}',
            })
        
    if variable == 'upward_air_velocity':
        opts.update({
            'constraint': 'upward_air_velocity',
            'plot_title': 'upward air velocity',
            'plot_fname': 'upward_air_velocity',
            'units'     : 'm s-1',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : -1,
            'vmax'      : 1,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })
        
    elif variable == 'updraft_helicity_max':
        opts.update({
            'constraint': 'm01s20i080',
            'plot_title': 'maximum updraft helicity 2000-5000m',
            'plot_fname': 'updraft_helicity_2000_5000m_max',
            'units'     : 'm2 s-2',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pg',
            'vmin'      : 0,
            'vmax'      : 25,
            'cmap'      : 'turbo',
            'fmt'       : '{:.1f}',
            })
        
    if variable == 'surface_altitude':
        opts.update({
            'constraint': 'surface_altitude',
            'units'     : 'm',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pa000',
            'vmin'      : 0,
            'vmax'      : 2000,
            'cmap'      : 'twilight',
            'dtype'     : 'int16',
            'fmt'       : '{:.0f}',
            })
        
    elif variable == 'dew_point_temperature':
        opts.update({
            'constraint': 'dew_point_temperature',
            'plot_title': 'dew point_temperature (1.5 m)',
            'plot_fname': 'dew_point_temperature_1p5m',
            'units'     : '°C',
            'obs_key'   : 'Tdp',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : -10,
            'vmax'      : 30,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'relative_humidity':
        opts.update({
            'constraint': 'relative_humidity',
            'plot_title': 'relative humidity (1.5 m)',
            'plot_fname': 'relative_humidity_1p5m',
            'units'     : '%',
            'obs_key'   : 'RH',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.3f}',
            'dtype'     : 'float32',
            })

    elif variable == 'specific_humidity':
        opts.update({
            'constraint': 'm01s03i237',
            'plot_title': 'specific humidity (1.5 m)',
            'plot_fname': 'specific_humidity_1p5m',
            'units'     : 'kg/kg',
            'obs_key'   : 'Qair',
            'fname'     : 'umnsaa_psurfc',
            'vmin'      : 0.004,
            'vmax'      : 0.020,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.4f}',
            })

    elif variable == 'evaporation_from_soil_surface':
        opts.update({
            'constraint': 'Evaporation from soil surface',
            'plot_title': 'Evaporation from soil surface',
            'plot_fname': 'Evap_soil',
            'units'     : 'kg/m2/s',
            'fname'     : 'umnsaa_psurfc',
            'vmin'      : 0, 
            'vmax'      : 0.0002,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.4f}',
            })

    elif variable == 'latent_heat_flux':
        opts.update({
            'constraint': 'surface_upward_latent_heat_flux',
            'plot_title': 'Latent heat flux',
            'plot_fname': 'latent_heat_flux',
            'units'     : 'W/m2',
            'obs_key'   : 'Qle',
            # 'fname'     : 'umnsaa_pvera',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : -100, 
            'vmax'      : 500,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.1f}',
            })
        
    elif variable == 'sensible_heat_flux':
        opts.update({
            'constraint': 'surface_upward_sensible_heat_flux',
            'plot_title': 'Sensible heat flux',
            'plot_fname': 'sensible_heat_flux',
            'units'     : 'W/m2',
            'obs_key'   : 'Qh',
            # 'fname'     : 'umnsaa_pvera',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : -100, 
            'vmax'      : 600,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'soil_moisture_l1':
        opts.update({
            'constraint': 'moisture_content_of_soil_layer',
            'plot_title': 'soil moisture (layer 1)',
            'plot_fname': 'soil_moisture_l1',
            'units'     : 'kg/m2',
            'obs_key'   : 'soil_moisture_l1',
            'level'     : 0,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 50,
            'cmap'      : 'turbo_r',
            })

    elif variable == 'soil_moisture_l2':
        opts.update({
            'constraint': 'moisture_content_of_soil_layer',
            'plot_title': 'soil moisture (layer 2)',
            'plot_fname': 'soil_moisture_l2',
            'units'     : 'kg/m2',
            'obs_key'   : 'soil_moisture_l2',
            'level'     : 1,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'turbo_r',
            })
        
    elif variable == 'soil_moisture_l3':
        opts.update({
            'constraint': 'moisture_content_of_soil_layer',
            'plot_title': 'soil moisture (layer 3)',
            'plot_fname': 'soil_moisture_l3',
            'units'     : 'kg/m2',
            'obs_key'   : 'soil_moisture_l3',
            'level'     : 2,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 5,
            'vmax'      : 100,
            'cmap'      : 'turbo_r',
            })
        
    elif variable == 'soil_moisture_l4':
        opts.update({
            'constraint': 'moisture_content_of_soil_layer',
            'plot_title': 'soil moisture (layer 4)',
            'plot_fname': 'soil_moisture_l4',
            'units'     : 'kg/m2',
            'obs_key'   : 'soil_moisture_l4',
            'level'     : 3,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 5,
            'vmax'      : 100,
            'cmap'      : 'turbo_r',
            })

    elif variable == 'surface_temperature':
        opts.update({
            'constraint': 'surface_temperature',
            'plot_title': 'land surface temperature',
            'plot_fname': 'surface_temperature',
            'units'     : '°C',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 70,
            'cmap'      : 'inferno',
            })

    elif variable == 'soil_temperature_5cm':
        opts.update({
            'constraint': 'soil_temperature',
            'plot_title': 'soil temperature (5cm)',
            'plot_fname': 'soil_temperature_5cm',
            'units'     : '°C',
            'level'     : 0.05,
            'obs_key'   : 'Tsoil05',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 10,
            'vmax'      : 40,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'toa_outgoing_longwave_flux':
        opts.update({
            'constraint': 'toa_outgoing_longwave_flux',
            'plot_title': 'outgoing longwave flux (top of atmosphere)',
            'plot_fname': 'toa_longwave',
            'units'     : 'W/m2',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 50,
            'vmax'      : 400,
            'cmap'      : 'Greys_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'wind_speed_of_gust':
        opts.update({
            'constraint': 'wind_speed_of_gust',
            'plot_title': 'wind speed of gust',
            'plot_fname': 'wind_gust',
            'units'     : 'm/s',
            'obs_key'   : 'Wind_gust',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 10,
            'vmax'      : 50,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'wind':
        opts.update({
            'constraint': 'wind',
            'plot_title': 'wind speed',
            'plot_fname': 'wind',
            'units'     : 'm/s',
            'obs_key'   : 'wind',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 25,
            'cmap'      : 'turbo',
            'threshold' : 2.57,
            'fmt'       : '{:.2f}',
            })

    elif variable == 'ics_soil_albedo':
        opts.update({
            'constraint': 'soil_albedo',
            'plot_title': 'soil albedo (initial conditions)',
            'plot_fname': 'soil_albdo_ics',
            'units'     : '-',
            'fname'     : 'astart',
            'vmin'      : 0,
            'vmax'      : 0.5,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'radar_reflectivity':
        opts.update({
            'constraint': 'radar_reflectivity_due_to_all_hydrometeors_at_1km_altitude',
            'plot_title': 'Radar reflectivity at 1km',
            'plot_fname': 'radar_reflectivity',
            'units'     : 'dBZ',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0.,
            'vmax'      : 25.,
            'cmap'      : 'Greys_r',
            'fmt'       : '{:.1f}',
            })
        
    elif variable == 'air_pressure_at_sea_level':
        opts.update({
            'constraint': 'air_pressure_at_sea_level',
            'plot_title': 'air pressure at sea level',
            'plot_fname': 'air_pressure_at_sea_level',
            'units'     : 'Pa',
            'obs_key'   : 'SLP',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 96000,
            'vmax'      : 105000,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.1f}',
            })
        
    elif variable == 'surface_runoff_amount':
        opts.update({
            'constraint': 'surface_runoff_amount',
            'plot_title': 'surface runoff amount',
            'plot_fname': 'surface_runoff_amount',
            'units'     : 'kg m-2',
            'fname'     : 'umnsaa_psurfb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'Blues',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'fog_area_fraction':
        opts.update({
            'constraint': 'fog_area_fraction',
            'plot_title': 'fog fraction',
            'plot_fname': 'fog_fraction',
            'units'     : '-',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'Greys',
            'fmt'       : '{:.2f}',
            })
        
    elif variable == 'surface_net_downward_longwave_flux':
        opts.update({
            'constraint': 'surface_net_downward_longwave_flux',
            'plot_title': 'surface net longwave flux',
            'plot_fname': 'surface_net_longwave_flux',
            'units'     : 'W m-2',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : -250,
            'vmax'      : 50,
            'cmap'      : 'inferno',
            'fmt'       : '{:.1f}',
            })
        
    elif variable == 'visibility':
        opts.update({
            'constraint': 'visibility_in_air',
            'plot_title': 'visibility',
            'plot_fname': 'visibility',
            'units'     : 'm',
            'obs_key'   : 'visibility',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 12000,
            'cmap'      : 'viridis_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'total_precipitation_rate':
        opts.update({
            'constraint': iris.Constraint(
                name='precipitation_flux', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'precipitation rate',
            'plot_fname': 'prcp_rate',
            'units'     : 'kg m-2',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.1f}',
            })
                
    elif variable == 'precipitation_amount_accumulation':
        opts.update({
            'constraint': iris.Constraint(
                name='precipitation_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'precipitation accumulation',
            'plot_fname': 'prcp_accum',
            'units'     : 'kg m-2',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.1f}',
            })
        
    elif variable == 'convective_rainfall_amount_accumulation':
        opts.update({
            'constraint': iris.Constraint(
                name='convective_rainfall_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'convective rainfall amount accumulation',
            'plot_fname': 'conv_rain_accum',
            'units'     : 'kg m-2',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.1f}',
            })
        
    elif variable == 'stratiform_rainfall_amount_accumulation':
        opts.update({
            'constraint': iris.Constraint(
                name='stratiform_rainfall_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'stratiform rainfall amount accumulation',
            'plot_fname': 'strat_rain_accum',
            'units'     : 'kg m-2',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.1f}',
            })
        
    elif variable == 'daily_precipitation_amount':
        opts.update({
            'constraint': iris.Constraint(
                name='precipitation_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'daily precipitation amount',
            'plot_fname': 'daily_prcp',
            'units'     : 'mm per day',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.1f}',
            })
    
    elif variable == 'stratiform_rainfall_amount_10min':
        opts.update({
            'constraint': iris.Constraint(
                name='stratiform_rainfall_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'rain accumulation',
            'plot_fname': 'rain_accum_10min',
            'units'     : 'kg m-2',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_*_spec',
            'vmin'      : 0,
            'vmax'      : 200,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'stratiform_rainfall_flux_mean':
        opts.update({
            'constraint': iris.Constraint(
                name='stratiform_rainfall_flux', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'rain flux',
            'plot_fname': 'rain_flux',
            'units'     : r'mm h${^-1}$',
            'obs_key'   : 'precip_hour',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 32,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.5f}',
            })


    elif variable == 'low_type_cloud_area_fraction':
        opts.update({
            'constraint': iris.Constraint(
                name='low_type_cloud_area_fraction', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'low cloud fraction',
            'plot_fname': 'low_cloud_fraction',
            'units'     : '0-1',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.8f}',
            })

    elif variable == 'subsurface_runoff_amount':
        opts.update({
            'constraint': 'subsurface_runoff_amount',
            'plot_title': 'subsurface runoff amount',
            'plot_fname': 'subsurface_runoff_amount',
            'fname'     : 'umnsaa_psurfb',
            'vmin'      : None,
            'vmax'      : None,
            'cmap'      : 'cividis',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'landfrac':
        opts.update({
            'constraint': 'm01s00i216',
            'plot_title': 'land fraction',
            'plot_fname': 'land_fraction',
            'units'     : '1',
            'fname'     : 'astart',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'viridis',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'orography':
        opts.update({
            'constraint': 'surface_altitude',
            'plot_title': 'orography',
            'plot_fname': 'orography',
            'units'     : 'm',
            'fname'     : 'umnsaa_pa000',
            'vmin'      : 0,
            'vmax'      : 2500,
            'cmap'      : 'terrain',
            'fmt'       : '{:.0f}',
            })

    elif variable == 'land_sea_mask':
        opts.update({
            'constraint': 'land_binary_mask',
            'plot_title': 'land sea mask',
            'plot_fname': 'land_sea_mask',
            'units'     : 'm',
            'fname'     : 'umnsaa_pa000',
            'vmin'      : 0,
            'vmax'      : 1,
            'fmt'       : '{:.1f}',
            'dtype'     : 'int16',
            })

    # add variable to opts
    opts.update({'variable':variable})

    return opts