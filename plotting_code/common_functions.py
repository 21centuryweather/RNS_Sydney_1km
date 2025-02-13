import os
import iris
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
import glob

def plot_spatial(exps, dss, opts, sids, stations, obs, cbar_loc='right', slabels=False,
                 fill_obs=False, distance=100, fill_size=15, ncols=2, fill_diff=False,
                 diff_vals=2, show_mean=True, suffix=''):

    # limit obs to pass to those matching ds timestamps
    if not obs.empty:
        obs_to_pass = obs.loc[dss.time.values]
    else:
        obs_to_pass = pd.DataFrame()

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

    tz = 'AEST' if local_time else 'UTC'

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
        im = dss[exp].plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False, transform=proj)

        # # for cartopy
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.coastlines(color='0.85',linewidth=0.5,zorder=5)
        left, bottom, right, top = get_bounds(dss)
        ax.set_extent([left, right, bottom, top], crs=proj)

        ax = distance_bar(ax,distance)
        ax.set_title(exp_plot_titles[exp])
        
        # show ticks on all subplots but labels only on first column and last row
        subplotspec = ax.get_subplotspec()
        ax.set_ylabel('latitude [degrees]') if subplotspec.is_first_col() else ax.set_ylabel('')
        ax.set_xlabel('longitude [degrees]') if subplotspec.is_last_row() else ax.set_xlabel('')
        ax.tick_params(axis='y', labelleft=subplotspec.is_first_col(), labelright=False, labelsize=8)
        ax.tick_params(axis='x', labelbottom=subplotspec.is_last_row(), labeltop=False, labelsize=8) 

        # station labels and values
        if len(sids) > 0:
            # remove sids outside ds extent
            xdsmin,ydsmin,xdsmax,ydsmax = get_bounds(dss)
            sids = [sid for sid in sids if (stations.loc[sid,'lon']>xdsmin) and (stations.loc[sid,'lon']<xdsmax)
                        and (stations.loc[sid,'lat']>ydsmin) and (stations.loc[sid,'lat']<ydsmax)]
            # if fill_diff: # difference between obs and sim
            #     for sid in sids:
            #         lat,lon = stations.loc[sid,'lat'],stations.loc[sid,'lon']
            #         obs_to_pass[sid] = dss[exp].sel(latitude=lat,longitude=lon,method='nearest').to_pandas() - obs_to_pass[sid]
            print_station_labels(ax,im,obs_to_pass,sids,stations,stime,etime,opts,slabels,fill_obs,fill_size,fill_diff,diff_vals)

        # print mean spatial value on plot
        if show_mean:
            value_str = f"model domain mean: {opts['fmt'].format(dss[exp].mean().values)} [{opts['units']}]"
            ax.text(0.01,0.99, value_str, fontsize=6, ha='left', va='top', color='0.65', transform=ax.transAxes)
            if len(sids) > 0:
                # add mean obs value if obs at least 95% complete
                obs_mean = obs_to_pass.mean(axis=None)
                value_str = f"obs mean: {opts['fmt'].format(obs_mean)} [{opts['units']}]"
                ax.text(0.01,0.96, value_str, fontsize=6, ha='left', va='top', color='0.65', transform=ax.transAxes)

                # calculate model bias at station locations
                sid_mean_list = []
                for sid in sids:
                    lat,lon = stations.loc[sid,'lat'],stations.loc[sid,'lon']
                    sid_mean = dss[exp].sel(latitude=lat,longitude=lon,method='nearest').to_pandas() - obs_to_pass[sid]
                    if sid_mean.size > 1:
                        sid_mean = sid_mean.mean()
                    sid_mean_list.append(sid_mean)
                sid_mean = pd.Series(sid_mean_list).mean()
                value_str = f"bias at stations: {opts['fmt'].format(sid_mean)} [{opts['units']}]"
                ax.text(0.01,0.93, value_str, fontsize=6, ha='left', va='top', color='0.65', transform=ax.transAxes)                

        if subplotspec.is_last_col():
            cbar = custom_cbar(ax,im,cbar_loc)  
            cbar.ax.set_ylabel(cbar_title)
            cbar.ax.tick_params(labelsize=8)

    title = f"{opts['plot_title'].capitalize()} [{opts['units']}] {timestamp} [{tz}]"
    fig.suptitle(title,y=0.99)
    fig.subplots_adjust(left=0.08,right=0.9,bottom=0.06,top=0.90,wspace=0.05,hspace=0.15)

    timestamp = timestamp.replace(' ','_').replace(':','')
    fname = f"{opts['plot_fname']}_spatial_{timestamp}_{tz}{suffix}.png"
    print('fname:', fname)

    return fig,fname

def print_station_labels(ax,im,obs_to_pass,sids,stations,stime,etime,opts,
                         slabels,fill_obs,fill_size,fill_diff=False,diff_vals=2):

    obs_val = None

    for sid in sids:
        lat,lon,name = stations.loc[sid,'lat'],stations.loc[sid,'lon'],stations.loc[sid,'name'].strip()
        if slabels:
            ax.annotate(text=name[:3], xycoords='data', xy=(lon,lat), xytext=(0,3),
                textcoords='offset points', fontsize=4,color='k',ha='center', zorder=10)
        if fill_obs:
            try:
                obs_val = obs_to_pass[sid]
                if isinstance(obs_val,pd.Series):
                    obs_val = obs_val.mean()
                if np.isnan(obs_val):
                    ax.plot(lon,lat,marker='.',mfc='None',mec='k',ms=2,mew=0.5)
                elif fill_diff:
                    im = ax.scatter(lon,lat, c=obs_val, cmap='coolwarm', vmin=-diff_vals, vmax=diff_vals,
                                    marker='o', s=fill_size, edgecolors='k', linewidth=0.5, zorder=10)
                    ax.annotate(text=opts['fmt'].format(obs_val), xycoords='data', xy=(lon,lat), xytext=(0,-3),
                        textcoords='offset points', fontsize=4,color='k',ha='center',va='top',zorder=10)
                    if ax.get_subplotspec().is_last_col():
                        cbar = custom_cbar(ax,im,cbar_loc='far_right')  
                        cbar.ax.set_ylabel('diff (sim - obs)')
                        cbar.ax.tick_params(labelsize=8)
                else:
                    ax.scatter(lon,lat, c=obs_val, norm=im.norm, cmap=im.cmap, marker='o', s=fill_size, edgecolors='k', linewidth=0.5, zorder=10)
            except Exception as e:
                print(f'exception plotting {sid} obs at {stime} to {etime}: {e}')
                ax.plot(lon,lat,marker='.',mfc='None',mec='k',ms=2,mew=0.5)
        else:
            ax.plot(lon,lat,marker='.',mfc='None',mec='k',ms=2,mew=0.5)

    return

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

    return f'completed, see: {plotpath}/{fname}'

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
        print_station_labels(ax,im,obs,sids,stations,stime,etime,opts,slabels,fill_obs,fill_size,fill_diff,diff_vals)
        # if dss.time.size == 1:
        #     print('plotting observation timestep')
        #     print_station_labels(ax,im,obs,sids,stations,stime,etime,opts,slabels,fill_obs,fill_size)
        # elif dss.time.size > 1:
        #     print('calculating mean of observations')
        #     print_mean_station_labels(ax,im,da,obs,sids,stations,time_start,time_end,opts['obs_key'],slabels,fill_obs,fill_size,
        #                         print_diff=False,print_val=True)

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

###### timeseries plots ######

def plot_all_station_timeseries(ds, obs, sids, exps, stations, opts, ncols=3, suffix=''):

    '''
    plot all station timeseries for a list of stations
    Args:
        ds (xr.Dataset): model data
        obs (dict): observations
        sids (list): station ids
        exps (list): experiments
        stations (pd.DataFrame): station metadata
        opts (dict): variable options
        ncols (int): number of columns for the plot
        suffix (str): suffix to add to the plot filename
    Returns:
        fig (plt.figure): figure object
        fname (str): filename of the plot
    '''

    plt.close('all')
    ncols = min(len(sids),ncols)   # min three columns
    nrows = ( len(sids) % ncols + len(sids) + 1) // ncols 
    fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1+ncols*5,1+nrows*3), sharey=True)
   
    all_stats = calc_all_stats(ds, obs, sids, exps, stations, opts)

    # long simulation
    if len(ds.time) > 100:
        marker = ''
    else: 
        marker = 'o'

    for sid,ax in zip(sids,axes.flatten()):
        station = stations.loc[sid,'name'].strip()
        lat,lon = stations.loc[sid,'lat'],stations.loc[sid,'lon']

        print(f'plotting {station} {opts["variable"]}')
        o = obs[sid]

        for exp in exps:

            s = ds[exp].sel(latitude=lat, longitude=lon,method='nearest').to_series()
            s.plot(ax=ax, label=exp, color=exp_colours[exp], alpha=0.8, marker=marker, ms=3)

        ####################################
        ### STATS ON PLOT 
        try:
            stats = all_stats[[sid]].T.stack(future_stack=True).loc[sid].T
            df = stats.map(lambda x: opts['fmt'].format(x)).rename(columns={
                    'MAE': f"MAE [{opts['units']}]",
                    'MBE': f"MBE [{opts['units']}]",
                    'threshold': f"<±{opts['threshold']} {opts['units']} [%]",
                    })
            yloc=0.99
            # exp names on plot
            result_str = '\n'.join(df.index.to_list())
            ax.text(0.01,yloc, '\n'+result_str, fontsize=7, ha='left', va='top', transform=ax.transAxes)
            # MAE stats on plot
            result_str = df.iloc[:,0].to_frame().to_string(index=False,header=True, justify='right')
            ax.text(0.22,yloc, result_str, fontsize=7, ha='right', va='top', transform=ax.transAxes)
            # MBE stats on plot
            result_str = df.iloc[:,1].to_frame().to_string(index=False,header=True, justify='right')
            ax.text(0.33,yloc, result_str, fontsize=7, ha='right', va='top', transform=ax.transAxes)
            # threshold stats on plot
            if opts['threshold'] is not None:
                result_str = df.iloc[:,-1].to_frame().to_string(index=False,header=True, justify='right')
                ax.text(0.47,yloc, result_str, fontsize=7, ha='right', va='top', transform=ax.transAxes)
        except Exception:
            pass
        ####################################

        if o.count()==0:
            print('no observations')
        else:
            o.plot(ax=ax, color='k', marker='o', ms=2, label='AWS observations')

        ax.set_xlim([s.index[0],s.index[-1]])
        ax.set_ylim((opts['vmin'],opts['vmax']))

        ax.set_title(station)

        ax.set_xlabel('')
        ax.grid(color='0.8',linewidth=0.5,which='both')

        ax.tick_params(axis='x',labelsize=8, which='both')


    handles, labels = ax.get_legend_handles_labels()
    leg = dict(zip(labels, handles))
    fig.legend(leg.values(),leg.keys(),loc='lower center',bbox_to_anchor=(0.5, -0.03),
               fontsize=10,ncol=len(exps)+1,framealpha=1)

    ####################################
    ### MEAN STATS ON PLOT 
    ax = axes.flatten()[-1]
    df = all_stats.mean(axis=1).unstack().map(lambda x: opts['fmt'].format(x)).rename(columns={
            'MAE': f"MAE [{opts['units']}]",
            'MBE': f"MBE [{opts['units']}]",
            'threshold': f"<±{opts['threshold']} {opts['units']} [%]",
            })
    yloc=-0.26
    xloc=0.50
    # exp names on plot
    result_str = '\n'.join(df.index.to_list())
    fig.text(0.01+xloc,yloc, '\n'+result_str, fontsize=7, ha='left', va='top', transform=ax.transAxes)
    # MAE stats on plot
    result_str = df.iloc[:,0].to_frame().to_string(index=False,header=True, justify='right')
    fig.text(0.22+xloc,yloc, result_str, fontsize=7, ha='right', va='top', transform=ax.transAxes)
    # MBE stats on plot
    result_str = df.iloc[:,1].to_frame().to_string(index=False,header=True, justify='right')
    fig.text(0.33+xloc,yloc, result_str, fontsize=7, ha='right', va='top', transform=ax.transAxes)
    # threshold stats on plot
    if opts['threshold'] is not None:
        result_str = df.iloc[:,-1].to_frame().to_string(index=False,header=True, justify='right')
        fig.text(0.47+xloc,yloc, result_str, fontsize=7, ha='right', va='top', transform=ax.transAxes)

    ####################################

    title = f"{opts['plot_title'].capitalize()} [{opts['units']}]"
    fig.suptitle(title,y=0.99,fontsize=14)

    fig.tight_layout(w_pad=0.05)

    print('mean stats')
    print(df)

    fname = f"{plotpath}/{opts['plot_fname']}_vs_obs_{len(sids)}_sites{suffix}.png"

    return fig,fname,all_stats

def plot_station_data_func_timeseries(ds, obs, sids, exps, stations, opts, func='mean', suffix=''):
    '''
    plot station data for a list of stations
    Args:
        ds (xr.Dataset): model data
        obs (dict): observations
        sids (list): station ids
        exps (list): experiments
        stations (pd.DataFrame): station metadata
        opts (dict): variable options
        func (str): function to apply to the data (mean, median, max, min)
        suffix (str): suffix to add to the plot filename
    Returns:
        fig (plt.figure): figure object
        fname (str): filename of the plot
    '''

    if len(sids)==0:
        print('no sids passed, exiting')

    o = []
    s = {exp:[] for exp in exps}

    lw, whichgrid, figwidth, yoff = 1, 'both', 9, 0.10
    if isinstance(ds.time.min().values,np.datetime64):
        sdate,edate = pd.Timestamp(ds.time.min().values), pd.Timestamp(ds.time.max().values)
        if (edate - sdate).days > 30.: 
            lw, whichgrid, figwidth, yoff = 0.5, 'major', 18, 0.06
    else:
        sdate,edate = ds.time.min().values.item(), ds.time.max().values.item()

    for sid in sids:
        station = stations.loc[sid,'name'].strip()
        print(f'gathering {station} data')
        o.append(obs.loc[sdate:edate,sid])
        lat,lon = stations.loc[sid,'lat'],stations.loc[sid,'lon']

        for exp in exps:
            s[exp].append(ds[exp].sel(latitude=lat, longitude=lon,method='nearest').to_series())
    s = {exp:pd.concat(s[exp],axis=1) for exp in exps}
    o = pd.concat(o,axis=1)

    #########################

    o = o.aggregate(func=func,axis=1)
    stats = pd.DataFrame()
    for exp in exps:  
        s[exp] = s[exp].aggregate(func=func,axis=1)

        mae = calc_MAE(s[exp],o)
        mbe = calc_MBE(s[exp],o)
        r = calc_R(s[exp],o)
        stats[exp] = pd.Series([mae,mbe,r],index=['MAE','MBE','R'])
        if opts['threshold'] is not None:
            within_threshold,_ = calc_percent_within_threshold(s[exp],o,threshold=2)
            stats.loc['threshold',exp] = within_threshold

    #########################

    plt.close('all')
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(figwidth,4))

    # metric_str = all_stats.unstack()[sids][metrics] # .to_string(float_format='%.2f',col_space=20,justify='right')

    o.plot(ax=ax, color='k', lw=lw, label=f'AWS observations ({func} avg)', zorder=10)

    for exp in exps:
        s[exp].plot(ax=ax,label=exp_plot_titles[exp], 
            color=exp_colours[exp], alpha=0.75 ,marker='o', ms=1, lw=0.5,
            ls='dotted' if 'sm' in exp else 'solid')

    ax.set_xlim([s[exp].index[0],s[exp].index[-1]])

    ymin,ymax = ax.get_ylim()
    rng = (ymax - ymin)*0.10
    ax.set_ylim((ymin,ymax+rng))

    # if opts['constraint'] == 'air_temperature':
    #     ax.set_ylim((opts['vmin']-5,opts['vmax']+5))
    # else:
    #     ax.set_ylim((opts['vmin'],opts['vmax']))

    ax.set_xlabel('')
    ax.grid(color='0.8',linewidth=0.5,which=whichgrid,axis='both')

    ax.tick_params(axis='x',labelsize=10, which='both')
    ax.set_ylabel(f"{opts['variable']} [{opts['units']}]")

    handles, labels = ax.get_legend_handles_labels()
    leg = dict(zip(labels, handles))

    # loc = 'lower left' if constraint in ['soil_temperature','specific_humidity','relative_humidity'] else 'upper left'
    # ax.legend(leg.values(),leg.keys(),loc=loc,fontsize=8,framealpha=1)

    fig.legend(leg.values(),leg.keys(),loc='lower center',bbox_to_anchor=(0.5, -0.03),
            fontsize=8,ncol=len(exps)+1,framealpha=1)

    if len(sids)==1:
        title_str = f"{opts['plot_title'].capitalize()} [{opts['units']}]: {stations.loc[sid]['name']}"
        fname_suffix = f'{sid}{suffix}'
    else:
        title_str = f"{opts['plot_title'].capitalize()} [{opts['units']}]: {len(sids)} station {func}"
        fname_suffix = f'{len(sids)}_{func}{suffix}'
    ax.set_title(title_str)

    ##### MEAN STATS ON PLOT
    df = stats.T.map(lambda x: opts['fmt'].format(x)).rename(columns={
        'MAE': f"MAE [{opts['units']}]",
        'MBE': f"MBE [{opts['units']}]",
        'threshold': f"<±{opts['threshold']} {opts['units']} [%]",
        })

    yloc=0.99
    # exp names on plot
    result_str = '\n'.join(df.index.to_list())
    ax.text(0.01,yloc, '\n'+result_str, fontsize=7, ha='left', va='top', transform=ax.transAxes)
    # MAE stats on plot
    result_str = df.iloc[:,0].to_frame().to_string(index=False,header=True, justify='right')
    ax.text(0.01+yoff*2,yloc, result_str, fontsize=7, ha='right', va='top', transform=ax.transAxes)
    # MBE stats on plot
    result_str = df.iloc[:,1].to_frame().to_string(index=False,header=True, justify='right')
    ax.text(0.01+yoff*3,yloc, result_str, fontsize=7, ha='right', va='top', transform=ax.transAxes)
    # threshold stats on plot
    if opts['threshold'] is not None:
        result_str = df.iloc[:,-1].to_frame().to_string(index=False,header=True, justify='right')
        ax.text(0.01+yoff*4,yloc, result_str, fontsize=7, ha='right', va='top', transform=ax.transAxes)

    print(f'{func} stats')
    print(df)

    #########################

    # fig.tight_layout()

    fname = f"{plotpath}/{opts['plot_fname']}_vs_obs_{fname_suffix}.png"
    
    return fig,fname,stats

def plot_station_data_avg_timeseries(ds,obs,sids,exps,stations,opts,suffix=''):

    if len(sids)==0:
        print('no sids passed, exiting')

    o = []
    s = {exp:[] for exp in exps}

    lw,whichgrid,figwidth,yoff = 1,'both',9,0.10
    if isinstance(ds.time.min().values,np.datetime64):
        sdate,edate = pd.Timestamp(ds.time.min().values), pd.Timestamp(ds.time.max().values)
        if (edate - sdate).days > 30.: 
            lw,whichgrid,figwidth,yoff = 0.5,'major',18,0.06
    else:
        sdate,edate = ds.time.min().values.item(), ds.time.max().values.item()

    all_stats = calc_all_stats(ds, obs, sids, exps, stations, opts)

    for sid in sids:
        station = stations.loc[sid,'name'].strip()
        print(f'gathering {station} data')
        o.append(obs.loc[sdate:edate,sid])
        lat,lon = stations.loc[sid,'lat'],stations.loc[sid,'lon']

        for exp in exps:
            s[exp].append(ds[exp].sel(latitude=lat, longitude=lon,method='nearest').to_series())
    
    s = {exp:pd.concat(s[exp],axis=1) for exp in exps}
    o = pd.concat(o,axis=1)


    #########################

    plt.close('all')
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(figwidth,4))

    o.mean(axis=1).plot(ax=ax,color='k',lw=lw,marker='',
                        label='AWS observations (mean)',zorder=10)

    for exp in exps:
        s[exp].mean(axis=1).plot(ax=ax,label=exp_plot_titles[exp], 
            color=exp_colours[exp], alpha=0.75 ,marker='o', ms=1, lw=0.5,
            ls='dotted' if next((True for v in ['_sm','_1p5'] if v in exp), False) else 'solid')
            
        
    ax.set_xlim([s[exp].index[0],s[exp].index[-1]])

    ymin,ymax = ax.get_ylim()
    yrng = (ymax - ymin)*0.10
    ax.set_ylim((ymin,ymax+yrng))

    # if opts['constraint'] == 'air_temperature':
    #     ax.set_ylim((opts['vmin']-5,opts['vmax']+5))
    # else:
    #     ax.set_ylim((opts['vmin'],opts['vmax']))

    ax.set_xlabel('')
    ax.grid(color='0.8',linewidth=0.5,which='major',axis='both')

    ax.tick_params(axis='x',labelsize=10, which='both')
    ax.set_ylabel(f"{opts['variable']} [{opts['units']}]")

    handles, labels = ax.get_legend_handles_labels()
    leg = dict(zip(labels, handles))

    # loc = 'lower left' if constraint in ['soil_temperature','specific_humidity','relative_humidity'] else 'upper left'
    # ax.legend(leg.values(),leg.keys(),loc=loc,fontsize=8,framealpha=1)

    fig.legend(leg.values(),leg.keys(),loc='lower center',bbox_to_anchor=(0.5, -0.03),
            fontsize=8,ncol=len(exps)+1,framealpha=1)

    if len(sids)==1:
        title_str = f"{opts['plot_title'].capitalize()} [{opts['units']}]: {stations.loc[sid]['name']}"
        fname_suffix = f'{sid}{suffix}'
    else:
        title_str = f"{opts['plot_title'].capitalize()} [{opts['units']}]: {len(sids)} station average"
        fname_suffix = f'{len(sids)}_avg{suffix}'
    ax.set_title(title_str)

    ##### MEAN STATS ON PLOT
    df = all_stats.mean(axis=1).unstack().applymap(lambda x: opts['fmt'].format(x)).rename(columns={
        'MAE': f"MAE [{opts['units']}]",
        'MBE': f"MBE [{opts['units']}]",
        'threshold': f"<±{opts['threshold']} {opts['units']} [%]",
        })
    
    yloc=0.99
    # exp names on plot
    result_str = '\n'.join(df.index.to_list())
    ax.text(0.01,yloc, '\n'+result_str, fontsize=7, ha='left', va='top', transform=ax.transAxes)
    # MAE stats on plot
    result_str = df.iloc[:,0].to_frame().to_string(index=False,header=True, justify='right')
    ax.text(0.01+yoff*2,yloc, result_str, fontsize=7, ha='right', va='top', transform=ax.transAxes)
    # MBE stats on plot
    result_str = df.iloc[:,1].to_frame().to_string(index=False,header=True, justify='right')
    ax.text(0.01+yoff*3,yloc, result_str, fontsize=7, ha='right', va='top', transform=ax.transAxes)
    # threshold stats on plot
    if opts['threshold'] is not None:
        result_str = df.iloc[:,-1].to_frame().to_string(index=False,header=True, justify='right')
        ax.text(0.01+yoff*4,yloc, result_str, fontsize=7, ha='right', va='top', transform=ax.transAxes)

    print('mean stats')
    print(df)

    #########################

    # fig.tight_layout()

    fname = f"{plotpath}/{opts['plot_fname']}_vs_obs_{fname_suffix}.png"

    return fig,fname,all_stats

def plot_station_data_bias_timeseries(ds,obs,sids,exps,stations,opts,suffix,resample=None):

    o = []
    s = {exp:[] for exp in exps}

    all_stats_list = []

    for sid in sids:
        station = stations.loc[sid,'name'].strip()
        print(f'gathering {station} data')
        o.append(obs[sid])
        lat,lon = stations.loc[sid,'lat'],stations.loc[sid,'lon']

        stats = pd.DataFrame()
        for exp in exps:
            stmp = ds[exp].sel(latitude=lat, longitude=lon,method='nearest').to_series()
            s[exp].append(stmp)
            
            mae = calc_MAE(stmp,obs[sid])
            mbe = calc_MBE(stmp,obs[sid])
            rmse = calc_RMSE(stmp,obs[sid])
            r = calc_R(stmp,obs[sid])

            stats[exp] = pd.Series([mae,mbe,rmse,r],index=['MAE','MBE','RMSE','R'])

        all_stats_list.append(stats.unstack())
    all_stats = pd.concat(all_stats_list,axis=1)
    o = pd.concat(o,axis=1)
    s = {exp:pd.concat(s[exp],axis=1) for exp in exps}

    #########################

    plt.close('all')
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(9,5))

    diff = {}
    metric = 'MAE'
    for exp in exps:

        diff[exp] = (s[exp].mean(axis=1)-o.mean(axis=1)).dropna()
        if diff[exp].count() == 0:
            continue

        if resample is not None:
            diff[exp] = diff[exp].shift(0.5,freq=resample).resample(resample,offset='3H').mean()

        result_str = opts['fmt'].format(all_stats.loc[(exp,metric)].mean()) + f' [{opts["units"]}]'

        diff[exp].plot(ax=ax,label=f"{exp_plot_titles[exp]}: {metric} = {result_str}",
            color=exp_colours[exp],alpha=0.8,marker='o',ms=3,
            ls='dotted' if next((True for v in ['_sm','_1p5'] if v in exp), False) else 'solid')

        # plot bias indicator
        ax.axhline(y=diff[exp].mean(), xmin=1.0,xmax=1.025,clip_on=False,
            color=exp_colours[exp],ls='dotted' if 'sm' in exp else 'solid')
        text_str = opts['fmt'].format(diff[exp].mean())
        ax.text(diff[exp].index[-1],diff[exp].mean(),text_str,
            color=exp_colours[exp],va='bottom')

    print('MAE:')
    # print(all_stats.xs('MAE',level=1).mean(axis=1).to_string(float_format=opts['fmt']))
    print(all_stats.xs('MAE',level=1).mean(axis=1).map(lambda x: opts['fmt'].format(x)))

    ax.set_xlim([s[exp].index[0],s[exp].index[-1]])
    ax.axhline(y=0,color='k',lw=1)

    ax.set_xlabel('')
    ax.grid(color='0.8',linewidth=0.5,which='both')

    ax.tick_params(axis='x',labelsize=10, which='both')
    ax.set_ylabel(f"{opts['variable']} bias [{opts['units']}]")

    handles, labels = ax.get_legend_handles_labels()
    leg = dict(zip(labels, handles))

    ax.legend(leg.values(),leg.keys(),loc='upper center',ncol=2,bbox_to_anchor=(0.5, -0.12),fontsize=7,framealpha=1)

    if len(sids)==1:
        title_str = f"{opts['plot_title'].capitalize()} [{opts['units']}]: {stations.loc[sid]['name']} bias"
        fname_suffix = f'{sid}{suffix}'
    else:
        title_str = f"{opts['plot_title'].capitalize()} [{opts['units']}]: {len(sids)} station mean bias"
        fname_suffix = f'{len(sids)}_mean{suffix}'
    ax.set_title(title_str)

    # fig.tight_layout()

    fname = f"{plotpath}/{opts['plot_fname']}_vs_obs_bias_{fname_suffix}.png"
    
    return fig,fname,all_stats


def calc_all_stats(ds, obs, sids, exps, stations, opts):

    all_stats_list = []

    for sid in sids:
        stats = pd.DataFrame()

        station = stations.loc[sid,'name'].strip()
        lat,lon = stations.loc[sid,'lat'],stations.loc[sid,'lon']
        
        o = obs[sid]

        for exp in exps:
            s = ds[exp].sel(latitude=lat, longitude=lon,method='nearest').to_series()
            stats[exp] = pd.Series([
                calc_MAE(s,o), 
                calc_MBE(s,o),
                calc_RMSE(s,o),
                calc_R(s,o)
                ], index=['MAE','MBE','RMSE','R'])
            if opts['threshold'] is not None:
                within_threshold,sim_within = calc_percent_within_threshold(s,o,opts['threshold'])
                stats.loc['threshold',exp] = within_threshold
    
        all_stats_list.append(stats.unstack())
    all_stats = pd.concat(all_stats_list,axis=1)

    return all_stats

###### hourly mean ######

def calc_diurnal_obs(obs,ds):

    diurnal_obs = pd.DataFrame()

    sdate, edate = pd.Timestamp(ds.time.min().values), pd.Timestamp(ds.time.max().values)

    for sid in obs.columns:
        # select obs and ds with same timestamps
        diurnal_obs[sid] = obs[sid].loc[sdate:edate].groupby(obs.index.hour).mean()

    return diurnal_obs

def calc_diurnal_sim(ds):

    hourly_list = []
    groups = ds.groupby('time.time')
    for hour,item in groups:
        dss = item.mean(dim='time')
        # add time coordinate
        hourly_list.append(dss.assign_coords(time=int(hour.strftime('%H'))).expand_dims('time')) # .strftime('%H:%M')
        
    diurnal_ds = xr.concat(hourly_list,dim='time')

    return diurnal_ds

###### obsevations ######

def process_station_netcdf(variable, stationpath, sdate='2013-01-01', edate=None, local_time_offset=10):

    '''
    Opens all station 5min netcdf files, collects single variable and returns xarray dataset
    Arguments:
        variable (str): variable to extract from netcdf files
        stationpath (str): path to station netcdf files
        sdate (str): start date
        edate (str): end date
    Returns:
        obs (xarray dataset): xarray dataset of all stations for variable
        stations (pandas dataframe): dataframe of station metadata
    '''

    fname = f'{stationpath}/all_stations_{variable}_from_{sdate}.nc'

    # check if file has already been processed
    if os.path.exists(fname):
        print(f'opening {fname}')
        obs_ds = xr.open_dataset(fname)

    # else:    # previous processing
        # variable_map = {
        #     'air_temperature' : 'Temperature',
        #     'dew_point_temperature' : 'dew_point_temperature',
        #     }
    #     # get list of files and variable name
    #     obs_var = variable_map[variable]
    #     fnames = glob.glob(f'{stationpath}/*.nc')

    #     da_list = []

    #     for i, fname in enumerate(fnames):
    #         print(f'{i}/{len(fnames)}: {fname}')
    #         station_obs = xr.open_dataset(fname)
    #         # rename Time to time
    #         station_obs = station_obs.rename({'Time':'time'})
    #         if obs_var in station_obs.data_vars:
    #             print(f'  found {variable}')
    #             da = station_obs[obs_var].sel(time=slice(sdate, edate))
    #             da.attrs.update(station_obs.attrs)
    #             da.name = da.attrs['Station_number']
    #             da_list.append(da)

    #     print('merging station data into one dataset')
    #     obs_ds = xr.merge(da_list)
    #     obs_ds.attrs = {}

    #     # set time dtype encoding to normal integer
    #     obs_ds.time.encoding.update({'dtype':'int32'})

    #     # set all data_vars to float32
    #     for var in obs_ds.data_vars:
    #         obs_ds[var].encoding.update({'dtype':'float32'})

    #     # chunk for optimising timeseries
    #     obs_ds = obs_ds.chunk({'time': -1})

    #     # save to netcdf
    #     obs_ds.to_netcdf(fname)

    # convert to dataframe
    obs = obs_ds.to_dataframe()

    # hour offset from local to UTC
    obs.index = obs.index + pd.DateOffset(hours=-local_time_offset)

    station_data_list = []
    for sid in obs_ds.data_vars:
        station_data = pd.Series([
            obs_ds[sid].attrs['latitude'], 
            obs_ds[sid].attrs['longitude'],
            obs_ds[sid].attrs['elevation'],
            obs_ds[sid].attrs['state'],
            obs_ds[sid].attrs['station_name'],
            ], name=sid)
        station_data_list.append(station_data)

    stations = pd.DataFrame(station_data_list)
    # stations.columns = ['lat','lon','height','opened','closed','name']
    stations.columns = ['lat','lon','elevation','state','name']

    return obs, stations

def get_barra_data(ds,opts,exp):

    print(f'loading {exp}')

    ekeys = {'BARRA-R2': 'AUS-11',
             'BARRA-C2': 'AUST-04',
            }

    varkeys = {
        'precipitation_amount_accumulation':'pr',
        'total_precipitation_rate':'pr',
        'air_temperature': 'tas',
        'sensible_heat_flux' : 'hfss',
        'latent_heat_flux': 'hfls',
        'relative_humidity' : 'hurs',
        'specific_humidity': 'huss',
        'surface_temperature' : 'ts',
        'soil_moisture_l1': 'mrsos',
        'soil_moisture_l2': 'mrsol',
        'soil_moisture_l3': 'mrsol',
        'soil_moisture_l4': 'mrsol',
        }

    varkey = varkeys[opts['variable']]


    if varkey == 'mrsol':
        period = 'day'
    else:
        period = '1hr'

    sdate,edate = pd.Timestamp(ds.time.min().values), pd.Timestamp(ds.time.max().values)

    fpath = f'/g/data/ob53/BARRA2/output/reanalysis/{ekeys[exp]}/BOM/ERA5/historical/hres/{exp}/v1/{period}/{varkey}/latest/'
    yms = sdate.strftime('%Y%m')
    fname = f'{varkey}_{ekeys[exp]}_ERA5_historical_hres_BOM_{exp}_v1_{period}_{yms}-{yms}.nc'
    da = xr.open_dataset(f'{fpath}/{fname}')[varkey]
    da = da.rename({'lon':'longitude','lat':'latitude'})

    # check if ds extends into next month
    yme = edate.strftime('%Y%m')
    if yms != yme:
        print('loading second BARRA month')
        fname = f'{varkey}_{ekeys[exp]}_ERA5_historical_hres_BOM_{exp}_v1_{period}_{yme}-{yme}.nc'
        dae = xr.open_dataset(f'{fpath}/{fname}')[varkey]
        dae = dae.rename({'lon':'longitude','lat':'latitude'})
        da = xr.concat([da,dae],dim='time')

    if opts['variable'] in ['precipitation_amount_accumulation','total_precipitation_rate']:
        # adjust time to be at end of accumulation period
        da['time'] = da['time'] + pd.Timedelta('30Min')

    if opts['variable'] in ['air_temperature']:
        # convert from K to C
        da = da - 273.15

    if varkey == 'mrsos':
        da = da.expand_dims('depth')

    # # select soil layer
    # if varkey == 'mrsol':
    #     level = int(opts['variable'][-1]) - 1
    #     da = da.isel(depth=level)

    da = da.sel(time=slice(sdate,edate))

    if opts['variable'] in ['precipitation_amount_accumulation']:
        print('converting from hourly flux to accumulation')
        da = (da*3600.).cumsum(dim='time')

    return da

def get_flux_obs(variable, local_time_offset=None):
    '''
    Contact: Siyuan Tian siyuan.tian@bom.gov.au
    OzFlux data from: https://dap.ozflux.org.au/thredds/dodsC/ozflux/sites/
    Hourly averaged sensible and latent heat flux observations converted to UTC
    Arguments:
        variable (str): e.g. 'sensible_heat_flux' or 'latent_heat_flux'
    station data:
    sitename,lat,lon,Surface_SM,Rootzone_SM,SSM_depths(cm),RZSM_depths(cm)
        AliceSpringsMulga,-22.2828,133.2493,TRUE,FALSE,10,nan
        Calperum,-34.0027,140.5877,TRUE,FALSE,10,100
        CapeTribulation,-16.1056,145.3778,TRUE,TRUE,10,75
        CowBay,-16.1032,145.4469,TRUE,TRUE,10,75
        CumberlandPlain,-33.6152,150.7236,TRUE,FALSE,8,20
        DalyPasture,-17.1507,133.3502,TRUE,TRUE,5,50
        DalyUncleared,-14.1592,131.3881,TRUE,TRUE,5,50
        DryRiver,-15.26,132.41,TRUE,TRUE,5,50
        Emerald,-23.85872,148.4746,TRUE,FALSE,5,30
        Gingin,-31.3764,115.7139,TRUE,TRUE,10,80
        GreatWesternWoodlands,-30.1913,120.6541,TRUE,TRUE,10,110
        HowardSprings,-12.4943,131.1523,TRUE,TRUE,10,100
        Litchfield,-13.179,130.7945,TRUE,TRUE,5,100
        RedDirtMelonFarm,-14.563639,132.477567,TRUE,TRUE,5,50
        Ridgefield,-32.506102,116.966827,TRUE,TRUE,5,80
        RiggsCreek,-36.6499,145.576,TRUE,FALSE,5,50
        RobsonCreek,-17.1175,145.6301,TRUE,TRUE,6,75
        Samford,-27.3881,152.877,TRUE,FALSE,5,nan
        SturtPlains,-17.1507,133.3502,TRUE,TRUE,5,50
        TiTreeEast,-22.287,133.64,TRUE,FALSE,10,100
        Tumbarumba,-35.6566,148.1517,TRUE,FALSE,10,nan
        WallabyCreek,-37.4259,145.1878,TRUE,FALSE,10,nan
        Warra,-43.09502,146.65452,TRUE,TRUE,8,80
        Whroo,-36.6732,145.0294,TRUE,FALSE,10,nan
        WombatStateForest,-37.4222,144.0944,TRUE,TRUE,10,95
        Yanco,-34.9893,146.2907,TRUE,TRUE,10,75
    JULES soil model depths:
        0.05,0.225,0.675,2
    5-6cm cm surface soil moisture sites:
        DalyPasture, DalyUncleared, DryRiver, Emerald, Litchfield, RedDirtMelonFarm,
        Ridgefield, RiggsCreek, RobsonCreek, Samford, SturtPlains
    '''

    import re

    stationpaths = {
        'sensible_heat_flux':'/g/data/ce10/Insitu-Observations/OzFlux/Fluxes/Hourly_Fh_UTC',
        'latent_heat_flux':'/g/data/ce10/Insitu-Observations/OzFlux/Fluxes/Hourly_Fe_UTC',
        'soil_moisture_l1': '/g/data/ce10/Insitu-Observations/OzFlux/Soil_Moisture/Hourly_SurfaceSM_UTC',
        'soil_moisture_l2': '/g/data/ce10/Insitu-Observations/OzFlux/Soil_Moisture/Hourly_SurfaceSM_UTC',
        'soil_moisture_l3': '/g/data/ce10/Insitu-Observations/OzFlux/Soil_Moisture/Hourly_RootzoneSM_UTC',
        # 'soil_moisture_l4': '/g/data/ce10/Insitu-Observations/OzFlux/Soil_Moisture/Hourly_RootzoneSM_UTC',
        }

    stations = pd.read_csv(f'/g/data/ce10/Insitu-Observations/OzFlux/Soil_Moisture/OzFlux_sites_info_processed.csv')
    stations = stations.rename(columns={'sitename':'sid'})
    
    # create new column "name" which converts camel case to space separated
    stations['name'] = stations['sid'].apply(lambda x: re.sub('([a-z0-9])([A-Z])', r'\1 \2', x).title())
    stations.set_index('sid', inplace=True)

    obs_list = []
    for station in stations.index.to_list():
        print(f'loading {variable}: {station}')
        try:
            # get fname
            fname = glob.glob(f'{stationpaths[variable]}/{station}_*')[0]
            # load csv and rename column to variable
            tmp = pd.read_csv(fname,usecols=[0,1],names=['time',station],skiprows=1,index_col=0,parse_dates=True)
            tmp.index = pd.to_datetime(tmp.index, format='%Y-%m-%d %H:%M:%S+00:00')
            # set index as UTC and localize
            tmp.index = tmp.index.tz_convert('UTC').tz_localize(None)
            if local_time_offset:
                tmp.index = offset_time_index(tmp.index, local_time_offset)
            obs_list.append(tmp)

        except Exception as e:
            print(f"problem loading {variable} at {station}:")
            print(e)
            obs_list.append(pd.DataFrame([np.nan], columns=[station], index=[pd.Timestamp(2010,1,1)]))
    
    obs = pd.concat(obs_list, axis=1)

    return obs, stations

def get_station_obs(stationpath,opts,local_time_offset=None,resample='1H',method='instant'):

    obs_key = opts['obs_key']

    fname = glob.glob(f'{stationpath}/*StnDet*')[0]
    stations = pd.read_csv(fname,header=None,index_col=0,usecols=[1,3,6,7],names=['sid','name','lat','lon'])

    obs = {}

    if obs_key not in [None,'None','Qh','Qle']:

        print(f'getting {obs_key} observations')

        obs_names = ['year', 'month', 'day', 'hour', 'minute']
        ymdhm = [' Year Month Day Hour Minutes in YYYY.1','MM.1','DD.1','HH24.1','MI format in Universal coordinated time']

        # for special mapping
        if 'Qair' == obs_key:
            obs_names = obs_names+['Tair','e','PSurf']
            obs_cols = ymdhm+['Air Temperature in degrees Celsius','Vapour pressure in hPa','Station level pressure in hPa']
        elif 'Tsoil' == obs_key:
            obs_names = obs_names+['Tsoil05','Tsoil10','Tsoil20','Tsoil50','Tsoil100']
            obs_cols = ymdhm+['Temperature of soil at 5cm depth in degrees Celsius','Temperature of soil at 10cm depth in degrees Celsius','Temperature of soil at 20cm depth in degrees Celsius','Temperature of soil at 50cm depth in degrees Celsius','Temperature of soil at 100cm depth in degrees Celsius']
        else:
            obs_names = obs_names+[obs_key]
            obs_cols = ymdhm+[get_tidy_aws_map(reversed=True)[obs_key]]

        for sid in stations.index:
            fname = glob.glob(f'{stationpath}/*{sid}*')[0]

            try:
                tmp = pd.read_csv(fname,usecols=obs_cols, na_values='', low_memory=False)
            except Exception as e:
                print(f"no data found for {obs_key} at {sid}: {stations.loc[sid]['name'].strip()}")
                tmp = pd.DataFrame(columns=obs_names)

            if obs_key == 'all':
                tmp_all = pd.read_csv(fname,na_values='', low_memory=False).iloc[:,15:]
                tmp = pd.concat([tmp,tmp_all],axis=1)

            col_map = get_tidy_aws_map()

            tmp.columns = tmp.columns.str.strip()
            tmp = tmp.rename(columns=col_map)

            # set datetime, replace missing with nan and convert to float
            tmp.index = pd.to_datetime(tmp[['year','month','day','hour','minute']]).values
            tmp = tmp.drop(columns=['year','month','day','hour','minute'])
            tmp = tmp.replace(r'^\s*$', np.nan, regex=True).astype('float')

            for key in ['PSurf','e','e_sat','SLP']: # hPa -> Pa
                if key in tmp.columns:
                    tmp[key] = tmp[key]*100

            for key in ['visibility']: # km -> m
                if key in tmp.columns:
                    tmp[key] = tmp[key]*1000

            # for key in ['precip_hour']: # mm/h -> mm/s
            #     if key in tmp.columns:
            #         tmp[key] = tmp[key]/3600

            if method=='instant':
                # resample to INSTANTANEOUS (per model output)
                tmp = tmp.resample(resample).asfreq()
            if method=='mean':
                # resample to MEAN (per model output)
                tmp = tmp.resample(resample,closed='right',label='right').mean()

            if obs_key == 'None':
                tmp['None'] = np.nan

            if local_time:
                tmp.index = tmp.index + pd.to_timedelta(local_time_offset,'h')

            obs[sid] = tmp

        if obs_key == 'Qair':
            # calculate Qair
            for sid in stations.index:
                obs[sid]['Qair'] =  convert_vapour_pressure_to_qair(obs[sid]['e'], obs[sid]['Tair']+273.15, obs[sid]['PSurf'])
                obs[sid] = obs[sid][['Qair']]

    else:
        for sid in stations.index:
            obs[sid] = pd.DataFrame(columns=[obs_key])

    return obs,stations

def get_tidy_aws_map(reversed=False):

    col_map = {
       'Latitude to four decimal places - in degrees'   : 'latitude',
       'Longitude to four decimal places - in degrees'  : 'longitude',
       'Year Month Day Hour Minutes in YYYY'            : 'year',
       'MM'                                             : 'month',
       'DD'                                             : 'day',
       'HH24'                                           : 'hour',
       'MI format in Local standard time'               : 'minute',
       'Year Month Day Hour Minutes in YYYY.1'          : 'year',
       'MM.1'                                           : 'month',
       'DD.1'                                           : 'day',
       'HH24.1'                                         : 'hour',
       'MI format in Universal coordinated time'        : 'minute',
       'Precipitation since last (AWS) observation in mm': 'precip_last_aws_obs',
       'Total precipitation in last 60 minutes in mm where observations count >= 24': 'precip_hour',
       'Air Temperature in degrees Celsius'                     : 'Tair',
       'Temperature of soil at 5cm depth in degrees Celsius'    : 'Tsoil05',
       'Temperature of soil at 10cm depth in degrees Celsius'   : 'Tsoil10',
       'Temperature of soil at 20cm depth in degrees Celsius'   : 'Tsoil20',
       'Temperature of soil at 50cm depth in degrees Celsius'   : 'Tsoil50',
       'Temperature of soil at 100cm depth in degrees Celsius'  : 'Tsoil100',
       'Wet bulb temperature in degrees Celsius'                : 'Tw',
       'Dew point temperature in degrees Celsius'               : 'Tdp',
       'Relative humidity in percentage %'                      : 'RH',
       'Vapour pressure in hPa'                                 : 'e',
       'Saturated vapour pressure in hPa'                       : 'e_sat',
       'Wind (1 minute) speed in m/s'                           : 'wind',
       'Average wind speed in last 10 minutes in m/s where observations count >= 4': 'Wind_10min',
       'Highest wind speed in last 10 minutes in m/s where observations count >= 4': 'Wind_10min_max',
       'Highest maximum 3 sec wind gust in last 10 minutes in m/s where observations count >= 4'    : 'Wind_gust',
       'Average direction of wind in last 10 minutes in degrees true where observations count >= 4' : 'wdir',
       'Visibility (automatic - 10 minute data mean) in km'     : 'visibility',
       'Station level pressure in hPa': 'PSurf'
    }

    if reversed:
        col_map = {v:k for k,v in col_map.items()}

    return col_map

def calc_esat(temp,pressure,mode=0):
    '''Calculates vapor pressure at saturation

    From Weedon 2010, through Buck 1981: 
    New Equations for Computing Vapor Pressure and Enhancement Factor, Journal of Applied Meteorology
    ----------
    temp        [K]     2m air temperature
    pressure    [Pa]    air pressure
    mode        [0,1]   two different methods to calculate:
        mode=0: from Wheedon et al. 2010
        mode=1: from Ukkola et al., 2017
    NOTE: mode 0 and 1 nearly identical
          Ukkola et al uses the ws=qs approximation (which is not used here, see Weedon 2010)
    '''
   
    # constants
    Rd = 287.05  # specific gas constant for dry air
    Rv = 461.52  #specific gas constant for water vapour
    Epsilon = Rd/Rv  # = 0.622...
    Beta = (1.-Epsilon) # = 0.378 ...

    temp_C = temp - 273.15  # temperature conversion to [C]

    if mode == 0: # complex calculation from Weedon et al. 2010

        # values when over:         water,  ice
        A = np.where( temp_C > 0., 6.1121,  6.1115 )
        B = np.where( temp_C > 0., 18.729,  23.036 )
        C = np.where( temp_C > 0., 257.87,  279.82 )
        D = np.where( temp_C > 0., 227.3,   333.7 )
        X = np.where( temp_C > 0., 0.00072, 0.00022 )
        Y = np.where( temp_C > 0., 3.2E-6,  3.83E-6 )
        Z = np.where( temp_C > 0., 5.9E-10, 6.4E-10 )

        esat = A * np.exp( ((B - (temp_C/D) ) * temp_C)/(temp_C + C))

        enhancement = 1. + X + pressure/100. * (Y + (Z*temp_C**2))

        esat = esat*enhancement*100.

    elif mode == 1: 
        '''simpler calculation from Ukkola et al., 2017
        From Jones (1992), Plants and microclimate: A quantitative approach 
        to environmental plant physiology, p110 '''

        esat = 613.75*np.exp( (17.502*temp_C)/(240.97+temp_C) )

    else:
        raise SystemExit(0)

    return esat

def calc_qsat(esat,pressure):
    '''Calculates specific humidity at saturation

    Parameters
    ----------
    esat        [Pa]    vapor pressure at saturation
    pressure    [Pa]    air pressure

    Returns
    -------
    qsat        [g/g] specific humidity at saturation

    '''
    # constants
    Rd = 287.05  # specific gas constant for dry air
    Rv = 461.52  #specific gas constant for water vapour
    Epsilon = Rd/Rv  # = 0.622...
    Beta = (1.-Epsilon) # = 0.378 ...

    qsat = (Epsilon*esat)/(pressure - Beta*esat)

    return qsat

def convert_rh_to_qair(rh,temp,pressure):
    ''' using equations from Weedon 2010 & code from Cucchi 2020 '''

    assert rh.where(rh<=105).all(), 'relative humidity values > 105. check input'
    assert rh.where(rh>0).any(), 'relative humidity values < 0. check input'
    assert rh.max()>1.05, 'relative humidity values betwen 0-1 (should be 0-100)'

    # calculate saturation vapor pressure
    esat = calc_esat(temp,pressure)
    # calculate saturation specific humidity
    qsat = calc_qsat(esat,pressure)
    # calculate specific humidity
    qair = qsat*rh/100.

    return qair

def convert_vapour_pressure_to_qair(e,temp,pressure):
    ''' using equations from Weedon 2010 & code from Cucchi 2020 '''

    # calculate saturation vapor pressure
    esat = calc_esat(temp,pressure)
    # calculate saturation specific humidity
    qsat = calc_qsat(esat,pressure)
    # calculate specific humidity

    rh = e/esat*100
    qair = qsat*rh/100.

    return qair

def convert_dewtemp_to_qair(dewtemp,temp,pressure):
    ''' using equations from Weedon 2010 & code from Cucchi 2020 '''

    # calculate saturation vapor pressure
    esat = calc_esat(temp,pressure)
    # calculate saturation vapor pressure at dewpoint
    esat_dpt = calc_esat(dewtemp,pressure)
    # calculate saturation specific humidity
    qsat = calc_qsat(esat,pressure)
    # calculate specific humidity
    qair = qsat * (esat_dpt/esat)

    return qair

def convert_dewtemp_to_rh(dewtemp,temp,pressure):
    ''' using equations from Weedon 2010 & code from Cucchi 2020 '''

    # calculate saturation vapor pressure
    esat = calc_esat(temp,pressure)
    # calculate saturation vapor pressure at dewpoint
    esat_dpt = calc_esat(dewtemp,pressure)
    # calculate saturation specific humidity
    qsat = calc_qsat(esat,pressure)
    # calculate relative humidity
    rh = esat_dpt/esat*100

    assert rh.where(rh<=105).all(), 'relative humidity values > 105. check input'
    assert rh.where(rh>0).any(), 'relative humidity values < 0. check input'
    assert rh.max()>1.05, 'relative humidity values betwen 0-1 (should be 0-100)'

    return rh

##### error metrics #####

def calc_MAE(sim,obs):
    '''Calculate Mean Absolute Error'''

    sim = sim.where(obs.notna()).dropna()
    obs = obs.where(sim.notna()).dropna()
    metric = abs(sim-obs).mean()

    return metric

def calc_nMAE(sim,obs):
    '''Calculate Mean Absolute Error normalised by average observation'''

    sim = sim.where(obs.notna()).dropna()
    obs = obs.where(sim.notna()).dropna()
    metric = abs(sim-obs).mean()/obs.mean()

    return metric

def calc_MBE(sim,obs):
    '''Calculate Mean Bias Error from Best et al 2015'''

    sim = sim.where(obs.notna()).dropna()
    obs = obs.where(sim.notna()).dropna()
    metric = np.mean(sim-obs)

    return metric

def calc_R(sim,obs):
    '''cacluate normalised correlation coefficient (pearsons)'''

    sim = sim.where(obs.notna()).dropna()
    obs = obs.where(sim.notna()).dropna()
    metric = sim.corr(obs, method='pearson')

    return metric

def calc_nSD(sim,obs):
    '''calculate normalised standard deviation'''

    sim = sim.where(obs.notna()).dropna()
    obs = obs.where(sim.notna()).dropna()
    metric = sim.std()/obs.std()

    return metric

def calc_RMSE(sim,obs):
    '''Calculate Mean Absolute Error'''

    sim = sim.where(obs.notna()).dropna()
    obs = obs.where(sim.notna()).dropna()
    metric = np.sqrt( ((sim-obs)**2).mean() )

    return metric

def calc_percent_within_threshold(sim,obs,threshold=2):

    sim = sim.where(obs.notna()).dropna()
    obs = obs.where(sim.notna()).dropna()
    sim_within = sim.where( ((sim-obs)<threshold) & ((sim-obs)>-threshold) )

    if sim.count() == 0:
        metric = np.nan
    else:
        metric = 100*sim_within.count()/sim.count()

    return metric,sim_within

def trim_sids(ds, obs, sids, stations):

    '''
    Remove station ids (sids) that are outside the model domain or have no obs data
    '''
    
    # remove sids outside ds extent
    xdsmin,ydsmin,xdsmax,ydsmax = get_bounds(ds)
    sids = [sid for sid in sids if (stations.loc[sid,'lon']>xdsmin) and (stations.loc[sid,'lon']<xdsmax)
                and (stations.loc[sid,'lat']>ydsmin) and (stations.loc[sid,'lat']<ydsmax)]
    
    # remove those without obs data
    sdate,edate = pd.Timestamp(ds.time.min().values),pd.Timestamp(ds.time.max().values)
    # remove any column that is all nan between sdate and edate
    sids = [sid for sid in sids if not (obs.loc[sdate:edate, sid].isna().all())]

    return sids

def open_output_netcdf(exps,opts,variable,datapath):
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
            da = get_barra_data(ds,opts,exp)
            ds[exp] = da.interp_like(ds[template_exp], method='nearest')

        else:
            print('no file found')

    # update the ds timezone attribute as "UTC"
    ds.time.attrs.update({'timezone':'UTC'})

    return ds

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
        
    if variable == 'anthropogenci_heat_flux':
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
            'plot_title': 'dew point temperature (1.5 m)',
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
            'fmt'       : '{:.2f}',
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

    elif variable == 'specific_humidity_lowest_atmos_level':
        opts.update({
            'constraint': 'm01s00i010',
            'plot_title': 'specific humidity (lowest atmos. level)',
            'plot_fname': 'specific_humidity_lowest_atmos_level',
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

    elif variable == 'surface_net_longwave_flux':
        opts.update({
            'constraint': 'surface_net_downward_longwave_flux',
            'plot_title': 'surface net longwave flux',
            'plot_fname': 'surface_net_longwave_flux',
            'units'     : 'W m-2',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : -250,
            'vmax'      : 50,
            'cmap'      : 'inferno',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'surface_net_shortwave_flux':
        opts.update({
            'constraint': 'surface_net_downward_shortwave_flux',
            'plot_title': 'surface net shortwave flux',
            'plot_fname': 'surface_net_shortwave_flux',
            'units'     : 'W m-2',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : -250,
            'vmax'      : 50,
            'cmap'      : 'inferno',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'surface_downwelling_shortwave_flux':
        opts.update({
            'constraint': 'surface_downwelling_shortwave_flux_in_air',
            'units'     : 'W m-2',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : -250,
            'vmax'      : 50,
            'cmap'      : 'inferno',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'surface_downwelling_longwave_flux':
        opts.update({
            'constraint': 'surface_downwelling_longwave_flux_in_air',
            'units'     : 'W m-2',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : -250,
            'vmax'      : 50,
            'cmap'      : 'inferno',
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
            'units'     : '°C',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 70,
            'cmap'      : 'inferno',
            })

    elif variable == 'boundary_layer_thickness':
        opts.update({
            'constraint': 'm01s00i025',
            'plot_title': 'boundary layer thickness',
            'plot_fname': 'boundary_layer_thickness',
            'units'     : '°C',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 3000,
            'cmap'      : 'turbo_r',
            })

    elif variable == 'surface_air_pressure':
        opts.update({
            'units'     : 'Pa',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 88000,
            'vmax'      : 104000,
            'cmap'      : 'viridis',
            })

    elif variable == 'soil_temperature_l1':
        opts.update({
            'constraint': 'soil_temperature',
            'plot_title': 'soil temperature (5cm)',
            'plot_fname': 'soil_temperature_l1',
            'units'     : '°C',
            'level'     : 0.05,
            'obs_key'   : 'Tsoil05',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 10,
            'vmax'      : 40,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'soil_temperature_l2':
        opts.update({
            'constraint': 'soil_temperature',
            'plot_title': 'soil temperature (22.5cm)',
            'plot_fname': 'soil_temperature_l2',
            'units'     : '°C',
            'level'     : 0.225,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : None,
            'vmax'      : None,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'soil_temperature_l3':
        opts.update({
            'constraint': 'soil_temperature',
            'plot_title': 'soil temperature (67.5cm)',
            'plot_fname': 'soil_temperature_l3',
            'units'     : '°C',
            'level'     : 0.675,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : None,
            'vmax'      : None,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'soil_temperature_l4':
        opts.update({
            'constraint': 'soil_temperature',
            'plot_title': 'soil temperature (200cm)',
            'plot_fname': 'soil_temperature_l4',
            'units'     : '°C',
            'level'     : 2,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : None,
            'vmax'      : None,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'toa_outgoing_shortwave_flux':
        opts.update({
            'constraint': 'm01s01i208',
            'plot_title': 'outgoing shortwave radiation flux (top of atmosphere)',
            'plot_fname': 'toa_shortwave',
            'units'     : 'W/m2',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 50,
            'vmax'      : 400,
            'cmap'      : 'Greys_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'toa_outgoing_shortwave_flux_corrected':
        opts.update({
            'constraint': 'm01s01i205',
            'plot_title': 'outgoing shortwave radiation flux corrected (top of atmosphere)',
            'plot_fname': 'toa_shortwave_corrected',
            'units'     : 'W/m2',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 50,
            'vmax'      : 400,
            'cmap'      : 'Greys_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'toa_outgoing_longwave_flux':
        opts.update({
            'constraint': 'toa_outgoing_longwave_flux',
            'plot_title': 'outgoing longwave radiation flux (top of atmosphere)',
            'plot_fname': 'toa_longwave',
            'units'     : 'W/m2',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 100,
            'vmax'      : 350,
            'cmap'      : 'Greys',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'toa_outgoing_shortwave_radiation_flux':
        opts.update({
            'constraint': 'm01s01i205',
            'plot_title': 'outgoing shortwave radiation flux (top of atmosphere)',
            'plot_fname': 'toa_shortwave',
            'units'     : 'W/m2',
            'fname'     : 'umnsaa_vera',
            'vmin'      : 0, 
            'vmax'      : 1000,
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

    elif variable == 'wind_u':
        opts.update({
            'constraint': 'm01s03i225',
            'plot_title': '10 m wind: U-component',
            'plot_fname': 'wind_u_10m',
            'units'     : 'm/s',
            'obs_key'   : 'wind',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 25,
            'cmap'      : 'turbo',
            'threshold' : 2.57,
            'fmt'       : '{:.2f}',
            })

    elif variable == 'wind_v':
        opts.update({
            'constraint': 'm01s03i226',
            'plot_title': '10 m wind: V-component',
            'plot_fname': 'wind_v_10m',
            'units'     : 'm/s',
            'obs_key'   : 'wind',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 25,
            'cmap'      : 'turbo',
            'threshold' : 2.57,
            'fmt'       : '{:.2f}',
            })
    
    elif variable == 'wind_speed':
        opts.update({
            'plot_title': '10 m wind speed',
            'plot_fname': 'wind_speed_10m',
            'units'     : 'm/s',
            'obs_key'   : 'wind',
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
            'vmin'      : 97000,
            'vmax'      : 103000,
            'cmap'      : 'viridis',
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

    elif variable == 'cloud_area_fraction':
        opts.update({
            'constraint': 'm01s09i217',
            'units'     : '1',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'Greys',
            'fmt'       : '{:.3f}',
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
        
    elif variable == 'stratiform_rainfall_amount':
        opts.update({
            'constraint': iris.Constraint(
                name='stratiform_rainfall_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'units'     : 'kg m-2',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'stratiform_rainfall_flux':
        opts.update({
            'constraint': iris.Constraint(
                name='stratiform_rainfall_flux', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'units'     : 'kg m-2 s-1',
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
    

    elif variable == 'subsurface_runoff_amount':
        opts.update({
            'constraint': 'subsurface_runoff_amount',
            'plot_title': 'subsurface runoff amount',
            'plot_fname': 'subsurface_runoff_amount',
            'units'     : 'kg m-2',
            'fname'     : 'umnsaa_psurfb',
            'vmin'      : None,
            'vmax'      : None,
            'cmap'      : 'cividis',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'surface_runoff_flux':
        opts.update({
            'constraint': 'surface_runoff_flux',
            'plot_title': 'surface runoff flux',
            'plot_fname': 'surface_runoff_flux',
            'units'     : 'kg m-2 s-1',
            'fname'     : 'umnsaa_psurfb',
            'vmin'      : 0,
            'vmax'      : 0.01,
            'cmap'      : 'Blues',
            'fmt'       : '{:.4f}',
            })
    
    elif variable == 'subsurface_runoff_flux':
        opts.update({
            'constraint': 'subsurface_runoff_flux',
            'plot_title': 'subsurface runoff flux',
            'plot_fname': 'subsurface_runoff_flux',
            'units'     : 'kg m-2 s-1',
            'fname'     : 'umnsaa_psurfb',
            'vmin'      : None,
            'vmax'      : 0.001,
            'cmap'      : 'cividis',
            'fmt'       : '{:.5f}',
            })

    elif variable == 'surface_total_moisture_flux':
        opts.update({
            'constraint': 'm01s03i223',
            'units'     : 'kg m-2 s-1',
            'fname'     : 'umnsaa_psurfc',
            'vmin'      : None,
            'vmax'      : 0.0002,
            'cmap'      : 'cividis',
            'fmt'       : '{:.6f}',
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