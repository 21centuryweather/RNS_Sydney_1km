__version__ = "2024-12-17"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

"""
To plot the ancillary domains
"""

import os
import iris
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.geodesic as cgeo

def plot_domain_orography():
    """
    Plot the orography of the different domains
    Mask out the ocean with the land sea mask ancil
    """

    ancil_path = '/home/561/mjl561/cylc-run/ancils_SY_1km/share/data/ancils/SY_CCI'
    domains = 'SY_11p1','SY_5','SY_1_L','SY_1'
    plot_path = f'{os.getenv("HOME")}/git/Sydney_1km/plotting_code/figures'


    data = {}
    for domain in domains:
        print(domain)

        # get land sea mask ancil
        opts = get_variable_opts('lsm','ancil')
        fname = f'{ancil_path}/{domain}/{opts["fname"]}'
        cb = iris.load_cube(fname, constraint=opts["constraint"])
        lsm = xr.DataArray().from_iris(cb)

        # get orography ancil
        opts = get_variable_opts('surface_altitude','ancil')
        fname = f'{ancil_path}/{domain}/{opts["fname"]}'
        cb = iris.load_cube(fname, constraint=opts["constraint"])
        # convert to xarray and constrain to lsm
        data[domain] = xr.DataArray().from_iris(cb)
        # reindex lsm (rounding errors)
        lsm = lsm.reindex_like(data[domain],method='nearest')
        data[domain] = data[domain].where(lsm>0)

    #############################################

    print(f"plotting")

    proj = ccrs.PlateCarree()
    opts = get_variable_opts('surface_altitude','ancil')
    cmap = plt.get_cmap(opts['cmap'])
    # cmap = replace_cmap_min_with_white(cmap)

    plt.close('all')
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(11,9),
                            sharey=True,sharex=True,
                            subplot_kw={'projection': proj},
                            )
    for domain in domains:
        print(f'plotting {domain}')
        im = data[domain].plot(ax=ax,cmap=cmap, vmin=opts['vmin'],vmax=opts['vmax'],add_colorbar=False, transform=proj)
        # draw rectangle around domain
        left, bottom, right, top = get_bounds(data[domain])
        ax.plot([left, right, right, left, left], [bottom, bottom, top, top, bottom], color='0.65', linewidth=1, linestyle='dashed' if domain=='SY_1' else 'solid')
        # label domain with white border around black text
        domain_text = f'{domain}: {data[domain].shape[0]}x{data[domain].shape[1]}'
        ax.text(right-0.1, top-0.1, f'{domain_text}', fontsize=8, ha='right', va='top', color='k',
                path_effects=[path_effects.withStroke(linewidth=1.5, foreground='w')])

    cbar_title = f"{opts['plot_title'].capitalize()} [{opts['units']}]"
    cbar = custom_cbar(ax,im,cbar_loc='right')  
    cbar.ax.set_ylabel(cbar_title)
    cbar.ax.tick_params(labelsize=8)

    # # for cartopy
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.coastlines(color='k',linewidth=0.5,zorder=5)
    left, bottom, right, top = get_bounds(data['SY_11p1'])
    ax.set_extent([left, right, bottom, top], crs=proj)
    
    ax = distance_bar(ax,distance=200)
    ax.set_title('Sydney domains')

    fig.savefig(f'{plot_path}/SY_domain_{opts["plot_fname"]}.png',dpi=300,bbox_inches='tight')

    return

def replace_cmap_min_with_white(cmap):
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[0] = [1, 1, 1, 1]  # RGBA for white
    cmap_new = plt.matplotlib.colors.ListedColormap(cmap_colors)
    return cmap_new

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
    # ydist = abs(ylims[1]-ylims[0])

    xdist = abs(xlims[1]-xlims[0])
    offset = 0.03*xdist

    # plot distance bar
    start = (xlims[0]+offset,ylims[0]+offset)
    end = cgeo.Geodesic().direct(points=start,azimuths=90,distances=distance*1000).flatten()
    ax.plot([start[0],end[0]],[start[1],end[1]], color='0.65', linewidth=1.5)
            # path_effects=[pe.Stroke(linewidth=2.5, foreground='w'), pe.Normal()])
    ax.text(start[0]+offset/7,start[1]+offset/5, f'{distance} km', 
        fontsize=9, ha='left',va='bottom', color='0.65')
        # path_effects=[pe.Stroke(linewidth=0.75, foreground='w'), pe.Normal()])

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

def get_variable_opts(variable,case):
    '''standard variable options for plotting. to be updated within master script as needed'''

    # standard ops
    opts = {
        'constraint': variable,
        'plot_title': variable,
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
        'case'      : case,
        }

    if variable == 'surface_altitude':
        opts.update({
        'constraint': 'surface_altitude',
        'plot_title': 'surface altitude',
        'plot_fname': 'surface_altitude',
        'units'     : 'm',
        'obs_key'   : 'None',
        'fname'     : 'qrparm.orog.mn',
        'vmin'      : 0,
        'vmax'      : 1500,
        'cmap'      : 'terrain',
        })

    elif variable == 'lsm':
        opts.update({
            'constraint': 'm01s00i030',
            'plot_title': 'land sea mask',
            'plot_fname': 'land_sea_mask',
            'units'     : '1',
            'fname'     : 'qrparm.mask',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'viridis',
            'fmt'       : '{:.2f}',
            })

    # add variable to opts
    opts.update({'variable':variable})
    
    return opts

if __name__ == '__main__':
    plot_domain_orography()
