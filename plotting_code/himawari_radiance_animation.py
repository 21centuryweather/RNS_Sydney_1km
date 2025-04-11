__title__ = "Extract and reproject himawari, create animation for custom domain"
__version__ = "2025-04-11"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

'''
Gadi notes:
 - You need access to ra22 project and gdata storage for Himawari images
 - Update plotpath, domain and dates of interest
 - Use analysis3:
        module use /g/data/hh5/public/modules; module load conda/analysis3
 - WARNING: Filenames change from HIMAWARI8 to HIMAWARI9 some time in June 2022
   refer to satver variable
'''

import os
import pandas as pd
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

################################################################################

# set paths for plots (e.g. /g/data/dp9/USER/himawari)
plotpath = '/g/data/ce10/users/mjl561/cylc-run/rns_ostia_SY_1km/himawari'

# suffix for file names (e.g. "aus_domain")
suffix = 'aus_domain'

# subset domain of interest
xmin, xmax, ymin, ymax = 140.95, 154.30, -39.35, -24.54  # ECL subdomain
xmin, xmax, ymin, ymax = 110, 155, -45, -9               # ACCESS-A domain

# dates of interest
sdate = '2017-02-01 00:00'
edate = '2017-02-07 00:00'

# whether to create gif or mp4 animations
gif=False
mp4=True

# save netcdf of reprojection (not required for plot by maybe useful for later analysis)
netcdf=False

################################################################################

def main(gif, mp4):
    '''reproject Himawari 8 over domain of interest, plot radiance and create an animation as gif or mp4'''

    datapath = '/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest'

    satver = 'HIMAWARI8' ### WARNING: becomes HIMAWARI9 at some point in 2022/06

    dates = pd.date_range(sdate, edate, freq='1h')
    for date in dates:
        year, month, day, time = date.strftime('%Y'), date.strftime('%m'), date.strftime('%d'), date.strftime('%H%M')
        ymdhm = date.strftime('%Y%m%d%H%M')

        nc_fname_in = f"{datapath}/{year}/{month}/{day}/{time}/{ymdhm}00-P1S-ABOM_OBS_B02-PRJ_GEOS141_1000-{satver}-AHI.nc"
        nc_fname_out = f"{plotpath}/{satver}_{ymdhm}_{suffix}.nc"

        print(f'processing {satver} {date}')
        sat = convert_sat(nc_fname_in, nc_fname_out)

        print(f'plotting {satver} {date}')
        fig = plot_sat(sat, date)
        fig.savefig(f"{plotpath}/SAT_{date.strftime('%Y-%m-%d-%H%M')}_{suffix}.png", dpi=300, bbox_inches='tight')

    # create animations
    plt_fname_in = f"{plotpath}/SAT*_{suffix}.png"
    plt_fname_out = f"{plotpath}/SAT_{date.strftime('%Y%m')}_{suffix}"

    if gif:
        print('creating gif')
        command = f'convert -delay 20 -loop 0 {plt_fname_in} {plt_fname_out}.gif'
        os.system(command)

    if mp4:
        print('creating mp4')
        make_mp4(plt_fname_in, plt_fname_out, fps=12, quality=26)

    return

################################################################################

def convert_sat(fname_in, fname_out):
    '''reproject Himawari
        fname_in (string) : filename for himawari full disk data
        fname_out (string) : filename for netcdf (if saved)
    '''

    # first check if file is already generated
    if os.path.exists(fname_out):
        print('this date already processed')
        return xr.open_dataset(fname_out)

    # open himawary with rioxarray (which is projection-aware extension of xarray):
    orig = rxr.open_rasterio(fname_in, masked=True)

    # reproject to a normal latitude and longitude coordinate system
    ds = orig.rio.reproject('epsg:4326')

    # select a subset based on your desired lat/lon
    subset = ds.sel(y=slice(ymin,ymax),x=slice(xmin,xmax))

    # check if latitude bounds need reversing
    if len(subset.y)==0:
        subset = ds.sel(y=slice(ymax,ymin),x=slice(xmin,xmax))

    if netcdf:
        # save to netcdf (setting compression, fillvalue and datatype)
        subset.encoding.update({'zlib':'True', '_FillValue':-999., 'dtype':'float32'})
        print('saving netcdf')
        subset.to_netcdf(fname_out,format='NETCDF4')

    return subset

def plot_sat(sat, date):
    '''plots reprojected himawari from saved netcdf
        sat (xarray dataset): reprojected himawari data
        date (pd.datetime) : date of plot
    '''

    data = sat['channel_0002_scaled_radiance'].fillna(0)

    ## normally a simple plot with xarray inbuilt function:
    # fig,ax = plt.subplots()
    # data.plot(ax=ax, vmin=0, vmax=0.5, cmap='Greys_r')
    # fig.savefig(f'{plotpath}/sat_subset_{date}_simple.png', dpi=300, bbox_inches='tight')

    # here we create a more complicated plot with cartopy on a map projection to get coastlines
    plt.close('all')
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())

    data.plot(ax=ax, vmin=0, vmax=0.7, cmap='Greys_r',extend='max', add_colorbar=False)

    ax.coastlines(resolution='10m',color='r',lw=0.5)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels   = False
    gl.right_labels = False
    gl.ylines       = False
    gl.xlines       = False
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    ax.set_title(f"Himawari scaled radiance: {date.strftime('%Y-%m-%d %H%M')}Z",fontsize=10,y = 1.02)

    return fig

def make_mp4(fnamein,fnameout,fps=12,quality=26):
    '''
    Uses ffmpeg to create mp4 with custom codec and options for maximum compatability across OS.
        fnamein (string): The image files to create animation from, with glob wildcards (*) accepted.
        fnameout (string): The output filename (excluding extension)
        fps (float): The frames per second. Default 6.
        quality (float): quality ranges 0 to 51, 51 being worst.
    '''

    import glob
    import imageio.v2 as imageio

    # collect animation frames
    fnames = sorted(glob.glob(fnamein))
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

################################################################################

if __name__ == '__main__':

    main(gif,mp4)


