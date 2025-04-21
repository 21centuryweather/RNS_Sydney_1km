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
plotpath = '/g/data/ce10/users/mjl561/cylc-run/rns_MC_202002/himawari'

# suffix for file names (e.g. "aus_domain")
suffix = 'MC_domain'

# subset domain of interest
xmin, xmax, ymin, ymax = 140.95, 154.30, -39.35, -24.54  # ECL subdomain
xmin, xmax, ymin, ymax = 110, 155, -45, -9               # ACCESS-A domain
xmin, xmax, ymin, ymax = 108, 157, -56.5, 56.5           # MC (Maxime Colin)

# dates of interest
sdate = '2020-02-01 00:00'
edate = '2020-02-04 00:00'

channel, chname = 'P1S-ABOM_OBS_B15-PRJ_GEOS141_2000', 'infrared'  # Band 15: 12.38 um (IR)
channel, chname = 'P1S-ABOM_OBS_B02-PRJ_GEOS141_1000', 'visible'   # Band 2: 0.51 um (visible)

satver = 'HIMAWARI8' ### WARNING: becomes HIMAWARI9 at some point in 2022/06

# whether to create gif or mp4 animations
gif=False
mp4=True

# save netcdf of reprojection (not required for plot by maybe useful for later analysis)
netcdf=False

################################################################################

def main(gif, mp4):
    '''reproject Himawari 8 over domain of interest, plot radiance and create an animation as gif or mp4'''

    datapath = '/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest'

    dates = pd.date_range(sdate, edate, freq='1h')
    for date in dates:
        
        # if fig_fname exists, skip this date
        fig_fname = f"{plotpath}/HIM_{chname}_{date.strftime('%Y-%m-%d-%H%M')}_{suffix}.png"
        if os.path.exists(fig_fname):
            print(f'{fig_fname} already exists, skipping')
            continue

        year, month, day, time = date.strftime('%Y'), date.strftime('%m'), date.strftime('%d'), date.strftime('%H%M')
        ymdhm = date.strftime('%Y%m%d%H%M')

        nc_fname_in = f"{datapath}/{year}/{month}/{day}/{time}/{ymdhm}00-{channel}-{satver}-AHI.nc"
        nc_fname_out = f"{plotpath}/{satver}_{ymdhm}_{suffix}.nc"

        print(f'processing {satver} {date}')
        sat = convert_sat(nc_fname_in, nc_fname_out)
        fig = plot_sat(sat, date)
        fig.savefig(fig_fname, dpi=300, bbox_inches='tight')

    # create animations
    plt_fname_in = f"{plotpath}/HIM_{chname}*_{suffix}.png"
    plt_fname_out = f"{plotpath}/HIM_{chname}_{date.strftime('%Y%m')}_{suffix}"

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

    # get the first data_var name
    data_var = list(sat.data_vars)[0]
    print(f'plotting {data_var}: {date}')

    data = sat[data_var].fillna(0)

    ## normally a simple plot with xarray inbuilt function:
    # fig,ax = plt.subplots()
    # data.plot(ax=ax, vmin=0, vmax=0.5, cmap='Greys_r')
    # fig.savefig(f'{plotpath}/sat_subset_{date}_simple.png', dpi=300, bbox_inches='tight')

    # here we create a more complicated plot with cartopy on a map projection to get coastlines
    plt.close('all')
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())

    if chname == 'visible':
        vmin, vmax = 0, 0.7
        cmap = 'Greys_r'
        title_suffix = '[0.51 um]'
    elif chname == 'infrared':
        vmin, vmax = 150, 320
        cmap = 'Greys'
        title_suffix = '[12.38 um]'
    
    data.plot(ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,extend='max', add_colorbar=False)

    ax.coastlines(resolution='50m',color='0.85',lw=0.4,zorder=5)
    # ax.coastlines(resolution='10m',color='r',lw=0.5)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels   = False
    gl.right_labels = False
    gl.ylines       = False
    gl.xlines       = False
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    ax.set_title(f"Himawari {chname} {title_suffix}:\n{date.strftime('%Y-%m-%d %H%M')}Z",fontsize=10,y = 1.02)

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

def stitch_two_videos(fname1, fname2, fnameout, quality=26):
    '''
    Uses ffmpeg to stitch two videos together
        fname1 (string): The first video file
        fname2 (string): The second video file
        fnameout (string): The output filename (excluding extension)
    '''

    # quality ranges 0 to 51, 51 being worst.
    assert 0 <= quality <= 51, "quality must be between 1 and 51 inclusive"

    # Get video dimensions using ffprobe to ensure same height
    h1 = int(os.popen(f'ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 {fname1}').read().strip())
    h2 = int(os.popen(f'ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 {fname2}').read().strip())
    # w1 = int(os.popen(f'ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 {fname1}').read().strip())
    # w2 = int(os.popen(f'ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 {fname2}').read().strip())
    min_height = min(h1, h2)

    # print dimensions
    print(f'Video 1 height: {h1}, Video 2 height: {h2}, using min height: {min_height}')
    # print(f'Video 1 width: {w1}, Video 2 width: {w2}')

    # # Ensure width is divisible by 2 for both videos
    # adjusted_width1 = w1 if w1 % 2 == 0 else w1 - 1
    # adjusted_width2 = w2 if w2 % 2 == 0 else w2 - 1

    adjusted_width1 = -1
    adjusted_width2 = -1

    # ffmpeg to create mp4
    command = f'ffmpeg -i {fname1} -i {fname2} -filter_complex \
        "[0:v]scale={adjusted_width1}:{min_height}[v0];[1:v]scale={adjusted_width2}:{min_height}[v1];[v0][v1]hstack=inputs=2" -c:v libx264 -crf {quality} -pix_fmt yuv420p -y {fnameout}.mp4'
    os.system(command)

    return f'completed, see: {fnameout}.mp4'


################################################################################

if __name__ == '__main__':

    # make plotpath if it doesn't exist
    if not os.path.exists(plotpath):
        os.makedirs(plotpath)

    main(gif,mp4)


