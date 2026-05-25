__title__ = "Plot all variables of the new analysis runs"
__version__ = "2026-05-17"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

'''
Environment:
    module use /g/data/xp3/public/modules; module load conda/analysis3

This script will plot all variables, comparing the two experiments:
1. get variables to plot based on files in each exp subdirectory, check if they exist in both exp subdirs
2. collect var metadata, including units, standard_name and if dimensions are standard (latitude, longitude, time)
3. loop through variables, opening files in both exp subdir
4. calcululate a mean in time for each variable
5. If variable exists in both exp subdirs, calculate a mean difference (exp1 - exp2)
6. If the variable has standard dimensions, plot the mean for each variable, 
    Also plot the difference if the variable exists in both exp subdirs
    plots should be three panels side by side, with last panel a difference plot (exp1 - exp2)
    save plots to plot_dir, with filename format: varname_exp1_exp2.png
7. If an extra dimension is depth or model_level_number, plot first level only e.g. (isel(depth=0))
'''

import os
import sys
import importlib.util
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colorbar import ColorbarBase
import cartopy.crs as ccrs
import rioxarray as rxr
import glob

# Set to None to plot full domain, or provide lat/lon bounds as shown below.
SPATIAL_SUBSET_BOUNDS = {'lat_min': -34.5, 'lat_max': -33.0, 'lon_min': 150.1, 'lon_max': 151.9}
SPATIAL_SUBSET_BOUNDS = None

def main(exps, output_root_dir, plot_dir, variables_to_plot=None, spatial_subset_bounds=None, time_hour=None):

    os.makedirs(plot_dir, exist_ok=True)
    print(f"plot_dir: {plot_dir}")
    print(f"output_root_dir: {output_root_dir}")
    print(f"experiments: {', '.join(exps)}")

    # Split wind from other requested vars so wind-only runs skip full metadata scans.
    vars_without_wind, wants_wind = normalize_requested_vars(variables_to_plot)
    if time_hour is not None:
        print(f"time_hour: {time_hour:02d}")

    if variables_to_plot and not vars_without_wind and wants_wind:
        exp_var_files = {exp: {} for exp in exps}
        var_meta = {}
        all_vars = []
    else:
        exp_var_files, var_meta, all_vars = collect_metadata(exps, output_root_dir, vars_without_wind)
    plot_all_variables(
        exps,
        exp_var_files,
        var_meta,
        all_vars,
        plot_dir,
        output_root_dir,
        variables_to_plot,
        spatial_subset_bounds,
        time_hour,
    )

    return 0


def collect_metadata(exps, output_root_dir, variables_to_plot=None):
    name_map = load_name_map()
    exp_dirs = {exp: os.path.join(output_root_dir, exp) for exp in exps}
    for exp, exp_dir in exp_dirs.items():
        print(f"scanning exp dir: {exp} -> {exp_dir}")
    exp_var_files = {
        exp: find_variable_files(exp_dirs[exp], variables_to_plot)
        for exp in exps
    }

    # collect variable metadata across experiments
    var_meta = {}
    for exp in exps:
        for var_name, fname in exp_var_files[exp].items():
            if var_name not in var_meta:
                meta = get_variable_metadata(fname, var_name, name_map)
                var_meta[var_name] = meta

    # variables present in at least one experiment
    all_vars = sorted(var_meta.keys())
    print(f"found {len(all_vars)} variables across experiments")

    return exp_var_files, var_meta, all_vars


def plot_all_variables(exps, exp_var_files, var_meta, all_vars, plot_dir, output_root_dir, variables_to_plot=None, spatial_subset_bounds=None, time_hour=None):
    prefix = os.path.basename(os.path.normpath(output_root_dir))
    subset_suffix = '_subset' if spatial_subset_bounds is not None else ''
    hour_suffix = f"_hour{time_hour:02d}" if time_hour is not None else ''
    suffix = f"{subset_suffix}{hour_suffix}"
    requested, wants_wind = normalize_requested_vars(variables_to_plot)
    if requested:
        available = [name for name in requested if name in all_vars]
        missing = [name for name in requested if name not in all_vars]
        if missing:
            print(f"requested variables not found: {', '.join(missing)}")
        all_vars = available
        print(f"plotting {len(all_vars)} requested variables")
        if not all_vars and not wants_wind:
            return

    if wants_wind:
        plot_wind_speed(exps, exp_var_files, output_root_dir, plot_dir, prefix, suffix, spatial_subset_bounds, time_hour)
    for var_name in all_vars:
        print(f"\nprocessing variable: {var_name}")
        exp_has_var = [exp for exp in exps if var_name in exp_var_files[exp]]
        if len(exp_has_var) == 0:
            continue
        print(f"available in: {', '.join(exp_has_var)}")

        meta = var_meta[var_name]
        if not meta['standard_dims']:
            print(f"skipping {var_name}: non-standard dimensions {meta['dims']}")
            continue

        # load and time-mean each experiment
        exp_means = {}
        for exp in exp_has_var:
            fname = exp_var_files[exp][var_name]
            print(f"opening {var_name} from {exp}: {fname}")
            da = open_variable(fname, var_name)
            da = select_first_level(da)
            da = mean_in_time(da, time_hour=time_hour)
            da = apply_spatial_subset(da, spatial_subset_bounds)
            exp_means[exp] = da

        # compute difference if both experiments present
        diff_da = None
        if len(exps) >= 2 and all(exp in exp_means for exp in exps[:2]):
            print(f"computing difference: {exps[0]} - {exps[1]}")
            da1, da2 = xr.align(exp_means[exps[0]], exp_means[exps[1]], join='inner')
            diff_da = da1 - da2

        print("plotting panels")
        meta_plot = dict(meta)
        if time_hour is not None:
            meta_plot['plot_title'] = f"{meta['plot_title']} (hour {time_hour:02d})"
        plot_variable_panels(var_name, exp_means, diff_da, exps, plot_dir, meta_plot, prefix, suffix)

        # break # for testing, remove to run all variables


def normalize_requested_vars(variables_to_plot):
    if not variables_to_plot:
        return [], False

    requested = [var for var in variables_to_plot if var.lower() != 'wind']
    wants_wind = any(var.lower() == 'wind' for var in variables_to_plot)
    return requested, wants_wind


def plot_wind_speed(exps, exp_var_files, output_root_dir, plot_dir, prefix, subset_suffix, spatial_subset_bounds, time_hour=None):
    var_u = 'uwnd10m_b'
    var_v = 'vwnd10m_b'
    exp_means = {}
    units = None
    dims = None

    for exp in exps:
        exp_files = exp_var_files.get(exp, {})
        if var_u not in exp_files or var_v not in exp_files:
            exp_dir = os.path.join(output_root_dir, exp)
            exp_files = find_variable_files(exp_dir, [var_u, var_v])
        if var_u not in exp_files or var_v not in exp_files:
            print(f"skipping wind for {exp}: missing {var_u} or {var_v}")
            continue

        fname_u = exp_files[var_u]
        fname_v = exp_files[var_v]
        print(f"opening wind components from {exp}: {fname_u}, {fname_v}")
        da_u = open_variable(fname_u, var_u)
        da_v = open_variable(fname_v, var_v)
        da_u = select_first_level(da_u)
        da_v = select_first_level(da_v)
        da_u = mean_in_time(da_u, time_hour=time_hour)
        da_v = mean_in_time(da_v, time_hour=time_hour)
        da_u = apply_spatial_subset(da_u, spatial_subset_bounds)
        da_v = apply_spatial_subset(da_v, spatial_subset_bounds)
        da_u, da_v = xr.align(da_u, da_v, join='inner')
        da_speed = (da_u ** 2 + da_v ** 2) ** 0.5
        exp_means[exp] = da_speed

        if units is None:
            units = da_u.attrs.get('units', '')
        if dims is None:
            dims = list(da_speed.dims)

    if not exp_means:
        print("no wind data available")
        return

    meta = {
        'dims': dims or [],
        'units': units or '',
        'standard_name': '',
        'bom_name': 'wind',
        'bom_description': '10 m wind speed (sqrt(uwnd10m_b^2 + vwnd10m_b^2))',
        'stash_long_name': '10 m wind speed',
        'plot_title': '10 m wind speed',
        'standard_dims': has_standard_dims(dims or []),
    }

    diff_da = None
    if len(exps) >= 2 and all(exp in exp_means for exp in exps[:2]):
        print(f"computing difference: {exps[0]} - {exps[1]}")
        da1, da2 = xr.align(exp_means[exps[0]], exp_means[exps[1]], join='inner')
        diff_da = da1 - da2

    print("plotting panels")
    meta_plot = dict(meta)
    if time_hour is not None:
        meta_plot['plot_title'] = f"{meta['plot_title']} (hour {time_hour:02d})"
    plot_variable_panels('wind', exp_means, diff_da, exps, plot_dir, meta_plot, prefix, subset_suffix)

def plot_styling(ax, dss, proj, title):

    # # for cartopy
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ax.coastlines(color='0.85',linewidth=0.5,zorder=5)
    left, bottom, right, top = get_bounds(dss)
    ax.set_extent([left, right, bottom, top], crs=proj)

    # ax = distance_bar(ax,distance)
    ax.set_title(title, fontsize=8)
    # ax.set_title('')
    
    # show ticks on all subplots but labels only on first column and last row
    subplotspec = ax.get_subplotspec()
    ax.set_ylabel('latitude [degrees]', fontsize=6) if subplotspec.is_first_col() else ax.set_ylabel('')
    ax.set_xlabel('longitude [degrees]', fontsize=6) if subplotspec.is_last_row() else ax.set_xlabel('')
    ax.tick_params(axis='y', labelleft=subplotspec.is_first_col(), labelright=False, labelsize=6)
    ax.tick_params(axis='x', labelbottom=subplotspec.is_last_row(), labeltop=False, labelsize=6) 
    # ax.set_ylabel('')
    # ax.set_xlabel('')

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
            bbox_to_anchor=(0, -0.15, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
        cbar = ColorbarBase(cax, cmap=im.cmap, norm = im.norm, orientation='horizontal', ticks = ticks)

    return cbar

def get_bounds(ds):
    """
    Make sure that the bounds are in the correct order for cartopy
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


def find_variable_files(exp_dir, variables_to_plot=None):
    """Return a mapping of variable name to file path for a given experiment."""
    if not os.path.isdir(exp_dir):
        print(f"missing exp dir: {exp_dir}")
        return {}

    files = sorted(glob.glob(os.path.join(exp_dir, "*.nc*")))
    print(f"found {len(files)} files in {exp_dir}")
    var_files = {}
    requested = set(variables_to_plot) if variables_to_plot else None
    if requested:
        filtered = [
            fname for fname in files
            if any(var in os.path.basename(fname) for var in requested)
        ]
        if filtered:
            print(f"limiting inspection to {len(filtered)} candidate files")
            files = filtered
    for fname in files:
        try:
            print(f"inspecting file: {fname}")
            with xr.open_dataset(fname, decode_timedelta=True) as ds:
                for var_name in ds.data_vars:
                    if requested and var_name not in requested:
                        continue
                    if var_name not in var_files:
                        var_files[var_name] = fname
                if requested and requested.issubset(var_files.keys()):
                    break
        except Exception as exc:
            print(f"skipping file (open error): {fname} ({exc})")
            continue

    return var_files


def get_variable_metadata(fname, var_name, name_map):
    """Collect variable metadata from a single file."""
    with xr.open_dataset(fname, decode_timedelta=True) as ds:
        da = ds[var_name]
        dims = list(da.dims)
        units = da.attrs.get('units', '?')
        standard_name = da.attrs.get('standard_name', '')
        bom_name = da.attrs.get('bom_name', var_name)

    bom_meta = resolve_bom_meta(name_map, bom_name)
    bom_description = bom_meta.get('bom_description')
    stash_long_name = bom_meta.get('stash_long_name')
    plot_title = bom_description or stash_long_name or bom_name

    standard_dims = has_standard_dims(dims)

    return {
        'dims': dims,
        'units': units,
        'standard_name': standard_name,
        'bom_name': bom_name,
        'bom_description': bom_description,
        'stash_long_name': stash_long_name,
        'plot_title': plot_title,
        'standard_dims': standard_dims,
    }


def resolve_bom_meta(name_map, bom_name):
    if not name_map or not bom_name:
        return {}

    if bom_name in name_map:
        return name_map[bom_name]

    matches = [key for key in name_map.keys() if key and key in bom_name]
    if not matches:
        return {}

    best_key = max(matches, key=len)
    return name_map.get(best_key, {})


def load_name_map(path="/scratch/gb02/postproc_um2nc/src/name_map.py"):
    if not os.path.exists(path):
        print(f"name map not found: {path}")
        return {}

    spec = importlib.util.spec_from_file_location("name_map", path)
    if spec is None or spec.loader is None:
        print(f"unable to load name map: {path}")
        return {}

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = getattr(module, "names", {})
    bom_map = {}
    for _, meta in names.items():
        bom_name = meta.get("bom_name")
        if bom_name:
            bom_map[bom_name] = meta
    print(f"loaded {len(bom_map)} bom_name entries from name map")
    return bom_map


def has_standard_dims(dims):
    """Check if dims include lat/lon and optional time, with optional depth/level."""
    dims_set = set(dims)
    has_lat = 'latitude' in dims_set or 'lat' in dims_set
    has_lon = 'longitude' in dims_set or 'lon' in dims_set
    if not (has_lat and has_lon):
        return False

    allowed = {'time', 'latitude', 'longitude', 'lat', 'lon', 'depth', 'model_level_number'}
    return all(dim in allowed for dim in dims_set)


def open_variable(fname, var_name):
    """Open a single variable as a DataArray."""
    ds = xr.open_dataset(fname, decode_timedelta=True)
    return ds[var_name]


def select_first_level(da):
    """Select the first level if depth or model_level_number is present."""
    if 'depth' in da.dims:
        da = da.isel(depth=0)
    if 'model_level_number' in da.dims:
        da = da.isel(model_level_number=0)
    return da


def mean_in_time(da, time_hour=None):
    """Mean over time if available, optionally restricted to a specific hour."""
    if 'time' not in da.dims:
        return da

    if time_hour is None:
        return da.mean(dim='time', skipna=True)

    hourly = da.groupby('time.hour').mean(dim='time', skipna=True)
    if time_hour not in hourly['hour'].values:
        print(f"no data for hour {time_hour:02d}; falling back to full mean")
        return da.mean(dim='time', skipna=True)

    return hourly.sel(hour=time_hour)


def parse_time_hour(args):
    for arg in args:
        if arg.isdigit() and len(arg) <= 2:
            hour = int(arg)
            if 0 <= hour <= 23:
                return hour
        if arg.lower().startswith("hour="):
            value = arg.split("=", 1)[-1]
            if value.isdigit() and len(value) <= 2:
                hour = int(value)
                if 0 <= hour <= 23:
                    return hour
    return None


def is_time_hour_arg(arg, time_hour):
    if time_hour is None:
        return False
    if arg.isdigit() and len(arg) <= 2:
        return int(arg) == time_hour
    if arg.lower().startswith("hour="):
        value = arg.split("=", 1)[-1]
        return value.isdigit() and int(value) == time_hour
    return False


def apply_spatial_subset(da, spatial_subset_bounds=None):
    """Apply optional lat/lon subset bounds to a DataArray."""
    if spatial_subset_bounds is None:
        return da

    lat_name = 'latitude' if 'latitude' in da.coords else 'lat' if 'lat' in da.coords else None
    lon_name = 'longitude' if 'longitude' in da.coords else 'lon' if 'lon' in da.coords else None
    if lat_name is None or lon_name is None:
        return da

    try:
        lat_min = spatial_subset_bounds['lat_min']
        lat_max = spatial_subset_bounds['lat_max']
        lon_min = spatial_subset_bounds['lon_min']
        lon_max = spatial_subset_bounds['lon_max']
    except Exception:
        print(f"invalid SPATIAL_SUBSET_BOUNDS: {spatial_subset_bounds}; plotting full domain")
        return da

    # Build slices that respect coordinate ordering (ascending or descending).
    lat0 = float(da[lat_name].values[0])
    lat1 = float(da[lat_name].values[-1])
    lon0 = float(da[lon_name].values[0])
    lon1 = float(da[lon_name].values[-1])
    lat_slice = slice(lat_min, lat_max) if lat0 <= lat1 else slice(lat_max, lat_min)
    lon_slice = slice(lon_min, lon_max) if lon0 <= lon1 else slice(lon_max, lon_min)

    return da.sel({lat_name: lat_slice, lon_name: lon_slice})


def plot_variable_panels(var_name, exp_means, diff_da, exps, plot_dir, meta, prefix, subset_suffix=''):
    """Plot mean fields and optional difference for a variable."""
    exp1 = exps[0] if len(exps) > 0 else "exp1"
    exp2 = exps[1] if len(exps) > 1 else "exp2"
    titles = [exp1, exp2, f"{exp1} - {exp2}"]
    panels = [
        exp_means.get(exp1),
        exp_means.get(exp2),
        diff_da,
    ]
    ref_da = next((da for da in panels if da is not None), None)
    if ref_da is None:
        return

    shared_vmin = None
    shared_vmax = None
    base_panels = [da for da in panels[:2] if da is not None]
    if base_panels:
        vmins = [float(da.quantile(0.005)) for da in base_panels]
        vmaxs = [float(da.quantile(0.995)) for da in base_panels]
        shared_vmin = min(vmins)
        shared_vmax = max(vmaxs)

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(13.5, 4.5),
        sharex=True,
        sharey=True,
        subplot_kw={'projection': proj},
    )

    for i, (da, title) in enumerate(zip(panels, titles)):
        ax = axes[i]
        if da is None:
            plot_styling(ax, ref_da, proj, title)
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha='center', va='center', fontsize=8, color='0.5')
            continue
        if i == 2:
            vmax = float(da.quantile(0.995))
            vmin = float(da.quantile(0.005))
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
            im = da.plot(
                ax=ax,
                cmap='RdBu_r',
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
                transform=proj,
            )
        else:
            im = da.plot(
                ax=ax,
                vmin=shared_vmin,
                vmax=shared_vmax,
                add_colorbar=False,
                transform=proj,
            )

        plot_styling(ax, da, proj, title)
        cbar = custom_cbar(ax, im, cbar_loc='bottom')
        units = meta['units']
        bom_name = meta.get('bom_name', 'variable')
        label = f"{bom_name} difference [{units}]" if i == 2 else f"{bom_name} [{units}]"
        cbar.set_label(label, fontsize=6)
        cbar.ax.tick_params(labelsize=6)

    desc = meta['plot_title']
    if len(desc) > 178:
        desc = f"{desc[:178]}..."
    plot_title = f"{desc} [{meta['units']}]"
    fig.suptitle(plot_title, y=0.98, fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.90, wspace=0.05)

    if len(exps) >= 2:
        fname = f"{prefix}_{var_name}_{exps[0]}_{exps[1]}{subset_suffix}.png"
    else:
        fname = f"{prefix}_{var_name}_{exps[0]}{subset_suffix}.png"

    out_path = os.path.join(plot_dir, fname)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":

    exps = ['CTRL','NO-URBAN']
    output_root_dir = f'/scratch/gb02/mjl561/um2nc/SY/SY_1'
    if len(sys.argv) > 1:
        output_root_dir = sys.argv[1]
    variables_to_plot = sys.argv[2:] if len(sys.argv) > 2 else []
    time_hour = parse_time_hour(variables_to_plot)
    if time_hour is not None:
        variables_to_plot = [arg for arg in variables_to_plot if not is_time_hour_arg(arg, time_hour)]
    # if 'all' in variables_to_plot, plot all variables
    if any(arg.lower() == "all" for arg in variables_to_plot):
        variables_to_plot = []
    plot_dir = f'{output_root_dir}/plots/'

    main(
        exps,
        output_root_dir,
        plot_dir,
        variables_to_plot,
        spatial_subset_bounds=SPATIAL_SUBSET_BOUNDS,
        time_hour=time_hour,
    )

    # # Sam's dask setup https://github.com/21centuryweather/dask_setup
    # from dask_setup import setup_dask_client
    # client, cluster, dask_tmp = setup_dask_client(workload_type="mixed")
    # try:
    #     main(exps, output_root_dir, plot_dir)
    # finally:
    #     client.close()
    #     cluster.close()

