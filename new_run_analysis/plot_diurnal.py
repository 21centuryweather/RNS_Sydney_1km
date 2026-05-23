__title__ = "Plot diurnal cycles for new analysis runs"
__version__ = "2026-05-17"
__author__ = "Mathew Lipson"
__email__ = "m.lipson@unsw.edu.au"

'''
Environment:
    module use /g/data/xp3/public/modules; module load conda/analysis3
'''

import os
import sys
import importlib.util
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import glob


def main(exps, output_root_dir, plot_dir, lat, lon, variables_to_plot=None):

    os.makedirs(plot_dir, exist_ok=True)
    print(f"plot_dir: {plot_dir}")
    print(f"output_root_dir: {output_root_dir}")
    print(f"experiments: {', '.join(exps)}")
    print(f"point: {lat:.3f}, {lon:.3f}")

    # Split wind from other requested vars so wind-only runs skip full metadata scans.
    vars_without_wind, wants_wind = normalize_requested_vars(variables_to_plot)
    if variables_to_plot and not vars_without_wind and wants_wind:
        exp_var_files = {exp: {} for exp in exps}
        var_meta = {}
        all_vars = []
    else:
        exp_var_files, var_meta, all_vars = collect_metadata(exps, output_root_dir, vars_without_wind)
    plot_all_diurnal(
        exps,
        exp_var_files,
        var_meta,
        all_vars,
        plot_dir,
        output_root_dir,
        lat,
        lon,
        variables_to_plot,
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

    var_meta = {}
    for exp in exps:
        for var_name, fname in exp_var_files[exp].items():
            if var_name not in var_meta:
                meta = get_variable_metadata(fname, var_name, name_map)
                var_meta[var_name] = meta

    all_vars = sorted(var_meta.keys())
    print(f"found {len(all_vars)} variables across experiments")

    return exp_var_files, var_meta, all_vars


def plot_all_diurnal(exps, exp_var_files, var_meta, all_vars, plot_dir, output_root_dir, lat, lon, variables_to_plot=None):
    prefix = os.path.basename(os.path.normpath(output_root_dir))
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

    exp_colors = {exp: plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10] for i, exp in enumerate(exps)}

    if wants_wind:
        plot_wind_diurnal(exps, exp_var_files, output_root_dir, plot_dir, lat, lon, exp_colors, prefix)

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

        fig, ax = plt.subplots(figsize=(10, 4.5))
        has_data = False

        for exp in exp_has_var:
            fname = exp_var_files[exp][var_name]
            print(f"opening {var_name} from {exp}: {fname}")
            da = open_variable(fname, var_name)
            da = select_first_level(da)
            da = select_point(da, lat, lon)
            if 'time' not in da.dims:
                print(f"skipping {var_name} in {exp}: no time dimension")
                continue

            series = da.to_series().dropna()
            if series.empty:
                print(f"skipping {var_name} in {exp}: no data at point")
                continue

            series.index = pd.to_datetime(series.index)
            daily = series.groupby([series.index.date, series.index.hour]).mean().unstack(level=1)
            hourly_mean = series.groupby(series.index.hour).mean()

            color = exp_colors[exp]
            for _, row in daily.iterrows():
                ax.plot(row.index.values, row.values, color=color, alpha=0.12, linewidth=0.8)

            ax.plot(hourly_mean.index.values, hourly_mean.values, color=color, linewidth=2, label=exp)
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel('hour of day')
        ax.set_ylabel(f"{meta['bom_name']} [{meta['units']}]")
        ax.grid(True, alpha=0.2)
        ax.legend(loc='best', fontsize=8)

        desc = meta['plot_title']
        if len(desc) > 178:
            desc = f"{desc[:178]}..."
        title = f"{desc} [{meta['units']}]\nDiurnal at {lat:.3f}, {lon:.3f}"
        fig.suptitle(title, y=0.98, fontsize=10)

        if len(exps) >= 2:
            fname = f"{prefix}_{var_name}_diurnal_{exps[0]}_{exps[1]}.png"
        else:
            fname = f"{prefix}_{var_name}_diurnal_{exps[0]}.png"

        out_path = os.path.join(plot_dir, fname)
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)


def normalize_requested_vars(variables_to_plot):
    if not variables_to_plot:
        return [], False

    requested = [var for var in variables_to_plot if var.lower() != 'wind']
    wants_wind = any(var.lower() == 'wind' for var in variables_to_plot)
    return requested, wants_wind


def plot_wind_diurnal(exps, exp_var_files, output_root_dir, plot_dir, lat, lon, exp_colors, prefix):
    var_u = 'uwnd10m_b'
    var_v = 'vwnd10m_b'
    units = None
    fig, ax = plt.subplots(figsize=(10, 4.5))
    has_data = False

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
        da_u = select_point(da_u, lat, lon)
        da_v = select_point(da_v, lat, lon)
        if 'time' not in da_u.dims or 'time' not in da_v.dims:
            print(f"skipping wind in {exp}: no time dimension")
            continue

        da_u, da_v = xr.align(da_u, da_v, join='inner')
        da_speed = (da_u ** 2 + da_v ** 2) ** 0.5
        series = da_speed.to_series().dropna()
        if series.empty:
            print(f"skipping wind in {exp}: no data at point")
            continue

        series.index = pd.to_datetime(series.index)
        daily = series.groupby([series.index.date, series.index.hour]).mean().unstack(level=1)
        hourly_mean = series.groupby(series.index.hour).mean()

        color = exp_colors[exp]
        for _, row in daily.iterrows():
            ax.plot(row.index.values, row.values, color=color, alpha=0.12, linewidth=0.8)

        ax.plot(hourly_mean.index.values, hourly_mean.values, color=color, linewidth=2, label=exp)
        has_data = True

        if units is None:
            units = da_u.attrs.get('units', '')

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel('hour of day')
    ax.set_ylabel(f"wind [{units or ''}]")
    ax.grid(True, alpha=0.2)
    ax.legend(loc='best', fontsize=8)

    title = f"10 m wind speed [{units or ''}]\nDiurnal at {lat:.3f}, {lon:.3f}"
    fig.suptitle(title, y=0.98, fontsize=10)

    if len(exps) >= 2:
        fname = f"{prefix}_wind_diurnal_{exps[0]}_{exps[1]}.png"
    else:
        fname = f"{prefix}_wind_diurnal_{exps[0]}.png"

    out_path = os.path.join(plot_dir, fname)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def find_variable_files(exp_dir, variables_to_plot=None):
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
    dims_set = set(dims)
    has_lat = 'latitude' in dims_set or 'lat' in dims_set
    has_lon = 'longitude' in dims_set or 'lon' in dims_set
    if not (has_lat and has_lon):
        return False

    allowed = {'time', 'latitude', 'longitude', 'lat', 'lon', 'depth', 'model_level_number'}
    return all(dim in allowed for dim in dims_set)


def open_variable(fname, var_name):
    ds = xr.open_dataset(fname, decode_timedelta=True)
    return ds[var_name]


def select_first_level(da):
    if 'depth' in da.dims:
        da = da.isel(depth=0)
    if 'model_level_number' in da.dims:
        da = da.isel(model_level_number=0)
    return da


def select_point(da, lat, lon):
    if 'latitude' in da.coords:
        y_dim = 'latitude'
    elif 'lat' in da.coords:
        y_dim = 'lat'
    else:
        return da

    if 'longitude' in da.coords:
        x_dim = 'longitude'
    elif 'lon' in da.coords:
        x_dim = 'lon'
    else:
        return da

    return da.sel({y_dim: lat, x_dim: lon}, method='nearest')


def parse_float(value):
    try:
        return float(value)
    except Exception:
        return None


if __name__ == "__main__":

    exps = ['CTRL','NO-URBAN']
    output_root_dir = f'/scratch/gb02/mjl561/um2nc/SY/SY_1'
    default_lat = -33.813
    default_lon = 151.003

    args = sys.argv[1:]
    if len(args) >= 1:
        output_root_dir = args[0]

    variables_to_plot = []
    lat = default_lat
    lon = default_lon

    if len(args) >= 3:
        lat_val = parse_float(args[1])
        lon_val = parse_float(args[2])
        if lat_val is not None and lon_val is not None:
            lat = lat_val
            lon = lon_val
            variables_to_plot = args[3:]
        else:
            variables_to_plot = args[1:]
    elif len(args) == 2:
        variables_to_plot = args[1:]

    if any(arg.lower() == "all" for arg in variables_to_plot):
        variables_to_plot = []

    plot_dir = f'{output_root_dir}/plots/'

    main(exps, output_root_dir, plot_dir, lat, lon, variables_to_plot)
