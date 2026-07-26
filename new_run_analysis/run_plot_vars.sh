#!/bin/bash
#PBS -N plot_all_vars
#PBS -q normal
#PBS -P gb02
#PBS -l walltime=02:00:00
#PBS -l ncpus=12
#PBS -l mem=48GB
#PBS -l storage=gdata/xp65+scratch/gb02
#PBS -l wd
#PBS -l jobfs=10GB

set -euo pipefail

module use /g/data/xp65/public/modules
module load conda/analysis3

# location of post-processed netcdf data
output_root_dir="/g/data/gb02/mjl561/ram3_SY_urban/SY_djf/SY_1"
variables_to_plot="all"
# use "wind" to plot 10 m wind speed from uwnd10m_b/vwnd10m_b
# variables_to_plot="temp_scrn"
# movie settings (used for hour-by-hour frames)
movie_fps=6
movie_quality=26
# # diurnal for Summer Hill
lat="-33.891"
lon="151.137"
# Optional: provide AEST hours to plot hourly means, e.g. hours=(0 6 12 18).
# Leave empty to plot a single all-timesteps mean.
hours=()
hours=($(seq 0 23))

# spatial
if (( ${#hours[@]} > 0 )); then
  hours_csv=$(IFS=,; echo "${hours[*]}")
  echo "Processing AEST hours: ${hours_csv}"
  python /home/561/mjl561/git/RNS_Sydney_1km/new_run_analysis/plot_vars.py \
    "$output_root_dir" "hours=${hours_csv}" "lat=${lat}" "lon=${lon}" "$variables_to_plot"
else
  echo "No hours passed: plotting all-timesteps mean"
  python /home/561/mjl561/git/RNS_Sydney_1km/new_run_analysis/plot_vars.py \
    "$output_root_dir" "lat=${lat}" "lon=${lon}" "$variables_to_plot"
fi

# make movie from hourly frames (only when hour plots were generated)
if (( ${#hours[@]} > 0 )); then
  frame_glob="${output_root_dir}/plots/${variables_to_plot}/*_${variables_to_plot}_*hour*.png"
  movie_name="${variables_to_plot}_hours"
  echo "Creating movie from frames: ${frame_glob}"
  python /home/561/mjl561/git/RNS_Sydney_1km/new_run_analysis/make_movie.py \
    "$frame_glob" "$movie_name" "$movie_fps" "$movie_quality"
fi

# # diurnal for Parramatta CBD
# lat="-33.813"
# lon="151.003"

# # diurnal for Sydney CBD
# lat="-33.8688"
# lon="151.2093"

# # diurnal for Summer Hill
# lat="-33.891"
# lon="151.137"

# python /home/561/mjl561/git/RNS_Sydney_1km/new_run_analysis/plot_diurnal.py \
#   "$output_root_dir" "$lat" "$lon" $variables_to_plot
