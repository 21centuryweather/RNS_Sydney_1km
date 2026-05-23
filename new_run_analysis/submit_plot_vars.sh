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

# location of post-processed netcdf data
output_root_dir="/scratch/gb02/mjl561/um2nc/SY_djf/SY_1"
# variables_to_plot="all"
# use "wind" to plot 10 m wind speed from uwnd10m_b/vwnd10m_b
variables_to_plot="temp_scrn"
# time_hour can be "00"-"23" (optional). Leave empty for full-period mean.
time_hour="00"

module use /g/data/xp65/public/modules
module load conda/analysis3

# spatial
python /home/561/mjl561/git/RNS_Sydney_1km/new_run_analysis/plot_vars.py \
  "$output_root_dir" $time_hour $variables_to_plot

# diurnal for Parramatta CBD
lat="-33.813"
lon="151.003"

python /home/561/mjl561/git/RNS_Sydney_1km/new_run_analysis/plot_diurnal.py \
  "$output_root_dir" "$lat" "$lon" $variables_to_plot
