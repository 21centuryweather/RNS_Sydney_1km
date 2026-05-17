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
output_root_dir="/scratch/gb02/mjl561/um2nc/SY/SY_1"
# variables_to_plot="all"
variables_to_plot="sfc_pres"

module use /g/data/xp65/public/modules
module load conda/analysis3

# spatial
python /home/561/mjl561/git/RNS_Sydney_1km/new_run_analysis/plot_vars.py \
  "$output_root_dir" $variables_to_plot

# diurnal for Parramatta CBD
lat="-33.813"
lon="151.003"

python /home/561/mjl561/git/RNS_Sydney_1km/new_run_analysis/plot_diurnal.py \
  "$output_root_dir" "$lat" "$lon" $variables_to_plot
