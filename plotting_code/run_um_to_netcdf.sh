#!/bin/bash
#PBS -q normal
#PBS -l ncpus=12
#PBS -l mem=48GB
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/hh5+gdata/access+gdata/dp9+scratch/dp9+scratch/ce10+gdata/ce10+scratch/public+scratch/pu02
#PBS -l wd
#PBS -l jobfs=10GB
#PBS -P fy29

set -eu
module purge
module use /g/data/hh5/public/modules
module load conda/analysis3-24.01
module load dask-optimiser

python $HOME/git/RNS_Sydney_1km/plotting_code/um_to_netcdf.py
