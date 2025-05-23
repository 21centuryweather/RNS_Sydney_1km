#!/bin/bash
#PBS -q normal
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l walltime=08:00:00
#PBS -l storage=gdata/hh5+gdata/access+gdata/dp9+scratch/dp9+scratch/ce10+gdata/ce10+scratch/public+scratch/pu02+gdata/gb02+gdata/ob53+gdata/ra22
#PBS -l wd
#PBS -l jobfs=10GB
#PBS -P fy29

set -eu
module purge
module use /g/data/hh5/public/modules
module load conda/analysis3-24.01
module load dask-optimiser

python $HOME/git/RNS_Sydney_1km/plotting_code/himawari_radiance_animation.py