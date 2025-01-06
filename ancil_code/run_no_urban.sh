#!/bin/bash
#PBS -q normal
#PBS -l ncpus=4
#PBS -l mem=16GB
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/hh5+gdata/access+gdata/dp9+scratch/dp9+scratch/ce10+gdata/ce10+scratch/public+scratch/pu02
#PBS -l wd
#PBS -l jobfs=10GB
#PBS -P ce10

set -eu
module purge
module use /g/data/access/ngm/modules
module load ants/0.18

REGION_PATH=/home/561/mjl561/cylc-run/u-dl705/share/data/ancils/SY_no_urban
# for each directory in current directory
for dir in SY_5 SY_1 SY_1_L;
do
    cd $REGION_PATH/$dir
    echo processing $dir
    python $HOME/git/RNS_Sydney_1km/ancil_code/ancil_lct_postproc_no_urban.py \
    qrparm.veg.frac_cci_pre_c4 \
    --target-lsm qrparm.mask \
    --output qrparm.veg.frac_cci_pre_c4

done
