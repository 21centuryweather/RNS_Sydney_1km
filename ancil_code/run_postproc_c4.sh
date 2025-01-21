#PBS -N SY_no_urban_SY_
#PBS -o cylc-run/u-dl705/log/job/1/SY_no_urban_SY_1_L_ancil_lct_postproc_c4/02/job.out
#PBS -e cylc-run/u-dl705/log/job/1/SY_no_urban_SY_1_L_ancil_lct_postproc_c4/02/job.err
#PBS -l walltime=10800
#PBS -q normal
#PBS -P pu02
#PBS -l ncpus=16
#PBS -l mem=64gb
#PBS -l jobfs=10gb
#PBS -l storage=gdata/access+gdata/hr22+gdata/ki32+scratch/ce10
#PBS -W umask=0022
export CYLC_DIR='/g/data/hr22/apps/cylc7/cylc_7.9.7'
export CYLC_VERSION='7.9.7'
CYLC_FAIL_SIGNALS='EXIT ERR TERM XCPU'

cylc__job__inst__cylc_env() {
    # CYLC SUITE ENVIRONMENT:
    export CYLC_CYCLING_MODE="integer"
    export CYLC_SUITE_FINAL_CYCLE_POINT="1"
    export CYLC_SUITE_INITIAL_CYCLE_POINT="1"
    export CYLC_SUITE_NAME="u-dl705"
    export CYLC_UTC="True"
    export CYLC_VERBOSE="false"
    export TZ="UTC"

    export CYLC_SUITE_RUN_DIR="$HOME/cylc-run/u-dl705"
    export CYLC_SUITE_DEF_PATH="${HOME}/cylc-run/u-dl705"
    export CYLC_SUITE_DEF_PATH_ON_SUITE_HOST="/home/561/mjl561/cylc-run/u-dl705"
    export CYLC_SUITE_UUID="1e7f669e-b71d-495e-aa77-60ed25e20612"

    # CYLC TASK ENVIRONMENT:
    export CYLC_TASK_JOB="1/SY_no_urban_SY_1_L_ancil_lct_postproc_c4/02"
    export CYLC_TASK_NAMESPACE_HIERARCHY="root HOST_HPC HOST_ANTS_MPP HOST_ANTS_SERIAL ANCIL_ANTS ANCIL_LCT SY_no_urban_SY_1_L_ancil SY_no_urban_SY_1_L_ancil_lct_postproc_c4"
    export CYLC_TASK_DEPENDENCIES="SY_no_urban_SY_1_L_ancil_lct.1"
    export CYLC_TASK_TRY_NUMBER=1
}

cylc__job__inst__global_init_script() {
# GLOBAL-INIT-SCRIPT:
module use /g/data/hr22/modulefiles
        module load cylc7/23.09
	module load python2-as-python
        export CYLC_FLOW_VERSION=7.9.7
        export ROSE_VERSION=2019.01.7
        export CYLC_VERSION=$CYLC_FLOW_VERSION
        export PATH=$CYLC_DIR/../23.09/bin:$PATH
}

cylc__job__inst__user_env() {
    # TASK RUNTIME ENVIRONMENT:
    export TMPDIR TIDS ROSE_TASK_N_JOBS ANTS_NPROCESSES ANCIL_MASTER ANCIL_PREPROC_PATH TRANSFORM_DIR ANCIL_TARGET_PATH HORIZ_GRID VERT_LEV CAPGRID CAPHORIZGRID VARIABLE ROSE_TASK_APP TARGET_VEGFRAC TARGET_LSM
    TMPDIR="$CYLC_TASK_WORK_DIR"
    TIDS="/g/data/access/TIDS"
    ROSE_TASK_N_JOBS="${PBS_NCPUS:-1}"
    ANTS_NPROCESSES="1"
    ANCIL_MASTER=~/"$ROSE_SUITE_DIR_REL/share/data/etc/ancil_master_ants/"
    ANCIL_PREPROC_PATH="$ROSE_DATA/etc/ants_preproc"
    TRANSFORM_DIR="/g/data/access/TIDS/UM/ancil/data/transforms"
    ANCIL_TARGET_PATH="$ROSE_DATA/ancils/SY_no_urban/SY_1_L"
    HORIZ_GRID="$ROSE_DATA/ancils/SY_no_urban/SY_1_L/grid.nl"
    VERT_LEV="$ROSE_DATA/ancils/SY_no_urban/SY_1_L/L90_40km"
    CAPGRID="$ROSE_DATA/ancils/SY_no_urban/SY_1_L/grid.nl"
    CAPHORIZGRID=""
    VARIABLE="F"
    ROSE_TASK_APP="ancil_lct_postproc_c4"
    TARGET_VEGFRAC="qrparm.veg.frac_cci"
    TARGET_LSM="qrparm.mask_cci"
}

cylc__job__inst__init_script() {
# INIT-SCRIPT:
module use ~access/modules
# module load python3-as-python
module load cap/9.2
# ants module is loaded in site/nci_gadi/python_env
ulimit -s unlimited
}

cylc__job__inst__env_script() {
# ENV-SCRIPT:
eval $(rose task-env)
}

cylc__job__inst__script() {
# SCRIPT:
rose task-run -v --opt-conf-key='(nci-gadi)'
}

cylc__job__inst__post_script() {
# POST-SCRIPT:
cd $ROSE_DATA/ancils/SY_no_urban/SY_1_L && ln -sf qrparm.veg.frac_cci qrparm.veg.frac &&  ln -sf qrparm.veg.frac_cci.nc qrparm.veg.frac.nc
}

. "${CYLC_DIR}/lib/cylc/job.sh"
cylc__job__main

#EOF: 1/SY_no_urban_SY_1_L_ancil_lct_postproc_c4/02