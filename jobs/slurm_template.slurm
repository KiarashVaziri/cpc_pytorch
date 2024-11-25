#!/bin/bash 

# inputs to slurm (should be exported as enviroment variables)
: "${Q:=shanechi}"
: "${TIME:=72:00:00}"
: "${MEM:=32GB}"

: "${env_dir:=/scratch1/vazirigo/envs/392_venv_mdfine/bin/activate}"
: "${log_dir:=/scratch1/vazirigo/logs}"
: "${log_file:=log.txt}"
: "${code_dir:=/project/shanechi_280/vazirigo/cpc_pytorch/}"
: "${command:=python}"

# For discovery
module purge
module load gcc/8.3.0
module load cuda/10.0.130
module load cudnn/7.5.1-10.1

# mkdir -p $log_dir

mamba init bash
source ~/.bashrc
mamba activate ssl_py39
# echo $env_dir
# source $env_dir 

# export MPLCONFIGDIR=/scratch1/eerturk/python-cache/mpl-cache
export GIT_PYTHON_REFRESH=quiet

cd $code_dir
${command}