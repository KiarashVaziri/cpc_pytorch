#!/bin/bash

#SBATCH --partition=shanechi
#SBATCH --account=shanechi_280
#SBATCH --gres=none
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1:00:00

module purge
module load gcc/8.3.0
module load cuda/10.1.243
module load cudnn/8.0.2-10.1

mamba init bash
source ~/.bashrc
mamba activate ssl_py39

echo "Conda Environment: $CONDA_DEFAULT_ENV"
echo Loaded all modules successfully. Running the program...

python3 cpc_test_bench.py confs/rnds/ldm_ldmfcst0
# python3 train_cpc_model.py confs/conf_train_ldmcpc_rnd21.py