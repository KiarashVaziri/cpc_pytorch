#!/bin/bash

#SBATCH --partition=shanechi
#SBATCH --account=shanechi_280
#SBATCH --job-name=ldm_ldmfcst0_k2
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=5
#SBATCH --mem=64GB
#SBATCH --time=48:00:00

module purge
module load gcc/8.3.0
module load cuda/10.1.243
module load cudnn/8.0.2-10.1

mamba activate ssl_py39

echo "Conda Environment: $CONDA_DEFAULT_ENV"
echo Loaded all modules successfully. Running the program...

python cpc_concurrent_testbench.py confs/rnds/ldm_ldmfcst0/
# python3 cpc_test_bench.py confs/rnds/ldm_ldmfcst0/
