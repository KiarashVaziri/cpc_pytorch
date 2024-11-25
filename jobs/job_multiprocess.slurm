export code_dir="/project/shanechi_280/vazirigo/cpc_pytorch/"
cd /project/shanechi_280/vazirigo/cpc_pytorch/
# cd /scratch1/vazirigo/cpc_pytorch/hpc/slurm

export log_dir="/project/shanechi_280/vazirigo/cpc_pytorch/logs";
export command="python3 -u train_cpc_model.py confs/rnds/gru_ldmfcst0/conf_gru_ldmfcst0_rnd21.py"
sbatch --job-name gru_ldmfcst0_k4_21 --time 48:00:00 --mem 32GB --partition shanechi  --cpus-per-task 4 --mail-user vazirigo@usc.edu jobs/slurm_template.slurm;


export log_dir="/project/shanechi_280/vazirigo/cpc_pytorch/logs";
export command="python3 -u train_cpc_model.py confs/rnds/gru_ldmfcst0/conf_gru_ldmfcst0_rnd22.py"
sbatch --job-name gru_ldmfcst0_k4_22 --time 48:00:00 --mem 32GB --partition shanechi  --cpus-per-task 4 --mail-user vazirigo@usc.edu jobs/slurm_template.slurm;


export log_dir="/project/shanechi_280/vazirigo/cpc_pytorch/logs";
export command="python3 -u train_cpc_model.py confs/rnds/gru_ldmfcst0/conf_gru_ldmfcst0_rnd23.py"
sbatch --job-name gru_ldmfcst0_k4_23 --time 48:00:00 --mem 32GB --partition shanechi  --cpus-per-task 4 --mail-user vazirigo@usc.edu jobs/slurm_template.slurm;


export log_dir="/project/shanechi_280/vazirigo/cpc_pytorch/logs";
export command="python3 -u train_cpc_model.py confs/rnds/gru_ldmfcst0/conf_gru_ldmfcst0_rnd24.py"
sbatch --job-name gru_ldmfcst0_k4_24 --time 48:00:00 --mem 32GB --partition shanechi  --cpus-per-task 4 --mail-user vazirigo@usc.edu jobs/slurm_template.slurm;


export log_dir="/project/shanechi_280/vazirigo/cpc_pytorch/logs";
export command="python3 -u train_cpc_model.py confs/rnds/gru_ldmfcst0/conf_gru_ldmfcst0_rnd25.py"
sbatch --job-name gru_ldmfcst0_k4_25 --time 48:00:00 --mem 32GB --partition shanechi  --cpus-per-task 4 --mail-user vazirigo@usc.edu jobs/slurm_template.slurm;

##########
#ldm_ldmfcst0
##########
# export log_dir="/project/shanechi_280/vazirigo/cpc_pytorch/logs";
# export command="python3 -u train_cpc_model.py confs/rnds/ldm_ldmfcst0/conf_ldm_ldmfcst0_rnd21.py"
# sbatch --job-name ldm_ldmfcst0_k3_25 --time 48:00:00 --mem 32GB --partition shanechi  --cpus-per-task 4 --mail-user vazirigo@usc.edu jobs/slurm_template.slurm;


# export log_dir="/project/shanechi_280/vazirigo/cpc_pytorch/logs";
# export command="python3 -u train_cpc_model.py confs/rnds/ldm_ldmfcst0/conf_ldm_ldmfcst0_rnd22.py"
# sbatch --job-name ldm_ldmfcst0_k3_22 --time 48:00:00 --mem 32GB --partition shanechi  --cpus-per-task 4 --mail-user vazirigo@usc.edu jobs/slurm_template.slurm;


# export log_dir="/project/shanechi_280/vazirigo/cpc_pytorch/logs";
# export command="python3 -u train_cpc_model.py confs/rnds/ldm_ldmfcst0/conf_ldm_ldmfcst0_rnd23.py"
# sbatch --job-name ldm_ldmfcst0_k3_23 --time 48:00:00 --mem 32GB --partition shanechi  --cpus-per-task 4 --mail-user vazirigo@usc.edu jobs/slurm_template.slurm;


# export log_dir="/project/shanechi_280/vazirigo/cpc_pytorch/logs";
# export command="python3 -u train_cpc_model.py confs/rnds/ldm_ldmfcst0/conf_ldm_ldmfcst0_rnd24.py"
# sbatch --job-name ldm_ldmfcst0_k3_24 --time 48:00:00 --mem 32GB --partition shanechi  --cpus-per-task 4 --mail-user vazirigo@usc.edu jobs/slurm_template.slurm;


# export log_dir="/project/shanechi_280/vazirigo/cpc_pytorch/logs";
# export command="python3 -u train_cpc_model.py confs/rnds/ldm_ldmfcst0/conf_ldm_ldmfcst0_rnd25.py"
# sbatch --job-name ldm_ldmfcst0_k3_25 --time 48:00:00 --mem 32GB --partition shanechi  --cpus-per-task 4 --mail-user vazirigo@usc.edu jobs/slurm_template.slurm;

