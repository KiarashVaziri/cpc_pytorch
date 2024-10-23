# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

Modified to run multiple configurations in parallel.
"""

import sys
import os
from multiprocessing import Pool, cpu_count

# Function to run the model with a given configuration file
def run_experiment(conf_file):
    os.system(f'echo "Running experiment using configuration file {conf_file} (see log file for further details)"')
    os.system(f'python train_cpc_model.py {conf_file}')

# Read the configuration file
if len(sys.argv) != 2:
    sys.exit('''Usage:\n  python cpc_test_bench.py <dir_to_configuration_files>\n\n
             where <dir_to_configuration_files> is a directory containing at least one .py configuration file.''')
else:
    dir_of_conf_files = sys.argv[1]
    
# Check if the given directory is actually a directory
if not os.path.isdir(dir_of_conf_files):
    sys.exit(f'The given argument {dir_of_conf_files} is not a directory!')

# Find out the configuration files in the given directory
filenames_in_dir = os.listdir(dir_of_conf_files)

# Clean the list if there are other files than .py configuration files
conf_file_names = [os.path.join(dir_of_conf_files, filename) for filename in filenames_in_dir if filename.endswith('.py')]

# Run the experiments in parallel
if __name__ == '__main__':
    print(f"Running {len(conf_file_names)} CPC experiments in parallel...")
    
    # Create a pool of workers, default is the number of CPU cores
    with Pool(processes=cpu_count()) as pool:
        pool.map(run_experiment, conf_file_names)

    print("All experiments completed!")
