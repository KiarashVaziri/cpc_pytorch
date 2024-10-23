# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

Code for training and evaluating a number of different CPC models.
"""

import sys
import os
import subprocess
import concurrent.futures

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
conf_file_names = [filename for filename in filenames_in_dir if filename.endswith('.py')]

# Function to run the training script with a given configuration file
def run_experiment(conf_file, index, total):
    print(f'Running CPC experiment {index}/{total} using configuration file {conf_file}')
    command = f'python train_cpc_model.py {os.path.join(dir_of_conf_files, conf_file)}'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f'Experiment {conf_file} completed successfully.')
    else:
        print(f'Experiment {conf_file} failed with error:\n{result.stderr}')

# Go through each configuration file and run in parallel
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = {executor.submit(run_experiment, conf_file, index + 1, len(conf_file_names)): conf_file for index, conf_file in enumerate(conf_file_names)}

    # Optionally, you can wait for the results
    for future in concurrent.futures.as_completed(futures):
        conf_file = futures[future]
        try:
            future.result()  # Get the result to raise any exceptions that occurred
        except Exception as e:
            print(f'An error occurred while running {conf_file}: {e}')

print("All experiments submitted for execution!")
