# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

Code for training and evaluating a number of different CPC models.

"""
import os
import numpy as np
import time
import sys

from importlib.machinery import SourceFileLoader
import torch
torch.autograd.set_detect_anomaly(True)

conf_file_name = 'conf_train_cpc_model_orig_implementation.py'

def set_config_seed(filepath, seed):
    # Read the entire file into memory
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Find and update the line with the variable
    new_value = seed
    for i, line in enumerate(lines):
        if line.startswith('random_seed'):
            # Assuming variable_to_change is defined as a simple assignment
            lines[i] = f'random_seed = {new_value}\n'
            break

    # Write the modified content back to the file
    with open(filepath, 'w') as file:
        file.writelines(lines)

NUM_SEEDS = 1
for seed in range(1, NUM_SEEDS+1):
    set_config_seed(conf_file_name, seed)
    os.system(f'echo "Running CPC seed {seed}/{NUM_SEEDS}"')
    os.system(f'python train_cpc_model.py')

os.system('echo "All seeds completed!"')




