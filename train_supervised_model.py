# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

Code for training and evaluating a CPC model.

"""

import numpy as np
import time
import sys
import os

from importlib.machinery import SourceFileLoader
from copy import deepcopy
from torch import cuda, no_grad, save, load
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm 

from utils import convert_py_conf_file_to_text
# from utils import visualize_tsne
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read the configuration file
if len(sys.argv) > 2:
    sys.exit('Usage: \n1) python train_cpc_model.py \nOR \n2) python train_cpc_model.py <configuration_file>')
if len(sys.argv) == 2:
    conf = SourceFileLoader('', sys.argv[1]).load_module()
    conf_file_name = sys.argv[1]
else:
    try:
        # import conf_train_cpc_model_orig_implementation as conf
        # conf_file_name = 'conf_train_cpc_model_orig_implementation.py'
        from confs import conf_train_ldmcpc_model as conf
        conf_file_name = 'confs/conf_train_ldmcpc_model.py'
    except ModuleNotFoundError:
        sys.exit('''Usage: \n1) python train_cpc_model.py \nOR \n2) python train_cpc_model.py <configuration_file>\n\n
        By using the first option, you need to have a configuration file named "conf_train_cpc_model.py" in the same directory 
        as "train_cpc_model.py"''')


# Import our models
CPC_encoder = getattr(__import__('cpc_model', fromlist=[conf.encoder_name]), conf.encoder_name)
CPC_autoregressive_model = getattr(__import__('cpc_model', fromlist=[conf.autoregressive_model_name]), conf.autoregressive_model_name)

# Import our dataset for our data loader
CPC_dataset = getattr(__import__('cpc_data_loader', fromlist=[conf.dataset_name]), conf.dataset_name)

# Import our optimization algorithm
optimization_algorithm = getattr(__import__('torch.optim', fromlist=[conf.optimization_algorithm]), conf.optimization_algorithm)

# Import our learning rate scheduler
if conf.use_lr_scheduler:
    scheduler = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler]), conf.lr_scheduler)

if __name__ == '__main__':

    # Open the file for writing
    file = open(conf.name_of_log_textfile, 'w')
    file.close()
    
    # Read the text in the configuration file and add it to the logging file
    if conf.print_conf_contents:
        conf_file_lines = convert_py_conf_file_to_text(conf_file_name)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write(f'The configuration settings in the file {conf_file_name}:\n\n')
            for line in conf_file_lines:
                f.write(f'{line}\n')
            f.write('\n########################################################################################\n\n\n\n')
        
    
    # Use CUDA if it is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    with open(conf.name_of_log_textfile, 'a') as f:
        f.write(f'Process on {device}\n\n')

    # Initialize our models
    Encoder = CPC_encoder(**conf.encoder_params)
    AR_model = CPC_autoregressive_model(**conf.ar_model_params)
    
    # Pass the models to the available device
    Encoder = Encoder.to(device)
    AR_model = AR_model.to(device)

    # Give the parameters of our models to an optimizer
    model_parameters = list(Encoder.parameters()) + list(AR_model.parameters())
    optimizer = optimization_algorithm(params=model_parameters, **conf.optimization_algorithm_params)

    # Initialize the data loaders
    if conf.train_model or conf.test_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Initializing training set...\n')
        training_set = CPC_dataset(train_val_test='train', **conf.params_train_dataset)
        train_data_loader = DataLoader(training_set, **conf.params_train)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n')
            f.write('Initializing validation set...\n')
        validation_set = CPC_dataset(train_val_test='validation', **conf.params_validation_dataset)
        validation_data_loader = DataLoader(validation_set, **conf.params_train)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n')
    if conf.test_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Initializing test set...\n')
        test_set = CPC_dataset(train_val_test='test', **conf.params_test_dataset)
        test_data_loader = DataLoader(test_set, **conf.params_test)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n\n')

    # Start training our model
    if conf.train_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Starting training...\n')
        
        for epoch in range(1, conf.max_epochs + 1):
            
            start_time = time.time()
    
            # Lists containing the losses of each epoch
            epoch_loss_training = []
            epoch_loss_validation = []
    
            # Indicate that we are in training mode, so e.g. dropout will function
            Encoder.train()
            AR_model.train()

            # Open the log file for writing
            log_file = open('training_log.txt', "w")

            # Loop through every batch of our training data
            for train_data in tqdm(train_data_loader, desc=f"Epoch {epoch}/{conf.max_epochs}, training: ", unit="batch", file=log_file):
                    # The loss of the batch
                    loss_batch = 0.0
                    
                    # Get the batches
                    X_input, batch_labels = train_data
                    X_input = X_input.to(device)
                    
                    # Zero the gradient of the optimizer
                    optimizer.zero_grad()
                    
                    # Pass our data through the encoder
                    Z = Encoder(X_input)
                    
                    # Create the output of the AR model. Note that the default AR_model flips the dimensions of Z from the
                    # form [batch_size, num_features, num_frames_encoding] into [batch_size, num_frames_encoding, num_features])
                    if conf.rnn_models_used_in_ar_model:
                        C, hidden = AR_model(Z, hidden)
                    else:
                        C = AR_model(Z)

                    # Take the features
                    feats = torch.mean(C, dim=1)

                    # TODO: pytorch implementation of logistic regression

                    