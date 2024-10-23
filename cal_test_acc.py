import pickle
import numpy as np

# Define your variables
ar_type = 'gru'
use_ldm_params = 0
rnd_seeds = [21, 22, 23, 24, 25]
pickle_files = [f"metrics_{ar_type}_ldmfcst{use_ldm_params}_rnd{rnd_seed}" for rnd_seed in rnd_seeds]

# Initialize list to store test_acc values
test_accs = []

# Iterate through each pickle file
for pickle_file in pickle_files:
    try:
        # Open and load the pickle file
        with open(pickle_file, 'rb') as f:
            metrics = pickle.load(f)
        
        # Append the test_acc from the loaded metrics dictionary
        test_accs.append(metrics['test_acc'])  # Adjust the key based on your dictionary structure
        
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading {pickle_file}: {e}")

# Calculate and print the average test accuracy
if test_accs:
    avg_test_acc = np.mean(test_accs)
    print(f"Average test accuracy: {avg_test_acc}")
else:
    print("No valid test_acc found in the pickle files.")
