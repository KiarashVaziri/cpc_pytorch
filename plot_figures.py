import pickle
import numpy as np
import matplotlib.pyplot as plt

# Define your variables
ar_type = 'gru'
use_ldm_params = 0
k = 8
rnd_seeds = [21, 22, 23, 24, 25]
pickle_files = [f"metrics/metrics_{ar_type}_ldmfcst{use_ldm_params}_{8}_rnd{rnd_seed}" for rnd_seed in rnd_seeds]

# Initialize list to store test_acc values and dictionary for metrics
test_accs = []
combined_metrics = {}

# Iterate through each pickle file to load and accumulate the values
for file in pickle_files:
    try:
        # Open and load the pickle file
        with open(file, 'rb') as f:
            metrics = pickle.load(f)
        
        # Append the test_acc from the loaded metrics dictionary
        test_accs.append(metrics['test_acc'])  # Adjust the key based on your dictionary structure

        # For the first file, initialize combined_metrics with lists of zeros of the same length
        if not combined_metrics:
            combined_metrics = {key: np.zeros_like(value) for key, value in metrics.items()}
        
        # Add the values from each file to accumulate for averaging
        for key in metrics:
            combined_metrics[key] += np.array(metrics[key])

    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading {file}: {e}")

# Calculate and print the average test accuracy
if test_accs:
    avg_test_acc = np.mean(test_accs)
    print(f"Test accs: {test_accs}")
    print(f"Average test acc: {avg_test_acc}")
else:
    print("No valid test_acc found in the pickle files.")

# Calculate the average for each metric over the different runs
avg_metrics = {key: combined_metrics[key] / len(pickle_files) for key in combined_metrics}

# Compute min and max values for each metric over epochs
min_metrics = {key: np.min([pickle.load(open(file, 'rb'))[key] for file in pickle_files], axis=0) for key in combined_metrics}
max_metrics = {key: np.max([pickle.load(open(file, 'rb'))[key] for file in pickle_files], axis=0) for key in combined_metrics}

# Plot the average metrics, and mark the min and max values over epochs
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.plot(avg_metrics['epoch_loss_training'], label='Avg Training Loss', color='blue')
plt.plot(avg_metrics['epoch_loss_validation'], label='Avg Validation Loss', color='orange')
plt.fill_between(range(len(avg_metrics['epoch_loss_training'])), min_metrics['epoch_loss_training'], max_metrics['epoch_loss_training'], color='blue', alpha=0.2)
plt.fill_between(range(len(avg_metrics['epoch_loss_validation'])), min_metrics['epoch_loss_validation'], max_metrics['epoch_loss_validation'], color='orange', alpha=0.2)
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Average Loss vs Epochs'); plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.plot(avg_metrics['epoch_acc_training'], label='Avg Training Accuracy', color='green')
plt.plot(avg_metrics['epoch_acc_validation'], label='Avg Validation Accuracy', color='red')
plt.fill_between(range(len(avg_metrics['epoch_acc_training'])), min_metrics['epoch_acc_training'], max_metrics['epoch_acc_training'], color='green', alpha=0.2)
plt.fill_between(range(len(avg_metrics['epoch_acc_validation'])), min_metrics['epoch_acc_validation'], max_metrics['epoch_acc_validation'], color='red', alpha=0.2)
plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Average Accuracy vs Epochs'); plt.legend()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(f"figs/epochs_{ar_type}_ldmfcst{use_ldm_params}_k{k}.png")
plt.show()
