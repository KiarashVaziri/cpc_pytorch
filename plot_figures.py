import pickle
import numpy as np
import matplotlib.pyplot as plt

# List of paths to your saved pickle files
ar_type = 'ldm'
use_ldm_params = 1
rnd_seeds = [22, 23]
pickle_files = [f"metrics_{ar_type}_ldmfcst{use_ldm_params}_rnd{rnd_seed}" for rnd_seed in rnd_seeds]

# Initialize a dictionary to hold summed metrics for averaging
combined_metrics = {}

# Load each pickle file and accumulate the values
for file in pickle_files:
    with open(file, 'rb') as f:
        metrics = pickle.load(f)
    
    # For the first file, initialize combined_metrics with lists of zeros of the same length
    if not combined_metrics:
        combined_metrics = {key: np.zeros_like(value) for key, value in metrics.items()}
    
    # Add the values from each file
    for key in metrics:
        combined_metrics[key] += np.array(metrics[key])

# Calculate the average for each metric
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
plt.savefig(f"epochs_{ar_type}_ldmfcst{use_ldm_params}.png")
plt.show()
