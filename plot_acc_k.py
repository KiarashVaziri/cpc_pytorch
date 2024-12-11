import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the metrics files
metrics_dir = 'metrics/'

# Random seeds to average over
random_seeds = [21, 22, 23, 24, 25]

# k values to consider
k_values = [2, 4, 8]

# Helper function to load test accuracy from a metrics file
def load_test_accuracy(file_path):
    with open(file_path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics.get('test_acc', None)

# Dictionaries to store results
k_acc = {"gru": [], "ldm": [], "ldmw": []}
k_min_max = {"gru": {"min": [], "max": []},
             "ldm": {"min": [], "max": []},
             "ldmw": {"min": [], "max": []}}

# Process metrics files
for k in k_values:
    seed_accs = {"gru": [], "ldm": [], "ldmw": []}
    for seed in random_seeds:
        # GRU
        file_path_gru = f'metrics/metrics_gru_ldmfcst0_{k}_rnd{seed}'
        acc = load_test_accuracy(file_path=file_path_gru)
        seed_accs['gru'].append(acc)

        # LDM
        file_path_ldm = f'metrics/metrics_ldm_ldmfcst0_{k}_rnd{seed}'
        acc = load_test_accuracy(file_path=file_path_ldm)
        seed_accs['ldm'].append(acc)

        # LDMW
        file_path_ldmw = f'metrics/metrics_ldm_ldmfcst1_{k}_rnd{seed}'
        acc = load_test_accuracy(file_path=file_path_ldmw)
        seed_accs['ldmw'].append(acc)

    # Compute mean, min, and max for each configuration
    for model in ['gru', 'ldm', 'ldmw']:
        k_acc[model].append(np.mean(seed_accs[model]))
        k_min_max[model]['min'].append(np.min(seed_accs[model]))
        k_min_max[model]['max'].append(np.max(seed_accs[model]))

# Plotting
plt.figure(figsize=(12, 6))

# GRU
plt.plot(k_values, k_acc['gru'], label='(i) gru', color='red')
plt.fill_between(k_values, k_min_max['gru']['min'], k_min_max['gru']['max'], color='red', alpha=0.2)

# LDM
plt.plot(k_values, k_acc['ldm'], label='(ii) ldm + learnable w', color='blue')
plt.fill_between(k_values, k_min_max['ldm']['min'], k_min_max['ldm']['max'], color='blue', alpha=0.2)

# LDMW
plt.plot(k_values, k_acc['ldmw'], label='(iii) ldm', color='green')
plt.fill_between(k_values, k_min_max['ldmw']['min'], k_min_max['ldmw']['max'], color='green', alpha=0.2)

# Finalize plot
plt.legend()
plt.xlabel('k')
plt.ylabel('test acc')
plt.title("Speaker classification test acc")
plt.savefig('figs/combined_acc_with_margins.png')
# plt.show()
