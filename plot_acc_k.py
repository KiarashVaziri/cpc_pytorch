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

k_acc = {"gru":[],
             "ldm":[],
             "ldmw":[]}

for k in k_values:
    seed_accs = {"gru":[],
                "ldm":[],
                "ldmw":[]}
    for seed in random_seeds:
        file_path_gru = f'metrics/metrics_gru_ldmfcst0_{k}_rnd{seed}_mse'
        acc = load_test_accuracy(file_path=file_path_gru)
        seed_accs['gru'].append(acc)

        file_path_ldm = f'metrics/metrics_ldm_ldmfcst0_{k}_rnd{seed}_mse'
        acc = load_test_accuracy(file_path=file_path_ldm)
        seed_accs['ldm'].append(acc)

        file_path_ldmw = f'metrics/metrics_ldm_ldmfcst1_{k}_rnd{seed}_mse'
        acc = load_test_accuracy(file_path=file_path_ldmw)
        seed_accs['ldmw'].append(acc)

    k_acc['gru'].append(np.mean(seed_accs['gru']))
    k_acc['ldm'].append(np.mean(seed_accs['ldm']))
    k_acc['ldmw'].append(np.mean(seed_accs['ldmw']))

plt.figure(figsize=(12, 6))

plt.plot(k_values, k_acc['gru'], label='gru', color='red')
plt.plot(k_values, k_acc['ldm'], label='ldm', color='blue')
plt.plot(k_values, k_acc['ldmw'], label='ldmw', color='green')
plt.legend()
plt.xlabel('k'); plt.ylabel('test acc'); plt.title("Speaker classification test acc")
plt.savefig('figs/acc_overk_mse.png')