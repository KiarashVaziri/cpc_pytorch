import sys
import os
import subprocess
import concurrent.futures

# Function to get available GPU IDs
def get_available_gpus():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        # Split the output into a list of GPU IDs
        gpu_ids = result.stdout.strip().split('\n')
        return gpu_ids
    except subprocess.CalledProcessError as e:
        print(f"Error getting GPU information: {e}")
        return []

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

# Get available GPU IDs
available_gpus = get_available_gpus()

# Function to run the training script with a given configuration file
def run_experiment(conf_file, index, total, gpu):
    print(f'Running CPC experiment {index}/{total} using configuration file {conf_file} on GPU {gpu}')
    
    env = os.environ.copy()  # Copy the current environment
    env["CUDA_VISIBLE_DEVICES"] = gpu  # Set the GPU core

    command = f'python train_cpc_model.py {os.path.join(dir_of_conf_files, conf_file)}'
    
    # Run the command with the modified environment
    result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env)
    
    if result.returncode == 0:
        print(f'Experiment {conf_file} completed successfully.')
    else:
        print(f'Experiment {conf_file} failed with error:\n{result.stderr}')

# Go through each configuration file and run in parallel
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = {}
    
    # Assign GPU IDs to each configuration file
    for index, conf_file in enumerate(conf_file_names):
        gpu_index = index % len(available_gpus)  # Round-robin assignment
        gpu = available_gpus[gpu_index]
        futures[executor.submit(run_experiment, conf_file, index + 1, len(conf_file_names), gpu)] = conf_file

    # Optionally, you can wait for the results
    for future in concurrent.futures.as_completed(futures):
        conf_file = futures[future]
        try:
            future.result()  # Get the result to raise any exceptions that occurred
        except Exception as e:
            print(f'An error occurred while running {conf_file}: {e}')

print("All experiments submitted for execution!")
