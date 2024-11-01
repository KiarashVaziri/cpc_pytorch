import os
import re

# Dictionary of target words and their corresponding new values
target_values = {
    'future_predicted_timesteps': 2, 
    'ar_model_params': {'type': 'ldm'},
    'w_use_ldm_params': 1,
    'num_speakers': 10,
    'max_epochs': 100,
    # 'loss_name' : "'CPC_mse_loss'\n,
}

# Path to the 'rnds' folder
rnds_folder = os.path.join('confs', 'rnds', f"{target_values['ar_model_params']['type']}_ldmfcst{target_values['w_use_ldm_params']}")

# Iterate over all .py files in the 'rnds' folder
for idx, filename in enumerate(os.listdir(rnds_folder)):
    if filename.endswith('.py'):
        match = re.search(r'_rnd(\d+)\.py$', filename)
        if not match:
            continue  # Skip files that don't match the expected pattern
        
        rnd_value = match.group(1)
        file_path = os.path.join(rnds_folder, filename)
        
        # Read the content of the current file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Modify the content
        modified_lines = []
        for line in lines:
            modified = False
            # Change target values based on the dictionary
            for target_word, new_value in target_values.items():
                if line.strip().startswith(f'{target_word} ='):
                    modified_line = f"{target_word} = {new_value}\n"
                    modified_lines.append(modified_line)
                    modified = True
                    break  # Exit loop once a match is found for the current line
            
            # Specifically modify 'num_workers' if found
            if "'num_workers':" in line:
                line = line.replace("'num_workers': 0", "'num_workers': 4")
                modified = True
            
            # Specifically modify 'pin_memory' if found
            if "'pin_memory':" in line:
                line = line.replace("'pin_memory': False", "'pin_memory': True")
                modified = True

            # Modify encoder, AR, W model paths, and log file name
            if "encoder_best_model_name =" in line:
                modified_line = (
                    f"encoder_best_model_name = f\"models/{{num_speakers}}/CPC_Encoder_best_model_"
                    f"{{ar_model_params['type']}}_ldmfcst{{w_use_ldm_params}}_k{{future_predicted_timesteps}}_rnd{rnd_value}.pt\"\n"
                )
                modified_lines.append(modified_line)
                modified = True
            elif "ar_best_model_name =" in line:
                modified_line = (
                    f"ar_best_model_name = f\"models/{{num_speakers}}/CPC_AR_best_model_"
                    f"{{ar_model_params['type']}}_ldmfcst{{w_use_ldm_params}}_k{{future_predicted_timesteps}}_rnd{rnd_value}.pt\"\n"
                )
                modified_lines.append(modified_line)
                modified = True
            elif "w_best_model_name =" in line:
                modified_line = (
                    f"w_best_model_name = f\"models/{{num_speakers}}/W_best_model_"
                    f"{{ar_model_params['type']}}_ldmfcst{{w_use_ldm_params}}_k{{future_predicted_timesteps}}_rnd{rnd_value}.pt\"\n"
                )
                modified_lines.append(modified_line)
                modified = True
            elif "name_of_log_textfile =" in line:
                modified_line = (
                    f"name_of_log_textfile = f\"logs/trainlog_{{ar_model_params['type']}}_ldmfcst"
                    f"{{w_use_ldm_params}}_dtch{{w_params['detach']}}_k{{future_predicted_timesteps}}_rnd{rnd_value}.txt\"\n"
                )
                modified_lines.append(modified_line)
                modified = True

            if not modified:
                modified_lines.append(line)
        
        # Generate the new filename
        new_filename = f"conf_{target_values['ar_model_params']['type']}_ldmfcst{target_values['w_use_ldm_params']}_rnd{rnd_value}.py"
        new_file_path = os.path.join(rnds_folder, new_filename)
        
        # Check if the new file already exists
        if os.path.exists(new_file_path):
            print(f"File {new_filename} exists, updating its content.")
        else:
            print(f"File {new_filename} does not exist, creating a new file.")
        
        # Write the modified content (either update or create the file)
        with open(new_file_path, 'w') as new_file:
            new_file.writelines(modified_lines)
        
        print(f"Saved modified content as {new_filename}")
