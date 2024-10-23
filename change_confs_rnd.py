import os

# Path to the 'rnds' folder
rnds_folder = os.path.join('confs', 'rnds')

# Arrays of target words and new values
target_words = ['train_model', 'test_model', 'ar_model_params', 'num_speakers', 'max_epochs', 'patience', 'w_use_ldm_params']  # Replace with your variable names
new_values = [1, 1, {'type':'ldm'}, 10, 150, 150, 1]  # Replace with the corresponding new values

# Ensure the length of target words and new values are the same
if len(target_words) != len(new_values):
    raise ValueError("The length of target_words and new_values must be the same.")

# Iterate over all .py files in the 'rnds' folder
for filename in os.listdir(rnds_folder):
    if filename.endswith('.py'):
        file_path = os.path.join(rnds_folder, filename)
        
        # Read the content of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Modify the content
        modified_lines = []
        for line in lines:
            modified = False
            for i, target_word in enumerate(target_words):
                if line.strip().startswith(f'{target_word} ='):
                    # Modify the line by replacing the value with the corresponding new value
                    modified_line = f"{target_word} = {new_values[i]}\n"
                    modified_lines.append(modified_line)
                    modified = True
                    break  # Exit loop once a match is found for the current line
            if not modified:
                modified_lines.append(line)
        
        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

        print(f"Modified {filename}")
