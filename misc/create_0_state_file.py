import os
import yaml

# Define the directory path
base_dir = "/scratch/project_465001281/tokenized/final_data_merged"
out_dir = "/scratch/project_465001281/tokenized/final_data_sliced"

# List all files in the directory
files = os.listdir(base_dir)

# Extract languages from filenames
languages = set()
for file in files:
    if file.endswith(".bin"):
        lang_code = file.split(".")[0]  # Extract prefix before ".bin"
        languages.add(lang_code)

# Create YAML structure
yaml_data = {}
for lang in sorted(languages):  # Sort for consistency
    bin_file = f"{lang}.bin"

    yaml_data[lang] = {
        "datapath": os.path.join(base_dir, bin_file),
        "idxs": [0]  # State always begins from 0
    }

# Save to a YAML file
yaml_file_path = "state.0.yaml"
with open(yaml_file_path, "w") as yaml_file:
    yaml.dump(yaml_data, yaml_file, default_flow_style=False)

print(f" State file saved as {yaml_file_path}")
