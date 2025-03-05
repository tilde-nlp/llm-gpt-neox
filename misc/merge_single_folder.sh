#!/bin/bash

# Base directory containing the subfolders
# Example:
BASE_DIR="/scratch/project_465001281/tokenized/1B_gucci_sliced/merged/N"
OUTPUT_DIR="/scratch/project_465001281/tokenized/1B_gucci_sliced/merged/N/1B_N_4"
CONTAINER_PATH="/scratch/project_465001281/MK/rocm603_martin_local"
PROJECT_DIR="/project/project_465001281/llm-gpt-neox/tools/datasets"

# Get the first subfolder (for testing one iteration)
#folder=$(ls -d "$BASE_DIR"/*/ | head -n 1)

# Check if a subfolder was found
#if [ -z "$folder" ]; then
#    echo "No subfolder found in $BASE_DIR"
#    exit 1
#fi

folder=$BASE_DIR

# Remove trailing slash to get the folder name
folder_name=$(basename "$folder")

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR/$folder_name"

# Log the operation
echo "Merging '$folder_name'"

# Run the merge job using srun (not in background)
srun --account=project_465001281 \
     --partition=dev-g \
     --gpus-per-node=4 \
     --ntasks-per-node=1 \
     --cpus-per-task=28 \
     --mem-per-gpu=60G \
     --time=2:00:00 \
     --nodes=1 \
     singularity exec "$CONTAINER_PATH" \
     bash -c "cd $PROJECT_DIR; \$WITH_CONDA; python ../merge_datasets_mmap.py --input '$folder' --output-prefix '$OUTPUT_DIR/$folder_name'"

echo "Merging of '$folder_name' completed."