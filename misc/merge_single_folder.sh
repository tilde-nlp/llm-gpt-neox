#!/bin/bash

# Base directory containing the subfolders
# Example:
# folder to merge
BASE_DIR="/scratch/project_465001281/tokenized/final_data_sliced/warmup_0"
OUTPUT_DIR="/scratch/project_465001281/tokenized/final_data_sliced/merged/warmup_0" # this will create an unnecessary subfolder for some reason
CONTAINER_PATH="/scratch/project_465001281/containers/rocm603_flash.sif"
# needs to point to the folder where merge_datasets_mmap.py is located
PROJECT_DIR="/project/project_465001281/IP/llm-gpt-neox/tools/datasets"

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

# Define log file path
LOG_FILE="$OUTPUT_DIR/$folder_name.merge.log"

# Log hardcoded variables
echo "Logging merge process for '$folder_name'" > "$LOG_FILE"
echo "BASE_DIR: $BASE_DIR" >> "$LOG_FILE"
echo "OUTPUT_DIR: $OUTPUT_DIR" >> "$LOG_FILE"
echo "CONTAINER_PATH: $CONTAINER_PATH" >> "$LOG_FILE"
echo "PROJECT_DIR: $PROJECT_DIR" >> "$LOG_FILE"
echo "Folder being merged: $folder_name" >> "$LOG_FILE"

echo "Listing .bin and .idx files in $BASE_DIR" >> "$LOG_FILE"
ls "$BASE_DIR"/*.bin "$BASE_DIR"/*.idx 2>/dev/null >> "$LOG_FILE"

# Log the operation
echo "Starting merge process..." >> "$LOG_FILE"

# Run the merge job using srun (not in background)
srun --account=project_465001281 \
     --partition=dev-g \
     --gpus-per-node=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=28 \
     --mem-per-gpu=60G \
     --time=2:00:00 \
     --nodes=1 \
     singularity exec "$CONTAINER_PATH" \
     bash -c "cd $PROJECT_DIR; \$WITH_CONDA; python merge_datasets_mmap.py --input '$folder' --output-prefix '$OUTPUT_DIR/$folder_name'" &>> "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "Merging of '$folder_name' completed successfully." >> "$LOG_FILE"
else
    echo "Merging of '$folder_name' encountered an error." >> "$LOG_FILE"
fi