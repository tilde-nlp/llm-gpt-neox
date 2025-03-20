#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <BASE_DIR> <OUTPUT_DIR>"
    exit 1
fi

# Assign command-line arguments to variables
BASE_DIR="$1"
OUTPUT_DIR="$2"

# Other fixed variables
CONTAINER_PATH="/scratch/project_465001281/containers/rocm603_flash.sif"
PROJECT_DIR="/project/project_465001281/IP/llm-gpt-neox/tools/datasets"

# Get the first subfolder (for testing one iteration)
folder="$BASE_DIR"

# Remove trailing slash to get the folder name
folder_name=$(basename "$folder")

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR/$folder_name"

# Define log file path
LOG_FILE="$OUTPUT_DIR/$folder_name.merge.log"

# Log input variables
echo "Logging merge process for '$folder_name'" | tee "$LOG_FILE"
echo "BASE_DIR: $BASE_DIR" | tee -a "$LOG_FILE"
echo "OUTPUT_DIR: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "CONTAINER_PATH: $CONTAINER_PATH" | tee -a "$LOG_FILE"
echo "PROJECT_DIR: $PROJECT_DIR" | tee -a "$LOG_FILE"
echo "Folder being merged: $folder_name" | tee -a "$LOG_FILE"

echo "Listing .bin and .idx files in $BASE_DIR" | tee -a "$LOG_FILE"
ls "$BASE_DIR"/*.bin "$BASE_DIR"/*.idx 2>/dev/null | tee -a "$LOG_FILE"

echo "Starting merge process..." | tee -a "$LOG_FILE"

# Run the merge job using srun
srun --account=project_465001281 \
     --partition=dev-g \
     --gpus-per-node=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=28 \
     --mem-per-gpu=60G \
     --time=2:00:00 \
     --nodes=1 \
     singularity exec "$CONTAINER_PATH" \
     bash -c "cd $PROJECT_DIR; \$WITH_CONDA; python merge_datasets_mmap.py --input '$folder' --output-prefix '$OUTPUT_DIR/$folder_name'" 2>&1 | tee -a "$LOG_FILE"

# Check exit status and log success/failure
if [ $? -eq 0 ]; then
    echo "Merging of '$folder_name' completed successfully." | tee -a "$LOG_FILE"
else
    echo "Merging of '$folder_name' encountered an error." | tee -a "$LOG_FILE"
fi
