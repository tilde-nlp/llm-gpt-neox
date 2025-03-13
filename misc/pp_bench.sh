#!/bin/bash

# Ensure correct usage
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 <root_ckpt_dir> <config_file> <test_folder> <root_temp_folder> [start_iteration]"
    exit 1
fi

# Assign input arguments
ROOT_CKPT_DIR=$1
CONFIG_FILE=$2
TEST_FOLDER=$3
TMP_PATH=$4
START_ITERATION=0  # Default: process all checkpoints

# If start_iteration argument is provided, convert it to an integer
if [ -n "$5" ]; then
    START_ITERATION=$((10#$5))  # Ensure it handles leading zeros correctly
fi

# SLURM parameters
ACCOUNT="project_465001281"
PARTITION="standard-g"
GPUS_PER_NODE=1
NTASKS_PER_NODE=1
CPUS_PER_TASK=7
MEM_PER_GPU="60G"
TIME="01:00:00"
NODES=1
CONTAINER_PATH="/scratch/project_465001281/containers/rocm603_inference.sif"
PROJECT_DIR="/project/project_465001281/IP/llm-gpt-neox"

# Log script start
echo "[INFO] Starting checkpoint evaluation."
echo "[INFO] Checking for available checkpoints in: $ROOT_CKPT_DIR"
echo "[INFO] Only evaluating checkpoints from iteration $START_ITERATION onwards."

# Detect all checkpoints
CHECKPOINTS=($(ls -d "$ROOT_CKPT_DIR"/global_step* 2>/dev/null | sort -V))

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "[ERROR] No checkpoints found in $ROOT_CKPT_DIR. Exiting."
    exit 1
fi

# Print detected checkpoints
echo "[INFO] Detected checkpoints:"
for CKPT in "${CHECKPOINTS[@]}"; do
    echo "       - $CKPT"
done
echo ""

# Iterate through each checkpoint
for CKPT_DIR in "${CHECKPOINTS[@]}"; do
    # Extract iteration number and convert to an integer
    ITERATION=$(basename "$CKPT_DIR" | sed 's/global_step//')
    ITERATION=$((10#$ITERATION))  # Convert to integer correctly

    # Log whether the checkpoint is being evaluated or skipped
    if [ "$ITERATION" -lt "$START_ITERATION" ]; then
        echo "[SKIP] Skipping global_step$ITERATION (below threshold $START_ITERATION)."
        continue
    fi
    echo "[RUN] Evaluating global_step$ITERATION..."

    # Define output CSV file
    OUTPUT_CSV="$ROOT_CKPT_DIR/$ITERATION.csv"

    # Run the Python script using srun in a separate background instance
    srun --account="$ACCOUNT" \
         --partition="$PARTITION" \
         --gpus-per-node="$GPUS_PER_NODE" \
         --ntasks-per-node="$NTASKS_PER_NODE" \
         --cpus-per-task="$CPUS_PER_TASK" \
         --mem-per-gpu="$MEM_PER_GPU" \
         --time="$TIME" \
         --nodes="$NODES" \
         singularity exec "$CONTAINER_PATH" \
         bash -c "cd $PROJECT_DIR; \$WITH_CONDA; python pp_tester_single.py --config '$CONFIG_FILE' --architecture llama --test-folder '$TEST_FOLDER' --log-file '$OUTPUT_CSV' --ckpt_path '$CKPT_DIR' --tmp-path '$TMP_PATH' --neox-path '$PROJECT_DIR'" &

    # Brief pause to prevent overwhelming the system
    # sleep 1
done

# Wait for all background srun jobs to finish
wait

# Log script completion
echo "[INFO] All checkpoint evaluations submitted."
