#!/bin/bash

# Ensure correct usage
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <root_ckpt_dir> <config_file> <test_folder>"
    exit 1
fi

# Assign input arguments
ROOT_CKPT_DIR=$1
CONFIG_FILE=$2
TEST_FOLDER=$3

# SLURM parameters
ACCOUNT="project_465001281"
PARTITION="standard-g"
GPUS_PER_NODE=1
NTASKS_PER_NODE=1
CPUS_PER_TASK=7
MEM_PER_GPU="60G"
TIME="01:00:00"
NODES=1
CONTAINER_PATH="/scratch/project_465001281/containers/rocm603_flash.sif"
PROJECT_DIR="/project/project_465001281/llm-gpt-neox"

# Iterate through each checkpoint folder in the root checkpoint directory
for CKPT_DIR in $(ls -d "$ROOT_CKPT_DIR"/global_step* 2>/dev/null | sort -V); do
    # Extract iteration number
    ITERATION=$(basename "$CKPT_DIR" | sed 's/global_step//')

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
         bash -c "cd $PROJECT_DIR; \$WITH_CONDA; python pp_tester_single.py --config '$CONFIG_FILE' --architecture llama --test-folder '$TEST_FOLDER' --log-file '$OUTPUT_CSV' --ckpt_path '$CKPT_DIR' --tmp-path /scratch/project_465001281/MK/tmp" &

    # Brief pause to prevent overwhelming the system
    # sleep 1
done

# Wait for all background srun jobs to finish
wait

echo "All checkpoint evaluations submitted."
