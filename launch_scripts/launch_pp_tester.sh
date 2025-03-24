#!/bin/bash

#SBATCH --account project_465001281
#SBATCH --partition standard-g
#SBATCH --exclusive=user
#SBATCH --nodes=1
#SBATCH --gpus-per-node=mi250:8
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --hint=nomultithread
#SBATCH --job-name=pp_bench
#Comments:
#  partition - dev-g when debugging on up to 32 nodes. standard-g when working with more nodes.
#  exclusive=user - Slurm will alocate us full nodes.
#  gpus-per-node=mi250:8 - use all GPUs on the nodes.
#  tasks-per-node=8 - will launch one process per GPU.
#  cpus-per-task - 7 is the maximum that doesn't crash.
#  mem=0 - unlimited RAM. 512GB RAM per node.
#  hint=nomultithread - make it so that no cpu has more than one process running on it at the same time.

set -euo pipefail

export CC=gcc-12
export CXX=g++-12

#Don't understand, maybe necessary.
export MEMORY_OPT_ALLREDUCE_SIZE=100000000

#Don't understand, but necessary.
export CUDA_DEVICE_MAX_CONNECTIONS=1

#Setting up torch distributed env parameters.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=1337
export WORLD_SIZE=$SLURM_NTASKS

#Don't fully understand.
#Necessary for faster internode communication.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

#Don't understand. Necessary for faster internode communication, I think.
export OMP_NUM_THREADS=1

#Don't understand. Necessary for faster memory access between nodes, I think.
export NCCL_NET_GDR_LEVEL=PHB

#A variety of debug output flags.
#Hasn't really helped me much.
#export NCCL_DEBUG=INFO
#export RCCL_KERNEL_COLL_TRACE_ENABLE=1
#export NCCL_DEBUG_SUBSYS=INIT,COLL

#Activate a virtual environment for launching gpt-neox's slurm launcher.
#Load some lmod modules. Not sure if all were necessary.
module purge
module load CrayEnv
module load cray-python/3.9.13.1
#module load rocm/6.2.2
module load gcc/12.2.0
source /scratch/project_465001281/containers/launch_conda_env.sh

#Tell singularity to bind /project /scratch /flash folders etc.
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems

#Not sure if this is necessary.
#I think this was copied over from one of SILO.AIs configs.
mkdir -p workdir
wd=$(realpath workdir)
if [ ! -d "$wd"/cray-deps ] ; then
  rm -rf "$wd"/cray-deps
  mkdir "$wd"/cray-deps
  cp /usr/lib64/libcxi* $wd/cray-deps
fi

#Generating a file so deepy.py knows the available resources.
#(deepy.py is the gpt-neox launcher script.)
GPUS_PER_NODE=1
mkdir -p ./hostfiles
hostfile=./hostfiles/hosts_$SLURM_JOBID
# loop over the node names
for i in `scontrol show hostnames $SLURM_NODELIST`
do
    # add a line to the hostfile
    echo $i slots=$GPUS_PER_NODE >>$hostfile
done
export DLTS_HOSTFILE=./hostfiles/hosts_$SLURM_JOBID

NEOX_DIR="/project/project_465001281/IP/llm-gpt-neox"

# IMPORTANT: change these
CONFIG_FILE="$NEOX_DIR/launch_scripts/final_train/U1_1/30B_SOTA_U1_1.yml"
OUTPUT_CSV_FOLDER="/project/project_465001281/IP/llm-gpt-neox/launch_scripts/final_train_v2"

# IMPORTANT: most likely dont change these
TEST_FOLDER="/scratch/project_465001281/MK/data"
TMP_PATH="/scratch/project_465001281/MK/tmp"
CONTAINER_PATH="/scratch/project_465001281/containers/rocm603_inference.sif"


#This command will tell deepy.py to run training with the config 00_example.yml.
CMD="python pp_tester_sophisticated.py \
--config $CONFIG_FILE \
--architecture llama \
--test-folder $TEST_FOLDER \
--log-file $OUTPUT_CSV_FOLDER \
--tmp-path $TMP_PATH \
--neox-path $NEOX_DIR
"

srun singularity exec "$CONTAINER_PATH" \
  bash -c "cd $NEOX_DIR; \$WITH_CONDA; $CMD"