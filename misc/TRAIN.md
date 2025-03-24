## Introduction

This pipeline assumes data has been properly **upsampled** and is **sufficient**

## Slice data [done]

### Create slice file
This is the file describing the proportions for each language. 
```
This excercise is left to the reader ;^)
```

### Create state file
Used by the slicer for indexing into the bin file.
```bash
python3 create_0_state_file.py
```

### Run slicer
Takes ths state file and slice file, training phase proportions, and other info to produce unmerged data packs.
```bash
python3 make_data_packs.py
--tokens-per-iter 4718592 
--warmup-iters 2000 
--cd_phase 20 
--max_tokens_per_pack 165150720000 
--out_dir /scratch/project_465001281/tokenized/final_data_sliced 
--slices_json /scratch/project_465001281/tokenized/final_data_sliced/slices_final.json 
--state_file /scratch/project_465001281/tokenized/final_data_sliced/state.0.yaml
```

## Training

### Merge data pack for phase XX

All sliced data is stored per phase in:

```
/scratch/project_465001281/tokenized/final_data_sliced
│
├── warmup_0
│   ├── arxiv1_slice0_0_933600.bin
│   ├── arxiv1_slice0_0_933600.idx
│   ├── etc ...
│
└── U1_1
    ├── arxiv1_slice1_132_13905583.bin
    ├── arxiv1_slice1_132_13905583.idx
    ├── etc ...

```

Merged datapacks per phase are stored in:

```
/scratch/project_465001281/tokenized/final_data_sliced/merged
│
├── warmup_0
│   ├── warmup_0.bin
│   ├── warmup_0.idx
│
│
└── U1_1
    ├── U1_1.bin
    ├── U1_1.idx

```

To create a merged datapack from sliced data for a phase XX (i.e. training ready):

```bash
cd /project/project_465001281/IP/llm-gpt-neox/misc
source purge.sh
bash merge_single_folder.sh \
    /scratch/project_465001281/tokenized/final_data_sliced/XX \
    /scratch/project_465001281/tokenized/final_data_sliced/merged/XX

```
Output (~ 5-10 min) : 
- /scratch/project_465001281/tokenized/final_data_sliced/merged/XX/XX.bin
- /scratch/project_465001281/tokenized/final_data_sliced/merged/XX/XX.idx
- /scratch/project_465001281/tokenized/final_data_sliced/merged/XX/XX.merge.log


**/scratch/project_465001281/tokenized/final_data_sliced/merged/XX/XX.bin** should then be passed to the training script


### Estimate num iterations for phase XX

To estimate what value to use for *train_iters* for all merged phases:

```bash
bash iters_per_datapack.sh \
    /scratch/project_465001281/tokenized/final_data_sliced/merged
```

Output:
```
/scratch/project_465001281/tokenized/final_data_sliced/merged/U1_1/: 29789.3
/scratch/project_465001281/tokenized/final_data_sliced/merged/warmup_0/: 2000.3
/scratch/project_465001281/tokenized/final_data_sliced/merged/XX/: 1337.69
```

***NOTE***: estimation is hardcoded for 8192 * 576 (seq_len * batch_size) - this will not work for different seq_len or batch sizes.


### Create phase train folder & config

```bash
cd /project/project_465001281/IP/llm-gpt-neox/launch_scripts/final_train
mkdir XX
cd XX
cp ../../launch_30_SOTA_example.sh .
cp ../../launch_pp_tester.sh .
cp ../../../schedule.sh .
cp ../../../train_configs/30B_SOTA_U1_1.yml 30B_SOTA_XX.yml
```

Change/check the following keys in the XX yml config:

```yaml
# NOTE: no .bin extension
"train_data_paths": ["/scratch/project_465001281/tokenized/final_data_sliced/merged/XX/XX"],

# LR stuff
"lr_decay_style": "constant",
"warmup": 0,

# Datapack change stuff
"train_iters": 29790, # IMPORTANT: 2000 for warmup, {size of current datapack} otherwise (usually ~35 000)
"iteration_offset": 2000, # IMPORTANT: name of the last checkpoint of previous datapack:
                             # e.g. we trained N_1 till global_step60000, we are now starting N_2, so we need to set offset to 60 0000

# more LR stuff
"override_lr_scheduler": true, # false for warmup

# checkpoint stuff
"checkpoint_factor": 450, # IMPORTANT: this should be ~ 2 h of train, use 17s/iter estimation
"exit_interval": 8550, # IMPORTANT: (make sure this is multiple of checkpoint factor) exit every ~ 40 h
"keep_last_n_checkpoints": 100000, # IMPORTANT: unlimited basically, manage manually
"save": "/scratch/project_465001281/MK/checkpoints/final_train_v2",
"load": "/scratch/project_465001281/MK/checkpoints/final_train_v2",

```

Change #SBATCH --job-name (optionally) and edit **CMD command** in *launch_30_SOTA_example.sh* to point to your .yml file:

```bash
#SBATCH --job-name=phaseXX
#This command will tell deepy.py to run training with the config 00_example.yml.
CMD="$NEOX_DIR/deepy.py \
  $NEOX_DIR/train.py \
  $NEOX_DIR/launch_scripts/final_train/XX/30B_SOTA_XX.yml
  "
```

Change config/output paths in  *launch_pp_tester.sh* :

```bash
CONFIG_FILE="$NEOX_DIR/launch_scripts/final_train/XX/30B_SOTA_XX.yml"
OUTPUT_CSV_FOLDER="/project/project_465001281/IP/llm-gpt-neox/launch_scripts/final_train_v2"
```

### Begin training

```bash
bash schedule.sh launch_30_SOTA_example.sh {n_jobs} launch_pp_tester.sh
```

How to estimate {n_jobs}:
- n_jobs ~ ceil( {train_iters} / {exit_interval} )
- always queue extra jobs
- any extra jobs will simply load, eval and re-save the last checkpoint (~10 min?)

If anything crashes mid datapack, simply rerun the launch command (prolly adjust {n_jobs}) and it should 
automagically continue from latest checkpoint. **NOTE: you do NOT need to change "iteration_offset" or "train_iters"** 


