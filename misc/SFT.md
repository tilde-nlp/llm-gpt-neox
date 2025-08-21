# SFT on LUMI v1



## 1. Prepare data on locally or on LUMI

### Data format (pre-tokenization)

The training data must be a `.jsonl` file where each line is a JSON object with a `text` field.  
Dialogues are wrapped with control tokens (subject to change):

```aiignore
{"text": "<<<|user|>>>{Prompt}{Input}<<<|user|>>><<<|assistant|>>>{Output}<<<|assistant|>>>"}
```

Example — **English → Latvian translation**:

```json
{"text": "<<<|user|>>>Translate the following sentence from English to Latvian: Hello, how are you?<<<|user|>>><<<|assistant|>>>Sveiki, kā jums klājas?<<<|assistant|>>>"}
```
Ideally you should make sure that regex cannot match <<<|x|>>> in your training text, 
otherwise those matches will be treated as control tokens during tokenization.

NOTE: The real tokens that tokenizer will use are <|x|>, not <<<|x|>>>, the latter is used in non-tokenized format 
just so that there is a smaller chance of unintended regex matches.


## 2. Upload data to LUMI

scp -r -i ssh_key path/to/jsonl user@lumi.csc.fi:/scratch/project_465001281/path/to/jsonl

## 3. Tokenize on LUMI

Since control tokens (e.g., `<<<|user|>>>`, `<<<|assistant|>>>`) cannot be mapped directly by the SentencePiece tokenizer,  
they must be **manually inserted** into the tokenized sequence.  

`tools/datasets/preprocess_sft.py` does the following:
- Extracts the `text` field from each JSONL entry
- Splits text on the `<<<|x|>>>` markers
- Tokenizes the spans in between
- Inserts the correct control token IDs at the boundaries

Currently, the script has these tokens hard-coded:

```python
self.special_tokens = ["<<<|user|>>>", "<<<|assistant|>>>"]
self.special_ids    = [7, 8]
```
From repo root:
```bash
# Clean up environment, mount stuff
source misc/purge.sh

# Start interactive shell on LUMI
srun \
  --job-name=test-ft \
  --account=project_465001281 \
  --partition=dev-g \
  --gpus-per-node=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=56 \
  --mem-per-gpu=0 \
  --time=00:45:00 \
  --nodes=1 \
  singularity shell /scratch/project_465001281/containers/rocm603_flash.sif

# init python
$WITH_CONDA

# run tokenization
cd tools/datasets
python preprocess_sft.py \
  --input /scratch/project_465001281/path/to/data.jsonl \
  --tokenizer-type SPMTokenizer \
  --vocab-file /scratch/project_465001281/tokenizers/4B_Final/model.model \
  --append-eod \
  --output-prefix /scratch/project_465001281/path/to/output_prefix \
  --workers 8

```
Expected output:

- /scratch/project_465001281/path/to/output_prefix_text_document.bin
- /scratch/project_465001281/path/to/output_prefix_text_document.idx

You can then use this in the training yaml config:

```yaml
# NOTE: no .bin extension
"train_data_paths": ["/scratch/project_465001281/path/to/output_prefix_text_document"]
```

### Optional debug

Inspect first 100 documents in text, tokenized, and token id format:

```bash
# purge ...
# srun ...

python vis.py /scratch/project_465001281/path/to/output_prefix_text_document.bin /scratch/project_465001281/tokenizers/4B_Final/model.model
```

For each sample in .bin file count tokens and output the count to file:
```bash
# purge ...
# srun ...

python count_tokens.py /scratch/project_465001281/path/to/output_prefix_text_document.bin /path/to/out.txt
```

## 4. Launch training

### Preparation

Probably want to use the following NEOX DIR on LUMI: /project/project_465001281/IP/debug-llm-gpt-neox

Make sure you are on 'main' branch, if using git.

```bash
# setup train folder
mkdir /project/project_465001281/IP/debug-llm-gpt-neox/launch_scripts/finetune_XXX

# copy draft configs and helpers into the new train folder
cp /project/project_465001281/IP/debug-llm-gpt-neox/launch_scripts/finetune_latest_6/30B_SOTA_finetune_latest_6.yml 30B_SOTA_finetune_latest_XXX.yml
cp /project/project_465001281/IP/debug-llm-gpt-neox/launch_scripts/finetune_latest_6/launch_30_SOTA_example.sh .
cp /project/project_465001281/IP/debug-llm-gpt-neox/launch_scripts/finetune_latest_6/pp.sh .
cp /project/project_465001281/IP/debug-llm-gpt-neox/launch_scripts/finetune_latest_6/schedule.sh .
```

Now you must decide from which checkpoint you want to start finetuning and create a new checkpoint folder.
The new checkpoint folder will serve as load/save for checkpoints during training. 
It should contain symlink to the desired starting checkpoint and a file named 'latest' that contains the name of that checkpoint.
The folder must be on /scratch/.
E.g.

```bash
user @uan01:/scratch/project_465001281/MK/checkpoints/LATEST_translate_sft_v1> ls -la
total 3540
drwxrws--- 22 user project_465001281   4096 Aug 13 23:21 .
drwxrws--- 37 user project_465001281   4096 Aug 19 09:32 ..
lrwxrwxrwx  1 user project_465001281     35 Aug 13 17:54 global_step423858 -> ../final_train_v2/global_step423858
-rw-rw----  1 user project_465001281     17 Aug 13 23:21 latest

user@uan01:/scratch/project_465001281/MK/checkpoints/LATEST_translate_sft_v1> cat latest
global_step423858
user@uan01:/scratch/project_465001281/MK/checkpoints/LATEST_translate_sft_v1>
```

***NOTE***: Our current end of CD (i.e. final) checkpoints are:

- No layer norm average: /scratch/project_465001281/MK/checkpoints/final_train_v2/_global_step_423858_backup
- With averaged layer norms: /scratch/project_465001281/MK/checkpoints/final_train_v2/_global_step423858

### Setup the configs

- ***launch_30_SOTA_example.sh***

Must point the correct (new) train folder:

```bash
#This command will tell deepy.py to run training with the config 00_example.yml.
CMD="$NEOX_DIR/deepy.py \
  $NEOX_DIR/train.py \
  $NEOX_DIR/launch_scripts/finetune_latest_XXX/30B_SOTA_finetune_latest_XXX.yml
  "
```
The two (legacy) flags must be set to 1:

```bash
export CURSE_ATTENTION=1 #FIXME: remember this, 1 = masked attentions, 0 = poorly masked attentions
export SFT_ENABLED=1 # FIXME: remember this, 1 = isntruction loss masking, 0 = no instruction loss masking
```

Optionally change the job name and adjust sbatch parameters:

```bash
#SBATCH --account project_465001281
#SBATCH --partition standard-g
#SBATCH --exclusive=user
#SBATCH --nodes=192
#SBATCH --gpus-per-node=mi250:8
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --job-name=ft_latest_XXX
```

- ***30B_SOTA_finetune_latest_XXX.yml***

Change/check the following keys in the XXX yml config:

```yaml
# NOTE: no .bin extension
"train_data_paths": ["/scratch/project_465001281/path/to/output_prefix_text_document"],
  
"finetune": true, # setting this to true will discard optimiser state,
                  # it will also allow to change model parallelism/sharding (i.e. num nodes)
                  # NOTE: if you want to RESUME training (e.g. bcs of crash), this flag should be false

# optimiser stuff
"optimizer":
    {
      "type": "Adam",
      "params": { "lr": 8.0e-6, "betas": [0.9, 0.95], "eps": 1.0e-8 }, # this is our final CoolDown LR
    },

"clip_grad": 1.0, # feel free to change, one of these keys is useless, don't know which
"gradient_clipping": 1.0, # feel free to change, one of these keys is useless, don't know which


# Batch settings - adjust as needed
"train_micro_batch_size_per_gpu": 3,
"train_batch_size": 192,
"gradient_accumulation_steps": 1, # I would NOT touch this one

# important 
"train_iters": 1337, # how many iterations to do on the given .bin file
"iteration_offset": 423858, # IMPORTANT: name of the finetune starting checkpoint
                            # IMPORTANT: i am not 100% certain if this is strictly necessary when finetune=true, 
                            #            but better use it

# VERY important
"pack_impl": "k_bin_packed", # for SFT this MUST be 'k_bin_packed', otherwise training samples will be split/catted

# checkpoint load/save
"checkpoint_factor": 100, # dealers choice
"exit_interval": 8550, # probably irrelevant for SFT, set this to be > train_iters
"keep_last_n_checkpoints": 100000, # IMPORTANT: unlimited basically, manage manually
"save": "/scratch/project_465001281/path/to/your/ckpt_folder", 
"load": "/scratch/project_465001281//path/to/your/ckpt_folder", # IMPORTANT: contains finetune start ckpt and 'latest' file

```

### Launch training

```bash
cd /project/project_465001281/IP/debug-llm-gpt-neox/launch_scripts/finetune_XXX
bash schedule.sh launch_30_SOTA_example.sh 1 pp.sh
```

***NOTE***: do not schedule more than 1 job when using finetune=true. 
To resume a failed training that has produced at least 1 new checkpoint, 
set finetune=false and then you can schedule as many jobs as you want.

***NOTE***: the scheduler will complain about pp.sh (i.e. that job won't be scheduled), that is normal and desired

## 5. Convert checkpoint to HF

TODO

## 6. Debug inference on LUMI

TODO