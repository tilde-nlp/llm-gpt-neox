## Introduction

This pipeline assumes data has been properly **upsampled** and is **sufficient**

## Slice data

### Create slice file

```
This excercise is left to the reader ;^)
```

### Create state file

```
python3 create_0_state_file.py
```

### Run slicer

```
python3 make_data_packs.py
--tokens-per-iter 4194304 
--warmup-iters 2000 
--cd_phase 20 
--max_tokens_per_pack 125829120000 
--out_dir /scratch/project_465001281/tokenized/final_data_sliced 
--slices_json /scratch/project_465001281/tokenized/final_data_sliced/slices_final.json 
--state_file /scratch/project_465001281/tokenized/final_data_sliced/state.0.yaml
```
