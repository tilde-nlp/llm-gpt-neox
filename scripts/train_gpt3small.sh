mkdir logs
python3 prepare_dataset.py 
NCCL_SHM_DISABLE=1 NCCL_DEBUG=info MASTER_ADDR=127.0.0.1 MASTER_PORT=2000 deepspeed train.py --deepspeed --deepspeed_config configs/base_deepspeed.json
