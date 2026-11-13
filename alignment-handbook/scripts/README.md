# Scripts to Train  Models


### SFT 

Train with deepspeed offloading
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes 1 scripts/sft.py  --config recipes/sft_cte/qwen3-0.6b-savisql.yaml
```

Or train only GPU if have more VRAM

```shell
python scripts/sft.py  --config recipes/sft_cte/qwen3-0.6b-savisql.yaml
```

