#!/bin/bash
name="lra_path32_bid_l256"
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=.99 nohup python -u $ROOT_DIR/experiments/lra.py \
    --base_dir $ROOT_DIR/data/ \
    --config_file  $ROOT_DIR/experiments/config/lra_pathf128.yaml \
    --name $name \
   >$ROOT_DIR/'data/logs_latte/'$name'_'$BASHPID'.log' 2>&1 &
