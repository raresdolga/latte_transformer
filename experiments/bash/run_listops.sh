#!/bin/bash
name="lra_listops_bid_L40_cosine"
# CUDA_VISIBLE_DEVICES=1
XLA_PYTHON_CLIENT_MEM_FRACTION=.99 nohup python -u $ROOT_DIR/experiments/lra.py \
    --base_dir $ROOT_DIR/data/ \
    --config_file  $ROOT_DIR/experiments/config/listops.yaml \
    --name $name \
    >$ROOT_DIR/'data/logs_latte/'$name'_'$BASHPID'.log' 2>&1 &
