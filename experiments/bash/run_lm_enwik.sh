#!/bin/bash
name="enwiki_extend2_latte"
# CUDA_VISIBLE_DEVICES=1
XLA_PYTHON_CLIENT_MEM_FRACTION=.99 nohup python -u $ROOT_DIR/experiments/lm.py \
    --base_dir $ROOT_DIR/data/ \
    --config_file  $ROOT_DIR/experiments/config/lm_scale_enwik.yaml \
    --name $name \
    >$ROOT_DIR/'data/logs_latte/'$name'_'$BASHPID'.log' 2>&1 &
