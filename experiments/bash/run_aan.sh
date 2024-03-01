#!/bin/bash
name="lra_aan_bid_mean_5e_5_20_epochs_no_logits_no_kast_norm" # "test_aan" #"lra_aan_bid_cos_256_last_tok"
# CUDA_VISIBLE_DEVICES=1
#XLA_PYTHON_CLIENT_MEM_FRACTION=.99 CUDA_VISIBLE_DEVICES=0 
XLA_PYTHON_CLIENT_MEM_FRACTION=.99 nohup python -u $ROOT_DIR/experiments/lra.py \
    --base_dir /mnt/data/ --config_file  $ROOT_DIR/experiments/config/lra_aan.yaml \
    --name $name \
    >$ROOT_DIR/'data/logs_latte/'$name'_'$BASHPID'.log' 2>&1 &
