#!/bin/bash
name="stable_latte_256_bsz_1_steps_160K_drop"
# nohup
XLA_PYTHON_CLIENT_MEM_FRACTION=.999 python -u $ROOT_DIR/experiments/lm.py \
    --base_dir $ROOT_DIR/data/ \
    --config_file  $ROOT_DIR/experiments/config/lm_scale_enwik.yaml\
    --name $name \
   #>$ROOT_DIR/'data/logs_latte/debug/'$name'_'$BASHPID'.log' 2>&1 &
