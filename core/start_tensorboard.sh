#!/bin/bash
# 说明，脚本在物理机上启动，而非在docker container中启动!!!

LOG_FILE="tensorboard_log_$(date +%Y%m%d_%H%M%S).log"
# 获取脚本的绝对路径
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
logdir_suffix=${SCRIPT_DIR}/log_dir
nohup tensorboard --host 0.0.0.0 --logdir=${logdir_suffix} --reload_multifile=True --reload_interval=5 > "$LOG_FILE" 2>&1 &