#!/usr/bin/env bash
set -ex
uv run torchrun --nnodes=2 --nproc_per_node=1 --node_rank=$([ "$(hostname)" = "pinky" ] && echo 0 || echo 1) --master_addr=169.254.239.145 --master_port=29500 main.py
