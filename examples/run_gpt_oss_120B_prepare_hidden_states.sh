#!/usr/bin/env bash
set -euo pipefail

# ----- Persistent, explicit paths -----
export DATAPATH="/root/.cache/user_artifacts/gpt-oss-120b/dataset/perfect-blend-gptoss-20B-1M.jsonl"
export CACHE_DIR="/root/.cache"
export OUTPUT_PATH="/root/data/yilian/data/gpt_oss_120B"
export MODEL="openai/gpt-oss-120B"
export CHAT_TEMPLATE="gpt-oss"
export TP=4

# ----- Sanitize distributed/RDZV env (prevents connecting to remote host like g258.voltagepark.net) -----
unset MASTER_ADDR || true
unset MASTER_PORT || true
unset TORCHELASTIC_RUN_ID || true
unset etcd_master_addr || true

# ----- GPU binding & NCCL safety knobs -----
# Select the exact GPUs you want this job to use.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Helpful debug; restrict NCCL to local loopback unless you set a real NIC (e.g., eth0)
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# ----- Run -----
uv run -- \
torchrun --standalone --nproc_per_node=4 \
         --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:0 \
  scripts/prepare_hidden_states.py \
  --data-path "$DATAPATH" \
  --cache-dir "$CACHE_DIR" \
  --output-path "$OUTPUT_PATH" \
  --model "$MODEL" \
  --chat-template "$CHAT_TEMPLATE" \
  --tp "$TP" \
  --enable-aux-hidden-states \
  --dist-timeout 7200 \
  --aux-hidden-states-layers 1,17,33
#  --num-samples 10